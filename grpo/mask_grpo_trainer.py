import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
import wandb
import math
from models.sampling import mask_by_random_topk
from training.prompting_utils import UniversalPrompting
from transformers import AutoTokenizer

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def linear_schedule(t):
    mask_ratio = 1 - t
    mask_ratio = mask_ratio.clamp(min=1e-6, max=1.0)
    return mask_ratio

def sample_categorical(categorical_probs):
    # A simple sample function based on probability distribution
    *sample_shape, C = categorical_probs.shape
    return torch.multinomial(categorical_probs.reshape(-1, C), num_samples=1).reshape(*sample_shape)

def get_cur_masks(sampled_ids, num_block, block_length, mask_id, shift):
    """
    Returns a boolean mask of shape [bs, L] where:
    - positions are within the block defined by `shift`, `num_block`, and `block_length`, AND
    - values at those positions are equal to `mask_id`
    
    Args:
        sampled_ids (Tensor): Tensor of shape [bs, L]
        num_block (int): Index of the current block
        block_length (int): Length of each block
        mask_id (int): The ID to match
        shift (int): Starting offset (e.g., input_ids.shape[1])
    
    Returns:
        Tensor: Boolean tensor of shape [bs, L]
    """
    bs, L = sampled_ids.shape
    start = shift + num_block * block_length
    end = shift + (num_block + 1) * block_length

    positions = torch.arange(L, device=sampled_ids.device)
    block_mask = (positions >= start) & (positions < end)  # shape: [L]

    return block_mask.unsqueeze(0) & (sampled_ids == mask_id)  # shape: [bs, L]

if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

def pref_score(scores: torch.Tensor) -> torch.Tensor:
    # scores: [batch_size, num_items]
    # return 1 for the best, 0 for the worst.
    ranks = scores.argsort(dim=1, descending=True).argsort(dim=1)  # rank 0 = best
    num_items = scores.size(1)
    normalized_scores = (num_items - 1 - ranks).float() / (num_items - 1)
    return normalized_scores.view(-1)

class MaskGRPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Discrete Diffusion Models.

    This class is based on diffu-GRPO, which extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities with masked tokens.
    MaskGRPO introduce additional mask designs and handles the advantage calculation and update
    among multiple devices.

    Key features:
    - Random or AR-like masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
        uni_processing_class: Any = None,
        vq_model: Any = None,
    ):
        def data_collator(features):
            # Override the original GRPOTrainer collator, in case of interleaved data.
            if "images" in features[0]:
                batch={}
                # batch["images"] = torch.stack([f["images"] for f in features if "images" in f])
                # batch = [{"prompts": f["prompts"][0]} for f in features if "prompts" in f]
                breakpoint()
                return batch
            else:
                return features
            
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self.data_collator = data_collator
        self.uni_prompting = uni_processing_class
        self.version = args.version
        self.remask_method = args.remasking
        self.task = args.task
        self.vq_model = vq_model
        self.mean_method = args.mean_method
        self.use_pref = args.pref_reward
        
        print(f"The current get_logp method is {self.version} with task {self.task}.")
        print(f"The current loss calculation method is {self.mean_method}.")
        if self.use_pref:
            print(f"The current reward is ranked rather than original score.")

    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
                            # Handle classifier-free guidance more efficiently
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)

                                # Get logits in a single forward pass
                                logits = model(x_).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(x).logits

                            # Apply Gumbel noise for sampling
                            logits_with_noise = self.add_gumbel_noise(
                                logits, temperature=temperature, dtype=dtype
                            )
                            x0 = torch.argmax(logits_with_noise, dim=-1)
                            del logits_with_noise

                            # We set it as default and use the remasking args to get logps
                            p = F.softmax(logits.to(dtype), dim=-1)
                            x0_p = torch.squeeze(
                                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                            )
                            # if remasking == "low_confidence":
                            #     p = F.softmax(logits.to(dtype), dim=-1)
                            #     x0_p = torch.squeeze(
                            #         torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                            #     )
                            # elif remasking == "random":
                            #     x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                            # else:
                            #     raise NotImplementedError(remasking)

                            # Ensure we don't process tokens beyond the current block
                            x0_p[:, end_idx:] = -np.inf

                            # Update masked tokens
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            # Select tokens to transfer based on confidence
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(confidence.shape[0]):
                                num_tokens = num_transfer_tokens[j, i].item()
                                if num_tokens > 0:
                                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                                    transfer_index[j, select_index] = True

                            x[transfer_index] = x0[transfer_index]
                            del x0, confidence, transfer_index

            return x
        
    def generate_v2(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        """
        A low-discrepancy emerge sampler modified by MaskGRPO.
        Also see the original ReDDiT (MDLM style) code:
        https://github.com/martian422/ReDDiT/blob/main/diffusion.py#L565
        """
        with torch.amp.autocast('cuda', enabled=True):
            noise_schedule = linear_schedule # tbd
            batch_size = prompt.shape[0]
            dtype = model.dtype
            sampled_ids = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(prompt.device)
            sampled_ids[:, :prompt.shape[1]] = prompt.clone() # only copy the inputs
            prompt_index = (sampled_ids != mask_id)

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            per_block_steps = max(1, steps // num_blocks)

            for num_block in range(num_blocks):

                update_masks = get_cur_masks(sampled_ids, num_block, block_length, mask_id, shift = prompt.shape[1])

                for step in range(per_block_steps):
                    torch.cuda.empty_cache()

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.amp.autocast('cuda', enabled=self.args.fp16):
                            # Handle classifier-free guidance more efficiently
                            if cfg_scale > 0.0:
                                un_cond = sampled_ids.clone()
                                un_cond[prompt_index] = mask_id
                                x_ = torch.cat([sampled_ids, un_cond], dim=0)
                                logits = self(x_).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(sampled_ids).logits

                            probs = logits.softmax(dim=-1)

                            if step < per_block_steps - 1:
                                k_t = noise_schedule(torch.tensor(1.0 * (step) / per_block_steps))
                                k_s = noise_schedule(torch.tensor(1.0 * (step + 1) / per_block_steps))
                                probs = probs * (k_t - k_s)
                                p_mask = k_s / k_t # for uni-mask tokens
                                probs[:, :, mask_id] = p_mask

                                new_pred_with_masks = sample_categorical(probs)

                                sampled_ids = torch.where((sampled_ids==mask_id) & update_masks, new_pred_with_masks, sampled_ids)
                                # sampled_ids = updated_sampled_ids
                            else:
                                new_pred = sample_categorical(probs)
                                sampled_ids = torch.where((sampled_ids==mask_id) & update_masks, new_pred, sampled_ids)
                            # print(f'decoded tokens:{(sampled_ids < codebook_size).sum().item()}')

            return sampled_ids
    @torch.no_grad()    
    def t2i_generate(
            self,
            model,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=12,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = self.uni_prompting
        vocab_shift = len(uni_prompting.text_tokenizer) + num_new_special_tokens
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)
        to_cycle_id = input_ids.clone()

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, to_cycle_id[:, resolution + 1:]], dim=1)
                model_input = torch.cat([to_cycle_id, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = model(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, vocab_shift: vocab_shift + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = model(to_cycle_id, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, vocab_shift: vocab_shift + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            if step < timesteps - 1:
            # Masks tokens with lower confidence.
                to_cycle_id[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                            sampled_ids + len(uni_prompting.text_tokenizer)
                                                            + num_new_special_tokens)
                input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
            else:
                to_cycle_id[:, -(num_vq_tokens + 1):-1] = sampled_ids + len(uni_prompting.text_tokenizer) + num_new_special_tokens
                input_ids_minus_lm_vocab_size = sampled_ids

        return to_cycle_id
    @torch.no_grad()
    def t2i_generate_emerge(
            self,
            model, 
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=12,  
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            **kwargs,
    ):
        """
        A low-discrepancy emerge sampler modified by MaskGRPO.
        """

        # begin with all image token ids masked
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = self.uni_prompting
        vocab_shift = len(uni_prompting.text_tokenizer) + num_new_special_tokens
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        sampled_ids = input_ids[:, -(num_vq_tokens + 1):-1].clone() # all equals to mask_token_ids.
        to_cycle_ids = input_ids.clone() # avoid manipulating the original input_ids.
        # this results in the uni-mask canvas

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if guidance_scale > 0 and uncond_input_ids is not None:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, to_cycle_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([to_cycle_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = model(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, vocab_shift: vocab_shift + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = model(to_cycle_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, vocab_shift: vocab_shift + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            if step < timesteps - 1:
                k_t = noise_schedule(torch.tensor(1.0 * (step) / timesteps))
                k_s = noise_schedule(torch.tensor(1.0 * (step + 1) / timesteps))
                p_mask = (k_s / k_t).expand(probs.shape[0], probs.shape[1], 1).to(probs.device) # for uni-mask tokens

                probs_with_mask_index = torch.cat([probs * (k_t - k_s) / k_t, p_mask], dim=-1)
                new_pred_with_masks = sample_categorical(probs_with_mask_index)

                updated_sampled_ids = torch.where(sampled_ids > codebook_size -1, new_pred_with_masks, sampled_ids)
                # at the first pass, the sampled_ids transfer into the codebook (8192) range, then it works as expected.
                sampled_ids = updated_sampled_ids
                sampled_ids_to_paste = torch.where(sampled_ids > codebook_size -1, mask_token_id - vocab_shift, updated_sampled_ids)
                to_cycle_ids[:, -(num_vq_tokens + 1):-1] = sampled_ids_to_paste + vocab_shift
            else:
                new_pred = sample_categorical(probs)
                sampled_ids_to_paste = torch.where(sampled_ids > codebook_size -1, new_pred, sampled_ids)
                to_cycle_ids[:, -(num_vq_tokens + 1):-1] = sampled_ids_to_paste + vocab_shift
        return to_cycle_ids

    def forward_process(self, batch, prompt_index, mask_id, seed=None):
        """
        Create a noisy batch where prompt tokens are randomly masked at ratio=p_mask. 
        The completions are always all-masked.
        The original d1 style implementation of remasking to get logps.
        """
        set_seed(seed.item())
        b, l = batch.shape
        t_p = torch.ones(b, device=batch.device) * self.args.p_mask_prompt

        # Create a random matrix to decide whether each prompt token is masked
        random_matrix = torch.rand((b, l), device=batch.device)

        # For prompt tokens: mask if random_matrix < t_p
        # For completion tokens: always mask
        is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index  # all completion tokens are masked
        is_mask = is_mask_prompt | is_mask_completion

        # Create a noisy (masked) batch
        noisy_batch = torch.where(is_mask, mask_id, batch)

        # Build p_mask, the probability that each token is masked under this scheme
        #   - p_mask[i, j] = t_p[i] if it's a prompt token
        #   - p_mask[i, j] = 1      if it's a completion token
        p_mask = torch.where(
            prompt_index,
            t_p.unsqueeze(1),  # prompt token probability
            torch.ones_like(t_p).unsqueeze(1),  # completion token probability
        )

        return noisy_batch, p_mask
    
    def remask(self, batch, prompt_index, mask_id, seed=None, ratio = 0.0):
        """
        Control the remask methods under uni settings.
        """
        if self.task == 't2t':
            if self.remask_method == 'random':
                # falls back to UniGRPO's random remasking strategy.
                return self.remask_t2t(batch, prompt_index, mask_id, seed=seed, ratio = ratio)
            elif self.remask_method == 'arlike':
                # MaskGRPO's default remasking strategy for t2t.
                return self.remask_ar(batch, prompt_index, mask_id, seed=seed, ratio = ratio)
        elif self.task == 't2i':
            return self.remask_t2i(batch, prompt_index, mask_id, seed=seed, ratio = ratio)
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        
    def remask_t2t(self, batch, prompt_index, mask_id, seed=None, ratio = 0.0):
        """
        Create a noisy batch where non-prompt tokens are randomly remasked with different ratio.
        """
        # when using the same seed, the masks generated will be identical.
        set_seed(seed.item())
        b, l = batch.shape
        # t_p = torch.ones(b, device=batch.device) * ratio

        # Create a random matrix to decide whether each prompt token is masked
        random_matrix = torch.rand((b, l), device=batch.device)

        # For prompt tokens: never mask
        # For completion tokens: mask with ratio

        # prompt_index: [prompt_len * True, False]
        # -> to_mask: [prompt_len * False, ratio True]
        to_mask = ~prompt_index & (random_matrix < ratio)

        # Create a noisy (masked) batch
        noisy_batch = torch.where(to_mask, mask_id, batch)

        # Build p_mask, the probability that each token is masked under this scheme
        #   - p_mask[i, j] = t_p[i] if it's a prompt token
        #   - p_mask[i, j] = 1      if it's a completion token
        # p_mask = torch.where( ~prompt_index, ratio, 0.0)

        return noisy_batch, to_mask
    
    def remask_t2i(self, batch, prompt_index, mask_id, seed=None, ratio = 0.0):
        """
        Create a noisy batch where non-prompt tokens are randomly remasked with different ratio.
        Specially shifted for t2i tasks.
        """
        # when using the same seed, the masks generated will be identical.
        set_seed(seed.item())
        b, l = batch.shape
        # t_p = torch.ones(b, device=batch.device) * ratio

        # Create a random matrix to decide whether each prompt token is masked
        random_matrix = torch.rand((b, l), device=batch.device)

        # For prompt tokens: never mask
        # For completion tokens: mask with ratio

        # prompt_index: [prompt_len * True, False]
        # -> to_mask: [prompt_len * False, ratio True]
        to_mask = ~prompt_index & (random_matrix < ratio)

        # Create a noisy (masked) batch
        noisy_batch = torch.where(to_mask, mask_id, batch)
        # Build p_mask, the probability that each token is masked under this scheme
        #   - p_mask[i, j] = t_p[i] if it's a prompt token
        #   - p_mask[i, j] = 1      if it's a completion token
        # p_mask = torch.where( ~prompt_index, ratio, 0.0)

        return noisy_batch, to_mask
    
    def remask_ar(self, batch, prompt_index, mask_id, seed=None, ratio = 0.3):
        """
        Auto-regressive re-masking strategy:
        Tokens later in the sequence are more likely to be masked.
        Note that this function do not perform well when ratio < 0.3.
        """
        # when using the same seed, the masks generated will be identical.
        set_seed(seed.item())
        B, L = batch.shape
        device = batch.device
        prompt_len = prompt_index[0].sum().item()
        non_prompt_len = L - prompt_len
        # print(f'origin ratio:{ratio}')
        ratio = 1 - ratio # the reversed ratio works as expected.

        ar_curve = torch.linspace(1, 0, non_prompt_len, device=device)

        # Generate increasing position-based mask probability [0, 1] shape: (L,)
        ar_probs_tail = ar_curve * (ratio * non_prompt_len) / ar_curve.sum()  # shape: (L,)
        ar_probs_tail = torch.clamp(ar_probs_tail, max=1.0)

        # Expand to batch size -> shape: (B, L)
        ar_probs = torch.cat([
            torch.zeros(prompt_len, device=device),
            ar_probs_tail
        ])  # shape: (L,)
        ar_probs_batch = ar_probs.unsqueeze(0).expand(B, -1)

        # Sample random matrix
        random_matrix = torch.rand((B, L), device=device)

        # Mask where not in prompt, and prob increase along length.
        to_mask = (~prompt_index) & (random_matrix > ar_probs_batch)

        # Apply masking
        noisy_batch = torch.where(to_mask, mask_id, batch)
        # print(to_mask[0].sum().item()/256)

        return noisy_batch, to_mask

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        # if cfg_scale > 0.0 and self.task =='t2i': # FIXME
        #     # raise ValueError('Please handle this before running.')
        #     cond_logits = model(batch).logits #Now its a hack to save memory. 
        #     uncond_batch = batch.clone()
        #     uncond_batch[:,prompt_index.bool()]=126093
        #     uncond_batch[:,-1029:-1025] = torch.tensor([126088,126080,126081,126084]).to(batch.device)
        #     uncond_logits = model(uncond_batch).logits
        #     logits = (1 + cfg_scale) * cond_logits - cfg_scale * uncond_logits
        #     return logits
        # elif cfg_scale > 0.0:
        #     # we have not tested cfg_scale for t2t tasks.
        #     assert len(prompt_index) == batch.shape[1]
        #     prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        #     un_batch = batch.clone()
        #     un_batch[prompt_index] = mask_id
        #     batch = torch.cat([batch, un_batch])

        logits = model(batch).logits

        # if cfg_scale > 0.0:
        #     logits, un_logits = torch.chunk(logits, 2, dim=0)
        #     logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        # Create tensor once and modify in-place
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)
    
    def _get_per_token_logps(self, model, input_ids, logits_to_keep, mask_seeds, cur_iter=0):
        """
        Control the logp methods. We omit the TraceRL method for simplicity.
        """
        if self.version == 'd1':
            if self.task=='t2i':
                raise ValueError('Not implemented for d1-style t2i tasks!')
            else:
                return self._get_per_token_logps_d1(model, input_ids, logits_to_keep, mask_seeds, cur_iter)
        elif self.version == 'uni':
            return self._get_per_token_logps_uni(model, input_ids, logits_to_keep, mask_seeds, cur_iter)
        else:
            raise ValueError(f"Unknown version: {self.version}")

    def _get_per_token_logps_d1(self, model, input_ids, logits_to_keep, mask_seeds, cur_iter=0):
        """
        Calculate per-token log probabilities, in d1 style.
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        per_token_logps = torch.zeros(num_iterations, batch_size, logits_to_keep, device=device)

        # Verify mask_seeds length: one seed per iteration
        assert (
            len(mask_seeds) == num_iterations
        ), f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}"

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True  # Mark prompt tokens as True

        # applying masks
        all_perturbed_seqs = []
        all_expanded_inputs = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
            perturbed_seq, _ = self.forward_process(
                expanded_input, prompt_index, self.args.mask_id, seed=mask_seed
            )
            all_perturbed_seqs.append(perturbed_seq)
            all_expanded_inputs.append(expanded_input)

        # Concatenate all iterations into a single batch
        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # [num_iterations * batch_size, seq_len]
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # [num_iterations * batch_size, seq_len]

        # Get model predictions for the combined batch
        logits = self.get_logits(
            model, perturbed_seq, prompt_index, self.args.cfg_scale, self.args.mask_id
        )  # [num_iterations * batch_size, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        completion_logits = logits[
            :, -logits_to_keep:, :
        ]  # [num_iterations * batch_size, logits_to_keep, vocab_size]
        completion_targets = expanded_input[
            :, -logits_to_keep:
        ]  # [num_iterations * batch_size, logits_to_keep]
        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_targets = completion_targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        # Convert to log probabilities and reshape
        completion_log_probs = -loss.view(num_iterations * batch_size, logits_to_keep)
        per_token_logps = completion_log_probs.view(num_iterations, batch_size, logits_to_keep)

        # Clean up memory
        del perturbed_seq, logits, all_perturbed_seqs, all_expanded_inputs
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps, None
    

    def _get_per_token_logps_uni(self, model, input_ids, logits_to_keep, mask_seeds, cur_iter):
        """
        Calculate per-token log probabilities with new method.
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        per_token_logps = torch.zeros(num_iterations, batch_size, logits_to_keep, device=device)
        compress_rate = self.args.min_mask_rate # the mask ratio starting point(hyper-param).
        # Verify mask_seeds length: one seed per iteration
        assert (
            len(mask_seeds) == num_iterations
        ), f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}"

        prompt_length = seq_len - logits_to_keep
        if self.task == 't2t':
            prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
            prompt_index[:prompt_length] = True  # Mark prompt tokens as True
        elif self.task == 't2i':
            prompt_index = torch.ones(seq_len, dtype=torch.bool, device=device)
            prompt_index[-(logits_to_keep+1):-1] = False # the image is wrapped with eoi token
        else:
            raise ValueError('NOT implemented method.')

        # applying masks
        all_perturbed_seqs = []
        all_expanded_inputs = []
        all_mask = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]

            # Here, we rewrite the perturb process so that the prompt is never masked.
            # only the answer will be masked with a varying mask ratio.
            if num_iterations == 1 and self.args.num_iterations != 1:
                iter_idx = cur_iter # in case of compute_loss, the iter is specified.
            ratio = min(compress_rate + (1 - compress_rate) * (iter_idx + 1) / self.args.num_iterations, 0.999)
            perturbed_seq, to_mask = self.remask(
                expanded_input, prompt_index, self.args.mask_id, seed=mask_seed, ratio=ratio
            )
            if self.task=='t2t':
                all_mask.append(to_mask[:,-logits_to_keep:].unsqueeze(0))
            elif self.task=='t2i':
                all_mask.append(to_mask[:,-(logits_to_keep+1):-1].unsqueeze(0))
            else:
                raise ValueError
            all_perturbed_seqs.append(perturbed_seq)
            all_expanded_inputs.append(expanded_input)

        # Concatenate all iterations into a single batch
        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # [num_iterations * batch_size, seq_len]
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # [num_iterations * batch_size, seq_len]
        perturb_mask = torch.cat(all_mask, dim=0)

        # Get model predictions for the combined batch
        logits = self.get_logits(
            model, perturbed_seq, prompt_index, self.args.cfg_scale, self.args.mask_id
        )  # [num_iterations * batch_size, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        if self.task=='t2t':
            completion_logits = logits[:, -logits_to_keep:, :]  # [num_iterations * batch_size, logits_to_keep, vocab_size]
            completion_targets = expanded_input[:, -logits_to_keep:]  # [num_iterations * batch_size, logits_to_keep]
        elif self.task=='t2i':
            completion_logits = logits[:, -(logits_to_keep+1):-1, 126349:134541]  # [num_iterations * batch_size, logits_to_keep, vocab_size]
            completion_targets = expanded_input[:, -(logits_to_keep+1):-1] - 126349  # [num_iterations * batch_size, logits_to_keep]
        else:
            raise ValueError            
        
        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_targets = completion_targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        # Convert to log probabilities and reshape
        completion_log_probs = -loss.view(num_iterations * batch_size, logits_to_keep)
        per_token_logps = completion_log_probs.view(num_iterations, batch_size, logits_to_keep)

        # Clean up memory
        del perturbed_seq, logits, all_perturbed_seqs, all_expanded_inputs
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32) # shape in [num_iterations, batch_size, logits_to_keep]
        return per_token_logps, perturb_mask

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs
    
    @torch.no_grad()
    def prepare_inputs_uni(
        self,
        texts,
        task,
        device,
        num_vq_tokens=1024,
        mask_token_id=126336
    ):
        if task=='t2i':
            image_tokens = torch.ones((len(texts), num_vq_tokens), dtype=torch.long) * mask_token_id
            input_ids, masks = self.uni_prompting((texts, image_tokens), 't2i_gen')
            if self.args.cfg_scale > 0.0:  
                uncond_input_ids, uncond_masks = self.uni_prompting(([''] * image_tokens.shape[0], image_tokens), 't2i_gen')
                # specifically, the things that matters is wrapped by 126080 and 126081, text null = 126093.
                # if you want to directly produce uncond_input_ids:
                # replace the 2 tokens before 126081 into [126088, 126080], before all 126093.
                input_ids = torch.concat([input_ids, uncond_input_ids],dim=0)
                masks = torch.concat([masks, uncond_masks], dim=0) 
            return input_ids.to(device), masks.to(device), None
        elif task=='t2t':
            prompts_text = self.processing_class.apply_chat_template(texts, add_generation_prompt=True, tokenize=False)
            prompt_ids = self.processing_class(text=prompts_text, return_tensors="pt", padding=True, padding_side="left")['input_ids']
            input_ids = torch.tensor(prompt_ids).to(device)
            return input_ids, None, prompts_text


    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        The rollout function in dLLM.
        """
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompt_ids, prompt_mask, prompts_text = self.prepare_inputs_uni(prompts, self.task, device)

        if self.max_prompt_length is not None and self.task=='t2t':
            # do not trim explicitly while doing t2i tasks.
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            # prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale
        num_vq_tokens = 1024

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            # Roll out G = num_generation completions for each prompt.
            # In case of OOM, we can separate the total rollout into num_generation // num_per_pass splits.
            num_per_pass = self.args.generation_batch_size
            prompt_completion_ids_all = []
            if self.task=='t2t':
                # Process in batches
                for i in range(0, prompt_ids.size(0), num_per_pass):
                    end_idx = min(i + num_per_pass, prompt_ids.size(0))
                    batch_prompt_ids = prompt_ids[i:end_idx]
                    # batch_prompt_mask = prompt_mask[i:end_idx]
                    # This works fine if you set num_generations == per_device_train_batch_size.
                    # if you set num_generations a multiple of per_device_train_batch_size
                    # the group-wise reward calculation has to be handled carefully.
                    batch_prompt_completion_ids = self.generate(
                        model=unwrapped_model,
                        prompt=batch_prompt_ids,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=temperature,
                        cfg_scale=cfg_scale,
                        remasking=self.args.remasking,
                        mask_id=self.args.mask_id,
                    )
                    prompt_completion_ids_all.append(batch_prompt_completion_ids)

                    # del batch_prompt_ids, batch_prompt_mask, batch_prompt_completion_ids
                    del batch_prompt_ids, batch_prompt_completion_ids
                    torch.cuda.empty_cache()
            elif self.task=='t2i':
                if self.args.cfg_scale > 0:
                    prompt_ids, uncond_prompt_ids = torch.split(prompt_ids, prompt_ids.shape[0]//2, dim=0)
                    prompt_mask, uncond_prompt_mask = torch.split(prompt_mask, prompt_mask.shape[0]//2, dim=0)
                for i in range(0, prompt_ids.size(0), num_per_pass):
                    end_idx = min(i + num_per_pass, prompt_ids.size(0))
                    batch_prompt_ids = prompt_ids[i:end_idx]
                    batch_prompt_mask = prompt_mask[i:end_idx]
                    if self.args.cfg_scale > 0:
                        batch_uncond_prompt_ids = uncond_prompt_ids[i:end_idx]
                        batch_uncond_prompt_mask = uncond_prompt_mask[i:end_idx]
                    else:
                        batch_uncond_prompt_ids = None
                        batch_uncond_prompt_mask = None

                    batch_prompt_completion_ids = self.t2i_generate_emerge(
                        model=unwrapped_model,
                        input_ids=batch_prompt_ids,
                        uncond_input_ids=batch_uncond_prompt_ids,
                        attention_mask=batch_prompt_mask,
                        uncond_attention_mask=batch_uncond_prompt_mask,
                        temperature=temperature,
                        guidance_scale=self.args.cfg_scale,#cfg_scale
                        resolution=self.uni_prompting.max_text_len
                    ) # you have to return the answer with the question, not just answers.
                    prompt_completion_ids_all.append(batch_prompt_completion_ids)

                # del batch_prompt_ids, batch_prompt_mask, batch_prompt_completion_ids
                del batch_prompt_ids, batch_prompt_completion_ids, batch_prompt_mask
                torch.cuda.empty_cache()

            else:
                raise ValueError

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        if self.task=='t2t':
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            # In d1 setting, pre-set the all-mask for completions. 
            # Note: When using uni, we will replace it with masks from get_per_token_logps. 
            # Refer to the end of this func.
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            logits_to_keep = completion_ids.size(1)  
            # we only need to compute the logits for the completion tokens
        elif self.task=='t2i':
            prompt_length = self.uni_prompting.max_text_len # the padded prompt length.
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, -(num_vq_tokens + 1) : -1]
            logits_to_keep = completion_ids.size(1)
            # we will use perturb_mask to replace the completion_mask.
            completion_mask = (prompt_completion_ids < 0).int()
            completion_mask[:,-(logits_to_keep+1):-1] = 1

        if self.args.random_masking:
            # default turned on.
            # use random seeds for every iterations in GRPO iterations
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device)
        else:
            # use fixed seeds for every iterations in GRPO iterations
            mask_seeds = [42] * self.num_iterations

        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        with torch.no_grad():
            if self.num_iterations > 1:
                # repeat prompt completion ids self.num_iterations times
                # note that at current design, on each device, the model handles only one question with G samples.
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                ) # [num_iters, G, padded_ids_length]
                old_per_token_logps, perturb_mask = self._get_per_token_logps(
                    self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds
                )
                all_old_per_token_logps = old_per_token_logps
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    # the fixed seed ensures that the masks at this step are identical to the perturb_mask
                    ref_per_token_logps, _ = self._get_per_token_logps(
                        self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds
                    )
                    all_ref_per_token_logps = ref_per_token_logps
        if self.task=='t2t':
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        elif self.task=='t2i':
            completions_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        
            completions_images = completion_ids - len(self.uni_prompting.text_tokenizer)
            completions_images = torch.clamp(completions_images, max=self.model.config.codebook_size - 1, min=0)
            completions_images = self.vq_model.decode_code(completions_images)
        else:
            raise ValueError("Unsupported tasks!")

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            if self.task=='t2t':
                completions = completions_text
            elif self.task=='t2i':
                completions = completions_images
            else:
                raise ValueError('Unsupported completion evaluation!')

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        if self.use_pref:
            # the Pref-GRPO design, default off.
            ori_rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            rewards = pref_score(ori_rewards.view(-1, self.num_generations))
        else:
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        # FIXME:why the reward_weights is not specified throughout the process?

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        # advantages = rewards - mean_grouped_rewards
        # FIXME: d1 removed the regularization, this may NOT be right. The original should be:
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-5)
        # This is different from d1 design, which computes advantages on each device.
        # MixGRPO (arXiv 2507.21802) for t2i reports that epsilon 1e-3. To be observed.
        # which is not common. 0.2 is the default choice.
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["advantages_max"].append(advantages.max().item())
        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            # "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask if self.version=='d1' else perturb_mask.int(),
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,  # Store all mask seeds for consistent mask patterns
        }
    
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids = inputs["prompt_ids"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        mask_seeds = inputs["mask_seeds"]

        # Combine prompt and completion
        # For t2t, this is simple. for t2i, the wrap tokens 126084(*)126085 need to be included.
        if self.task == 't2t':
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        elif self.task == 't2i':
            input_ids = torch.cat(
                [prompt_ids,
                torch.full((prompt_ids.size(0), 1), 126084, dtype=prompt_ids.dtype, device=prompt_ids.device),
                completion_ids,
                torch.full((prompt_ids.size(0), 1), 126085, dtype=prompt_ids.dtype, device=prompt_ids.device)],
                dim=1
            )
        else:
            raise ValueError('Be careful with the final steps!')
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens

        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations - 1
        if this_itr_idx == -1:
            this_itr_idx = self.args.num_iterations - 1 # do not return -1
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        input_ids = input_ids.unsqueeze(0)
        per_token_logps, mask_check = self._get_per_token_logps(model, input_ids, logits_to_keep, [this_itr_mask_seed], this_itr_idx)
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        if self.mean_method=='grpo':
            # original GRPO calculation.
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

            is_clipped = (per_token_loss1 < per_token_loss2).float()
            clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()

        elif self.mean_method=='gmpo' or self.mean_method=='gspo':
            # following GMPO https://github.com/callsys/GMPO. use geometry mean.
            cliprange = self.epsilon
            low_cliprange = torch.tensor(-cliprange, device=per_token_logps.device)
            high_cliprange = torch.tensor(cliprange, device=per_token_logps.device)

            logprobs_diff = per_token_logps - old_per_token_logps

            # sign of advantage (broadcast across tokens)
            sgn_advantage = torch.where(advantages.unsqueeze(1) >= 0, -1.0, 1.0)

            sgn_logprobs_diff = sgn_advantage * logprobs_diff
            sgn_logprobs_diff_clamp = torch.clamp(sgn_logprobs_diff, low_cliprange, high_cliprange)
            sgn_logprobs_diff_max = torch.max(sgn_logprobs_diff, sgn_logprobs_diff_clamp)
            logprobs_diff_max = sgn_advantage * sgn_logprobs_diff_max

            # reconstruct ratio from modified log-diffs
            ratio = torch.exp(logprobs_diff_max)

            per_token_loss = -advantages.unsqueeze(1) * ratio
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

            # Clipping indicator: true if clamp modified the value
            is_clipped = (sgn_logprobs_diff != sgn_logprobs_diff_clamp).float()
            clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()    
        else:
            raise ValueError(f"Not supported mean method:{self.mean_method}!")
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss

