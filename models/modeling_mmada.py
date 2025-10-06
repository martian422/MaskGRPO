from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
from PIL import Image
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from .modeling_llada import LLaDAModelLM
from .sampling import cosine_schedule, linear_schedule, mask_by_random_topk
from transformers import PretrainedConfig

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

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

class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        allowed_keys = [
            "vocab_size",
            "llm_vocab_size",
            "llm_model_path",
            "codebook_size",
            "num_vq_tokens",
            "num_new_special_tokens",
            "gradient_checkpointing",
            "new_vocab_size",
        ]

        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])



class MMadaModelLM(LLaDAModelLM):
    config_class = MMadaConfig
    base_model_prefix = "model"
    def __init__(self, config: MMadaConfig, *args, **kwargs):
        # print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)

        # # resize token embeddings
        # print(f"Resizing token embeddings to {config.new_vocab_size}")
        # self.resize_token_embeddings(config.new_vocab_size)

    @torch.no_grad()
    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
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
        uni_prompting = kwargs.get("uni_prompting", None)
        vocab_shift = len(uni_prompting.text_tokenizer) + num_new_special_tokens
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                # cur_cfg = (step/timesteps) * (guidance_scale - 1) + 1.0
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                # logits = (1 + cur_cfg) * cond_logits - cur_cfg * uncond_logits# FIXME: do not use this for RL rollout.
                logits = logits[:, -(num_vq_tokens + 1):-1, vocab_shift: vocab_shift + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
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
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    @torch.no_grad()
    def t2i_generate_emerge(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  
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
        Also see the original ReDDiT (MDLM style) code:
        https://github.com/martian422/ReDDiT/blob/main/diffusion.py#L565
        """

        # begin with all image token ids masked
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        vocab_shift = len(uni_prompting.text_tokenizer) + num_new_special_tokens
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        sampled_ids = input_ids[:, -(num_vq_tokens + 1):-1].clone() # all equals to mask_token_ids.
        # this results in the uni-mask canvas

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                # cur_cfg = (step/timesteps) * (guidance_scale - 1) + 1.0
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                # logits = (1 + cur_cfg) * cond_logits - cur_cfg * uncond_logits# FIXME: do not use this for RL rollout.
                logits = logits[:, -(num_vq_tokens + 1):-1, vocab_shift: vocab_shift + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
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
                input_ids[:, -(num_vq_tokens + 1):-1] = sampled_ids_to_paste + vocab_shift
            else:
                new_pred = sample_categorical(probs)
                sampled_ids = torch.where(sampled_ids > codebook_size -1, new_pred, sampled_ids)
        return sampled_ids
    
    @torch.no_grad()
    def t2i_edit_emerge(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,
            repair_from=9,
            mode='simple',  
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
        A low-discrepancy sampler modified by ReDDiT.
        https://github.com/martian422/ReDDiT/blob/main/diffusion.py#L565
        """

        # begin with all image token ids masked
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        vocab_shift = len(uni_prompting.text_tokenizer) + num_new_special_tokens
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        sampled_ids = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        if mode == 'simple':
            # for the edit task, you have to handled it, as some tokens are decoded.
            sampled_ids = torch.where(sampled_ids!=mask_token_id, sampled_ids - vocab_shift, 8192)
            print(f'remained tokens:{(sampled_ids != 8192).sum().item()}')

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        if mode =='prob_remask':
            uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
            model_input = torch.cat([input_ids, uncond_input_ids])
            attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            logits = self(model_input, attention_bias=attention_bias).logits 
            # print(f"logits.shape: {logits.shape}")
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            # it seems that muse has a different cfg setting
            logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
            logits = logits[:, -(num_vq_tokens + 1):-1, vocab_shift: vocab_shift + codebook_size]
            probs = logits.softmax(dim=-1)

            k_repair = noise_schedule(torch.tensor(repair_from / timesteps)) 
            k_remain = 1 - k_repair
            p_mask = k_repair.expand(probs.shape[0], probs.shape[1], 1).to(probs.device)

            probs_with_mask_index = torch.cat([probs * k_remain, p_mask], dim=-1)

            new_pred_with_masks = sample_categorical(probs_with_mask_index)
            sampled_ids = new_pred_with_masks
            print(f'remained tokens:{(sampled_ids != 8192).sum().item()}')
            sampled_ids_to_paste = torch.where(sampled_ids > codebook_size -1, mask_token_id - vocab_shift, sampled_ids)
            input_ids[:, -(num_vq_tokens + 1):-1] = sampled_ids_to_paste + vocab_shift

        elif mode =='flow_remask':
            uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
            model_input = torch.cat([input_ids, uncond_input_ids])
            attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            logits = self(model_input, attention_bias=attention_bias).logits 
            # print(f"logits.shape: {logits.shape}")
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            # it seems that muse has a different cfg setting
            logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
            logits = logits[:, -(num_vq_tokens + 1):-1, vocab_shift: vocab_shift + codebook_size]
            probs = logits.softmax(dim=-1)
            sampled_ids = sampled_ids - vocab_shift

            origin_delta = F.one_hot(sampled_ids, num_classes=codebook_size)
            flow = torch.where(origin_delta.to(dtype=torch.bool), torch.zeros_like(origin_delta), probs)
            flow_out = flow.sum(dim=-1)

            k_repair = noise_schedule(torch.tensor(repair_from / timesteps))
            mask_len = int((k_repair * seq_len).item())
            _, topk_indices = torch.topk(flow_out, k=mask_len, dim=-1, largest=True)
            mask = torch.zeros_like(flow_out, dtype=torch.bool)
            batch_indices = torch.arange(flow_out.size(0), device=flow_out.device).unsqueeze(1)
            mask[batch_indices, topk_indices] = True

            sampled_ids = torch.where(mask, 8192, sampled_ids)
            # print(f'remained tokens:{(sampled_ids != 8192).sum().item()}')

            sampled_ids_to_paste = torch.where(sampled_ids > codebook_size -1, mask_token_id - vocab_shift, sampled_ids)
            input_ids[:, -(num_vq_tokens + 1):-1] = sampled_ids_to_paste + vocab_shift
        elif mode == 'simple':
            pass
        else:
            raise ValueError(f'Sample method {mode} not implemented!')


        for step in range(repair_from, timesteps, 1):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, vocab_shift: vocab_shift + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
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
                sampled_ids = updated_sampled_ids
                print(f'decoded tokens:{(sampled_ids != 8192).sum().item()}')
                sampled_ids_to_paste = torch.where(sampled_ids > codebook_size -1, mask_token_id - vocab_shift, updated_sampled_ids)
                input_ids[:, -(num_vq_tokens + 1):-1] = sampled_ids_to_paste + vocab_shift
            else:
                new_pred = sample_categorical(probs)
                sampled_ids = torch.where(sampled_ids > codebook_size -1, new_pred, sampled_ids)
                print(f'decoded tokens:{(sampled_ids != 8192).sum().item()}')
        return sampled_ids
    
    def forward_process(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            p_mask_lm=None,
            p_mask_mmu=None,
            answer_lengths=None,
            t2i_masks=None,
            answer_lengths_lm=None
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        # print(f"logits shape: {logits.shape}") B, 359, vocab_size

        if batch_size_t2i == 0:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)
        else:
            # t2i loss
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
        
        # llada loss  
        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_lm = masked_indices[batch_size_t2i:batch_size_t2i + batch_size_lm]
        # 新增调试代码：统计每行mask数量
        # if masked_indices_lm.numel() > 0:
        #     mask_counts = torch.sum(masked_indices_lm, dim=1)
        #     logging.info(f"[LM mask nums]: {mask_counts.cpu()}.")
        # else:
        #     logging.info("[LM mask nums] no LM sample.")
        masked_indices_mmu = masked_indices[-batch_size_mmu:]
        p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
        p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)       
        answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
        loss_lm = F.cross_entropy(
            logits[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
            labels[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_lm[masked_indices_lm]
        # print(f"logits lm shape: {logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape}")
        loss_lm = loss_lm.sum() / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0] * logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[1])

        # llm loss 
        answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
        loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0])  
        
        loss_mmu = F.cross_entropy(
            logits[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1, self.output_size),
            labels[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_mmu[masked_indices_mmu]
        loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[-batch_size_mmu:].shape[0])
        
        return logits, loss_t2i, loss_lm, loss_mmu

    def forward_process_with_r2i(
            self,
            input_ids, 
            labels,
            t2i_masks=None,
            max_seq_length=128,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            batch_size_r2i=0,
            p_mask_lm=None,
            p_mask_mmu=None,
            p_mask_r2i=None,
            answer_lengths=None,
            answer_lengths_lm=None,
            answer_lengths_r2i=None,
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        # print(f"logits shape: {logits.shape}") B, 359, vocab_size

        if batch_size_t2i == 0:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)
        else:
            # t2i loss
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
        
        # llada loss  

        start_lm = batch_size_t2i
        end_lm = start_lm + batch_size_lm
        start_mmu = end_lm
        end_mmu = start_mmu + batch_size_mmu
        start_r2i = end_mmu
        end_r2i = start_r2i + batch_size_r2i

        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_lm = masked_indices[start_lm:end_lm]
        masked_indices_mmu = masked_indices[start_mmu:end_mmu]
        masked_indices_r2i = masked_indices[start_r2i:end_r2i]

        p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
        p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)
        p_mask_r2i = p_mask_r2i.to(masked_indices_r2i.device)

        answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
        answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
        answer_lengths_r2i = answer_lengths_r2i.to(masked_indices_r2i.device)

        loss_lm = F.cross_entropy(
            logits[start_lm:end_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
            labels[start_lm:end_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_lm[masked_indices_lm]
        # print(f"logits lm shape: {logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape}")
        loss_lm = loss_lm.sum() / (logits[start_lm:end_lm].shape[0] * logits[start_lm:end_lm].shape[1])
        loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[start_lm:end_lm].shape[0]) 

        loss_mmu = F.cross_entropy(
            logits[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1, self.output_size),
            labels[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_mmu[masked_indices_mmu]
        loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[start_mmu:end_mmu].shape[0])
        
        loss_r2i = F.cross_entropy(
            logits[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1, self.output_size),
            labels[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_r2i[masked_indices_r2i]
        loss_r2i = torch.sum(loss_r2i/answer_lengths_r2i[masked_indices_r2i]) / (logits[start_r2i:end_r2i].shape[0])
        
        return logits, loss_t2i, loss_lm, loss_mmu, loss_r2i


    def forward_t2i(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            max_seq_length=128,
            t2i_masks=None
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        # print(f"logits shape: {logits.shape}") B, 359, vocab_size

        loss_t2i = F.cross_entropy(
            logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
            labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )
        
        return loss_t2i





    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        # print(f"num_blocks: {num_blocks}, steps: {steps}")
        # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
            # print(f"num_transfer_tokens: {num_transfer_tokens}, num_transfer_tokens.shape: {num_transfer_tokens.shape}")
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
                
            
            # logits = logits[:, -1, :] / temperature
            # # optionally crop the logits to only the top k options
            # if top_k is not None:
            #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            #     logits[logits < v[:, [-1]]] = -float('Inf')
            # # apply softmax to convert logits to (normalized) probabilities
            # probs = F.softmax(logits, dim=-1)
            # # sample from the distribution
            # idx_next = torch.multinomial(probs, num_samples=1)
            # result.append(idx_next[0][0])
            # # append sampled index to the running sequence and continue
            # if self.config.w_clip_vit:
            #     idx_next_embeddings = self.mmada.model.embed_tokens(idx_next)
            #     input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            # else:
            #     idx = torch.cat((idx, idx_next), dim=1)

            # if eot_token is not None and idx_next.cpu() == eot_token:
            #     break

        return x

    @torch.no_grad()
    def mmu_generate_emerge(
        self, 
        input_ids=None, 
        input_embeddings=None, 
        max_new_tokens=512, 
        steps=128, 
        block_length=128, 
        temperature=0.0, 
        top_k=None, 
        eot_token=None, 
        cfg_scale=0.0, 
        remasking='low_confidence', 
        mask_id=126336, 
        attention_mask=None
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = input_ids.device
        except:
            device = input_embeddings.device

        noise_schedule = linear_schedule # to be verified: is cosine or other scheduler ok in language generation?
        batch_size = input_ids.shape[0]
        sampled_ids = torch.full((batch_size, input_ids.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        sampled_ids[:, :input_ids.shape[1]] = input_ids.clone() # only copy the inputs
        prompt_index = (sampled_ids != mask_id) 
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        # we use probability, so do not need strict division.
        per_block_steps = int(steps/num_blocks)
        
        for num_block in range(num_blocks):
            #determine the indices to decode at this block (usually the entire).
            update_masks = get_cur_masks(sampled_ids, num_block, block_length, mask_id, shift = input_ids.shape[1])

            for step in range(per_block_steps):
                # mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_cond = sampled_ids.clone()
                    un_cond[prompt_index] = mask_id
                    x_ = torch.cat([sampled_ids, un_cond], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(sampled_ids, attention_bias=attention_bias).logits
                # apply our masked categorical sampling.
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
                # print(f'decoded tokens:{(sampled_ids != mask_id).sum().item()}')             

        return sampled_ids

    @torch.no_grad()
    def mmu_generate_fast(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
            if eot_token is not None:
                last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
                if last_token_index_in_current_block < x.shape[1]:
                    tokens_at_block_end = x[:, last_token_index_in_current_block]
                    if torch.all(tokens_at_block_end == eot_token):
                        break
        return x

    @torch.no_grad()
    def t2i_generate_decoding_stepwise(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            vq_model = None,
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
        uni_prompting = kwargs.get("uni_prompting", None)
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, vocab_shift: vocab_shift + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
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
            current_image_vq_indices = sampled_ids.clone()
            # print(f"current_image_vq_indices: {current_image_vq_indices}")
            current_image_vq_indices = torch.clamp(current_image_vq_indices, 0, 8192 - 1)
            current_image = vq_model.decode_code(current_image_vq_indices)
            images = torch.clamp((current_image + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = Image.fromarray(images[0]) 
            yield pil_images, f"Step {step + 1}/{timesteps}"
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
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
            

        return sampled_ids
    

AutoConfig.register("mmada", MMadaConfig)
AutoModelForCausalLM.register(MMadaConfig, MMadaModelLM)
AutoModel.register(MMadaConfig, MMadaModelLM)
