import os
import inspect

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from models import MAGVITv2, get_mask_schedule, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting
from training.utils import flatten_omega_conf, image_transform
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch.nn.functional as F
# from training.data import Text2ImageDataset, GenevalPromptDataset

from omegaconf import OmegaConf

import json
from accelerate import PartialState
distributed_state = PartialState()

def resize_vocab(model, config):
    print(f"Resizing token embeddings to {config.new_vocab_size}")
    model.resize_token_embeddings(config.new_vocab_size)


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=['ImageReward','GenEval','DPG','HPS','HPSv3'], help="Task to run.")
    parser.add_argument("--path", default='/data/models/mmada-8b-base', help="The folder to load the model.")
    parser.add_argument("--ckpt_path", default=None, help="The folder to load the lora model.")
    parser.add_argument("--cfg", default=4.0, type=float, help="cfg for t2i generation.")
    parser.add_argument("--sample_steps", default=16, type=int, help="number of sampling steps for t2i generation.")
    parser.add_argument("--sample_per_prompt", default=5, type=int, help="number of samples per prompt for t2i generation.")
    parser.add_argument("--dual", action="store_true", help="push to inference both method.")
    args = parser.parse_args()

    config = OmegaConf.load('configs/mmada_demo.yaml')
    config.mode='t2i'

    sample_steps = args.sample_steps
    cfg = args.cfg
    sample_per_prompt=args.sample_per_prompt # shall be larger than 1 for reward calculation

    config.training.guidance_scale = cfg
    config.training.generation_timesteps = sample_steps
    model_path = args.path
    signature = args.task

    print(f"\033[32mInferencing with cfg {cfg}, sample_steps {sample_steps}, sample_per_prompt {sample_per_prompt} under task {signature}\033[0m")
    
    if signature not in ['ImageReward', 'GenEval', 'DPG', 'HPS', 'HPSv3']:
        raise ValueError(f"task {signature} not supported for t2i generation.")

    eval_prompts = []
    if signature == 'ImageReward':
        with open('dataset/ImageReward/benchmark-prompts.json', "r") as f:
            eval_prompts = json.load(f)
            print(f"Loaded {len(eval_prompts)} prompts for ImageReward evaluation.")
    elif signature =='GenEval':
        with open('dataset/GenEval/evaluation_metadata_merged.jsonl', 'r', encoding='utf-8') as f:
            eval_prompts = [json.loads(line) for line in f]
            print(f"Loaded {len(eval_prompts)} prompts for GenEval evaluation.")
    elif signature =='DPG':
        with open('dataset/DPG/dpg_bench.jsonl', 'r', encoding='utf-8') as f:
            eval_prompts = [json.loads(line) for line in f]
            print(f"Loaded {len(eval_prompts)} prompts for DPG evaluation.")
    elif signature =='HPS':
        with open('dataset/HPS/HPSv2_eval.jsonl', 'r', encoding='utf-8') as f:
            eval_prompts = [json.loads(line) for line in f]
            print(f"Loaded {len(eval_prompts)} prompts for HPSv2 evaluation.")
    elif signature =='HPSv3':
        with open('dataset/HPS/v3_eval_sample.jsonl', 'r', encoding='utf-8') as f:
            eval_prompts = [json.loads(line) for line in f]
            print(f"Loaded {len(eval_prompts)} prompts for HPSv3 evaluation.")
    
    save_dir = os.path.join('generated_samples', signature, f's{sample_steps}-cfg{cfg}')
    os.makedirs(save_dir, exist_ok=True)
    if args.dual:
        mvtm_dir = os.path.join(save_dir, 'mvtm-lora-base')
        os.makedirs(mvtm_dir, exist_ok=True)
    ddpm_dir = os.path.join(save_dir, 'ddpm-lora-base')
    os.makedirs(ddpm_dir, exist_ok=True)

    device = distributed_state.device
    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length, special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    model = MMadaModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    if args.ckpt_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.ckpt_path, torch_dtype=torch.bfloat16)

    model.to(device)

    mask_token_id = model.config.mask_token_id

    with distributed_state.split_between_processes(eval_prompts) as process_data:
        
        for idx, batch in tqdm(enumerate(process_data)):
            
            batch = [batch] * sample_per_prompt
            if signature =='ImageReward':
                prompt_ids = [item["id"] for item in batch]
                prompts = [item["prompt"] for item in batch]
            elif signature == 'GenEval':
                prompt_ids = [item["prompt"] for item in batch]
                prompts = [item["prompt"] for item in batch]
            elif signature == 'DPG':
                prompt_ids = [item["item_id"] for item in batch]
                prompts = [item["prompt"] for item in batch]
            elif signature == 'HPS':
                prompt_ids = [item["prompt"] for item in batch]
                prompts = [item["prompt"] for item in batch]
            elif signature == 'HPSv3':
                prompt_ids = [item["prompt_id"] for item in batch]
                prompts = [item["caption"] for item in batch]
            
            image_tokens = torch.ones((len(prompts), config.model.mmada.num_vq_tokens),
                                        dtype=torch.long, device=device) * mask_token_id
            input_ids, attention_mask = uni_prompting((prompts, image_tokens), 't2i_gen')
            if config.training.guidance_scale > 0:
                uncond_input_ids, uncond_attention_mask = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
            else:
                uncond_input_ids = None
                uncond_attention_mask = None

            mask_schedule = get_mask_schedule("cosine")
            if args.dual:
                with torch.no_grad():
                    gen_token_ids = model.t2i_generate(
                        input_ids=input_ids,
                        uncond_input_ids=uncond_input_ids,
                        attention_mask=attention_mask,
                        uncond_attention_mask=uncond_attention_mask,
                        guidance_scale=config.training.guidance_scale,
                        temperature=config.training.get("generation_temperature", 1.0),
                        timesteps=config.training.generation_timesteps,
                        noise_schedule=mask_schedule,
                        noise_type=config.training.get("noise_type", "mask"),
                        seq_len=config.model.mmada.num_vq_tokens,
                        uni_prompting=uni_prompting,
                        config=config,
                    )

                gen_token_ids = torch.clamp(gen_token_ids, max=config.model.mmada.codebook_size - 1, min=0)
                images = vq_model.decode_code(gen_token_ids)

                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images *= 255.0
                images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                pil_images = [Image.fromarray(image) for image in images]
                
                for i, (id, img) in enumerate(zip(prompt_ids, pil_images)):
                    img.save(f'{mvtm_dir}/{id}_{i}.png')

            with torch.no_grad():
                gen_token_ids = model.t2i_generate_emerge(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    uncond_attention_mask=uncond_attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", "mask"),
                    seq_len=config.model.mmada.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )

            gen_token_ids = torch.clamp(gen_token_ids, max=config.model.mmada.codebook_size - 1, min=0)
            images = vq_model.decode_code(gen_token_ids)

            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = [Image.fromarray(image) for image in images]

            for i, (id, img) in enumerate(zip(prompt_ids, pil_images)):
                img.save(f'{ddpm_dir}/{id}_{i}.png')
