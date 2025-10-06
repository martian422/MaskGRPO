# coding=utf-8
# Copyright 2025 MMaDA Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import MAGVITv2, get_mask_schedule, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting
from training.utils import flatten_omega_conf, image_transform
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch.nn.functional as F

from omegaconf import OmegaConf
from torchvision.utils import save_image

def resize_vocab(model, config):
    print(f"Resizing token embeddings to {config.new_vocab_size}")
    model.resize_token_embeddings(config.new_vocab_size)


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':

    config = OmegaConf.load('configs/mmada_demo.yaml')
    sample_steps=[50]
    cfg=3.5

    config.mode='t2i'
    config.training.batch_size=4
    config.training.guidance_scale = cfg
    config.training.generation_timesteps = sample_steps


    validation_prompts = [
        'An intricate Chinese ink and wash painting that depicts a majestic tiger, its fur rendered in delicate brush strokes, wearing a traditional train conductor\'s hat atop its head. The tiger\'s piercing eyes gaze forward as it firmly grasps a skateboard, which features a prominent yin-yang symbol in its design, symbolizing balance. The background of the painting is a subtle wash of grays, suggesting a misty and timeless landscape.'
    ]
    signature = 'tiger'
    save_dir = os.path.join('generated_samples/simple', signature)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length, special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    model = MMadaModelLM.from_pretrained('/mydata/models/models/mmada-8b-base', trust_remote_code=True, torch_dtype=torch.bfloat16)

    model.to(device)

    mask_token_id = model.config.mask_token_id

    for step in tqdm(range(0, len(sample_steps), config.training.batch_size)):
        prompts = validation_prompts

        image_tokens = torch.ones((len(prompts), config.model.mmada.num_vq_tokens),
                                    dtype=torch.long, device=device) * mask_token_id
        input_ids, attention_mask = uni_prompting((prompts, image_tokens), 't2i_gen')
        if config.training.guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
        else:
            uncond_input_ids = None
            uncond_attention_mask = None

        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_schedule(schedule, **args)
        else:
            mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))
        with torch.no_grad():
            gen_token_ids = model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                guidance_scale=config.training.guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps[step],
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                seq_len=config.model.mmada.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )

        gen_token_ids = torch.clamp(gen_token_ids, max=config.model.mmada.codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)

        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        # images *= 255.0
        # images = images.permute(0, 2, 3, 1)

        save_image(images, os.path.join(save_dir, f"mvtm-sample-s{sample_steps[step]}-cfg{cfg}-{step}.png"))

        with torch.no_grad():
            gen_token_ids = model.t2i_generate_emerge(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                guidance_scale=config.training.guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps[step],
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                seq_len=config.model.mmada.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )

        gen_token_ids = torch.clamp(gen_token_ids, max=config.model.mmada.codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)

        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        # images *= 255.0
        # images = images.permute(0, 2, 3, 1)

        save_image(images, os.path.join(save_dir, f"ddpm-sample-s{sample_steps[step]}-cfg{cfg}-{step}.png"))

        # wandb_images = [wandb.Image(image, caption=prompts[i]) for i, image in enumerate(pil_images)]
        # wandb.log({"generated_images": wandb_images}, step=step)
