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

def random_corrupt(x, mask_index_range, ratio):

    move_indices = torch.rand(
            * x.shape, device=x.device) < ratio
        
    mask_index = torch.randint(*mask_index_range, size=x.shape, dtype=x.dtype, device=x.device)

    xt = torch.where(move_indices, mask_index, x)
    return xt
    
if __name__ == '__main__':

    config = OmegaConf.load('configs/mmada_demo.yaml')
    sample_steps=[16]
    cfg=4.0

    config.mode='t2i'
    config.training.batch_size=1
    config.training.guidance_scale = cfg
    config.training.generation_timesteps = sample_steps
    image_path = '/nfs/mtr/code/MMaDA/generated_samples/ImageReward/s16-cfg4.0/ddpm/000539-0052_0.png'

    # mode = 'simple'
    mode = 'prob_remask'
    # mode = 'flow_remask'
    backward_t = 0.6
    # t indicates the timeline backward (needs to be projected by the scheduler)
    # e.g., for ratio = 0.2, the timeline will step back to t = 0.2, sample the tokens
    # (the remasked tokens in the first step is larger than 20% in cosine schedule!)
    # and finish the sample process from t = 0.2 to t = 0

    validation_prompts = [
        'old man ( long white beard and a hood ) riding on elephants back.'
    ]
    signature = 'elephant'
    save_dir = os.path.join('generated_samples/edit', signature)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length, special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    model.to(device)

    mask_token_id = model.config.mask_token_id
    mask_index_range = (mask_token_id, mask_token_id+1)

    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

    image_ori = Image.open(image_path).convert("RGB")
    image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
    image = image.unsqueeze(0)
    origin_image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)

    for step in tqdm(range(0, len(sample_steps), config.training.batch_size)):
        prompts = validation_prompts

        # image_tokens = torch.ones((len(prompts), config.model.mmada.num_vq_tokens),dtype=torch.long, device=device) * mask_token_id
        if mode =='simple':
            remask_ratio = mask_schedule(torch.tensor(1.0 - backward_t))
            image_tokens = random_corrupt(origin_image_tokens, mask_index_range, remask_ratio)
        else:
            image_tokens = origin_image_tokens

        input_ids, attention_mask = uni_prompting((prompts, image_tokens), 't2i_gen')
        if config.training.guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
        else:
            uncond_input_ids = None
            uncond_attention_mask = None

        # with torch.no_grad():
        #     gen_token_ids = model.t2i_generate(
        #         input_ids=input_ids,
        #         uncond_input_ids=uncond_input_ids,
        #         attention_mask=attention_mask,
        #         uncond_attention_mask=uncond_attention_mask,
        #         guidance_scale=config.training.guidance_scale,
        #         temperature=config.training.get("generation_temperature", 1.0),
        #         timesteps=config.training.generation_timesteps[step],
        #         noise_schedule=mask_schedule,
        #         noise_type=config.training.get("noise_type", "mask"),
        #         seq_len=config.model.mmada.num_vq_tokens,
        #         uni_prompting=uni_prompting,
        #         config=config,
        #     )

        # gen_token_ids = torch.clamp(gen_token_ids, max=config.model.mmada.codebook_size - 1, min=0)
        # images = vq_model.decode_code(gen_token_ids)

        # images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        # # images *= 255.0
        # # images = images.permute(0, 2, 3, 1)

        # save_image(images, os.path.join(save_dir, f"mvtm-sample-s{sample_steps[step]}-cfg{cfg}.png"))

        with torch.no_grad():
            gen_token_ids = model.t2i_edit_emerge(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                guidance_scale=config.training.guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps[step],
                repair_from=int((1 - backward_t) * config.training.generation_timesteps[step]),
                mode=mode,
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

        save_image(images, os.path.join(save_dir, f"{mode}-edit-s{sample_steps[step]}-cfg{cfg}-ratio{backward_t}.png"))

        # wandb_images = [wandb.Image(image, caption=prompts[i]) for i, image in enumerate(pil_images)]
        # wandb.log({"generated_images": wandb_images}, step=step)
