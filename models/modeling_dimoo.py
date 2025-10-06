from typing import List
import torch
from .modeling_llada import LLaDAModelLM
from .configuration_llada import LLaDAConfig
__all__ = ["DiMooModelLM"]


### the Lumina-DiMoo model is not fundamentally different from MMaDA, except for the training data and some details. 
### however, its generation process is too long for RL, therefore we only copy the model structure here.
### if you want to do RL with DiMoo, please refer to the MMaDA implementation.
### do not forget to change the config class, model class, and VQ_model in your training!


def create_attention_mask(original_lengths, max_tokens, device):
    batch_size = len(original_lengths)
    attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=device)
    for i, length in enumerate(original_lengths):
        attention_mask[i, :length] = 1  # 有效位置设为1
    return attention_mask

class DiMooModelLM(LLaDAModelLM):
    config_class = LLaDAConfig
    base_model_prefix = "model"
    def __init__(self, config: LLaDAConfig, *args, **kwargs):
        print(f"Initializing DiMooModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)
    
    def forward(self, input_ids=None, labels=None, infer=False, use_cache=False, to_compute_mask=None, cat='', **kwargs):
        input_ids = input_ids.tolist()
        # ========================================================
        # padding input batch len & attention bias for attention mask
        # ========================================================
        max_tokens = max([len(_) for _ in input_ids])
        original_lengths = [len(example) for example in input_ids] # every sample len --> record for attention mask
        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids] # padding 0 to right --> max length
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device) 
        # attn mask
        attention_mask = create_attention_mask(original_lengths, max_tokens, self.device)

        # ========================================================
        # model output 
        # ========================================================
        output = LLaDAModelLM.forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                      use_cache=use_cache, to_compute_mask=to_compute_mask, cat=cat)
        if infer:
            return output
    
    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.transformer.blocks), self.model.transformer.ff_out]
        return modules