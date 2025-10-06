import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from reward_func import code_reward
from models.generate_llada import generate
from accelerate import PartialState
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np
from accelerate.utils import gather
state = PartialState()
device = state.device

MODEL_NAME = ""

GEN_LEN = 256
STEPS = 256
BLK_LEN = 32
CKPT = 5000
# 加载模型和 tokenizer
# model_path = f"/home/ZhangMu/Workspace/repos/mmada-mcat/outputs/acecode_maskgrpo_min0.6_256_max/checkpoint-{CKPT}"
model_path = "/ssd/models/llada-8b-instruct"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载 MBPP 数据集
dataset = load_dataset("json", data_files="dataset/MBPP/sanitized-mbpp.json")["train"]

def make_conversation(example, prompt_column: str = "prompt"):
    prompt = []
    # prompt=[
    #     {'role': 'user', 'content': 'You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n'},
    #     {'role': 'assistant', 'content': "```python\ndef similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)' \n```"},

    #     {'role': 'user', 'content': 'You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n assert is_not_prime(2) == False \nassert is_not_prime(10) == True \nassert is_not_prime(35) == True \n'},
    #     {'role': 'assistant', 'content': "```python\nimport math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result' \n```"},

    #     {'role': 'user', 'content': 'You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n'},
    #     {'role': 'assistant', 'content': "```python\nimport heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums' \n```"}
    # ]
    prompt.append({"role": "user", "content": "You are an expert Python programmer, and here is your task: " + example[prompt_column] + ' Your code should pass these tests:\n\n' + "\n".join(example['test_list'])})
    return {"prompt": prompt}

dataset = dataset.map(make_conversation)
# 只取前 N 条用于测试
# N = 500
# dataset = dataset.select(range(N))

def collate_fn(batch):
    # batch 是 list of dict，每个 dict 里有 'prompt' 和 'test_list'
    prompts = [item['prompt'] for item in batch]
    test_lists = [item.get('test_list', []) for item in batch]
    return {'prompts': prompts, 'test_lists': test_lists}

sampler = DistributedSampler(
    dataset,
    num_replicas=state.num_processes,
    rank=state.process_index,
    shuffle=False
)

batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)

# 生成模型输出
all_rewards = []
with state.split_between_processes(dataloader) as process_data:
    
    for batch in tqdm(process_data):
        prompts_texts = tokenizer.apply_chat_template(batch['prompts'], add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(text=prompts_texts, return_tensors="pt", padding=True, padding_side="left")['input_ids']
        input_ids = torch.tensor(prompt_ids).to(device)
        with torch.no_grad():
            output = generate(
                model,
                input_ids,
                tokenizer,
                steps=STEPS,
                gen_length=GEN_LEN,
                block_length=BLK_LEN,
            )
        prompt_len = input_ids.shape[1]
        output = output[:, prompt_len:]
        completion_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        completion = [[{"role": "assistant", "content": text}] for text in completion_text]

        # 计算准确率
        rewards = code_reward(completion, test_cases=batch['test_lists'])
        all_rewards.extend(rewards)

# 收集所有进程的分数
all_rewards_tensor = torch.tensor([r if r is not None else float('nan') for r in all_rewards], dtype=torch.float32, device=device)
gathered_rewards = gather(all_rewards_tensor).cpu().numpy()

# 主进程统计（忽略 None/NaN）
if state.is_main_process:
    valid_rewards = gathered_rewards[~np.isnan(gathered_rewards)]
    print(f"平均分: {valid_rewards.mean():.4f}")
    
state.destroy_process_group()