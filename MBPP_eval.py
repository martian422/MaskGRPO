import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from grpo.reward_func import code_reward
from models.generate_llada import generate
from accelerate import PartialState
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np
from accelerate.utils import gather
import json
import os
import pickle

# -----------------------------
# CLI Arguments
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="acecode",
                        help="Base experiment/model folder name under outputs/")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Manually specify model path; if not provided use outputs/{model_name}/checkpoint-*")
    parser.add_argument("--gen_len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block_len", type=int, default=32)
    parser.add_argument("--dataset_file", type=str, default="dataset/MBPP/mbpp.jsonl")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ckpt_start", type=int, default=1)
    parser.add_argument("--ckpt_end", type=int, default=0)
    parser.add_argument("--ckpt_step", type=int, default=1)

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    state = PartialState()
    device = state.device

    GEN_LEN = args.gen_len
    STEPS = args.steps
    BLK_LEN = args.block_len

    checkpoint_range = range(args.ckpt_start, args.ckpt_end, -args.ckpt_step)

    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset_file)["train"]

    def make_conversation(example, prompt_column: str = "text"):
        prompt = [{
            "role": "user",
            "content": "You are an expert Python programmer, and here is your task: "
                       + example[prompt_column]
                       + ' Your code should pass these tests:\n\n'
                       + "\n".join(example['test_list'])
        }]
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)
    dataset = dataset.select(range(args.num_samples))

    def collate_fn(batch):
        prompts = [item['prompt'] for item in batch]
        test_lists = [item.get('test_list', []) for item in batch]
        return {'prompts': prompts, 'test_lists': test_lists}

    sampler = DistributedSampler(
        dataset,
        num_replicas=state.num_processes,
        rank=state.process_index,
        shuffle=False
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            collate_fn=collate_fn, sampler=sampler)

    all_results = {}

    # -----------------------------
    # Evaluate checkpoints
    # -----------------------------
    for CKPT in checkpoint_range:
        if state.is_main_process:
            print(f"\n=== {CKPT} ===")

        if args.model_path:
            model_path = args.model_path
        else:
            model_path = f"outputs/{args.model_name}/checkpoint-{CKPT}"

        if not os.path.exists(model_path):
            if state.is_main_process:
                print(f"Skipping non-existent ckpt: {model_path}")
            continue

        try:
            model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).to(device).eval()

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

        except Exception as e:
            if state.is_main_process:
                print(f"Fail loading {CKPT}: {e}")
            continue

        all_rewards = []
        all_samples = []

        # Iterate dataset for this checkpoint
        with state.split_between_processes(dataloader) as process_data:
            for batch in tqdm(process_data, desc=f"CKPT-{CKPT}"):

                prompts_texts = tokenizer.apply_chat_template(
                    batch['prompts'], add_generation_prompt=True, tokenize=False
                )
                prompt_ids = tokenizer(text=prompts_texts, return_tensors="pt",
                                       padding=True, padding_side="left")['input_ids']

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

                rewards = code_reward(completion, test_cases=batch['test_lists'], num_parallel=1)
                all_rewards.extend(rewards)

                for prompt, response, reward, test_list in zip(
                    batch['prompts'], completion_text, rewards, batch['test_lists']
                ):
                    all_samples.append({
                        "question": prompt[0]["content"],
                        "response": response,
                        "reward": float(reward) if reward is not None else None,
                        "test_cases": test_list,
                        "sample_id": len(all_samples)
                    })

        # Gather rewards
        all_rewards_tensor = torch.tensor(
            [r if r is not None else float('nan') for r in all_rewards],
            dtype=torch.float32, device=device
        )
        gathered_rewards = gather(all_rewards_tensor).cpu().numpy()

        if state.is_main_process:
            valid_rewards = gathered_rewards[~np.isnan(gathered_rewards)]
            avg_score = valid_rewards.mean()

            all_results[CKPT] = {
                "avg_score": float(avg_score),
                "total_samples": len(valid_rewards),
                "samples": all_samples
            }

            print(f"CKPT {CKPT}: average={avg_score:.4f}")

        del model, tokenizer
        torch.cuda.empty_cache()

    # -----------------------------
    # Save results
    # -----------------------------
    if state.is_main_process:
        save_folder = "generated_samples/MBPP"
        os.makedirs(save_folder, exist_ok=True)

        summary_file = f"summary_{args.model_name}_steps{STEPS}_genlen{GEN_LEN}.json"
        detail_file = f"{args.model_name}_steps{STEPS}_genlen{GEN_LEN}_detailed.json"

        summary = {ckpt: {"avg_score": res["avg_score"],
                          "total_samples": res["total_samples"]}
                   for ckpt, res in all_results.items()}

        with open(os.path.join(save_folder, summary_file), "w") as f:
            json.dump(summary, f, indent=2)

        with open(os.path.join(save_folder, detail_file), "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n=== Summary saved to {summary_file} ===")
        print(f"=== Details on main process saved to {detail_file} ===")

    state.destroy_process_group()


if __name__ == "__main__":
    main()
