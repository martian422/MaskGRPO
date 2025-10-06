import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig
from models import MMadaModelLM, MAGVITv2, LLaDAModelLM
from training.prompting_utils import UniversalPrompting
from training.data import Text2ImageDataset
import os

# Custom imports
from grpo.mask_grpo_trainer import MaskGRPOTrainer
from grpo.grpo_config import MaskGRPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    # countdown_reward_func,
    correctness_reward_func_math,
    # sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
    reward_clip_score,
    hpsv2,
    hpsv3_remote,
    unifiedreward_score,
)
from data_utils import (
    get_gsm8k_questions_llada,
    get_gsm8k_questions_mmada,
    # get_countdown_questions,
    # get_sudoku_questions,
    set_random_seed,
    get_math_questions_llada,
    get_webdata_prompts,
    ImagePromptDataset,
    HfPromptDataset
)

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")
    

def main(grpo_config, model_config):

    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = int(os.environ.get("WORLD_SIZE"))

    # Load dataset based on configuration
    if grpo_config.task=="t2t":
        if grpo_config.dataset == "gsm8k":
            dataset = get_gsm8k_questions_llada("train")
            reward_functions = [
                # xmlcount_reward_func,
                soft_format_reward_func,
                # strict_format_reward_func,
                int_reward_func,
                correctness_reward_func,
                # boxed_and_answer_tags_format_reward,
            ]
        # elif grpo_config.dataset == "countdown":
        #     dataset = get_countdown_questions("train")
        #     reward_functions = [countdown_reward_func]
        # elif grpo_config.dataset == "sudoku":
        #     dataset = get_sudoku_questions()
        #     reward_functions = [sudoku_reward_func]
        elif grpo_config.dataset == "math500":
            dataset = get_math_questions_llada("train")
            reward_functions = [
                correctness_reward_func_math,
                boxed_and_answer_tags_format_reward,
            ]

        elif grpo_config.dataset == "acecode":
            from reward_func import code_reward, get_code_format_reward
            # we thank the diffucoder authors for this part of script.
            # https://github.com/apple/ml-diffucoder
            from datasets import load_dataset
            dataset = load_dataset("json", data_files="dataset/acecode/acecode_hard.jsonl")["train"]
            def make_conversation(example, prompt_column: str = "question"):
                prompt = []
                # prompt.append({"role": "user", "content": "You are an expert Python programmer, and here is your task: " + example[prompt_column] + (' Your code should pass these tests:\n\n' + example['test_list'][0] if '\n```\n' not in example[prompt_column] else '')})
                prompt.append({"role": "user", "content": example[prompt_column] + ('\nTest cases: ' + example['test_cases'][0] if '\n```\n' not in example[prompt_column] else '')})
                return {"prompt": prompt}

            dataset = dataset.map(make_conversation)
            # dataset = dataset.select(range(600, len(dataset)))
            reward_functions = [
                code_reward, 
                get_code_format_reward("python"),
            ]
        else:
            raise ValueError("Unsupported text-to-text tasks!")
        
        # Shuffle dataset with fixed seed for reproducibility
        dataset = dataset.shuffle(seed=grpo_config.seed)
        
    elif grpo_config.task== "t2i":
        # we suggest you configure your own dataset here.
        dataset = ImagePromptDataset("dataset/GenEval", "train")
        # dataset = get_webdata_prompts("train")
        # dataset = HfPromptDataset(grpo_config.dataset, "train")
        reward_functions = [unifiedreward_score(device)]
        # reward_functions = [hpsv2(device)]

    else:
        raise ValueError("Unsupported task type!")


    # Split dataset if needed
    # if grpo_config.dataset in ["countdown", "sudoku"]:
    #     train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    # else:
    #     train_set = dataset
    train_set = dataset

    # Set up device

    # 4 bit quantization configuration
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    # Load model and tokenizer
    if grpo_config.task=='t2i':
        print("Training for text-to-image tasks, loading MMaDA model...")
        model = MMadaModelLM.from_pretrained(
            grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # quantization_config=bnb_config,
        ).to(device)
        vq_model = get_vq_model_class('magvitv2')
        vq_model = vq_model.from_pretrained(grpo_config.vq_model_path).to(device)
        vq_model.eval()
        vq_model.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, padding_side="left")
    elif grpo_config.task=='t2t':
        print("Training for pure language tasks, Loading LLaDA model...")
        model = LLaDAModelLM.from_pretrained(
            grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # quantization_config=bnb_config,
        ).to(device)
        vq_model = None
        tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("Unsupported task type!")

    print("The padded prompt length is 256.")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=256,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=0.1, use_reserved_token=True)
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    # Initialize and run trainer
    trainer = MaskGRPOTrainer(
        args=grpo_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
        processing_class=uni_prompting.text_tokenizer,
        uni_processing_class=uni_prompting,
        vq_model=vq_model
    )

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((MaskGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
