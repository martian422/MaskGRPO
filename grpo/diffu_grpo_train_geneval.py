import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig
from models import MMadaModelLM, MAGVITv2
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
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
    reward_clip_score,
    hpsv2,
    hpsv3,
    hpsv3_remote,
    geneval_score,
    unifiedreward_score
)
from data_utils import (
    set_random_seed,
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
        # dataset = dataset.shuffle(seed=grpo_config.seed)
        raise ValueError("Unsupported text-to-text tasks!")
        
        # Shuffle dataset with fixed seed for reproducibility
        
        
    elif grpo_config.task== "t2i":
        dataset = ImagePromptDataset("dataset/GenEval", "train")
        # dataset = HfPromptDataset(grpo_config.dataset, "train")
        reward_functions = [reward_clip_score(device), hpsv3_remote(device)]
        # reward_functions = [reward_clip_score(device), unifiedreward_score(device)]
        # reward_functions = [hpsv2(device)]

    else:
        raise ValueError("Unsupported task type!")


    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
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
    model = MMadaModelLM.from_pretrained(
        grpo_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config,
    ).to(device)

    if grpo_config.task=='t2i':
        vq_model = get_vq_model_class('magvitv2')
        vq_model = vq_model.from_pretrained(grpo_config.vq_model_path).to(device)
        vq_model.eval()
        vq_model.requires_grad_(False)
    else:
        vq_model = None

    # tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # /mydata/models/models/mmada-8b-base
    # tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, padding_side="left")
    tokenizer = AutoTokenizer.from_pretrained('/mydata/models/models/mmada-8b-base', padding_side="left")
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
