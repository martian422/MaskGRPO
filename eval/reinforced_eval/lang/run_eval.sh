#!/bin/bash

# Configuration variables
GPU_IDS=(0 1 2 3 4 5 6 7)

MASTER_PORT=29412

# Arrays of tasks and generation lengths
TASKS=("gsm8k")
GEN_LENGTHS=(512)

# Set GPU IDs from command line if provided
if [ $# -gt 0 ]; then
  # Clear default GPU list and add provided GPUs
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Set batch size based on generation length
    if [ "$gen_length" -eq 512 ]; then
      batch_size=16
    else
      batch_size=8
    fi
    
    echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size"
    
    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval/eval.py \
      --dataset $task \
      --batch_size $batch_size \
      --gen_length $gen_length \
      --output_dir "eval_results_fs_5_new_rl" \
      --model_path "/home/ZhangMu/Workspace/models/mmada-8b-mixcot" \
      --few_shot 5 \
      --checkpoint_path "/home/ZhangMu/Workspace/repos/mmada_grpo/checkpoints/gsm8k_base_bs12/checkpoint-1500"
  done
done


echo "All evaluations completed!"
