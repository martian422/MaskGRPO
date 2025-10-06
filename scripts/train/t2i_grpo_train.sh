#!/bin/bash
export WANDB_PROJECT=HPS
export WANDB_BASE_URL=https://api.bandw.top

DATASET="/data/dataset/flux-dev-portrait/wds/part{0..9}.tar"
BLOCK_LEN=4
MARK='HPS'

MODEL_PATH=/data/models/mmada-8b-base
NUM_ITER=6 # number of policy gradient inner updates iterations, do not cause OOM.
PER_DEVICE_BS=3 # for t2i, to avoid OOM, set to 6 at MAX.
MULTI=3
ROLLOUTS=$(( MULTI * PER_DEVICE_BS )) # may lead to OOM if large and not optimized.
ACCUMULATION=2
NUM_PROCESS=6

RUN_NAME=${MARK}

# The actual global batch size (which means how many data is used for one gradient update) 
# is gradient_accumulation_steps * num_processes * per_device_train_batch_size.
echo "==============================="
echo "The Global Batch Size is $(( ACCUMULATION * PER_DEVICE_BS * NUM_PROCESS ))"
echo "The Roll Out for each prompt is $ROLLOUTS."
echo "The Update Iteration is $NUM_ITER."
echo "==============================="

# And the NUM_ITER reuses the rollout completions. The longer you set it, more global steps it will take.

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

accelerate launch \
    --config_file grpo/accelerate.yaml \
    --num_processes $NUM_PROCESS \
    --main_process_port 23333 grpo/mask_grpo_train.py \
    --config configs/rl.yaml \
    --version uni \
    --report_to none \
    --min_mask_rate 0.6 \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --num_generations $ROLLOUTS \
    --generation_batch_size 1\
    --per_device_train_batch_size $PER_DEVICE_BS \
    --gradient_accumulation_steps $ACCUMULATION \
    --block_length $BLOCK_LEN \
    --dataset $DATASET \
    --task t2i \
    --cfg_scale 3.0 \
    --run_name $RUN_NAME \
    --output_dir outputs/$RUN_NAME 
