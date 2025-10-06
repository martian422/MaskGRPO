#!/bin/bash
# in case of WANDB error
export WANDB_BASE_URL=https://api.bandw.top

DATASET=gsm8k
BLOCK_LEN=16
MARK='mark'

MODEL_PATH=/ssd/models/llada-8b-instruct
NUM_ITER=12 # number of policy gradient inner updates iterations, do not cause OOM.
PER_DEVICE_BS=6
MULTI=1
ROLLOUTS=$(( MULTI * PER_DEVICE_BS )) # may lead to OOM if large and not optimized.
NUM_PER_PASS=6 # better set it as divider of PER_DEVICE_BS, or the same. For t2t, OOM is less likely to happen.
ACCUMULATION=2
NUM_PROCESS=8
MIN_MASK=0.6 # for d1, it do not work.

RUN_NAME=${DATASET}_${MARK}_min${MIN_MASK}

# The actual global batch size (which means how many data is used for one gradient update) 
# is gradient_accumulation_steps * num_processes * per_device_train_batch_size.
echo "==============================="
echo "The Global Batch Size is $(( ACCUMULATION * PER_DEVICE_BS * NUM_PROCESS ))"
echo "The Roll Out for each prompt is $ROLLOUTS."
echo "The Update Iteration is $NUM_ITER."
echo "==============================="

# And the NUM_ITER reuses the rollout completions. The longer you set it, more global steps it will take.

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch \
    --config_file grpo/accelerate.yaml \
    --num_processes $NUM_PROCESS \
    --main_process_port 23333 grpo/mask_grpo_train.py \
    --config configs/rl.yaml \
    --version uni \
    --min_mask_rate $MIN_MASK \
    --remasking arlike \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --num_generations $ROLLOUTS \
    --generation_batch_size $NUM_PER_PASS \
    --per_device_train_batch_size $PER_DEVICE_BS \
    --gradient_accumulation_steps $ACCUMULATION \
    --max_completion_length 256 \
    --diffusion_steps 128 \
    --block_length $BLOCK_LEN \
    --dataset $DATASET \
    --task t2t \
    --run_name $RUN_NAME \
    --mean_method grpo \
    --output_dir outputs/$RUN_NAME 
