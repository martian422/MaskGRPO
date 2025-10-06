#!/bin/bash
export WANDB_PROJECT=T2I
export WANDB_BASE_URL=https://api.bandw.top
export HF_ENDPOINT=https://hf-mirror.com

DATASET="/mydata/datasets/mj-prompts"
MARK='GenEval_HPSv3_CLIP'

MODEL_PATH=/mydata/models/models/mmada-8b-base
# NUM_ITER=8   # number of policy gradient inner updates iterations, do not cause OOM.
MIN_MASK=0.8
PER_DEVICE_BS=3  # to avoid OOM, set to 6 at MAX. if cfg enabled, 3 at max.
MULTI=3
ROLLOUTS=$(( MULTI * PER_DEVICE_BS ))
NUM_PROCESS=6
ACCUMULATION=4
CFG=3.5
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7  # we use 2 machines to only run server.

RUN_NAME=${MARK}_CFG${CFG}_itr${NUM_ITER}

echo "==============================="
echo "MIN_MASK = $MIN_MASK."
echo "The Global Batch Size is $(( ACCUMULATION * PER_DEVICE_BS * NUM_PROCESS ))"
echo "The Roll Out for each prompt is $ROLLOUTS."
echo "The Update Iteration is $NUM_ITER."
echo "==============================="

accelerate launch \
    --config_file grpo/accelerate.yaml \
    --num_processes $NUM_PROCESS \
    --main_process_port 23333 grpo/mask_grpo_train.py \
    --config configs/rl.yaml \
    --version uni \
    --min_mask_rate $MIN_MASK \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --num_generations $ROLLOUTS \
    --generation_batch_size 1 \
    --per_device_train_batch_size $PER_DEVICE_BS \
    --gradient_accumulation_steps $ACCUMULATION \
    --dataset $DATASET \
    --task t2i \
    --epsilon 0.001 \
    --mean_method grpo \
    --cfg_scale $CFG \
    --run_name $RUN_NAME \
    --output_dir outputs/$RUN_NAME
