#!/bin/bash
GEN_LENGTH=256
STEP=128
BLOCK_LEN=16
#
MARK=mark
OUTPUT_NAME=$MARK-$GEN_LENGTH-$STEP

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node 8 \
    --master_port 23333 \
    distributed_inference_t2t.py \
    --dataset math \
    --batch_size 8 \
    --gen_length $GEN_LENGTH \
    --diffusion_steps $STEP \
    --block_length $BLOCK_LEN \
    --output_dir generated_samples/math500/$OUTPUT_NAME \
    --model_path /ssd/models/llada-8b-instruct \
    --checkpoint_path outputs/math500_$MARK/checkpoint-4500

python eval/get_math_accuracy.py \
    --mark $OUTPUT_NAME
    --task math500
