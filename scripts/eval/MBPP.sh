#!/bin/bash

# Remember to launch the e2b sandbox first by running scripts/run_e2b.sh
MODEL_PATH=/ssd/models/llada-8b-instruct

accelerate launch \
    --main_process_port=23333 \
    --num_processes 8 \
    MBPP_eval.py \
    --model_path "$MODEL_PATH" \
    --gen_len 256 \
    --steps 256 \
    --block_len 32 \
    --batch_size 8 \
    --num_samples 500
    # --ckpt_start 5000 \
    # --ckpt_end 0 \
    # --ckpt_step 500
