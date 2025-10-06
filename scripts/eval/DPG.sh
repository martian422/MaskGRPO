accelerate launch \
    --main_process_port=29500 \
    --num_processes=8 \
    distributed_inference_t2i.py \
    --task DPG \
    --path /mydata/models/models/mmada-8b-base \
    --sample_per_prompt 4 \
    --sample_steps 32 \
    --cfg 3.5 \

# for t2i bench, you'd better launch a new environment and test it.