DATASET="blip3o"
MARK='min0.5_cst'
RUN_NAME=CT_${DATASET}_${MARK}

accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
    --main_process_port 23333 \
    training/train_mmada_sft.py \
    config=configs/sft.yaml \
    dataset.gen_type=t2i \
    dataset.params.train_t2i_shards_path_or_url='/mydata/datasets/blip3o-instruct/*.tar' \
    dataset.params.external_caption_path='' \
    run_name=$RUN_NAME \
    output_dir=outputs/$RUN_NAME