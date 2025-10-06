## ImageReward

```
accelerate launch --main_process_port=29500 --num_processes=8 distributed_inference_t2i.py --task ImageReward --sample_per_prompt 5
```

## GenEval

```
accelerate launch --main_process_port=29500 --num_processes=8 distributed_inference_t2i.py --task GenEval --sample_per_prompt 4
```