#!/bin/bash
# Pre-training configuration of the released VISTA-PATH checkpoint:
# 4 GPUs x 32 per-device batch x 4 gradient-accumulation steps = 512 effective batch.
# Run from the repository root. Build train.csv first with build_train_csv.py,
# or point --csv_file at examples/train_toy.csv for a quick check.

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

exp_name=VISTA-PATH

accelerate launch --num_processes 4 --num_machines 1 --main_process_port=${MASTER_PORT} \
    train.py \
    --csv_file ./train.csv \
    --json_dir ./idx_to_names \
    --output_dir ./checkpoints/${exp_name}/ \
    --batch_size 32 \
    --grad_accum_steps 4 \
    --num_epochs 10 \
    --crop_size 512 \
    --tune_mode freeze \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --augmentations \
    --num_workers 8 \
    --resume
