#!/bin/bash

base_dir="./output_dir_mesorch_seed_47"
mkdir -p ${base_dir}

# 只用 1 张 GPU
export CUDA_VISIBLE_DEVICES=0

python3 train.py \
    --model Mesorch \
    --conv_pretrain True \
    --seg_pretrain_path "./mit_b3.pth" \
    --batch_size 12 \
    --data_path "/data/jdon492/Mesorch_with_pretrain_weight/balanced_dataset.json" \
    --epochs 150 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --test_data_path "/data/jdon492/Mesorch_with_pretrain_weight/public_datasets/IML/CASIA1_processed" \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 2 \
    --seed 47 \
    --test_period 2 \
    --num_workers 1 \
    2> ${base_dir}/error.log 1>${base_dir}/logs.log
