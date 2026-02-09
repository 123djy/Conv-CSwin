
base_dir="./ccswin"
mkdir -p ${base_dir}

export CUDA_VISIBLE_DEVICES=0

python ./train.py \
    --model Mesorch_ConvNeXt_CSWinB \
    --conv_pretrain True \
    --image_size 512 \
    --data_path "/data/jdon492/Mesorch_with_pretrain_weight/balanced_dataset.json" \
    --test_data_path "/data/jdon492/Mesorch_with_pretrain_weight/public_datasets/IML/CASIA1_processed" \
    --batch_size 12 \
    --if_resizing \
    --epochs 150 \
    --lr 1e-4 \
    --min_lr 5e-7 \
    --warmup_epochs 2 \
    --weight_decay 0.05 \
    --accum_iter 2 \
    --seed 46 \
    --test_period 2 \
    --num_workers 1 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1> ${base_dir}/logs.log
