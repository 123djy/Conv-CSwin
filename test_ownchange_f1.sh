base_dir="./测试结果/ccswin_消融_只看cswin结构_无adapter_seed46"
mkdir -p ${base_dir}


CUDA_VISIBLE_DEVICES=0 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=1 \
./test.py \
    --model Mesorch_ConvNeXt_CSWinB \
    --world_size 1 \
    --test_data_json "./test_datasets.json" \
    --checkpoint_path "./训练结果/ccswin_消融_只看cswin结构_无adapter_seed46/checkpoint-134.pth" \
    --test_batch_size 2 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log