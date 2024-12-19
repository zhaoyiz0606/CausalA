# The name of experiment
gpuid=${1:-0}
export PATH="~/anaconda/bin:$PATH"
source activate vlt5
export CUDA_VISIBLE_DEVICES=$gpuid
name=VLT5

output=/data/***/vlt5/snap/transfomer_open_train_vlt5_vqav2

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 29231 \
    src/vqa.py \
        --distributed --multiGPU \
        --train vqav2_open_train \
        --valid vqav2_open_val \
        --test vqav2_open_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 10 \
        --backbone "t5-base" \
        --output $output ${@:2} \
        --load VLT5/pretrain_without_vqa \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 100
