# The name of experiment
gpuid=${1:-1}
export PATH="~/anaconda/bin:$PATH"
source activate vlt5
export CUDA_VISIBLE_DEVICES=$gpuid
name=VLBart

output=/data/***/vlt5/snap/transfomer_open_train_vlbart_vqav2

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 29441 \
    src/vqa.py \
        --distributed --multiGPU \
        --train vqav2_open_train \
        --valid vqav2_open_val \
        --test vqav2_open_test \
        --optim adamw \
        --epochs 5 \
        --num_workers 10 \
        --backbone 'facebook/bart-base' \
        --individual_vis_layer_norm False \
        --output $output ${@:2} \
        --load /data/***/VQACL/snap/pretrain/VLBart/pretrain_without_vqa \
        --num_beams 5 \
        --batch_size 200 \
        --valid_batch_size 100 \
