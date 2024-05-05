#!/bin/bash

python3 main.py \
    --image_size 224 \
    --clip_pretrained_model "openai/clip-vit-base-patch32" \
    --map_dim 768 \
    --num_pre_output_layers 1 \
    --drop_probs 0.1 0.4 0.2 \
    --gpus 0 \
    --max_steps -1 \
    --max_epochs -1 \
    --batch_size 9 \
    --lr 1e-4 \
    --weight_image_loss 1.0 \
    --weight_text_loss 1.0 \
    --weight_decay 1e-4
