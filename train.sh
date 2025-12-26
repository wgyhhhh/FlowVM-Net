#!/bin/bash
python train.py \
    --data_path "./data/vessel/" \
    --batch_size 16 \
    --epochs 300 \
    --num_frames 2 \
    --num_classes 1 \
    --gpu_id "0" \
    --cfg "./configs/spring-M.json" \
    --model "./pre_trained_weights/Tartan-C-T-TSKH-spring540x960-M.pth" \
