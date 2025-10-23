#!/bin/bash
python train.py \
    --data_path "/home/test3/test3/test3/wgy/MS/FlowVM-Net-main/data/vessel/" \
    --batch_size 4 \
    --epochs 50 \
    --num_frames 2 \
    --num_classes 1 \
    --gpu_id "3" \
    --cfg "./configs/spring-M.json" \
    --model "./pre_trained_weights/Tartan-C-T-TSKH-spring540x960-M.pth" \
    --device "cuda:3"
