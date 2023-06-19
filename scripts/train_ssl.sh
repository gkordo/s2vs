#!/bin/sh
VIDEO_DATASET=/path/to/extracted/frames
EXPERIMENTS=./experiments/

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 train.py \
          --dataset_path $VIDEO_DATASET \
          --experiment_path $EXPERIMENTS \
          --augmentations GT,FT,TT,ViV \
          --iter_epochs 30000 \
          --batch_sz 64 \
          --batch_sz_fe 512 \
          --workers 12 \
          --log_step 100 \
          --window_sz 32 \
          --learning_rate 5e-5 \
          --weight_decay 0.01 \
          --temperature 0.03 \
          --lambda_parameter 3. \
          --r_parameter 1. \
          --use_fp16 true \
