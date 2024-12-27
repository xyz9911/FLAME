#!/bin/bash

GPUS=${1:-"0"}
CONFIG_FILE="ds_zero1_config.json"

deepspeed --include "localhost:$GPUS" train_flame_deepspeed.py \
    --deepspeed $CONFIG_FILE \
    --legacy_navigation_mode 1 \
    --train_if_data_path "dataset/touchdown/ft_data/train.json" \
    --eval_if_data_path "dataset/touchdown/ft_data/dev.json" \
    --eval_split "dev" \
    --dataset "dataset/touchdown" \
    --num_train_epochs 25 \
