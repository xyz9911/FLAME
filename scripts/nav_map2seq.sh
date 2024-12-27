#!/bin/bash

GPU=$1
CHECKPOINT_DIR=$2
SPLIT=$3
shift 3

for CHECKPOINT_NUM in "$@"; do
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint-${CHECKPOINT_NUM}"

    echo "====================================="
    echo "Evaluating checkpoint: $CHECKPOINT_PATH"
    echo "====================================="

    CUDA_VISIBLE_DEVICES=$GPU python navigate.py \
        --legacy_navigation_mode=1 \
        --checkpoint_path="$CHECKPOINT_PATH" \
        --train_if_data_path="dataset/map2seq/ft_data/train.json" \
        --eval_if_data_path="dataset/map2seq/ft_data/$SPLIT.json" \
        --eval_split="$SPLIT" \
        --dataset="dataset/map2seq" \
        --env_batch_size=4
done
