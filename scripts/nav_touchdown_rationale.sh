#!/bin/bash

GPU=$1
CHECKPOINT_DIR=$2
SPLIT=$3
TEMPERATURE=$4
DECODING_PATHS=$5
shift 5

for CHECKPOINT_NUM in "$@"; do
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint-${CHECKPOINT_NUM}"

    echo "====================================="
    echo "Evaluating checkpoint: $CHECKPOINT_PATH"
    echo "Temperature: $TEMPERATURE"
    echo "Decoding paths: $DECODING_PATHS"
    echo "====================================="

    CUDA_VISIBLE_DEVICES=$GPU python navigate.py \
        --legacy_navigation_mode=0 \
        --checkpoint_path="$CHECKPOINT_PATH" \
        --train_if_data_path="dataset/touchdown/ft_data_rationale/train.json" \
        --eval_if_data_path="dataset/touchdown/ft_data_rationale/$SPLIT.json" \
        --eval_split="$SPLIT" \
        --dataset="dataset/touchdown" \
        --env_batch_size=1 \
        --temperature=$TEMPERATURE \
        --decoding_paths=$DECODING_PATHS
done
