#!/bin/bash

TEMP_DIR="/data2/xd_zm/tmp"
mkdir -p $TEMP_DIR

export TMPDIR=$TEMP_DIR
export TEMP=$TEMP_DIR
export TMP=$TEMP_DIR

find $TMPDIR -name "pymp-*" -type d -mtime +1 -exec rm -rf {} \; 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0,1

NNODES=1
NPROC_PER_NODE=2
MASTER_ADDR=127.0.0.1
MASTER_PORT=29501

nohup torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train.py \
  --ddp \
  --detector_path ./config/detector/intriface.yaml \
  --resume_checkpoint .../epoch_22.pth \
  --resume_mode continue \
  > train.log 2>&1 &

PID=$!