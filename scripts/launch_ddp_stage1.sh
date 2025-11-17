#!/bin/bash

#############################################################################
# Launch Script for Stage 1 Training with DistributedDataParallel (DDP)
#
# This script launches Stage 1 training on multiple GPUs using torchrun.
#
# Usage:
#   bash scripts/launch_ddp_stage1.sh --train_data data/stage1_train.jsonl --output_dir outputs/stage1
#
# Environment variables (set these if needed):
#   - NUM_GPUS: Number of GPUs to use (default: all available)
#   - MASTER_PORT: Port for communication (default: 29500)
#
# Example - 8 GPUs:
#   bash scripts/launch_ddp_stage1.sh \
#     --train_data data/stage1_train.jsonl \
#     --eval_data data/stage1_eval.jsonl \
#     --output_dir outputs/stage1 \
#     --batch_size 16 \
#     --num_epochs 1 \
#     --fp16 \
#     --gradient_checkpointing
#
# Example - 4 GPUs:
#   NUM_GPUS=4 bash scripts/launch_ddp_stage1.sh \
#     --train_data data/stage1_train.jsonl \
#     --output_dir outputs/stage1
#############################################################################

# Get number of GPUs (default: all available)
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi

# Get master port (default: 29500)
if [ -z "$MASTER_PORT" ]; then
    MASTER_PORT=29500
fi

echo "================================================================================"
echo "Launching Stage 1 DDP Training"
echo "================================================================================"
echo "GPUs: $NUM_GPUS"
echo "Master Port: $MASTER_PORT"
echo "Arguments: $@"
echo "================================================================================"
echo ""

# Launch training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    scripts/train_stage1.py "$@"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "Stage 1 Training Completed Successfully!"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "Stage 1 Training Failed!"
    echo "================================================================================"
    exit 1
fi
