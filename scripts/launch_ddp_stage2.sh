#!/bin/bash

#############################################################################
# Launch Script for Stage 2 Training with DistributedDataParallel (DDP)
#
# This script launches Stage 2 training on multiple GPUs using torchrun.
#
# Usage:
#   bash scripts/launch_ddp_stage2.sh \
#     --stage1_checkpoint outputs/stage1/best_model \
#     --train_data data/stage2_train.jsonl \
#     --output_dir outputs/stage2
#
# Environment variables (set these if needed):
#   - NUM_GPUS: Number of GPUs to use (default: all available)
#   - MASTER_PORT: Port for communication (default: 29500)
#
# Example - 8 GPUs:
#   bash scripts/launch_ddp_stage2.sh \
#     --stage1_checkpoint outputs/stage1/best_model \
#     --train_data data/stage2_train.jsonl \
#     --eval_data data/stage2_eval.jsonl \
#     --output_dir outputs/stage2 \
#     --batch_size 8 \
#     --num_epochs 1 \
#     --fp16 \
#     --gradient_checkpointing
#
# Example - 4 GPUs:
#   NUM_GPUS=4 bash scripts/launch_ddp_stage2.sh \
#     --stage1_checkpoint outputs/stage1/best_model \
#     --train_data data/stage2_train.jsonl \
#     --output_dir outputs/stage2
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
echo "Launching Stage 2 DDP Training"
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
    scripts/train_stage2.py "$@"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "Stage 2 Training Completed Successfully!"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "Stage 2 Training Failed!"
    echo "================================================================================"
    exit 1
fi
