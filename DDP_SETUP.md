# DistributedDataParallel (DDP) Training Guide

This document explains how to use the multi-GPU training implementation with PyTorch DistributedDataParallel (DDP).

## Overview

The implementation supports distributed training across multiple GPUs using PyTorch's DDP, following official best practices. This fixes OOM (Out of Memory) issues by properly distributing the model and data across all available GPUs.

## Features

✅ **DistributedDataParallel (DDP)** - Proper multi-GPU training
✅ **DistributedSampler** - Automatic dataset sharding across GPUs
✅ **Gradient Accumulation** - Controlled gradient synchronization
✅ **Mixed Precision (FP16)** - Memory-efficient training
✅ **Gradient Checkpointing** - Further memory savings
✅ **Rank-0 Saving** - Checkpoints saved only on main process
✅ **Proper Cleanup** - Distributed process group cleanup

## Quick Start

### 1. Single GPU Training (No Changes Needed)

```bash
# Works as before - automatically uses single GPU
python scripts/train_stage1.py \
    --train_data data/stage1_train.jsonl \
    --output_dir outputs/stage1 \
    --batch_size 32
```

### 2. Multi-GPU Training (8 GPUs)

```bash
# Use the launch script for DDP training
bash scripts/launch_ddp_stage1.sh \
    --train_data data/stage1_train.jsonl \
    --output_dir outputs/stage1 \
    --batch_size 16 \
    --fp16 \
    --gradient_checkpointing
```

**Important**: When using 8 GPUs, reduce batch_size per GPU to avoid OOM:
- **Single GPU**: batch_size=32
- **8 GPUs**: batch_size=16 (effective batch size = 16 × 8 = 128)

## Architecture

### Components

```
src/utils/distributed.py          # DDP utilities
src/training/trainer.py            # DDP-aware trainer
src/data/dataset.py                # DistributedSampler support
scripts/train_stage1.py            # DDP initialization
scripts/train_stage2.py            # DDP initialization
scripts/launch_ddp_stage1.sh       # Launch script for Stage 1
scripts/launch_ddp_stage2.sh       # Launch script for Stage 2
```

### How It Works

1. **Initialization**: `init_distributed()` sets up process group using NCCL backend
2. **Model Wrapping**: Model wrapped in `DistributedDataParallel` wrapper
3. **Data Sharding**: `DistributedSampler` splits data across GPUs
4. **Gradient Sync**: `model.no_sync()` controls when gradients are synchronized
5. **Checkpointing**: Only rank 0 saves checkpoints (unwraps DDP model first)
6. **Cleanup**: `cleanup_distributed()` destroys process group

## Detailed Usage

### Stage 1 Training

#### Basic 8 GPU Training
```bash
bash scripts/launch_ddp_stage1.sh \
    --train_data data/stage1_train.jsonl \
    --eval_data data/stage1_eval.jsonl \
    --output_dir outputs/stage1 \
    --batch_size 16 \
    --num_epochs 1 \
    --fp16 \
    --gradient_checkpointing
```

#### 4 GPU Training
```bash
NUM_GPUS=4 bash scripts/launch_ddp_stage1.sh \
    --train_data data/stage1_train.jsonl \
    --output_dir outputs/stage1 \
    --batch_size 24
```

#### Custom Port (if 29500 is busy)
```bash
MASTER_PORT=29501 bash scripts/launch_ddp_stage1.sh \
    --train_data data/stage1_train.jsonl \
    --output_dir outputs/stage1
```

### Stage 2 Training

```bash
bash scripts/launch_ddp_stage2.sh \
    --stage1_checkpoint outputs/stage1/best_model \
    --train_data data/stage2_train.jsonl \
    --eval_data data/stage2_eval.jsonl \
    --output_dir outputs/stage2 \
    --batch_size 8 \
    --num_epochs 1 \
    --fp16 \
    --gradient_checkpointing
```

## Effective Batch Size

When training with multiple GPUs, the **effective batch size** is:

```
Effective Batch Size = batch_size × num_gpus × gradient_accumulation_steps
```

### Example Configurations

#### Paper Settings (64 GPUs, batch_size=2048)
```
2048 = 32 × 64 × 1
```

#### Our 8 GPU Setup (effective batch_size=256)
```bash
# Option 1: batch_size=32, no gradient accumulation
bash scripts/launch_ddp_stage1.sh --batch_size 32
# Effective: 32 × 8 × 1 = 256

# Option 2: batch_size=16, gradient_accumulation=2
bash scripts/launch_ddp_stage1.sh --batch_size 16
# Effective: 16 × 8 × 2 = 256 (with gradient_accumulation_steps=2 in config)
```

## Memory Optimization

### Reduce Memory Usage

If you encounter OOM errors, try these strategies in order:

1. **Reduce batch_size per GPU**
```bash
--batch_size 8  # Instead of 16
```

2. **Enable FP16 mixed precision**
```bash
--fp16
```

3. **Enable gradient checkpointing**
```bash
--gradient_checkpointing
```

4. **Reduce sequence length**
```bash
--max_length 256  # Instead of 512
```

5. **Use gradient accumulation** (edit config to add `gradient_accumulation_steps=4`)

### Memory Estimates

For Llama-3.2-1B on A100 80GB GPUs:

| Configuration | Memory per GPU | Batch Size | Effective Batch |
|--------------|----------------|------------|-----------------|
| FP32, no checkpoint | ~45 GB | 8 | 64 (8 GPUs) |
| FP16, no checkpoint | ~25 GB | 16 | 128 (8 GPUs) |
| FP16 + checkpoint | ~15 GB | 32 | 256 (8 GPUs) |

## Monitoring

### Check GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use gpustat (install: pip install gpustat)
watch -n 1 gpustat -cp
```

### Expected Output

When training starts, you should see:

```
Distributed training initialized:
  Backend: nccl
  World size: 8
  GPUs per node: 8
  Device: cuda:0

Distributed training on 8 GPUs
  Batch size per GPU: 16
  Effective batch size: 128

Model wrapped in DistributedDataParallel
  World size: 8
  Rank: 0
  Device: cuda:0
```

## Troubleshooting

### Issue: Port Already in Use

**Error**:
```
RuntimeError: Address already in use
```

**Solution**:
```bash
MASTER_PORT=29501 bash scripts/launch_ddp_stage1.sh ...
```

### Issue: NCCL Timeout

**Error**:
```
RuntimeError: NCCL operation timed out
```

**Solutions**:
1. Check GPU connectivity: `nvidia-smi topo -m`
2. Ensure all GPUs are accessible: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
3. Increase timeout (add to script): `export NCCL_BLOCKING_WAIT=1`

### Issue: Different Output on Different GPUs

**Symptom**: GPUs show different losses or metrics

**Cause**: This is normal! Each GPU processes different data shards.

**Verification**: Only rank 0 prints metrics. All GPUs contribute to gradient updates.

### Issue: OOM on Some GPUs but Not Others

**Cause**: Uneven data distribution or model initialization

**Solution**:
1. Ensure `DistributedSampler` is used (automatic in our implementation)
2. Check that `batch_size` is the same for all GPUs
3. Verify no hardcoded device assignments in custom code

## Performance Tips

### 1. Optimal Number of Workers

```bash
--num_workers 4  # Good default (4 workers per GPU)
```

Too many workers → CPU overhead
Too few workers → GPU starvation

### 2. Pin Memory

Already enabled by default in `create_dataloader()`:
```python
pin_memory=True  # Faster host-to-device transfers
```

### 3. Avoid Synchronization Bottlenecks

The implementation already uses `model.no_sync()` during gradient accumulation to minimize synchronization overhead.

### 4. Use NCCL Optimizations

```bash
# Add to launch script for better performance
export NCCL_DEBUG=INFO  # For debugging (remove in production)
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=3  # Enable GPUDirect RDMA
```

## Comparison: Single GPU vs 8 GPUs

### Single GPU (32GB VRAM)

```bash
python scripts/train_stage1.py \
    --train_data data/stage1_train.jsonl \
    --output_dir outputs/stage1 \
    --batch_size 8 \
    --fp16 \
    --gradient_checkpointing
```

**Limitations**:
- Small batch size (8)
- Slower training
- May OOM with larger models

### 8 GPUs (8× 80GB = 640GB total)

```bash
bash scripts/launch_ddp_stage1.sh \
    --train_data data/stage1_train.jsonl \
    --output_dir outputs/stage1 \
    --batch_size 32 \
    --fp16 \
    --gradient_checkpointing
```

**Benefits**:
- Large effective batch size (32 × 8 = 256)
- 8× faster training (near-linear scaling)
- No OOM issues
- Matches paper methodology

## Implementation Details

### Gradient Synchronization

```python
# During gradient accumulation
if not should_sync:
    with self.model.no_sync():  # Don't sync gradients
        loss.backward()
else:
    loss.backward()  # Sync gradients (last accumulation step)
```

This minimizes communication overhead by only synchronizing when needed.

### Checkpoint Saving

```python
# Only rank 0 saves checkpoints
if is_main_process():
    # Unwrap DDP model before saving
    model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
    model_to_save.save_pretrained(checkpoint_dir)
```

This prevents conflicts and ensures only one checkpoint is written.

### DistributedSampler Epoch Setting

```python
# Set epoch for proper shuffling
if is_distributed() and hasattr(self.train_dataloader.sampler, 'set_epoch'):
    self.train_dataloader.sampler.set_epoch(epoch)
```

This ensures different shuffling in each epoch while maintaining reproducibility.

## Advanced Usage

### Manual Launch with torchrun

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    scripts/train_stage1.py \
    --train_data data/stage1_train.jsonl \
    --output_dir outputs/stage1 \
    --batch_size 16
```

### Multi-Node Training (Advanced)

For training across multiple machines:

**Node 0 (master)**:
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    scripts/train_stage1.py ...
```

**Node 1**:
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    scripts/train_stage1.py ...
```

## Verification

### Test DDP Setup

```bash
# Quick test with small batch
bash scripts/launch_ddp_stage1.sh \
    --train_data data/stage1_train.jsonl \
    --output_dir test_outputs \
    --batch_size 2 \
    --num_epochs 1 \
    --logging_steps 1

# Check output for:
# ✓ "Distributed training on X GPUs"
# ✓ "Model wrapped in DistributedDataParallel"
# ✓ All GPUs showing in nvidia-smi
# ✓ No OOM errors
```

### Benchmark Speed

```bash
# Single GPU
time python scripts/train_stage1.py --batch_size 8 ...

# 8 GPUs
time bash scripts/launch_ddp_stage1.sh --batch_size 8 ...

# Expected: 8 GPU version should be ~6-7x faster (due to communication overhead)
```

## References

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch DDP Best Practices](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)

---

**Last Updated**: 2025-01-17
**Tested On**: 8× NVIDIA A100 80GB GPUs, PyTorch 2.0+, CUDA 11.8+
