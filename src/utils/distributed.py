"""
Distributed Training Utilities

PyTorch DistributedDataParallel (DDP) helpers for multi-GPU training.
Follows PyTorch official best practices.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank."""
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """Get local rank (GPU ID on current node)."""
    if not is_distributed():
        return 0
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """Check if current process is main (rank 0)."""
    return get_rank() == 0


def init_distributed(backend: str = 'nccl') -> None:
    """
    Initialize distributed training.

    Args:
        backend: Backend to use ('nccl' for GPU, 'gloo' for CPU)

    Environment variables required:
        - RANK: Global rank of the process
        - WORLD_SIZE: Total number of processes
        - LOCAL_RANK: Local rank on the current node
        - MASTER_ADDR: IP address of rank 0 node
        - MASTER_PORT: Port for communication

    Usage:
        torchrun --nproc_per_node=8 train.py
    """
    if is_distributed():
        print(f"Distributed already initialized (rank {get_rank()}/{get_world_size()})")
        return

    # Check if we're in a distributed environment
    if 'RANK' not in os.environ:
        print("Not in distributed environment, running on single GPU/CPU")
        return

    # Get distributed parameters from environment
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )

    # Set device for current process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    if is_main_process():
        print(f"Distributed training initialized:")
        print(f"  Backend: {backend}")
        print(f"  World size: {world_size}")
        print(f"  GPUs per node: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        print(f"  Device: {device}")

    # Synchronize all processes
    barrier()


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    All-reduce operation across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation (SUM, AVG, etc.)

    Returns:
        Reduced tensor
    """
    if not is_distributed():
        return tensor

    dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes.

    Args:
        tensor: Tensor to gather (shape: [batch_size, ...])

    Returns:
        Gathered tensor (shape: [batch_size * world_size, ...])
    """
    if not is_distributed():
        return tensor

    world_size = get_world_size()

    # Create tensor list for gathering
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]

    # Gather
    dist.all_gather(tensor_list, tensor)

    # Concatenate
    gathered = torch.cat(tensor_list, dim=0)

    return gathered


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce dictionary of tensors across all processes.

    Args:
        input_dict: Dictionary of tensors to reduce
        average: If True, average the values; otherwise sum

    Returns:
        Reduced dictionary
    """
    if not is_distributed():
        return input_dict

    world_size = get_world_size()

    # Convert to tensor
    names = sorted(input_dict.keys())
    values = torch.stack([input_dict[k] for k in names])

    # All-reduce
    dist.all_reduce(values, op=dist.ReduceOp.SUM)

    if average:
        values /= world_size

    # Convert back to dict
    reduced_dict = {k: v.item() for k, v in zip(names, values)}

    return reduced_dict


def print_rank_0(message: str) -> None:
    """Print message only on rank 0."""
    if is_main_process():
        print(message)


def save_on_rank_0(obj, path: str) -> None:
    """Save object only on rank 0."""
    if is_main_process():
        torch.save(obj, path)


def setup_for_distributed(is_main: bool) -> None:
    """
    Disable printing when not in main process.

    This replaces the built-in print function to only print from rank 0.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# Memory optimization helpers

def get_memory_stats(device: Optional[torch.device] = None) -> dict:
    """Get GPU memory statistics."""
    if device is None:
        device = torch.cuda.current_device()

    if not torch.cuda.is_available():
        return {}

    stats = {
        'allocated': torch.cuda.memory_allocated(device) / 1024**3,  # GB
        'reserved': torch.cuda.memory_reserved(device) / 1024**3,  # GB
        'max_allocated': torch.cuda.max_memory_allocated(device) / 1024**3,  # GB
    }

    return stats


def print_memory_stats(prefix: str = "") -> None:
    """Print GPU memory statistics (rank 0 only)."""
    if not is_main_process():
        return

    if not torch.cuda.is_available():
        return

    stats = get_memory_stats()

    print(f"{prefix}GPU Memory:")
    print(f"  Allocated: {stats['allocated']:.2f} GB")
    print(f"  Reserved: {stats['reserved']:.2f} GB")
    print(f"  Max Allocated: {stats['max_allocated']:.2f} GB")


def synchronize() -> None:
    """
    Synchronize CUDA operations.
    Useful for accurate timing and memory measurements.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
