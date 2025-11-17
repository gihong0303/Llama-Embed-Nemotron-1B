"""
Utility functions module.
"""

from .model_merging import ModelMerger, merge_models_uniform, merge_models_weighted
from .distributed import (
    is_distributed,
    get_rank,
    get_world_size,
    get_local_rank,
    is_main_process,
    init_distributed,
    cleanup_distributed,
    barrier,
    all_reduce,
    all_gather,
    reduce_dict,
    print_rank_0,
    save_on_rank_0,
)

__all__ = [
    'ModelMerger',
    'merge_models_uniform',
    'merge_models_weighted',
    'is_distributed',
    'get_rank',
    'get_world_size',
    'get_local_rank',
    'is_main_process',
    'init_distributed',
    'cleanup_distributed',
    'barrier',
    'all_reduce',
    'all_gather',
    'reduce_dict',
    'print_rank_0',
    'save_on_rank_0',
]
