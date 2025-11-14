"""
Training module for embedding models.
"""

from .config import TrainingConfig, Stage1Config, Stage2Config, get_stage1_config, get_stage2_config
from .loss import InfoNCELoss, MultiTaskContrastiveLoss, TripletLoss
from .trainer import EmbeddingTrainer

__all__ = [
    'TrainingConfig',
    'Stage1Config',
    'Stage2Config',
    'get_stage1_config',
    'get_stage2_config',
    'InfoNCELoss',
    'MultiTaskContrastiveLoss',
    'TripletLoss',
    'EmbeddingTrainer',
]
