"""
Training Configuration

Based on Llama-Embed-Nemotron paper (Appendix A, Table 8)

Scaled down for Llama-3.2-1B (from 8B):
- Smaller batch size
- Adjusted learning rate
- Fewer GPUs required
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Base training configuration."""

    # Model
    model_name: str = "meta-llama/Llama-3.2-1B"
    max_length: int = 512
    hidden_size: int = 2048  # For Llama-3.2-1B

    # Training
    output_dir: str = "./outputs"
    num_epochs: int = 1
    batch_size: int = 32  # Per device
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Loss
    temperature: float = 0.02
    num_negatives: int = 1
    use_in_batch_negatives: bool = False

    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Hardware
    device: str = "cuda"
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    log_to_wandb: bool = False
    wandb_project: str = "llama-embed-nemotron-1b"
    wandb_run_name: Optional[str] = None

    # Data
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Misc
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None


@dataclass
class Stage1Config(TrainingConfig):
    """
    Stage 1: Retrieval Pretraining

    Based on paper Table 8:
    - Peak LR: 1e-5
    - Batch size: 2048 (we scale down to 256 for 1B model with 8 GPUs, 32 per device)
    - Steps: 5,773 (for 11.8M samples)
    - Hard negatives: 1
    - Temperature: 0.02
    """

    # Stage-specific
    stage: str = "stage1"
    output_dir: str = "./outputs/stage1"

    # Training (scaled for Llama-3.2-1B with fewer resources)
    num_epochs: int = 1
    batch_size: int = 32  # Per device (256 total with 8 GPUs)
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1

    # Loss
    num_negatives: int = 1  # From paper
    temperature: float = 0.02

    # Task
    task_type: str = "retrieval"
    instruction: str = "Retrieve relevant passages for this query"

    # Data (these would be set based on actual data)
    train_data_path: Optional[str] = None
    corpus_path: Optional[str] = None


@dataclass
class Stage2Config(TrainingConfig):
    """
    Stage 2: Multi-Task Fine-Tuning

    Based on paper Table 8:
    - Peak LR: 2e-6
    - Batch size: 128 (we scale down to 128 for 1B model with 4 GPUs, 32 per device)
    - Steps: 33,668 (for 4.3M samples)
    - Hard negatives: 4
    - Temperature: 0.02
    """

    # Stage-specific
    stage: str = "stage2"
    output_dir: str = "./outputs/stage2"

    # Training (scaled for Llama-3.2-1B)
    num_epochs: int = 1
    batch_size: int = 32  # Per device (128 total with 4 GPUs)
    learning_rate: float = 2e-6  # Lower LR for fine-tuning
    warmup_ratio: float = 0.05

    # Loss
    num_negatives: int = 4  # More negatives in stage 2
    temperature: float = 0.02

    # Multi-task
    task_types: List[str] = field(default_factory=lambda: ["retrieval", "sts", "classification"])
    task_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])

    # Data
    train_data_paths: Optional[List[str]] = None

    # Load from stage 1
    load_from_stage1: bool = True
    stage1_checkpoint_path: Optional[str] = None


@dataclass
class ModelMergingConfig:
    """
    Configuration for model merging (Model Soup).

    Based on paper:
    - Train 6 models with different hyperparameters/data mixes
    - Merge them using uniform averaging
    """

    # Models to merge
    model_paths: List[str] = field(default_factory=list)

    # Merging strategy
    merge_strategy: str = "uniform"  # uniform, weighted
    weights: Optional[List[float]] = None

    # Output
    output_path: str = "./outputs/merged_model"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    # Model
    model_path: str = "./outputs/stage2"

    # Evaluation
    batch_size: int = 64
    max_length: int = 512

    # Tasks
    eval_tasks: List[str] = field(default_factory=lambda: [
        "retrieval",
        "sts",
        "classification",
    ])

    # MTEB
    use_mteb: bool = True
    mteb_tasks: Optional[List[str]] = None  # None = all tasks

    # Output
    output_dir: str = "./evaluation_results"


def get_stage1_config(**kwargs) -> Stage1Config:
    """Get Stage 1 configuration with optional overrides."""
    config = Stage1Config()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def get_stage2_config(**kwargs) -> Stage2Config:
    """Get Stage 2 configuration with optional overrides."""
    config = Stage2Config()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def save_config(config: TrainingConfig, path: str):
    """Save configuration to file."""
    import json
    from dataclasses import asdict

    config_dict = asdict(config)

    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Config saved to {path}")


def load_config(path: str, config_class=TrainingConfig):
    """Load configuration from file."""
    import json

    with open(path, "r") as f:
        config_dict = json.load(f)

    config = config_class(**config_dict)

    print(f"Config loaded from {path}")
    return config
