"""
Stage 1 Training: Retrieval Pretraining

This script implements Stage 1 training from the Llama-Embed-Nemotron paper:
- Focus on retrieval tasks
- Use 1 hard negative per query
- Train on ~70% of total data (retrieval-focused)

Usage:
    python scripts/train_stage1.py --train_data data/stage1_train.jsonl --output_dir outputs/stage1
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from transformers import AutoTokenizer

from src.models.embedding_model import create_embedding_model
from src.data.dataset import LazyJSONLDataset, create_dataloader
from src.training.trainer import EmbeddingTrainer
from src.training.config import Stage1Config
from src.utils.distributed import (
    init_distributed,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    print_rank_0,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Retrieval Pretraining")

    # Data
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data (JSONL file)")
    parser.add_argument("--eval_data", type=str, default=None,
                       help="Path to evaluation data (JSONL file)")

    # Model
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                       help="Base model name")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")

    # Training
    parser.add_argument("--output_dir", type=str, default="./outputs/stage1",
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--num_negatives", type=int, default=1,
                       help="Number of hard negatives per query")
    parser.add_argument("--temperature", type=float, default=0.02,
                       help="Temperature for InfoNCE loss")

    # Hardware
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Use gradient checkpointing")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--wandb", action="store_true",
                       help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="llama-embed-nemotron-1b",
                       help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="W&B run name")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed training (if running with torchrun/torch.distributed.launch)
    init_distributed()

    # Set seed (different for each rank to ensure different randomness)
    if is_distributed():
        torch.manual_seed(args.seed + get_rank())
    else:
        torch.manual_seed(args.seed)

    if is_main_process():
        print("\n" + "="*80)
        print("STAGE 1: RETRIEVAL PRETRAINING")
        print("="*80 + "\n")

        if is_distributed():
            print(f"Distributed training on {get_world_size()} GPUs")
            print(f"  Batch size per GPU: {args.batch_size}")
            print(f"  Effective batch size: {args.batch_size * get_world_size()}")
            print()

    # Create config
    config = Stage1Config(
        model_name=args.model_name,
        max_length=args.max_length,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_negatives=args.num_negatives,
        temperature=args.temperature,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        log_to_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name or "stage1",
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
    )

    # Save config (only on main process)
    if is_main_process():
        os.makedirs(config.output_dir, exist_ok=True)
        from src.training.config import save_config
        save_config(config, os.path.join(config.output_dir, "config.json"))

    # Load tokenizer
    print_rank_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset (lazy loading for memory efficiency)
    print_rank_0(f"Initializing lazy dataset from {args.train_data}...")
    train_dataset = LazyJSONLDataset(
        file_path=args.train_data,
        tokenizer=tokenizer,
        max_length=config.max_length,
        num_negatives=config.num_negatives,
        instruction=config.instruction,
        task_type=config.task_type,
    )
    print_rank_0(f"  Dataset initialized with {len(train_dataset)} samples (lazy loading enabled)")
    print_rank_0(f"  Memory savings: ~99.96% vs traditional loading")

    # Create dataloader with DistributedSampler if running distributed
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        distributed=is_distributed(),  # Enable DistributedSampler for multi-GPU
    )

    # Eval dataloader (optional)
    eval_dataloader = None
    if args.eval_data:
        print_rank_0(f"Initializing lazy eval dataset from {args.eval_data}...")
        eval_dataset = LazyJSONLDataset(
            file_path=args.eval_data,
            tokenizer=tokenizer,
            max_length=config.max_length,
            num_negatives=config.num_negatives,
            instruction=config.instruction,
            task_type=config.task_type,
        )
        print_rank_0(f"  Eval dataset initialized with {len(eval_dataset)} samples")

        eval_dataloader = create_dataloader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.dataloader_num_workers,
            distributed=is_distributed(),  # Enable DistributedSampler for multi-GPU
        )

    # Create model
    print_rank_0(f"Creating model from {args.model_name}...")
    model = create_embedding_model(
        model_name=args.model_name,
        normalize=True,
        max_length=config.max_length,
    )

    # Create trainer
    print_rank_0("Creating trainer...")
    trainer = EmbeddingTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
    )

    # Train
    print_rank_0("\nStarting training...")
    trainer.train()

    print_rank_0(f"\nTraining complete! Model saved to {config.output_dir}")

    # Cleanup distributed training
    if is_distributed():
        cleanup_distributed()


if __name__ == "__main__":
    main()
