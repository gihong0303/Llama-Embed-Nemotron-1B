"""
Stage 2 Training: Multi-Task Fine-Tuning

This script implements Stage 2 training from the Llama-Embed-Nemotron paper:
- Multi-task learning (retrieval, STS, classification)
- Use 4 hard negatives per query
- Train on ~30% of total data (diverse tasks)
- Load from Stage 1 checkpoint

Usage:
    python scripts/train_stage2.py \
        --stage1_checkpoint outputs/stage1/best_model \
        --train_data data/stage2_train.jsonl \
        --output_dir outputs/stage2
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from transformers import AutoTokenizer

from src.models.embedding_model import InstructionAwareEmbeddingModel
from src.data.dataset import EmbeddingDataset, load_jsonl, create_dataloader, MultiTaskDataset
from src.training.trainer import EmbeddingTrainer
from src.training.config import Stage2Config


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Multi-Task Fine-Tuning")

    # Data
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data (JSONL file or comma-separated list)")
    parser.add_argument("--eval_data", type=str, default=None,
                       help="Path to evaluation data (JSONL file)")
    parser.add_argument("--task_weights", type=str, default=None,
                       help="Task weights (comma-separated, e.g., '0.5,0.3,0.2')")

    # Model
    parser.add_argument("--stage1_checkpoint", type=str, required=True,
                       help="Path to Stage 1 checkpoint")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")

    # Training
    parser.add_argument("--output_dir", type=str, default="./outputs/stage2",
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-6,
                       help="Learning rate (lower than Stage 1)")
    parser.add_argument("--num_negatives", type=int, default=4,
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

    # Set seed
    torch.manual_seed(args.seed)

    print("\n" + "="*80)
    print("STAGE 2: MULTI-TASK FINE-TUNING")
    print("="*80 + "\n")

    # Create config
    config = Stage2Config(
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
        wandb_run_name=args.wandb_run_name or "stage2",
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        stage1_checkpoint_path=args.stage1_checkpoint,
    )

    # Parse task weights
    if args.task_weights:
        config.task_weights = [float(w) for w in args.task_weights.split(",")]

    # Save config
    os.makedirs(config.output_dir, exist_ok=True)
    from src.training.config import save_config
    save_config(config, os.path.join(config.output_dir, "config.json"))

    # Load model from Stage 1
    print(f"Loading model from Stage 1 checkpoint: {args.stage1_checkpoint}")
    model = InstructionAwareEmbeddingModel.from_pretrained(args.stage1_checkpoint)
    print("Model loaded successfully")

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.stage1_checkpoint)
    if tokenizer.pad_token is None:
        # Fallback to loading tokenizer from base model
        from src.training.config import Stage1Config
        stage1_config_path = os.path.join(args.stage1_checkpoint, "..", "config.json")
        if os.path.exists(stage1_config_path):
            from src.training.config import load_config
            stage1_config = load_config(stage1_config_path, Stage1Config)
            tokenizer = AutoTokenizer.from_pretrained(stage1_config.model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print(f"Loading training data from {args.train_data}...")

    # Support multiple data files
    if "," in args.train_data:
        data_paths = args.train_data.split(",")
        all_data = []
        for path in data_paths:
            data = load_jsonl(path.strip())
            all_data.extend(data)
            print(f"  Loaded {len(data)} samples from {path}")
        train_data = all_data
    else:
        train_data = load_jsonl(args.train_data)

    print(f"  Total training samples: {len(train_data)}")

    # Create dataset
    # Determine if multi-task or single-task based on data
    task_types = set([sample.get("task_type", "retrieval") for sample in train_data])

    if len(task_types) > 1:
        print(f"  Detected multi-task data: {task_types}")

        # Create separate datasets for each task type
        datasets_by_task = {}
        for task_type in task_types:
            task_data = [s for s in train_data if s.get("task_type", "retrieval") == task_type]
            datasets_by_task[task_type] = EmbeddingDataset(
                data=task_data,
                tokenizer=tokenizer,
                max_length=config.max_length,
                num_negatives=config.num_negatives,
                task_type=task_type,
            )
            print(f"    {task_type}: {len(task_data)} samples")

        # Combine with weights
        dataset_list = [(ds, w) for ds, w in zip(datasets_by_task.values(), config.task_weights[:len(datasets_by_task)])]
        train_dataset = MultiTaskDataset(dataset_list, sampling_strategy="proportional")
    else:
        print(f"  Single task type: {list(task_types)[0]}")
        train_dataset = EmbeddingDataset(
            data=train_data,
            tokenizer=tokenizer,
            max_length=config.max_length,
            num_negatives=config.num_negatives,
        )

    # Create dataloader
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
    )

    # Eval dataloader (optional)
    eval_dataloader = None
    if args.eval_data:
        print(f"Loading evaluation data from {args.eval_data}...")
        eval_data = load_jsonl(args.eval_data)
        print(f"  Loaded {len(eval_data)} evaluation samples")

        eval_dataset = EmbeddingDataset(
            data=eval_data,
            tokenizer=tokenizer,
            max_length=config.max_length,
            num_negatives=config.num_negatives,
        )

        eval_dataloader = create_dataloader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.dataloader_num_workers,
        )

    # Create trainer
    print("Creating trainer...")
    trainer = EmbeddingTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    print(f"\nTraining complete! Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
