"""
Trainer for Text Embedding Model

Implements training loop with:
- InfoNCE contrastive loss
- Mixed precision training
- Gradient accumulation
- Logging and checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict
import os
from tqdm import tqdm
import json
from pathlib import Path

from ..models.embedding_model import InstructionAwareEmbeddingModel
from .loss import InfoNCELoss, MultiTaskContrastiveLoss
from .config import TrainingConfig


class EmbeddingTrainer:
    """
    Trainer for instruction-aware embedding model.

    Args:
        model: InstructionAwareEmbeddingModel
        train_dataloader: Training DataLoader
        eval_dataloader: Evaluation DataLoader (optional)
        config: TrainingConfig
    """

    def __init__(
        self,
        model: InstructionAwareEmbeddingModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: TrainingConfig = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()

        # Device
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss function
        self.loss_fn = MultiTaskContrastiveLoss(
            temperature=self.config.temperature,
            use_in_batch_negatives=self.config.use_in_batch_negatives,
        )

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs
        self.scheduler = self._create_scheduler(total_steps)

        # Mixed precision
        self.scaler = GradScaler() if self.config.fp16 else None

        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.encoder.gradient_checkpointing = True

        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Logging
        if self.config.log_to_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=self.config.__dict__,
                )
                self.use_wandb = True
            except ImportError:
                print("wandb not installed, skipping wandb logging")
                self.use_wandb = False
        else:
            self.use_wandb = False

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        print(f"Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Total training steps: {total_steps}")
        print(f"  Batch size (per device): {self.config.batch_size}")
        print(f"  Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Temperature: {self.config.temperature}")
        print(f"  Num negatives: {self.config.num_negatives}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        # Separate weight decay for different parameter groups
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

        return optimizer

    def _create_scheduler(self, total_steps: int):
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=0,
        )

        # Combine warmup + cosine
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        return scheduler

    def train(self):
        """Main training loop."""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50 + "\n")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Train one epoch
            train_loss = self.train_epoch()

            print(f"Epoch {epoch + 1} - Average train loss: {train_loss:.4f}")

            # Evaluate
            if self.eval_dataloader is not None:
                eval_loss = self.evaluate()
                print(f"Epoch {epoch + 1} - Eval loss: {eval_loss:.4f}")

                # Save best model
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.save_checkpoint("best_model")

            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")

        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50 + "\n")

        if self.use_wandb:
            import wandb
            wandb.finish()

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            loss = self.train_step(batch)

            total_loss += loss
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss:.4f}", "avg_loss": f"{total_loss/num_batches:.4f}"})

            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self.log_metrics({
                    "train/loss": loss,
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/epoch": self.epoch,
                })

            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(f"step_{self.global_step}")

            # Evaluate
            if self.eval_dataloader is not None and self.global_step % self.config.eval_steps == 0:
                eval_loss = self.evaluate()
                self.log_metrics({"eval/loss": eval_loss})
                self.model.train()  # Back to training mode

        return total_loss / num_batches

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Move to device
        query_input_ids = batch["query_input_ids"].to(self.device)
        query_attention_mask = batch["query_attention_mask"].to(self.device)
        pos_input_ids = batch["pos_input_ids"].to(self.device)
        pos_attention_mask = batch["pos_attention_mask"].to(self.device)
        neg_input_ids = batch["neg_input_ids"].to(self.device)  # [batch_size, num_negatives, seq_len]
        neg_attention_mask = batch["neg_attention_mask"].to(self.device)

        task_types = batch["task_types"]

        # Mixed precision
        with autocast(enabled=self.config.fp16):
            # Encode query
            query_embeds = self.model(query_input_ids, query_attention_mask)

            # Encode positive
            pos_embeds = self.model(pos_input_ids, pos_attention_mask)

            # Encode negatives
            batch_size, num_negatives, seq_len = neg_input_ids.shape

            # Reshape for batch encoding
            neg_input_ids_flat = neg_input_ids.view(batch_size * num_negatives, seq_len)
            neg_attention_mask_flat = neg_attention_mask.view(batch_size * num_negatives, seq_len)

            neg_embeds_flat = self.model(neg_input_ids_flat, neg_attention_mask_flat)

            # Reshape back
            neg_embeds = neg_embeds_flat.view(batch_size, num_negatives, -1)

            # Compute loss
            # For simplicity, use the first task type (in multi-task setting, batch would be homogeneous)
            task_type = task_types[0] if isinstance(task_types, list) else "retrieval"

            loss = self.loss_fn(query_embeds, pos_embeds, neg_embeds, task_type=task_type)

            # Gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.fp16:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1

        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            # Move to device
            query_input_ids = batch["query_input_ids"].to(self.device)
            query_attention_mask = batch["query_attention_mask"].to(self.device)
            pos_input_ids = batch["pos_input_ids"].to(self.device)
            pos_attention_mask = batch["pos_attention_mask"].to(self.device)
            neg_input_ids = batch["neg_input_ids"].to(self.device)
            neg_attention_mask = batch["neg_attention_mask"].to(self.device)
            task_types = batch["task_types"]

            # Forward pass
            query_embeds = self.model(query_input_ids, query_attention_mask)
            pos_embeds = self.model(pos_input_ids, pos_attention_mask)

            # Encode negatives
            batch_size, num_negatives, seq_len = neg_input_ids.shape
            neg_input_ids_flat = neg_input_ids.view(batch_size * num_negatives, seq_len)
            neg_attention_mask_flat = neg_attention_mask.view(batch_size * num_negatives, seq_len)
            neg_embeds_flat = self.model(neg_input_ids_flat, neg_attention_mask_flat)
            neg_embeds = neg_embeds_flat.view(batch_size, num_negatives, -1)

            # Loss
            task_type = task_types[0] if isinstance(task_types, list) else "retrieval"
            loss = self.loss_fn(query_embeds, pos_embeds, neg_embeds, task_type=task_type)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()

        torch.save(state, os.path.join(checkpoint_dir, "trainer_state.pt"))

        print(f"Checkpoint saved to {checkpoint_dir}")

        # Manage checkpoint limit
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save space."""
        if self.config.save_total_limit is None:
            return

        checkpoint_dirs = []
        for name in os.listdir(self.config.output_dir):
            if name.startswith("step_") or name.startswith("epoch_"):
                path = os.path.join(self.config.output_dir, name)
                if os.path.isdir(path):
                    checkpoint_dirs.append((path, os.path.getmtime(path)))

        # Sort by modification time
        checkpoint_dirs.sort(key=lambda x: x[1])

        # Remove oldest checkpoints
        while len(checkpoint_dirs) > self.config.save_total_limit:
            path_to_remove, _ = checkpoint_dirs.pop(0)
            import shutil
            shutil.rmtree(path_to_remove)
            print(f"Removed old checkpoint: {path_to_remove}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        # Load model
        self.model = InstructionAwareEmbeddingModel.from_pretrained(checkpoint_path)
        self.model.to(self.device)

        # Load training state
        state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)

            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_loss = state["best_loss"]
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])

            if self.scaler is not None and "scaler" in state:
                self.scaler.load_state_dict(state["scaler"])

            print(f"Checkpoint loaded from {checkpoint_path}")
            print(f"  Resuming from step {self.global_step}, epoch {self.epoch}")

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics."""
        # Console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Step {self.global_step}: {metrics_str}")

        # W&B
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=self.global_step)

        # Save to file
        log_file = os.path.join(self.config.output_dir, "training_log.jsonl")
        with open(log_file, "a") as f:
            log_entry = {"step": self.global_step, **metrics}
            f.write(json.dumps(log_entry) + "\n")
