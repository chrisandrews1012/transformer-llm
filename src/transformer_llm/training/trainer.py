"""
Trainer for language model training.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import LanguageModelingLoss, MetricsTracker, compute_all_metrics
from .scheduler import create_scheduler


class Trainer:
    """
    Trainer for language model training.

    Handles the complete training pipeline including training and validation
    loops, checkpointing, gradient clipping, gradient accumulation, and
    optional mixed-precision training.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config: dict,
        device: torch.device,
    ):
        """
        Initialize trainer.

        :param model: Language model to train.
        :type model: nn.Module
        :param train_dataloader: DataLoader for training data.
        :type train_dataloader: DataLoader
        :param val_dataloader: DataLoader for validation data.
        :type val_dataloader: DataLoader
        :param optimizer: PyTorch optimizer.
        :type optimizer: torch.optim.Optimizer
        :param scheduler: Learning rate scheduler (any object with a step() method),
            or None.
        :param config: Training configuration dictionary.
        :type config: dict
        :param device: Device to train on.
        :type device: torch.device
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        self.loss_fn = LanguageModelingLoss(
            pad_token_id=config.get("pad_token_id", 0)
        )

        self.use_amp = bool(config.get("use_amp", False) and device.type == "cuda")
        self.scaler = GradScaler(device="cuda", enabled=self.use_amp)

        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = config.get(
            "gradient_accumulation_steps", 1
        )

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> dict:
        """
        Run one training epoch.

        :returns: Dictionary of training metrics averaged over the epoch.
        :rtype: dict
        """
        self.model.train()
        metrics_tracker = MetricsTracker()
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")

        self.optimizer.zero_grad(set_to_none=True)
        num_batches = len(self.train_dataloader)

        for batch_idx, (input_ids, target_ids, attention_mask) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            if self.use_amp:
                with autocast(device_type="cuda", enabled=True):
                    logits, _ = self.model(input_ids, attention_mask=attention_mask)
                    loss = self.loss_fn(logits, target_ids)
            else:
                logits, _ = self.model(input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(logits, target_ids)

            raw_loss = loss.detach()
            loss = loss / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (
                (batch_idx + 1) % self.gradient_accumulation_steps == 0
                or (batch_idx + 1) == num_batches
            )
            if should_step:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1

            batch_metrics = compute_all_metrics(
                logits.detach(),
                target_ids,
                raw_loss,
                pad_token_id=self.config.get("pad_token_id", 0),
            )
            metrics_tracker.update(**batch_metrics)
            pbar.set_postfix(
                loss=f"{metrics_tracker.get('loss'):.4f}",
                ppl=f"{metrics_tracker.get('perplexity'):.2f}",
                acc=f"{metrics_tracker.get('accuracy'):.3f}",
            )

        return metrics_tracker.compute()

    def validate(self) -> dict:
        """
        Run one validation pass.

        :returns: Dictionary of validation metrics averaged over the dataset.
        :rtype: dict
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()

        with torch.no_grad():
            pbar = tqdm(self.val_dataloader, desc="Validation")

            for input_ids, target_ids, attention_mask in pbar:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                logits, _ = self.model(input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(logits, target_ids)

                batch_metrics = compute_all_metrics(
                    logits,
                    target_ids,
                    loss,
                    pad_token_id=self.config.get("pad_token_id", 0),
                )
                metrics_tracker.update(**batch_metrics)
                pbar.set_postfix(loss=f"{metrics_tracker.get('loss'):.4f}")

        return metrics_tracker.compute()

    def save_checkpoint(self, filename: str = "checkpoint.pt", is_best: bool = False) -> None:
        """
        Save model checkpoint to disk.

        :param filename: Checkpoint filename within checkpoint_dir.
        :type filename: str
        :param is_best: If True, also save a copy as best_model.pt.
        :type is_best: bool
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None and hasattr(self.scheduler, "state_dict")
                else None
            ),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model and optimizer state from a checkpoint file.

        :param checkpoint_path: Path to the checkpoint file.
        :type checkpoint_path: str
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (
            self.scheduler is not None
            and checkpoint.get("scheduler_state_dict") is not None
            and hasattr(self.scheduler, "load_state_dict")
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def train(self, num_epochs: int) -> None:
        """
        Run the full training loop for the specified number of epochs.

        :param num_epochs: Number of epochs to train.
        :type num_epochs: int
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Max gradient norm: {self.max_grad_norm}")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_metrics = self.train_epoch()
            print(f"Train metrics: {train_metrics}")

            val_metrics = self.validate()
            print(f"Val metrics: {val_metrics}")

            self.save_checkpoint(filename=f"checkpoint_epoch_{epoch + 1}.pt")

            val_loss = val_metrics.get("loss", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    filename=f"checkpoint_epoch_{epoch + 1}.pt", is_best=True
                )

        print("\nTraining complete!")


def create_trainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    config: dict,
    device: torch.device,
) -> Trainer:
    """
    Create a Trainer with optimizer and scheduler configured from config.

    :param model: Model to train.
    :type model: nn.Module
    :param train_dataloader: DataLoader for training data.
    :type train_dataloader: DataLoader
    :param val_dataloader: DataLoader for validation data.
    :type val_dataloader: DataLoader
    :param config: Training configuration dictionary (must include "learning_rate"
        and "scheduler_type").
    :type config: dict
    :param device: Device to train on.
    :type device: torch.device
    :returns: Configured Trainer instance.
    :rtype: Trainer
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config["scheduler_type"],
        warmup_steps=config.get("warmup_steps", 0),
        total_steps=config.get("total_steps", 1),
        min_lr=config.get("min_lr", 0.0),
    )

    return Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )
