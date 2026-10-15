"""
kiara/trainer/trainer.py — Deterministic Foundation Model Trainer
==================================================================
Manages pre-training loops, evaluation, gradient accumulation, gradient clipping,
metric tracking, and checkpoint persistence for Kiara.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kiara.model.transformer import KiaraForCausalLM
from kiara.trainer.checkpoint import load_checkpoint, save_checkpoint
from kiara.trainer.optimizer import build_optimizer
from kiara.trainer.scheduler import get_cosine_schedule_with_warmup


@dataclass
class TrainerConfig:
    """Configuration for model training."""

    output_dir: str = "checkpoints"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    max_steps: int = 1000
    gradient_accumulation_steps: int = 1
    eval_interval: int = 100
    save_interval: int = 200
    log_interval: int = 10
    seed: int = 1337


class Trainer:
    """Clean, deterministic training manager for Kiara."""

    def __init__(
        self,
        model: KiaraForCausalLM,
        config: TrainerConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model.to(self.device)

        # Build optimizer and scheduler
        self.optimizer = build_optimizer(
            self.model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

        self.global_step = 0
        self.epoch = 0
        self.history: list[dict[str, Any]] = []

    def train(self, resume_from_checkpoint: str | Path | None = None) -> dict[str, Any]:
        """Execute full training loop."""
        if resume_from_checkpoint is not None:
            meta = load_checkpoint(
                resume_from_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            self.global_step = meta.get("global_step", 0)
            self.epoch = meta.get("epoch", 0)

        self.model.train()
        accum_loss = 0.0
        start_time = time.time()

        while self.global_step < self.config.max_steps:
            self.epoch += 1
            for batch in self.train_dataloader:
                if self.global_step >= self.config.max_steps:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

                accum_loss += loss.item() * self.config.gradient_accumulation_steps

                # Check if accumulation window is complete
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        current_lr = self.scheduler.get_last_lr()[0]
                        step_time = time.time() - start_time
                        log_entry = {
                            "step": self.global_step,
                            "loss": accum_loss / self.config.log_interval,
                            "perplexity": math.exp(min(accum_loss / self.config.log_interval, 20.0)),
                            "lr": current_lr,
                            "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "elapsed_s": round(step_time, 2),
                        }
                        self.history.append(log_entry)
                        accum_loss = 0.0

                    # Evaluation
                    if self.eval_dataloader is not None and self.global_step % self.config.eval_interval == 0:
                        eval_metrics = self.evaluate()
                        self.model.train()

                    # Periodic Checkpoint Saving
                    if self.global_step % self.config.save_interval == 0:
                        ckpt_path = Path(self.config.output_dir) / f"checkpoint_step_{self.global_step}.pt"
                        save_checkpoint(
                            target_path=ckpt_path,
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            global_step=self.global_step,
                            epoch=self.epoch,
                            loss=self.history[-1]["loss"] if self.history else None,
                            config=self.model.config,
                        )

        # Final checkpoint at completion
        final_ckpt = Path(self.config.output_dir) / "checkpoint_final.pt"
        save_checkpoint(
            target_path=final_ckpt,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            epoch=self.epoch,
            loss=self.history[-1]["loss"] if self.history else None,
            config=self.model.config,
        )

        return {
            "global_step": self.global_step,
            "final_loss": self.history[-1]["loss"] if self.history else None,
            "history": self.history,
        }

    @torch.no_grad()
    def evaluate(self, eval_dataloader: DataLoader | None = None) -> dict[str, float]:
        """Evaluate model on evaluation dataset."""
        loader = eval_dataloader or self.eval_dataloader
        if loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_loss, 20.0))

        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
        }
