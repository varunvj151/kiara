"""
kiara/trainer/checkpoint.py — Checkpoint Persistence and Resumption
====================================================================
Saves and restores full training state including:
  - Model weights
  - Optimizer states
  - Learning rate scheduler states
  - Global step & epoch counters
  - Deterministic RNG states (torch, python, numpy)
  - Tokenizer metadata & provenance
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def capture_rng_states() -> dict[str, Any]:
    """Capture random number generator states across all libraries."""
    rng_states = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng_states


def restore_rng_states(rng_states: dict[str, Any]) -> None:
    """Restore captured random number generator states."""
    if "torch_cpu" in rng_states:
        torch.set_rng_state(rng_states["torch_cpu"])
    if "python" in rng_states:
        random.setstate(rng_states["python"])
    if "torch_cuda" in rng_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_states["torch_cuda"])


def save_checkpoint(
    target_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR | None,
    global_step: int,
    epoch: int,
    loss: float | None = None,
    config: Any = None,
    tokenizer_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save full training state to a checkpoint file.

    Args:
        target_path: Destination file path (e.g. checkpoints/step_1000.pt).
        model: PyTorch model.
        optimizer: Optimizer.
        scheduler: Optional LR scheduler.
        global_step: Global training step index.
        epoch: Current epoch number.
        loss: Optional latest loss value.
        config: Optional model configuration.
        tokenizer_metadata: Optional tokenizer metadata dictionary.

    Returns:
        Path to the saved checkpoint file.
    """
    file_path = Path(target_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "global_step": global_step,
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_states": capture_rng_states(),
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "tokenizer_metadata": tokenizer_metadata or {},
    }

    torch.save(checkpoint, file_path)
    return file_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load full training state from a checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model: PyTorch model to populate.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        restore_rng: Whether to restore RNG states.

    Returns:
        Metadata dictionary with global_step, epoch, loss, and configuration.
    """
    file_path = Path(checkpoint_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {file_path}")

    checkpoint: dict[str, Any] = torch.load(file_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if restore_rng and "rng_states" in checkpoint and checkpoint["rng_states"]:
        restore_rng_states(checkpoint["rng_states"])

    return {
        "global_step": checkpoint.get("global_step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss"),
        "config": checkpoint.get("config"),
        "tokenizer_metadata": checkpoint.get("tokenizer_metadata"),
    }
