"""
kiara/trainer/scheduler.py — Learning Rate Schedulers
======================================================
Implements linear warmup followed by cosine annealing decay down to min_lr.
"""

from __future__ import annotations

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: Optimizer instance.
        num_warmup_steps: Number of initial linear warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum LR multiplier at end of cosine decay (e.g. 0.1 = 10% of base lr).

    Returns:
        LambdaLR scheduler instance.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        if current_step > num_training_steps:
            return min_lr_ratio

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
