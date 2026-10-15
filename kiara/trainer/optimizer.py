"""
kiara/trainer/optimizer.py — AdamW Optimizer Factory with Weight Decay Splitting
=================================================================================
Configures decoupled AdamW optimization:
  - Applies weight decay to 2D weight matrices (projections, linear layers)
  - Excludes 1D biases and RMSNorm normalization gains from weight decay
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """Construct an AdamW optimizer with parameter-specific weight decay groups.

    Args:
        model: PyTorch model module.
        lr: Base learning rate.
        weight_decay: L2 penalty factor for matrix weights.
        betas: Adam beta coefficients (beta1, beta2).
        eps: Numerical stability constant.

    Returns:
        Configured AdamW optimizer.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    # Exclude RMSNorm gains, embedding padding, and biases from weight decay
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not name.endswith(".bias"):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(optim_groups, lr=lr, betas=betas, eps=eps)
