"""
kiara/model/mlp.py — SwiGLU Feed-Forward Network
=================================================
Implements SwiGLU Feed-Forward Network from first principles (Shazeer, 2020):
  SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kiara.model.config import KiaraConfig


class SwiGLUMLP(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU) Feed-Forward Network."""

    def __init__(self, config: KiaraConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (SiLU(gate(x)) * up(x)) @ down."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
