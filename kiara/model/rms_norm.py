"""
kiara/model/rms_norm.py — Root Mean Square Layer Normalization (RMSNorm)
========================================================================
Implements RMSNorm from first principles (Zhang & Sennrich, 2019):
  RMSNorm(x) = (x / RMS(x)) * weight
where RMS(x) = sqrt(mean(x^2) + eps).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization in float32 for numerical stability."""
        input_dtype = x.dtype
        # Compute mean square along the last dimension in float32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed.to(input_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm and learnable scaling."""
        return self._norm(x) * self.weight

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}"
