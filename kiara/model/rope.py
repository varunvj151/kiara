"""
kiara/model/rope.py — Rotary Position Embeddings (RoPE)
========================================================
Implements Rotary Position Embeddings from first principles (Su et al., 2021).
Rotates pairs of hidden dimensions according to position-dependent frequencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to query and key states.

    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine cache tensor of shape [seq_len, head_dim] or [batch, seq_len, head_dim]
        sin: Sine cache tensor of shape [seq_len, head_dim] or [batch, seq_len, head_dim]
        position_ids: Optional position indices tensor of shape [batch, seq_len]

    Returns:
        tuple of (rotated_q, rotated_k)
    """
    if position_ids is not None:
        # Index cos/sin by position_ids: [batch, seq_len, head_dim]
        cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    else:
        # Broadcast over batch and head dimensions: [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding module with precomputed frequencies."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Inverse frequency: theta_i = 1.0 / (base ** (2 * (i // 2) / dim))
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute initial cos and sin caches
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

        # Outer product of position t and inv_freq: [seq_len, dim / 2]
        freqs = torch.outer(t, self.inv_freq.to(device=device))

        # Concatenate freqs with itself to match head_dim: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin tensors up to seq_len."""
        if seq_len is None:
            seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )
