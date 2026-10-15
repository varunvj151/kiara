"""
kiara/model/attention.py — Grouped Query Attention (GQA) with RoPE and KV-Cache
================================================================================
Implements Grouped Query Attention from first principles (Ainslie et al., 2023).
Supports:
  - Causal masking
  - Rotary Position Embeddings (RoPE)
  - Configurable query-to-KV head ratios (MHA, GQA, MQA)
  - Isolated KV-cache for fast autoregressive generation
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kiara.model.config import KiaraConfig
from kiara.model.rope import apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key or value states along the head dimension for GQA.

    Args:
        hidden_states: Tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of times each KV head is repeated (num_heads // num_kv_heads).

    Returns:
        Tensor of shape [batch, num_heads, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    # Expand to [batch, num_kv_heads, n_rep, seq_len, head_dim]
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    # Reshape to [batch, num_heads, seq_len, head_dim]
    return expanded.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    """Multi-Head Grouped Query Attention (GQA) layer."""

    def __init__(self, config: KiaraConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass for Grouped Query Attention.

        Args:
            hidden_states: Tensor of shape [batch, seq_len, hidden_size]
            cos: Cosine cache tensor from RotaryEmbedding
            sin: Sine cache tensor from RotaryEmbedding
            attention_mask: Optional padding mask of shape [batch, 1, seq_len, kv_seq_len]
            position_ids: Optional position indices tensor [batch, seq_len]
            past_key_value: Optional tuple of (past_k, past_v)
            use_cache: If True, returns updated KV cache tuple.

        Returns:
            tuple of (output_tensor, present_kv_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2. Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply Rotary Position Embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

        # 4. Handle KV Cache for incremental decoding
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present_key_value = (k, v) if use_cache else None
        kv_seq_len = k.shape[2]

        # 5. Repeat KV heads to match query heads (GQA expansion)
        k_rep = repeat_kv(k, self.num_kv_groups)
        v_rep = repeat_kv(v, self.num_kv_groups)

        # 6. Compute Scaled Dot-Product Attention Scores
        # [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, kv_seq_len] -> [batch, num_heads, seq_len, kv_seq_len]
        attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale

        # 7. Apply Causal Mask if computing full attention over sequence
        if seq_len > 1:
            # Lower triangular causal mask: allow attending only to past and present tokens
            causal_mask = torch.triu(
                torch.full((seq_len, kv_seq_len), float("-inf"), device=hidden_states.device, dtype=attn_weights.dtype),
                diagonal=1 + (kv_seq_len - seq_len),
            )
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        # 8. Apply External Attention Mask (e.g. from padding tokens)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 9. Softmax & Dropout
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_probs = self.dropout(attn_probs)

        # 10. Compute Context Vector
        # [batch, num_heads, seq_len, kv_seq_len] @ [batch, num_heads, kv_seq_len, head_dim] -> [batch, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_probs, v_rep)

        # Reshape back to [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 11. Final Linear Projection
        output = self.o_proj(attn_output)

        return output, present_key_value
