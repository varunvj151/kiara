"""
kiara/model/block.py — Pre-LN Transformer Block
================================================
Combines RMSNorm, Grouped Query Attention, and SwiGLU MLP with residual connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kiara.model.attention import GroupedQueryAttention
from kiara.model.config import KiaraConfig
from kiara.model.mlp import SwiGLUMLP
from kiara.model.rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Decoder Transformer Block."""

    def __init__(self, config: KiaraConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

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
        """Forward pass through transformer block."""
        # 1. Pre-LN Self-Attention with Residual Connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_kv = self.self_attn(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # 2. Pre-LN SwiGLU MLP with Residual Connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_kv
