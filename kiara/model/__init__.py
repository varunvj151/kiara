"""
kiara/model — Decoder-Only Transformer Architecture Subsystem
==============================================================
Provides neural network modules and full language model classes for Kiara.
"""

from kiara.model.attention import GroupedQueryAttention
from kiara.model.block import TransformerBlock
from kiara.model.config import KiaraConfig
from kiara.model.mlp import SwiGLUMLP
from kiara.model.rms_norm import RMSNorm
from kiara.model.rope import RotaryEmbedding
from kiara.model.transformer import CausalLMOutput, KiaraForCausalLM, KiaraModel

__all__ = [
    "KiaraConfig",
    "KiaraModel",
    "KiaraForCausalLM",
    "CausalLMOutput",
    "TransformerBlock",
    "GroupedQueryAttention",
    "SwiGLUMLP",
    "RMSNorm",
    "RotaryEmbedding",
]
