"""
Kiara — Foundation Model Core Architecture and Infrastructure
==============================================================
"""

from kiara.data import (
    DataCollatorForCausalLM,
    PackedDataset,
    PaddedDataset,
    TokenizedDataset,
    build_training_pipeline,
    create_dataloader,
)
from kiara.model import (
    CausalLMOutput,
    KiaraConfig,
    KiaraForCausalLM,
    KiaraModel,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUMLP,
    TransformerBlock,
)
from kiara.tokenizer import KiaraTokenizer, TokenizerConfig
from kiara.trainer import Trainer, TrainerConfig, build_optimizer, get_cosine_schedule_with_warmup

__version__ = "0.1.0"

__all__ = [
    "KiaraTokenizer",
    "TokenizerConfig",
    "DataCollatorForCausalLM",
    "PackedDataset",
    "PaddedDataset",
    "TokenizedDataset",
    "create_dataloader",
    "build_training_pipeline",
    "KiaraConfig",
    "KiaraModel",
    "KiaraForCausalLM",
    "CausalLMOutput",
    "TransformerBlock",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLUMLP",
    "Trainer",
    "TrainerConfig",
    "build_optimizer",
    "get_cosine_schedule_with_warmup",
    "__version__",
]
