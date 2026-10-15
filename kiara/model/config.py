"""
kiara/model/config.py — Kiara Model Architecture Configuration
===============================================================
Encapsulates frozen architecture hyperparameters, validation rules,
and exact parameter count accounting for the Kiara decoder-only transformer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class KiaraConfig:
    """Configuration class for Kiara foundation models.

    Defaults represent the 'Kiara-Tiny' CPU-friendly development configuration.
    """

    vocab_size: int = 32768
    hidden_size: int = 512
    intermediate_size: int = 1536
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    attention_dropout: float = 0.0

    # Special token IDs (aligned with Decision 003)
    unk_token_id: int = 0
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3

    model_type: str = "kiara"

    def __post_init__(self) -> None:
        """Validate architectural divisibility and constraints."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads}) for Grouped Query Attention."
            )
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be even for Rotary Position Embeddings (RoPE)."
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        """Number of query heads per key/value head group."""
        return self.num_attention_heads // self.num_key_value_heads

    def calculate_expected_parameters(self, tie_word_embeddings: bool | None = None) -> dict[str, int]:
        """Mathematically compute exact theoretical parameter count.

        Args:
            tie_word_embeddings: If None, uses self.tie_word_embeddings.

        Returns:
            Dictionary with breakdown of parameter counts.
        """
        tied = self.tie_word_embeddings if tie_word_embeddings is None else tie_word_embeddings
        head_dim = self.head_dim

        # 1. Embeddings
        embedding_params = self.vocab_size * self.hidden_size

        # 2. Attention per block (Q, K, V, O)
        q_params = self.hidden_size * (self.num_attention_heads * head_dim)
        k_params = self.hidden_size * (self.num_key_value_heads * head_dim)
        v_params = self.hidden_size * (self.num_key_value_heads * head_dim)
        o_params = (self.num_attention_heads * head_dim) * self.hidden_size
        attn_params_per_block = q_params + k_params + v_params + o_params

        # 3. SwiGLU MLP per block (gate, up, down)
        gate_params = self.hidden_size * self.intermediate_size
        up_params = self.hidden_size * self.intermediate_size
        down_params = self.intermediate_size * self.hidden_size
        mlp_params_per_block = gate_params + up_params + down_params

        # 4. RMSNorms per block (input norm + post-attention norm)
        norm_params_per_block = 2 * self.hidden_size

        block_params = attn_params_per_block + mlp_params_per_block + norm_params_per_block
        all_blocks_params = block_params * self.num_hidden_layers

        # 5. Final RMSNorm
        final_norm_params = self.hidden_size

        # 6. LM Head
        lm_head_params = 0 if tied else (self.vocab_size * self.hidden_size)

        total = embedding_params + all_blocks_params + final_norm_params + lm_head_params
        non_embedding = all_blocks_params + final_norm_params

        return {
            "embedding": embedding_params,
            "attention_per_block": attn_params_per_block,
            "mlp_per_block": mlp_params_per_block,
            "norm_per_block": norm_params_per_block,
            "single_block": block_params,
            "all_blocks": all_blocks_params,
            "final_norm": final_norm_params,
            "lm_head": lm_head_params,
            "total": total,
            "non_embedding": non_embedding,
            "tied_embeddings": tied,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KiaraConfig:
        """Create configuration from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save configuration to config.json."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        config_file = out_dir / "config.json"
        config_file.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, load_directory: str | Path) -> KiaraConfig:
        """Load configuration from config.json."""
        in_path = Path(load_directory)
        if in_path.is_file():
            config_file = in_path
        else:
            config_file = in_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found at: {config_file}")
        data = json.loads(config_file.read_text(encoding="utf-8"))
        return cls.from_dict(data)
