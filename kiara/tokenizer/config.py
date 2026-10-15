"""
kiara/tokenizer/config.py — Tokenizer Configuration for Kiara
==============================================================
Encapsulates frozen configuration parameters for the Kiara Byte-Level BPE tokenizer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kiara.tokenizer.special_tokens import (
    BOS_TOKEN,
    BOS_TOKEN_ID,
    EOS_TOKEN,
    EOS_TOKEN_ID,
    PAD_TOKEN,
    PAD_TOKEN_ID,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    UNK_TOKEN_ID,
)


@dataclass
class TokenizerConfig:
    """Frozen configuration for Kiara Byte-Level BPE Tokenizer (Decision 003)."""

    tokenizer_type: str = "byte_bpe"
    vocab_size: int = 32768

    unk_token: str = UNK_TOKEN
    pad_token: str = PAD_TOKEN
    bos_token: str = BOS_TOKEN
    eos_token: str = EOS_TOKEN

    unk_token_id: int = UNK_TOKEN_ID
    pad_token_id: int = PAD_TOKEN_ID
    bos_token_id: int = BOS_TOKEN_ID
    eos_token_id: int = EOS_TOKEN_ID

    add_prefix_space: bool = False
    trim_offsets: bool = True
    special_tokens: list[str] = field(default_factory=lambda: list(SPECIAL_TOKENS))

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenizerConfig:
        """Create configuration from dictionary."""
        valid_keys = {
            "tokenizer_type",
            "vocab_size",
            "unk_token",
            "pad_token",
            "bos_token",
            "eos_token",
            "unk_token_id",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "add_prefix_space",
            "trim_offsets",
            "special_tokens",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json_file(cls, path: str | Path) -> TokenizerConfig:
        """Load configuration from a JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Tokenizer configuration file not found at: {file_path}")
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
