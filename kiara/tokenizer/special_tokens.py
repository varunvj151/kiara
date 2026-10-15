"""
kiara/tokenizer/special_tokens.py — Special Token Definitions for Kiara
========================================================================
Defines frozen special tokens and token IDs for the Kiara foundation model.
"""

from __future__ import annotations

from dataclasses import dataclass


UNK_TOKEN: str = "[UNK]"
PAD_TOKEN: str = "[PAD]"
BOS_TOKEN: str = "[BOS]"
EOS_TOKEN: str = "[EOS]"

SPECIAL_TOKENS: list[str] = [
    UNK_TOKEN,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
]

# Fixed canonical token IDs allocated in the vocabulary
UNK_TOKEN_ID: int = 0
PAD_TOKEN_ID: int = 1
BOS_TOKEN_ID: int = 2
EOS_TOKEN_ID: int = 3


@dataclass(frozen=True)
class SpecialTokens:
    """Container for special token representations and their canonical IDs."""

    unk: str = UNK_TOKEN
    pad: str = PAD_TOKEN
    bos: str = BOS_TOKEN
    eos: str = EOS_TOKEN

    unk_id: int = UNK_TOKEN_ID
    pad_id: int = PAD_TOKEN_ID
    bos_id: int = BOS_TOKEN_ID
    eos_id: int = EOS_TOKEN_ID

    @property
    def all_tokens(self) -> list[str]:
        return [self.unk, self.pad, self.bos, self.eos]

    @property
    def all_ids(self) -> list[int]:
        return [self.unk_id, self.pad_id, self.bos_id, self.eos_id]

    @property
    def id_to_token_map(self) -> dict[int, str]:
        return {
            self.unk_id: self.unk,
            self.pad_id: self.pad,
            self.bos_id: self.bos,
            self.eos_id: self.eos,
        }

    @property
    def token_to_id_map(self) -> dict[str, int]:
        return {
            self.unk: self.unk_id,
            self.pad: self.pad_id,
            self.bos: self.bos_id,
            self.eos: self.eos_id,
        }
