"""
kiara/data/dataset.py — Dataset Implementations for Kiara Training
===================================================================
Provides PyTorch Dataset abstractions for:
  1. TokenizedDataset: In-memory sequence dataset.
  2. PackedDataset: Concatenates sequences with [EOS] tokens and chunks into fixed seq_len blocks.
  3. PaddedDataset: Variable-length sequence dataset for dynamic padding.
"""

from __future__ import annotations

from typing import Any, Sequence
import torch
from torch.utils.data import Dataset

from kiara.tokenizer.tokenizer import KiaraTokenizer


class TokenizedDataset(Dataset):
    """Dataset storing pre-tokenized sequences of integer token IDs."""

    def __init__(self, sequences: list[list[int]]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> list[int]:
        return self.sequences[idx]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        tokenizer: KiaraTokenizer,
        add_special_tokens: bool = False,
        bos: bool = False,
        eos: bool = True,
    ) -> TokenizedDataset:
        """Tokenize a list of raw text documents into a TokenizedDataset."""
        sequences = [
            tokenizer.encode(text, add_special_tokens=add_special_tokens, bos=bos, eos=eos)
            for text in texts
        ]
        return cls(sequences)


class PackedDataset(Dataset):
    """Concatenates tokenized documents separated by [EOS] and chunks into fixed-length blocks.

    Prevents context window waste from padding during large-scale foundation model pre-training.
    """

    def __init__(
        self,
        chunks: list[list[int]],
        seq_len: int,
        total_tokens: int,
    ) -> None:
        self.chunks = chunks
        self.seq_len = seq_len
        self.total_tokens = total_tokens

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        chunk = self.chunks[idx]
        return {
            "input_ids": chunk,
            "attention_mask": [1] * len(chunk),
        }

    @property
    def num_tokens(self) -> int:
        return self.total_tokens

    @classmethod
    def from_tokenized_sequences(
        cls,
        sequences: list[list[int]],
        seq_len: int = 2048,
        eos_token_id: int | None = 3,
    ) -> PackedDataset:
        """Pack a list of token ID sequences into uniform chunks of size seq_len.

        Args:
            sequences: List of token ID sequences.
            seq_len: Target sequence length for training blocks.
            eos_token_id: Optional EOS token ID appended between sequences if not already present.

        Returns:
            PackedDataset instance.
        """
        all_tokens: list[int] = []
        for seq in sequences:
            if not seq:
                continue
            all_tokens.extend(seq)
            if eos_token_id is not None and (not seq or seq[-1] != eos_token_id):
                all_tokens.append(eos_token_id)

        total_tokens = len(all_tokens)
        num_chunks = total_tokens // seq_len

        chunks = [
            all_tokens[i * seq_len : (i + 1) * seq_len]
            for i in range(num_chunks)
        ]

        return cls(chunks=chunks, seq_len=seq_len, total_tokens=total_tokens)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        tokenizer: KiaraTokenizer,
        seq_len: int = 2048,
        bos: bool = False,
        eos: bool = True,
    ) -> PackedDataset:
        """Tokenize and pack a list of raw text documents into fixed-length blocks."""
        sequences = [
            tokenizer.encode(t, bos=bos, eos=eos)
            for t in texts
            if t.strip()
        ]
        return cls.from_tokenized_sequences(
            sequences=sequences,
            seq_len=seq_len,
            eos_token_id=tokenizer.eos_token_id,
        )

    def state_dict(self) -> dict[str, Any]:
        """Save dataset configuration state for training resumption."""
        return {
            "seq_len": self.seq_len,
            "num_chunks": len(self.chunks),
            "total_tokens": self.total_tokens,
        }


class PaddedDataset(Dataset):
    """Variable-length dataset where items are individual sequences to be dynamically padded."""

    def __init__(self, sequences: list[list[int]]) -> None:
        self.sequences = [seq for seq in sequences if seq]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        seq = self.sequences[idx]
        return {
            "input_ids": seq,
            "attention_mask": [1] * len(seq),
        }

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        tokenizer: KiaraTokenizer,
        add_special_tokens: bool = False,
        bos: bool = False,
        eos: bool = True,
    ) -> PaddedDataset:
        """Create PaddedDataset from raw texts."""
        sequences = [
            tokenizer.encode(t, add_special_tokens=add_special_tokens, bos=bos, eos=eos)
            for t in texts
            if t.strip()
        ]
        return cls(sequences)
