"""
kiara/tokenizer/tokenizer.py — Canonical KiaraTokenizer Implementation
========================================================================
Authoritative tokenizer interface for the Kiara foundation model.
Wraps the frozen 32,768 Byte-Level BPE artifact established in Decision 003.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from tokenizers import Tokenizer

from kiara.tokenizer.config import TokenizerConfig
from kiara.tokenizer.serialization import load_tokenizer_artifacts, save_tokenizer_artifacts


class KiaraTokenizer:
    """Canonical tokenizer class for Kiara (Byte-Level BPE, 32,768 vocabulary).

    Provides fast encoding, decoding, batch processing, padding, truncation,
    and special token management for dataset pre-processing, training, and inference.
    """

    def __init__(
        self,
        backend: Tokenizer,
        config: TokenizerConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the tokenizer.

        Args:
            backend: HuggingFace Tokenizer backend instance.
            config: TokenizerConfig holding frozen parameters.
            metadata: Optional provenance dictionary.
        """
        self._backend = backend
        self._config = config or TokenizerConfig()
        self._metadata = metadata or {}

        # Quick lookup for special token IDs to skip during decode
        self._special_token_id_set = {
            self._config.unk_token_id,
            self._config.pad_token_id,
            self._config.bos_token_id,
            self._config.eos_token_id,
        }

    # ---------------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------------

    @property
    def backend(self) -> Tokenizer:
        """Access the underlying HuggingFace Tokenizer backend."""
        return self._backend

    @property
    def config(self) -> TokenizerConfig:
        """Access the frozen TokenizerConfig."""
        return self._config

    @property
    def metadata(self) -> dict[str, Any]:
        """Access tokenizer metadata dictionary."""
        return self._metadata

    @property
    def vocab_size(self) -> int:
        """Return the total vocabulary size."""
        return self._backend.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        """Token ID for [PAD]."""
        return self._config.pad_token_id

    @property
    def unk_token_id(self) -> int:
        """Token ID for [UNK]."""
        return self._config.unk_token_id

    @property
    def bos_token_id(self) -> int:
        """Token ID for [BOS]."""
        return self._config.bos_token_id

    @property
    def eos_token_id(self) -> int:
        """Token ID for [EOS]."""
        return self._config.eos_token_id

    @property
    def pad_token(self) -> str:
        """String representation for [PAD]."""
        return self._config.pad_token

    @property
    def unk_token(self) -> str:
        """String representation for [UNK]."""
        return self._config.unk_token

    @property
    def bos_token(self) -> str:
        """String representation for [BOS]."""
        return self._config.bos_token

    @property
    def eos_token(self) -> str:
        """String representation for [EOS]."""
        return self._config.eos_token

    @property
    def special_tokens_map(self) -> dict[str, str]:
        """Map of special token names to token strings."""
        return {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Map of special token names to token IDs."""
        return {
            "unk_token_id": self.unk_token_id,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

    # ---------------------------------------------------------------------------
    # Core Encoding and Decoding
    # ---------------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        bos: bool = False,
        eos: bool = False,
    ) -> list[int]:
        """Encode a single text string into a list of token IDs.

        Args:
            text: Input string.
            add_special_tokens: If True, adds both [BOS] prefix and [EOS] suffix.
            bos: If True, prepends [BOS] token ID.
            eos: If True, appends [EOS] token ID.

        Returns:
            List of integer token IDs.
        """
        if not text:
            ids: list[int] = []
        else:
            ids = self._backend.encode(text).ids

        if add_special_tokens:
            bos = True
            eos = True

        if bos:
            ids = [self.bos_token_id] + ids
        if eos:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode a sequence of token IDs back into text.

        Args:
            token_ids: List or sequence of token IDs.
            skip_special_tokens: If True, removes special tokens ([UNK], [PAD], [BOS], [EOS]).

        Returns:
            Decoded text string.
        """
        if not token_ids:
            return ""

        ids_to_decode = list(token_ids)
        if skip_special_tokens:
            ids_to_decode = [i for i in ids_to_decode if i not in self._special_token_id_set]
            if not ids_to_decode:
                return ""

        return self._backend.decode(ids_to_decode, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
        bos: bool = False,
        eos: bool = False,
    ) -> list[list[int]]:
        """Encode a batch of strings into a list of token ID lists."""
        return [
            self.encode(t, add_special_tokens=add_special_tokens, bos=bos, eos=eos)
            for t in texts
        ]

    def decode_batch(
        self,
        sequences: list[Sequence[int]],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """Decode a batch of token ID sequences into strings."""
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    # ---------------------------------------------------------------------------
    # Padding and Truncation
    # ---------------------------------------------------------------------------

    def truncate(
        self,
        sequences: list[list[int]],
        max_length: int,
        truncation_side: str = "right",
    ) -> list[list[int]]:
        """Truncate a batch of token ID sequences to max_length.

        Args:
            sequences: List of token ID sequences.
            max_length: Maximum allowed sequence length.
            truncation_side: 'right' (truncate tail) or 'left' (truncate head).

        Returns:
            List of truncated token ID sequences.
        """
        truncated: list[list[int]] = []
        for seq in sequences:
            if len(seq) <= max_length:
                truncated.append(list(seq))
            elif truncation_side == "right":
                truncated.append(list(seq[:max_length]))
            elif truncation_side == "left":
                truncated.append(list(seq[-max_length:]))
            else:
                raise ValueError(f"Invalid truncation_side '{truncation_side}'. Choose 'left' or 'right'.")
        return truncated

    def pad(
        self,
        sequences: list[list[int]],
        max_length: int | None = None,
        padding_side: str = "right",
        pad_to_multiple_of: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        """Pad a batch of token ID sequences to a uniform length.

        Args:
            sequences: List of token ID sequences.
            max_length: Target sequence length (if None, uses length of longest sequence).
            padding_side: 'right' or 'left'.
            pad_to_multiple_of: If set, pads length to a multiple of this value (e.g. 8 for Tensor Cores).
            return_tensors: None (returns lists) or 'pt' (returns torch.Tensor).

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'.
        """
        if not sequences:
            if return_tensors == "pt":
                import torch

                return {
                    "input_ids": torch.empty((0, 0), dtype=torch.long),
                    "attention_mask": torch.empty((0, 0), dtype=torch.long),
                }
            return {"input_ids": [], "attention_mask": []}

        max_seq_len = max(len(seq) for seq in sequences)
        target_len = max_length if max_length is not None else max_seq_len
        target_len = max(target_len, max_seq_len)

        if pad_to_multiple_of is not None and pad_to_multiple_of > 0:
            remainder = target_len % pad_to_multiple_of
            if remainder != 0:
                target_len += pad_to_multiple_of - remainder

        padded_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        pad_id = self.pad_token_id

        for seq in sequences:
            pad_count = target_len - len(seq)
            if pad_count < 0:
                # Sequence longer than target_len (if max_length was smaller than longest seq)
                pad_count = 0
                seq = seq[:target_len]

            pad_tokens = [pad_id] * pad_count
            pad_mask = [0] * pad_count
            active_mask = [1] * len(seq)

            if padding_side == "right":
                padded_ids.append(list(seq) + pad_tokens)
                attention_masks.append(active_mask + pad_mask)
            elif padding_side == "left":
                padded_ids.append(pad_tokens + list(seq))
                attention_masks.append(pad_mask + active_mask)
            else:
                raise ValueError(f"Invalid padding_side '{padding_side}'. Choose 'left' or 'right'.")

        if return_tensors == "pt":
            import torch

            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_masks,
        }

    # ---------------------------------------------------------------------------
    # Token / ID Lookups
    # ---------------------------------------------------------------------------

    def token_to_id(self, token: str) -> int | None:
        """Return the token ID for a given token string, or None if absent."""
        return self._backend.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        """Return the token string for a given token ID, or None if absent."""
        return self._backend.id_to_token(token_id)

    def get_vocab(self) -> dict[str, int]:
        """Return full vocabulary dictionary mapping tokens to IDs."""
        return self._backend.get_vocab()

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save the tokenizer backend, config, and metadata to a directory."""
        return save_tokenizer_artifacts(
            backend=self._backend,
            config=self._config,
            target_dir=path,
            metadata=self._metadata,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> KiaraTokenizer:
        """Load the tokenizer from directory (defaults to artifacts/tokenizer)."""
        backend, config, metadata = load_tokenizer_artifacts(source_dir=path)
        return cls(backend=backend, config=config, metadata=metadata)

    @classmethod
    def from_pretrained(cls, path: str | Path | None = None) -> KiaraTokenizer:
        """Alias of KiaraTokenizer.load()."""
        return cls.load(path=path)

    # ---------------------------------------------------------------------------
    # Dunder Methods & Convenience Callable
    # ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"KiaraTokenizer(type='{self.config.tokenizer_type}', "
            f"vocab_size={self.vocab_size}, "
            f"pad_id={self.pad_token_id}, bos_id={self.bos_token_id}, eos_id={self.eos_token_id})"
        )

    def __call__(
        self,
        text_or_texts: str | list[str],
        padding: bool = False,
        max_length: int | None = None,
        truncation: bool = False,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
        padding_side: str = "right",
        truncation_side: str = "right",
    ) -> dict[str, Any]:
        """Convenience callable for encoding, truncation, and padding."""
        is_single = isinstance(text_or_texts, str)
        texts = [text_or_texts] if is_single else text_or_texts

        sequences = self.encode_batch(texts, add_special_tokens=add_special_tokens)

        if truncation and max_length is not None:
            sequences = self.truncate(sequences, max_length=max_length, truncation_side=truncation_side)

        if padding:
            return self.pad(
                sequences,
                max_length=max_length,
                padding_side=padding_side,
                return_tensors=return_tensors,
            )

        if return_tensors == "pt":
            import torch

            if is_single:
                return {"input_ids": torch.tensor(sequences[0], dtype=torch.long)}
            return {"input_ids": [torch.tensor(s, dtype=torch.long) for s in sequences]}

        if is_single:
            return {"input_ids": sequences[0]}
        return {"input_ids": sequences}
