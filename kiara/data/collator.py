"""
kiara/data/collator.py — Data Collator for Causal Language Modeling
====================================================================
Transforms raw dataset samples into standard training batch tensors:
  - input_ids:      torch.Tensor[B, T] (dtype=torch.long)
  - attention_mask: torch.Tensor[B, T] (dtype=torch.long)
  - labels:         torch.Tensor[B, T] (dtype=torch.long, with -100 for ignored/padding tokens)
"""

from __future__ import annotations

from typing import Any, Mapping
import torch

from kiara.data.constants import IGNORE_INDEX
from kiara.tokenizer.special_tokens import PAD_TOKEN_ID


class DataCollatorForCausalLM:
    """Collates and prepares training batches for autoregressive causal language modeling."""

    def __init__(
        self,
        pad_token_id: int = PAD_TOKEN_ID,
        padding_side: str = "right",
        pad_to_multiple_of: int | None = None,
        max_length: int | None = None,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length

    def __call__(
        self,
        features: list[dict[str, Any] | list[int] | torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Collate a list of dataset samples into batch tensors.

        Args:
            features: List of samples (dictionaries containing 'input_ids', or raw lists/tensors).

        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'labels' tensors.
        """
        if not features:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
                "labels": torch.empty((0, 0), dtype=torch.long),
            }

        # Extract sequence lists
        sequences: list[list[int]] = []
        for feat in features:
            if isinstance(feat, Mapping) and "input_ids" in feat:
                ids = feat["input_ids"]
                if isinstance(ids, torch.Tensor):
                    sequences.append(ids.tolist())
                else:
                    sequences.append(list(ids))
            elif isinstance(feat, torch.Tensor):
                sequences.append(feat.tolist())
            elif isinstance(feat, list):
                sequences.append(list(feat))
            else:
                raise TypeError(f"Unsupported sample type in collator: {type(feat)}")

        max_seq_len = max(len(seq) for seq in sequences)
        target_len = self.max_length if self.max_length is not None else max_seq_len
        target_len = max(target_len, max_seq_len)

        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            remainder = target_len % self.pad_to_multiple_of
            if remainder != 0:
                target_len += self.pad_to_multiple_of - remainder

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for seq in sequences:
            # Truncate if sequence exceeds target_len
            if len(seq) > target_len:
                seq = seq[:target_len]

            pad_count = target_len - len(seq)
            pad_ids = [self.pad_token_id] * pad_count
            pad_mask = [0] * pad_count
            pad_labels = [IGNORE_INDEX] * pad_count

            active_mask = [1] * len(seq)
            active_labels = list(seq)

            if self.padding_side == "right":
                batch_input_ids.append(list(seq) + pad_ids)
                batch_attention_mask.append(active_mask + pad_mask)
                batch_labels.append(active_labels + pad_labels)
            elif self.padding_side == "left":
                batch_input_ids.append(pad_ids + list(seq))
                batch_attention_mask.append(pad_mask + active_mask)
                batch_labels.append(pad_labels + active_labels)
            else:
                raise ValueError(f"Invalid padding_side '{self.padding_side}'. Choose 'left' or 'right'.")

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
