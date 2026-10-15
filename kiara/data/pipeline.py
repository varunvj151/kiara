"""
kiara/data/pipeline.py — Training & Evaluation DataLoader Pipeline
===================================================================
Constructs deterministic PyTorch DataLoaders for training and evaluation.
"""

from __future__ import annotations

from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset

from kiara.data.collator import DataCollatorForCausalLM
from kiara.data.dataset import PackedDataset, PaddedDataset
from kiara.tokenizer.tokenizer import KiaraTokenizer


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    collator: DataCollatorForCausalLM | None = None,
    seed: int = 1337,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a deterministic PyTorch DataLoader.

    Args:
        dataset: Dataset instance.
        batch_size: Batch size per iteration.
        shuffle: Whether to shuffle data every epoch.
        collator: DataCollator instance (defaults to DataCollatorForCausalLM).
        seed: Random seed for deterministic shuffling.
        num_workers: Number of background worker processes.
        drop_last: Whether to drop the last incomplete batch.
        pin_memory: Whether to pin memory for fast CUDA host-to-device transfers.

    Returns:
        PyTorch DataLoader instance.
    """
    if collator is None:
        collator = DataCollatorForCausalLM()

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


def build_training_pipeline(
    texts: list[str],
    tokenizer: KiaraTokenizer | None = None,
    seq_len: int = 2048,
    batch_size: int = 4,
    packed: bool = True,
    seed: int = 1337,
    pad_to_multiple_of: int | None = 8,
) -> tuple[Dataset, DataLoader]:
    """Convenience factory to create a training dataset and DataLoader from raw text documents.

    Args:
        texts: List of raw training documents.
        tokenizer: KiaraTokenizer instance (defaults to canonical load()).
        seq_len: Training sequence length block size.
        batch_size: Batch size per step.
        packed: If True, uses PackedDataset without padding waste.
        seed: Deterministic random seed.
        pad_to_multiple_of: Round length to multiple (e.g. 8).

    Returns:
        tuple of (dataset, dataloader)
    """
    if tokenizer is None:
        tokenizer = KiaraTokenizer.load()

    if packed:
        dataset = PackedDataset.from_texts(texts=texts, tokenizer=tokenizer, seq_len=seq_len)
    else:
        dataset = PaddedDataset.from_texts(texts=texts, tokenizer=tokenizer)

    collator = DataCollatorForCausalLM(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    loader = create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collator=collator,
        seed=seed,
    )

    return dataset, loader
