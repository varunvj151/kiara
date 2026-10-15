"""
kiara/data — Data Preprocessing, Packing, Collation, and Loading Subsystem
==========================================================================
Provides datasets, collators, and DataLoader pipelines for foundation model training.
"""

from kiara.data.collator import DataCollatorForCausalLM
from kiara.data.constants import IGNORE_INDEX
from kiara.data.dataset import PackedDataset, PaddedDataset, TokenizedDataset
from kiara.data.pipeline import build_training_pipeline, create_dataloader

__all__ = [
    "IGNORE_INDEX",
    "TokenizedDataset",
    "PackedDataset",
    "PaddedDataset",
    "DataCollatorForCausalLM",
    "create_dataloader",
    "build_training_pipeline",
]
