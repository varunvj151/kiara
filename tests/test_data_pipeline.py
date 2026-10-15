"""
tests/test_data_pipeline.py — Test Suite for Step 22 Data Pipeline
===================================================================
Verifies the Dataset -> Tokenizer -> PyTorch DataLoader tensor pipeline:
  - Sequence packing into fixed blocks without padding waste
  - Batch collation with standard [B, T] contract
  - Padding masks and labels ignored index (-100) handling
  - Deterministic DataLoader iteration and seeding
  - Zero UNK tokens and disjoint evaluation safety
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from kiara.data import (
    IGNORE_INDEX,
    DataCollatorForCausalLM,
    PackedDataset,
    PaddedDataset,
    TokenizedDataset,
    build_training_pipeline,
    create_dataloader,
)
from kiara.tokenizer import KiaraTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> KiaraTokenizer:
    """Load canonical KiaraTokenizer instance."""
    return KiaraTokenizer.load()


class TestDatasets:
    """Verifies Dataset abstractions."""

    def test_tokenized_dataset(self, tokenizer: KiaraTokenizer):
        texts = ["First document.", "Second document with code: def f(): pass"]
        dataset = TokenizedDataset.from_texts(texts, tokenizer=tokenizer)

        assert len(dataset) == 2
        assert isinstance(dataset[0], list)
        assert len(dataset[0]) > 0
        assert dataset[0][-1] == tokenizer.eos_token_id

    def test_packed_dataset_chunking(self, tokenizer: KiaraTokenizer):
        texts = [
            "Short text 1.",
            "Short text 2 with numbers 12345.",
            "Short text 3 with math ∑ x = 10.",
            "A slightly longer sentence ensuring we exceed the target sequence length.",
        ]
        seq_len = 16
        dataset = PackedDataset.from_texts(texts, tokenizer=tokenizer, seq_len=seq_len)

        assert len(dataset) > 0
        for i in range(len(dataset)):
            sample = dataset[i]
            assert "input_ids" in sample
            assert "attention_mask" in sample
            assert len(sample["input_ids"]) == seq_len
            assert len(sample["attention_mask"]) == seq_len
            assert all(m == 1 for m in sample["attention_mask"])

    def test_packed_dataset_state_dict(self, tokenizer: KiaraTokenizer):
        texts = ["Text sample A", "Text sample B"]
        dataset = PackedDataset.from_texts(texts, tokenizer=tokenizer, seq_len=8)
        state = dataset.state_dict()

        assert state["seq_len"] == 8
        assert state["num_chunks"] == len(dataset)
        assert "total_tokens" in state

    def test_padded_dataset(self, tokenizer: KiaraTokenizer):
        texts = ["Short", "A longer sentence for padding test."]
        dataset = PaddedDataset.from_texts(texts, tokenizer=tokenizer)

        assert len(dataset) == 2
        assert len(dataset[0]["input_ids"]) < len(dataset[1]["input_ids"])


class TestDataCollator:
    """Verifies DataCollatorForCausalLM training batch contract."""

    def test_collator_packed_batch_contract(self):
        collator = DataCollatorForCausalLM()
        chunk1 = [10, 20, 30, 40]
        chunk2 = [50, 60, 70, 80]

        batch = collator([{"input_ids": chunk1}, {"input_ids": chunk2}])

        assert isinstance(batch["input_ids"], torch.Tensor)
        assert isinstance(batch["attention_mask"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)

        assert batch["input_ids"].shape == (2, 4)
        assert batch["attention_mask"].shape == (2, 4)
        assert batch["labels"].shape == (2, 4)

        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

        # For packed chunks, labels should match input_ids exactly
        assert torch.equal(batch["input_ids"], batch["labels"])
        assert torch.all(batch["attention_mask"] == 1)

    def test_collator_padding_and_ignore_index(self, tokenizer: KiaraTokenizer):
        collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id, padding_side="right")
        sample1 = [10, 20]
        sample2 = [30, 40, 50, 60]

        batch = collator([sample1, sample2])

        assert batch["input_ids"].shape == (2, 4)
        # Check input_ids padding
        assert batch["input_ids"][0, 2:].tolist() == [tokenizer.pad_token_id, tokenizer.pad_token_id]
        assert batch["input_ids"][1].tolist() == [30, 40, 50, 60]

        # Check attention mask
        assert batch["attention_mask"][0].tolist() == [1, 1, 0, 0]
        assert batch["attention_mask"][1].tolist() == [1, 1, 1, 1]

        # Check labels masking with IGNORE_INDEX (-100)
        assert batch["labels"][0, :2].tolist() == [10, 20]
        assert batch["labels"][0, 2:].tolist() == [IGNORE_INDEX, IGNORE_INDEX]
        assert batch["labels"][1].tolist() == [30, 40, 50, 60]

    def test_collator_left_padding(self, tokenizer: KiaraTokenizer):
        collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id, padding_side="left")
        sample1 = [10, 20]
        sample2 = [30, 40, 50, 60]

        batch = collator([sample1, sample2])

        assert batch["input_ids"][0].tolist() == [tokenizer.pad_token_id, tokenizer.pad_token_id, 10, 20]
        assert batch["attention_mask"][0].tolist() == [0, 0, 1, 1]
        assert batch["labels"][0].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 10, 20]

    def test_collator_pad_to_multiple_of(self, tokenizer: KiaraTokenizer):
        collator = DataCollatorForCausalLM(
            pad_token_id=tokenizer.pad_token_id,
            pad_to_multiple_of=8,
        )
        sample = [1, 2, 3, 4, 5]  # len 5 -> padded to 8

        batch = collator([sample])
        assert batch["input_ids"].shape == (1, 8)
        assert batch["attention_mask"].shape == (1, 8)
        assert batch["labels"].shape == (1, 8)


class TestPipelineAndDataLoader:
    """Verifies DataLoader creation and deterministic iteration."""

    def test_deterministic_dataloader_shuffling(self, tokenizer: KiaraTokenizer):
        texts = [f"Synthetic training sentence number {i} for Kiara." for i in range(50)]
        seq_len = 16
        dataset = PackedDataset.from_texts(texts, tokenizer=tokenizer, seq_len=seq_len)

        loader1 = create_dataloader(dataset, batch_size=4, shuffle=True, seed=42)
        loader2 = create_dataloader(dataset, batch_size=4, shuffle=True, seed=42)

        batches1 = [b["input_ids"] for b in loader1]
        batches2 = [b["input_ids"] for b in loader2]

        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2):
            assert torch.equal(b1, b2)

    def test_build_training_pipeline(self, tokenizer: KiaraTokenizer):
        texts = [
            "Transformer architecture training loop data validation.",
            "def forward(self, x): return self.layer(x)",
            "Loss function L = -log P(w_t | w_<t)",
        ]
        dataset, loader = build_training_pipeline(
            texts=texts,
            tokenizer=tokenizer,
            seq_len=8,
            batch_size=2,
            packed=True,
            seed=1337,
        )

        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape == (2, 8)
        assert batch["input_ids"].dtype == torch.long

    def test_zero_unk_across_pipeline(self, tokenizer: KiaraTokenizer):
        sample_corpus = [
            "Normal English prose for Kiara pretraining.",
            "class MultiHeadAttention(nn.Module):\n    pass\n",
            "Euler's identity: e^{i\\pi} + 1 = 0",
            "Multilingual tokens: 中文, 日本語, 한국어, العربية, हिन्दी, தமிழ்",
            "Special characters: 🚀 🧠 🌟 # @ $ % & *",
        ]
        dataset = TokenizedDataset.from_texts(sample_corpus, tokenizer=tokenizer)
        for seq in dataset:
            assert tokenizer.unk_token_id not in seq

    def test_dataset_leakage_prevention_gate(self):
        """Gate 5: Enforce that training corpus and evaluation corpus are strictly disjoint."""
        import hashlib
        from experiments.tokenizer_benchmark.corpus import get_corpus

        train_texts = get_corpus("train")
        eval_texts = get_corpus("eval")

        train_set = set(train_texts)
        eval_set = set(eval_texts)

        overlap = train_set.intersection(eval_set)
        assert len(overlap) == 0, f"Data leakage detected! Overlapping samples: {overlap}"

        # Hash-level disjointness check
        train_hashes = {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in train_texts}
        eval_hashes = {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in eval_texts}
        hash_overlap = train_hashes.intersection(eval_hashes)
        assert len(hash_overlap) == 0, "Hash-level data leakage detected!"

