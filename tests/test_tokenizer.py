"""
tests/test_tokenizer.py — Canonical KiaraTokenizer Test Suite
==============================================================
Validates that KiaraTokenizer adheres to frozen Decision 003 specifications:
  - Exact 32,768 vocabulary size
  - Correct special token IDs ([UNK]=0, [PAD]=1, [BOS]=2, [EOS]=3)
  - Zero UNK tokens and exact round-trip reconstruction
  - Batching, padding (right/left), truncation, PyTorch tensor conversion
  - Serialization, loading, and configuration drift prevention
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from kiara.tokenizer import (
    BOS_TOKEN,
    BOS_TOKEN_ID,
    EOS_TOKEN,
    EOS_TOKEN_ID,
    PAD_TOKEN,
    PAD_TOKEN_ID,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    UNK_TOKEN_ID,
    KiaraTokenizer,
    TokenizerConfig,
)


@pytest.fixture(scope="module")
def tokenizer() -> KiaraTokenizer:
    """Load canonical KiaraTokenizer instance from artifacts/tokenizer."""
    return KiaraTokenizer.load()


class TestKiaraTokenizerFrozenConfig:
    """Verifies that the tokenizer configuration matches Decision 003."""

    def test_vocab_size_is_frozen_at_32768(self, tokenizer: KiaraTokenizer):
        assert tokenizer.vocab_size == 32768
        assert len(tokenizer) == 32768

    def test_special_token_ids(self, tokenizer: KiaraTokenizer):
        assert tokenizer.unk_token_id == UNK_TOKEN_ID == 0
        assert tokenizer.pad_token_id == PAD_TOKEN_ID == 1
        assert tokenizer.bos_token_id == BOS_TOKEN_ID == 2
        assert tokenizer.eos_token_id == EOS_TOKEN_ID == 3

    def test_special_token_strings(self, tokenizer: KiaraTokenizer):
        assert tokenizer.unk_token == UNK_TOKEN == "[UNK]"
        assert tokenizer.pad_token == PAD_TOKEN == "[PAD]"
        assert tokenizer.bos_token == BOS_TOKEN == "[BOS]"
        assert tokenizer.eos_token == EOS_TOKEN == "[EOS]"

    def test_special_tokens_in_vocab(self, tokenizer: KiaraTokenizer):
        assert tokenizer.token_to_id("[UNK]") == 0
        assert tokenizer.token_to_id("[PAD]") == 1
        assert tokenizer.token_to_id("[BOS]") == 2
        assert tokenizer.token_to_id("[EOS]") == 3

    def test_id_to_token_lookups(self, tokenizer: KiaraTokenizer):
        assert tokenizer.id_to_token(0) == "[UNK]"
        assert tokenizer.id_to_token(1) == "[PAD]"
        assert tokenizer.id_to_token(2) == "[BOS]"
        assert tokenizer.id_to_token(3) == "[EOS]"


class TestKiaraTokenizerEncodingDecoding:
    """Verifies lossless encoding and decoding across diverse domains."""

    @pytest.mark.parametrize(
        "text",
        [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "def scaled_dot_product_attention(q, k, v, mask=None):\n    return scores\n",
            "class TransformerBlock(nn.Module):\n    def __init__(self):\n        super().__init__()\n",
            r"\frac{\partial L}{\partial \theta} = -\frac{1}{N} \sum_{i=1}^{N} y_i \log \hat{y}_i",
            "∀ε > 0, ∃δ > 0: |x − a| < δ",
            "Hello 🌍🚀🧠 — multilingual: 中文, 日本語, 한국어, العربية, हिन्दी",
            "café naïve résumé façade coördinate Ångström",
            "https://github.com/kiara-project/kiara/blob/main/model.py#L42-L80",
            "$100.00 ± 0.05 (p < 0.001, n=1024)",
            "pneumonoultramicroscopicsilicovolcanoconiosis",
            "   leading and trailing spaces   ",
            "\n\n\tmultiple\n\twhitespace\n\n",
        ],
    )
    def test_exact_round_trip_reconstruction(self, tokenizer: KiaraTokenizer, text: str):
        ids = tokenizer.encode(text)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        # Byte-level BPE guarantees 0 UNK and exact text reconstruction
        assert tokenizer.unk_token_id not in ids
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_empty_string(self, tokenizer: KiaraTokenizer):
        ids = tokenizer.encode("")
        assert ids == []
        assert tokenizer.decode(ids) == ""

    def test_bos_and_eos_handling(self, tokenizer: KiaraTokenizer):
        text = "Kiara foundation model"
        ids_raw = tokenizer.encode(text)
        ids_bos = tokenizer.encode(text, bos=True)
        ids_eos = tokenizer.encode(text, eos=True)
        ids_both = tokenizer.encode(text, add_special_tokens=True)

        assert ids_bos == [tokenizer.bos_token_id] + ids_raw
        assert ids_eos == ids_raw + [tokenizer.eos_token_id]
        assert ids_both == [tokenizer.bos_token_id] + ids_raw + [tokenizer.eos_token_id]

        # Decode skipping special tokens
        assert tokenizer.decode(ids_both, skip_special_tokens=True) == text
        # Decode preserving special tokens
        decoded_with_special = tokenizer.decode(ids_both, skip_special_tokens=False)
        assert "[BOS]" in decoded_with_special and "[EOS]" in decoded_with_special

    def test_batch_encoding_and_decoding(self, tokenizer: KiaraTokenizer):
        texts = [
            "First sequence.",
            "Second longer sequence with code: x = y + 1",
            "Third sequence with emoji 🎉",
        ]
        batch_ids = tokenizer.encode_batch(texts)
        assert len(batch_ids) == len(texts)
        decoded = tokenizer.decode_batch(batch_ids)
        assert decoded == texts


class TestKiaraTokenizerPaddingAndTruncation:
    """Verifies padding and truncation operations."""

    def test_pad_right(self, tokenizer: KiaraTokenizer):
        seq1 = [10, 20]
        seq2 = [30, 40, 50, 60]
        res = tokenizer.pad([seq1, seq2], padding_side="right")

        assert res["input_ids"] == [
            [10, 20, tokenizer.pad_token_id, tokenizer.pad_token_id],
            [30, 40, 50, 60],
        ]
        assert res["attention_mask"] == [
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ]

    def test_pad_left(self, tokenizer: KiaraTokenizer):
        seq1 = [10, 20]
        seq2 = [30, 40, 50, 60]
        res = tokenizer.pad([seq1, seq2], padding_side="left")

        assert res["input_ids"] == [
            [tokenizer.pad_token_id, tokenizer.pad_token_id, 10, 20],
            [30, 40, 50, 60],
        ]
        assert res["attention_mask"] == [
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ]

    def test_pad_to_max_length(self, tokenizer: KiaraTokenizer):
        seq = [10, 20]
        res = tokenizer.pad([seq], max_length=5)
        assert len(res["input_ids"][0]) == 5
        assert res["input_ids"][0] == [10, 20, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id]
        assert res["attention_mask"][0] == [1, 1, 0, 0, 0]

    def test_pad_to_multiple_of(self, tokenizer: KiaraTokenizer):
        seq = [10, 20, 30]  # len = 3
        res = tokenizer.pad([seq], pad_to_multiple_of=8)
        assert len(res["input_ids"][0]) == 8
        assert res["attention_mask"][0] == [1, 1, 1, 0, 0, 0, 0, 0]

    def test_pad_return_tensors_pt(self, tokenizer: KiaraTokenizer):
        seq1 = [10, 20]
        seq2 = [30, 40, 50, 60]
        res = tokenizer.pad([seq1, seq2], return_tensors="pt")
        assert isinstance(res["input_ids"], torch.Tensor)
        assert isinstance(res["attention_mask"], torch.Tensor)
        assert res["input_ids"].shape == (2, 4)
        assert res["attention_mask"].shape == (2, 4)
        assert res["input_ids"].dtype == torch.long

    def test_truncate_right(self, tokenizer: KiaraTokenizer):
        seq = [1, 2, 3, 4, 5, 6]
        res = tokenizer.truncate([seq], max_length=4, truncation_side="right")
        assert res == [[1, 2, 3, 4]]

    def test_truncate_left(self, tokenizer: KiaraTokenizer):
        seq = [1, 2, 3, 4, 5, 6]
        res = tokenizer.truncate([seq], max_length=4, truncation_side="left")
        assert res == [[3, 4, 5, 6]]


class TestKiaraTokenizerCallInterface:
    """Verifies the unified __call__ interface."""

    def test_call_single_text(self, tokenizer: KiaraTokenizer):
        text = "Hello Kiara"
        out = tokenizer(text)
        assert "input_ids" in out
        assert out["input_ids"] == tokenizer.encode(text)

    def test_call_batch_with_padding_and_tensors(self, tokenizer: KiaraTokenizer):
        texts = ["Short", "A much longer sentence for batch test"]
        out = tokenizer(texts, padding=True, return_tensors="pt")
        assert isinstance(out["input_ids"], torch.Tensor)
        assert isinstance(out["attention_mask"], torch.Tensor)
        assert out["input_ids"].ndim == 2
        assert out["input_ids"].shape[0] == 2

    def test_call_with_truncation(self, tokenizer: KiaraTokenizer):
        text = "A very long sentence that needs to be truncated for context window."
        out = tokenizer(text, truncation=True, max_length=3)
        assert len(out["input_ids"]) == 3


class TestKiaraTokenizerPersistenceAndDrift:
    """Verifies that saving and loading preserves exact configuration and behavior."""

    def test_save_and_reload_identity(self, tokenizer: KiaraTokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_tokenizer"
            tokenizer.save(save_path)

            assert (save_path / "tokenizer.json").exists()
            assert (save_path / "tokenizer_config.json").exists()
            assert (save_path / "metadata.json").exists()

            reloaded = KiaraTokenizer.load(save_path)
            assert reloaded.vocab_size == tokenizer.vocab_size == 32768
            assert reloaded.pad_token_id == tokenizer.pad_token_id == 1

            sample = "Testing exact token parity across save/load cycles! 🚀"
            assert reloaded.encode(sample) == tokenizer.encode(sample)
            assert reloaded.decode(reloaded.encode(sample)) == sample

    def test_from_pretrained_alias(self, tokenizer: KiaraTokenizer):
        tok = KiaraTokenizer.from_pretrained()
        assert tok.vocab_size == 32768
        assert tok.config.tokenizer_type == "byte_bpe"
