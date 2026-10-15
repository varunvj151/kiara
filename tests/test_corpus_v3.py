"""
tests/test_corpus_v3.py — Corpus v3.0 Verification Tests
==========================================================
Verifies:
  - Train / Eval A / Eval B are pairwise disjoint (exact, normalized, SHA-256)
  - 0 UNK tokens across all corpora via KiaraTokenizer
  - Category coverage (all 5 training categories, both eval suites)
  - Reproducibility (same seed = identical corpus)
  - Non-trivial diversity (average sequence length, unique word ratio)
  - Corpus stats completeness
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.tokenizer_benchmark.corpus_v3 import (
    CORPUS_V3_VERSION,
    DEFAULT_SEED,
    OverlapReport,
    _normalize,
    _sha256,
    build_v3_training_corpus,
    compute_v3_corpus_stats,
    get_v3_eval_a_corpus,
    get_v3_eval_a_texts,
    get_v3_eval_b_corpus,
    verify_disjoint,
)
from kiara.tokenizer import KiaraTokenizer


@pytest.fixture(scope="module")
def small_train() -> list[str]:
    """Small training corpus for fast test execution."""
    return build_v3_training_corpus(seed=DEFAULT_SEED, target_samples=500)


@pytest.fixture(scope="module")
def eval_a_texts() -> list[str]:
    return get_v3_eval_a_texts()


@pytest.fixture(scope="module")
def eval_b_texts() -> list[str]:
    from experiments.tokenizer_benchmark.corpus_v3 import _get_eval_b_texts
    return _get_eval_b_texts()


@pytest.fixture(scope="module")
def tokenizer() -> KiaraTokenizer:
    return KiaraTokenizer.load()


# --------------------------------------------------------------------------
# Corpus version
# --------------------------------------------------------------------------

class TestCorpusVersion:
    def test_version_string(self):
        assert CORPUS_V3_VERSION == "3.0.0"


# --------------------------------------------------------------------------
# Category coverage
# --------------------------------------------------------------------------

class TestCategoryCoverage:
    def test_training_corpus_non_empty(self, small_train):
        assert len(small_train) >= 400, f"Expected ~500 samples, got {len(small_train)}"

    def test_eval_a_has_all_five_categories(self):
        corpus = get_v3_eval_a_corpus()
        required = {"natural", "technical", "code", "multilingual", "structured"}
        assert required == set(corpus.keys())

    def test_eval_a_non_empty(self, eval_a_texts):
        assert len(eval_a_texts) >= 30

    def test_eval_b_has_eight_categories(self):
        corpus = get_v3_eval_b_corpus()
        assert len(corpus) == 8

    def test_eval_b_non_empty(self, eval_b_texts):
        assert len(eval_b_texts) >= 100


# --------------------------------------------------------------------------
# Disjointness gate: exact, normalized, SHA-256
# --------------------------------------------------------------------------

class TestDisjointness:
    def test_train_eval_a_exact_disjoint(self, small_train, eval_a_texts):
        overlap = set(small_train) & set(eval_a_texts)
        assert len(overlap) == 0, f"Exact overlap train/eval_a: {list(overlap)[:3]}"

    def test_train_eval_b_exact_disjoint(self, small_train, eval_b_texts):
        overlap = set(small_train) & set(eval_b_texts)
        assert len(overlap) == 0, f"Exact overlap train/eval_b: {list(overlap)[:3]}"

    def test_eval_a_eval_b_exact_disjoint(self, eval_a_texts, eval_b_texts):
        overlap = set(eval_a_texts) & set(eval_b_texts)
        assert len(overlap) == 0

    def test_train_eval_a_normalized_disjoint(self, small_train, eval_a_texts):
        train_norm = {_normalize(t) for t in small_train}
        eval_norm = {_normalize(t) for t in eval_a_texts}
        overlap = train_norm & eval_norm
        assert len(overlap) == 0

    def test_train_eval_b_normalized_disjoint(self, small_train, eval_b_texts):
        train_norm = {_normalize(t) for t in small_train}
        eval_norm = {_normalize(t) for t in eval_b_texts}
        overlap = train_norm & eval_norm
        assert len(overlap) == 0

    def test_train_eval_a_sha256_disjoint(self, small_train, eval_a_texts):
        train_sha = {_sha256(t) for t in small_train}
        eval_sha = {_sha256(t) for t in eval_a_texts}
        assert len(train_sha & eval_sha) == 0

    def test_train_eval_b_sha256_disjoint(self, small_train, eval_b_texts):
        train_sha = {_sha256(t) for t in small_train}
        eval_sha = {_sha256(t) for t in eval_b_texts}
        assert len(train_sha & eval_sha) == 0

    def test_verify_disjoint_report_is_clean(self, small_train, eval_a_texts, eval_b_texts):
        report = verify_disjoint(small_train, eval_a_texts, eval_b_texts)
        assert report.is_clean, f"Corpus is NOT clean: {report}"
        assert report.train_eval_a_exact == 0
        assert report.train_eval_b_exact == 0
        assert report.eval_a_eval_b_exact == 0


# --------------------------------------------------------------------------
# Zero UNK gate
# --------------------------------------------------------------------------

class TestZeroUNK:
    def test_training_corpus_zero_unk(self, small_train, tokenizer):
        unk_id = tokenizer.unk_token_id
        unk_count = 0
        for text in small_train:
            ids = tokenizer.encode(text)
            unk_count += ids.count(unk_id)
        assert unk_count == 0, f"Found {unk_count} UNK tokens in training corpus"

    def test_eval_a_zero_unk(self, eval_a_texts, tokenizer):
        unk_id = tokenizer.unk_token_id
        unk_count = 0
        for text in eval_a_texts:
            ids = tokenizer.encode(text)
            unk_count += ids.count(unk_id)
        assert unk_count == 0, f"Found {unk_count} UNK tokens in Eval A"

    def test_eval_b_zero_unk(self, eval_b_texts, tokenizer):
        unk_id = tokenizer.unk_token_id
        unk_count = 0
        for text in eval_b_texts:
            ids = tokenizer.encode(text)
            unk_count += ids.count(unk_id)
        assert unk_count == 0, f"Found {unk_count} UNK tokens in Eval B"


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_produces_identical_corpus(self):
        corpus_a = build_v3_training_corpus(seed=DEFAULT_SEED, target_samples=200)
        corpus_b = build_v3_training_corpus(seed=DEFAULT_SEED, target_samples=200)
        assert corpus_a == corpus_b

    def test_different_seed_produces_different_corpus(self):
        corpus_a = build_v3_training_corpus(seed=DEFAULT_SEED, target_samples=200)
        corpus_b = build_v3_training_corpus(seed=DEFAULT_SEED + 1, target_samples=200)
        assert corpus_a != corpus_b


# --------------------------------------------------------------------------
# Diversity checks
# --------------------------------------------------------------------------

class TestDiversity:
    def test_unique_word_ratio(self, small_train):
        all_words = [w for t in small_train for w in t.split()]
        ratio = len(set(all_words)) / max(1, len(all_words))
        assert ratio >= 0.02, f"Vocabulary diversity too low: unique ratio = {ratio:.4f}"

    def test_average_sequence_length(self, small_train):
        avg_len = sum(len(t) for t in small_train) / max(1, len(small_train))
        assert avg_len >= 80, f"Average sequence length too short: {avg_len:.1f} chars"

    def test_no_empty_samples(self, small_train, eval_a_texts, eval_b_texts):
        for t in small_train:
            assert len(t.strip()) > 0
        for t in eval_a_texts:
            assert len(t.strip()) > 0
        for t in eval_b_texts:
            assert len(t.strip()) > 0

    def test_multilingual_variety(self):
        corpus = get_v3_eval_a_corpus()
        multi_samples = corpus["multilingual"]
        # Must include Tamil and Hindi specifically
        assert any("தமிழ்" in s or "இயந்திர" in s for s in multi_samples), "No Tamil in Eval A multilingual"
        assert any("मशीन" in s or "हानि" in s for s in multi_samples), "No Hindi in Eval A multilingual"


# --------------------------------------------------------------------------
# Corpus stats
# --------------------------------------------------------------------------

class TestCorpusStats:
    def test_stats_completeness(self, small_train, eval_a_texts, eval_b_texts):
        stats = compute_v3_corpus_stats(small_train, eval_a_texts, eval_b_texts)
        assert stats["version"] == CORPUS_V3_VERSION
        assert stats["is_clean"] is True
        assert stats["training"]["num_documents"] > 0
        assert stats["eval_a"]["num_documents"] > 0
        assert stats["eval_b"]["num_documents"] > 0

    def test_overlap_report_fields(self, small_train, eval_a_texts, eval_b_texts):
        stats = compute_v3_corpus_stats(small_train, eval_a_texts, eval_b_texts)
        report = stats["overlap_report"]
        for field in [
            "train_eval_a_exact", "train_eval_b_exact", "eval_a_eval_b_exact",
            "train_eval_a_normalized", "train_eval_b_normalized",
            "train_eval_a_sha256", "train_eval_b_sha256",
        ]:
            assert field in report
            assert report[field] == 0, f"Overlap in {field}: {report[field]}"
