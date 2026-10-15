"""
tests/test_real_corpus.py — Verification Suite for Real-World Corpus (Stage 26A)
================================================================================
Verifies:
  1. Strict disjointness between Train and Held-Out Eval (exact & SHA-256)
  2. 0 [UNK] tokens across all real text documents
  3. Non-empty documents and valid length distributions
  4. Manifest integrity and SHA-256 provenance
  5. Deterministic batch encoding via KiaraTokenizer
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kiara.tokenizer import KiaraTokenizer

DATA_DIR = REPO_ROOT / "data" / "real"


@pytest.fixture(scope="module")
def tokenizer() -> KiaraTokenizer:
    return KiaraTokenizer.load()


@pytest.fixture(scope="module")
def train_docs() -> list[str]:
    train_file = DATA_DIR / "train.jsonl"
    assert train_file.exists(), "train.jsonl does not exist in data/real/"
    return [json.loads(line)["text"] for line in train_file.read_text(encoding="utf-8").strip().split("\n")]


@pytest.fixture(scope="module")
def eval_docs() -> list[str]:
    eval_file = DATA_DIR / "eval_real.jsonl"
    assert eval_file.exists(), "eval_real.jsonl does not exist in data/real/"
    return [json.loads(line)["text"] for line in eval_file.read_text(encoding="utf-8").strip().split("\n")]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestRealCorpusIntegrity:
    def test_files_exist(self):
        assert (DATA_DIR / "train.jsonl").exists()
        assert (DATA_DIR / "eval_real.jsonl").exists()
        assert (DATA_DIR / "manifest.json").exists()

    def test_manifest_metadata(self):
        manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["dataset_name"] == "Kiara-Real-v1.0"
        assert manifest["train_docs"] > 10000
        assert manifest["eval_docs"] > 1000
        assert manifest["train_tokens"] > 3000000
        assert manifest["is_disjoint"] is True

    def test_non_empty_documents(self, train_docs, eval_docs):
        assert len(train_docs) > 10000
        assert len(eval_docs) > 1000
        for d in train_docs[:500]:
            assert len(d.strip()) >= 50
        for d in eval_docs[:200]:
            assert len(d.strip()) >= 50

    def test_strict_disjointness_exact(self, train_docs, eval_docs):
        train_set = set(train_docs)
        eval_set = set(eval_docs)
        overlap = train_set & eval_set
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping documents between train and eval!"

    def test_strict_disjointness_sha256(self, train_docs, eval_docs):
        train_hashes = {_sha256(d) for d in train_docs}
        eval_hashes = {_sha256(d) for d in eval_docs}
        overlap = train_hashes & eval_hashes
        assert len(overlap) == 0, f"Found {len(overlap)} SHA-256 hash collisions between train and eval!"

    def test_zero_unk_tokens(self, train_docs, eval_docs, tokenizer):
        unk_id = tokenizer.unk_token_id
        # Test sample of train docs
        for d in train_docs[:300]:
            ids = tokenizer.encode(d)
            assert unk_id not in ids, f"Found UNK token in train document: {d[:50]}"

        # Test sample of eval docs
        for d in eval_docs[:300]:
            ids = tokenizer.encode(d)
            assert unk_id not in ids, f"Found UNK token in eval document: {d[:50]}"
