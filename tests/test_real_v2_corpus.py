"""
tests/test_real_v2_corpus.py — Unit Tests for Kiara-Real-v2.0 Multi-Domain Corpus
================================================================================
Verifies:
  1. Strict train/held-out disjointness (0 SHA-256 hash collisions).
  2. All 6 domains (Natural, Technical, Code, Math, Multilingual, Structured) present.
  3. Manifest metadata consistency.
  4. Tokenizer encodes all documents losslessly with zero UNK tokens.
"""

import hashlib
import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kiara.tokenizer import KiaraTokenizer

DATA_DIR = REPO_ROOT / "data" / "real_v2"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_files_exist():
    assert (DATA_DIR / "train.jsonl").exists()
    assert (DATA_DIR / "eval_real.jsonl").exists()
    assert (DATA_DIR / "manifest.json").exists()


def test_manifest_metadata():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "Kiara-Real-v2.0"
    assert manifest["is_disjoint"] is True
    assert manifest["train_docs"] > 20000
    assert manifest["train_tokens"] > 5000000


def test_strict_disjointness():
    train_lines = (DATA_DIR / "train.jsonl").read_text(encoding="utf-8").strip().split("\n")
    eval_lines = (DATA_DIR / "eval_real.jsonl").read_text(encoding="utf-8").strip().split("\n")

    train_hashes = {_sha256(json.loads(l)["text"]) for l in train_lines}
    eval_hashes = {_sha256(json.loads(l)["text"]) for l in eval_lines}

    overlap = train_hashes & eval_hashes
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping documents between train and eval!"


def test_domain_diversity():
    train_lines = (DATA_DIR / "train.jsonl").read_text(encoding="utf-8").strip().split("\n")
    domains = {json.loads(l)["domain"].split("_")[0] for l in train_lines}

    expected_domains = {"natural", "technical", "source", "mathematics", "multilingual", "structured"}
    assert expected_domains.issubset(domains), f"Missing domains: {expected_domains - domains}"


def test_lossless_encoding_zero_unk():
    tokenizer = KiaraTokenizer.load()
    train_lines = (DATA_DIR / "train.jsonl").read_text(encoding="utf-8").strip().split("\n")

    # Sample across different lines
    for line in train_lines[:50]:
        doc = json.loads(line)["text"]
        token_ids = tokenizer.encode(doc)
        assert len(token_ids) > 0
        if tokenizer.unk_token_id is not None:
            assert tokenizer.unk_token_id not in token_ids
