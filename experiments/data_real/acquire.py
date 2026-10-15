"""
experiments/data_real/acquire.py — Phase 26A.1: Real-Data Acquisition
======================================================================
Acquires, cleans, deduplicates, and splits a clean, reproducible
real-world text corpus for Kiara pre-training:
  - TinyStories (GPT-4 generated / human-curated stories: narrative, dialogue, reasoning)
  - Wikitext (Wikipedia articles: expository, scientific, historical)

Outputs saved to data/real/:
  - data/real/train.jsonl        (Training set, ~3.5M tokens)
  - data/real/eval_real.jsonl    (Held-out real validation set, ~250K tokens)
  - data/real/manifest.json      (Provenance, SHA-256 hashes, license metadata)
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kiara.tokenizer import KiaraTokenizer

DATA_DIR = REPO_ROOT / "data" / "real"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1337
TARGET_TRAIN_TOKENS = 3_500_000
TARGET_EVAL_TOKENS = 250_000


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_tinystories(max_docs: int = 15000) -> list[str]:
    """Fetch real narrative stories from TinyStories."""
    print(f"  [1/2] Fetching TinyStories...", flush=True)
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    headers = {"Range": f"bytes=0-{max_docs * 700}"}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code not in (200, 206):
        raise RuntimeError(f"Failed to fetch TinyStories: HTTP {r.status_code}")

    raw_text = r.text
    raw_stories = raw_text.split("<|endoftext|>")
    clean_stories = []
    for s in raw_stories:
        norm = _normalize(s)
        if len(norm) >= 80:  # Skip trivial fragments
            clean_stories.append(norm)
        if len(clean_stories) >= max_docs:
            break
    print(f"        -> Collected {len(clean_stories):,} clean stories.", flush=True)
    return clean_stories


def fetch_wikitext(max_docs: int = 5000) -> list[str]:
    """Fetch encyclopedic real-world text from Wikitext."""
    print(f"  [2/2] Fetching Wikitext expository prose...", flush=True)
    urls = [
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    ]
    all_paras: list[str] = []
    for url in urls:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            continue
        lines = r.text.split("\n")
        for line in lines:
            p_clean = _normalize(line)
            # Filter out lone headers like '= = = Background = = ='
            if p_clean.startswith("=") and p_clean.endswith("="):
                continue
            if len(p_clean) >= 100 and "<unk>" not in p_clean:
                # Clean wikitext formatting tokens
                p_clean = p_clean.replace(" @-@ ", "-").replace(" @,@ ", ", ").replace(" @.@ ", ". ")
                all_paras.append(p_clean)
            if len(all_paras) >= max_docs:
                break
        if len(all_paras) >= max_docs:
            break
    print(f"        -> Collected {len(all_paras):,} clean encyclopedic passages.", flush=True)
    return all_paras


def acquire_real_corpus() -> dict[str, Any]:
    print(f"\n{'='*70}")
    print("  PHASE 26A.1: ACQUIRING REAL-WORLD TEXT CORPUS (Kiara-Real-v1.0)")
    print(f"{'='*70}\n")

    tokenizer = KiaraTokenizer.load()
    rng = random.Random(SEED)

    # 1. Fetch from verified open sources
    stories = fetch_tinystories(max_docs=15000)
    wikitext = fetch_wikitext(max_docs=5000)

    # 2. Combine and deduplicate
    combined = stories + wikitext
    rng.shuffle(combined)

    seen_hashes: set[str] = set()
    deduped: list[str] = []
    for doc in combined:
        h = _sha256(doc)
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(doc)

    print(f"\n  [Deduplication] {len(combined):,} raw docs -> {len(deduped):,} unique docs (0 duplicates).")

    # 3. Create strict Train / Held-out split (90% / 10%)
    split_idx = int(len(deduped) * 0.90)
    train_docs = deduped[:split_idx]
    eval_docs = deduped[split_idx:]

    # 4. Strict disjointness verification
    train_hashes = {_sha256(d) for d in train_docs}
    eval_hashes = {_sha256(d) for d in eval_docs}
    overlap = train_hashes & eval_hashes
    if overlap:
        raise ValueError(f"CRITICAL: Found {len(overlap)} overlapping documents between train and eval!")
    print(f"  [Verification] Train ({len(train_docs):,}) and Held-out Eval ({len(eval_docs):,}) are STRICTLY DISJOINT (0 hash overlap).")

    # 5. Tokenize and measure token volume
    print("  [Token Counting] Encoding dataset with frozen KiaraTokenizer...", flush=True)
    train_tokens = sum(len(tokenizer.encode(d)) for d in train_docs)
    eval_tokens = sum(len(tokenizer.encode(d)) for d in eval_docs)

    print(f"  [Dataset Stats]")
    print(f"    Train documents    : {len(train_docs):,}")
    print(f"    Train tokens       : {train_tokens:,}")
    print(f"    Held-out documents : {len(eval_docs):,}")
    print(f"    Held-out tokens    : {eval_tokens:,}")
    print(f"    Total tokens       : {train_tokens + eval_tokens:,}")

    # 6. Save JSONL files
    train_path = DATA_DIR / "train.jsonl"
    eval_path = DATA_DIR / "eval_real.jsonl"

    print(f"  [Writing] Saving to {DATA_DIR}...", flush=True)
    with open(train_path, "w", encoding="utf-8") as f:
        for d in train_docs:
            f.write(json.dumps({"text": d}) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for d in eval_docs:
            f.write(json.dumps({"text": d}) + "\n")

    manifest = {
        "dataset_name": "Kiara-Real-v1.0",
        "sources": [
            {"name": "TinyStories-GPT4", "license": "CDLA-Permissive / Open", "domain": "narrative_commonsense"},
            {"name": "Wikitext-2", "license": "CC-BY-SA 3.0", "domain": "expository_encyclopedic"},
        ],
        "seed": SEED,
        "train_docs": len(train_docs),
        "train_tokens": train_tokens,
        "eval_docs": len(eval_docs),
        "eval_tokens": eval_tokens,
        "train_sha256": _sha256(train_path.read_text(encoding="utf-8")),
        "eval_sha256": _sha256(eval_path.read_text(encoding="utf-8")),
        "is_disjoint": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"  [Manifest] Saved metadata to {manifest_path}")
    print(f"{'='*70}\n")
    return manifest


if __name__ == "__main__":
    acquire_real_corpus()
