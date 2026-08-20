"""
train.py — Kiara Tokenizer Benchmark Trainer
============================================
Trains tokenizer candidates using the benchmark corpus.

Supported algorithms:
  - byte_bpe  : Byte-level BPE via HuggingFace tokenizers
  - bpe       : Standard BPE via HuggingFace tokenizers
  - unigram   : Unigram via HuggingFace tokenizers

Each tokenizer is trained on the full benchmark corpus and saved to
  experiments/tokenizer_benchmark/data/models/<algo>_<vocab_size>/

Training is deterministic (fixed seed where possible).
Library versions and configuration are recorded.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).parent
MODELS_DIR = BENCHMARK_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_ALGOS = ("byte_bpe", "bpe", "unigram")
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"


# ---------------------------------------------------------------------------
# Library availability checks
# ---------------------------------------------------------------------------

def _check_tokenizers() -> str:
    """Return tokenizers library version, raise ImportError if missing."""
    try:
        import tokenizers
        return tokenizers.__version__
    except ImportError as e:
        raise ImportError(
            "HuggingFace 'tokenizers' library not found. "
            "Install it: pip install tokenizers"
        ) from e


def get_library_versions() -> dict[str, str]:
    """Return a dict of relevant library versions for reproducibility."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
    }
    try:
        import tokenizers
        versions["tokenizers"] = tokenizers.__version__
    except ImportError:
        versions["tokenizers"] = "NOT_INSTALLED"
    try:
        import sentencepiece
        versions["sentencepiece"] = sentencepiece.__version__
    except ImportError:
        versions["sentencepiece"] = "NOT_INSTALLED"
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = "NOT_INSTALLED"
    return versions


# ---------------------------------------------------------------------------
# Tokenizer trainers
# ---------------------------------------------------------------------------

def _write_corpus_to_file(texts: list[str], path: Path) -> None:
    """Write corpus texts as newline-separated strings to a file."""
    path.write_text("\n".join(texts), encoding="utf-8")


def train_byte_bpe(
    texts: list[str],
    vocab_size: int,
    save_dir: Path,
    seed: int = 1337,
) -> dict[str, Any]:
    """Train a Byte-level BPE tokenizer.

    Uses HuggingFace tokenizers with ByteLevel pre-tokenizer.
    Guaranteed zero UNK tokens.

    Returns a metadata dict.
    """
    _check_tokenizers()
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    save_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = save_dir / "corpus.txt"
    _write_corpus_to_file(texts, corpus_file)

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1,
        show_progress=False,
    )

    t0 = time.perf_counter()
    tokenizer.train(files=[str(corpus_file)], trainer=trainer)
    train_time_s = time.perf_counter() - t0

    actual_vocab_size = tokenizer.get_vocab_size()
    tokenizer_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    meta = {
        "algo": "byte_bpe",
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": actual_vocab_size,
        "tokenizer_path": str(tokenizer_path),
        "train_time_s": round(train_time_s, 3),
        "library": "tokenizers",
        "unk_possible": False,
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def train_bpe(
    texts: list[str],
    vocab_size: int,
    save_dir: Path,
    seed: int = 1337,
) -> dict[str, Any]:
    """Train a Standard (character-level) BPE tokenizer.

    Uses HuggingFace tokenizers with Whitespace pre-tokenizer.
    Can produce UNK tokens for characters not seen during training.

    Returns a metadata dict.
    """
    _check_tokenizers()
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    save_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = save_dir / "corpus.txt"
    _write_corpus_to_file(texts, corpus_file)

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1,
        show_progress=False,
    )

    t0 = time.perf_counter()
    tokenizer.train(files=[str(corpus_file)], trainer=trainer)
    train_time_s = time.perf_counter() - t0

    actual_vocab_size = tokenizer.get_vocab_size()
    tokenizer_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    unk_id = tokenizer.token_to_id(UNK_TOKEN)
    meta = {
        "algo": "bpe",
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": actual_vocab_size,
        "tokenizer_path": str(tokenizer_path),
        "train_time_s": round(train_time_s, 3),
        "library": "tokenizers",
        "unk_id": unk_id,
        "unk_possible": True,
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def train_unigram(
    texts: list[str],
    vocab_size: int,
    save_dir: Path,
    seed: int = 1337,
) -> dict[str, Any]:
    """Train a Unigram tokenizer.

    Uses HuggingFace tokenizers (Unigram model with Metaspace pre-tokenizer).
    This mirrors the SentencePiece Unigram approach without requiring a
    separate sentencepiece install.

    Returns a metadata dict.
    """
    _check_tokenizers()
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram
    from tokenizers.pre_tokenizers import Metaspace
    from tokenizers.trainers import UnigramTrainer
    from tokenizers.decoders import Metaspace as MetaspaceDecoder

    save_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = save_dir / "corpus.txt"
    _write_corpus_to_file(texts, corpus_file)

    tokenizer = Tokenizer(Unigram())
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")

    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        unk_token=UNK_TOKEN,
        show_progress=False,
        seed_size=vocab_size * 20,
        shrinking_factor=0.75,
        n_sub_iterations=2,
    )

    t0 = time.perf_counter()
    tokenizer.train(files=[str(corpus_file)], trainer=trainer)
    train_time_s = time.perf_counter() - t0

    actual_vocab_size = tokenizer.get_vocab_size()
    tokenizer_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    unk_id = tokenizer.token_to_id(UNK_TOKEN)
    meta = {
        "algo": "unigram",
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": actual_vocab_size,
        "tokenizer_path": str(tokenizer_path),
        "train_time_s": round(train_time_s, 3),
        "library": "tokenizers",
        "unk_id": unk_id,
        "unk_possible": True,
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# Main train dispatcher
# ---------------------------------------------------------------------------

TRAINER_MAP = {
    "byte_bpe": train_byte_bpe,
    "bpe": train_bpe,
    "unigram": train_unigram,
}


def train_tokenizer(
    algo: str,
    texts: list[str],
    vocab_size: int,
    seed: int = 1337,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """Train a tokenizer (or load from cache if already trained).

    Args:
        algo: One of 'byte_bpe', 'bpe', 'unigram'.
        texts: Training texts.
        vocab_size: Target vocabulary size.
        seed: Random seed for reproducibility.
        force_retrain: If True, retrain even if a cached model exists.

    Returns:
        Metadata dict including actual vocab size and save path.
    """
    if algo not in TRAINER_MAP:
        raise ValueError(f"Unknown algorithm '{algo}'. Choose from: {SUPPORTED_ALGOS}")

    save_dir = MODELS_DIR / f"{algo}_{vocab_size}"

    # Check cache
    meta_path = save_dir / "meta.json"
    if meta_path.exists() and not force_retrain:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"  [cache] {algo} vocab={vocab_size} → actual={meta['vocab_size_actual']} (loaded from cache)")
        return meta

    print(f"  [train] {algo} vocab={vocab_size} ...")
    meta = TRAINER_MAP[algo](texts=texts, vocab_size=vocab_size, save_dir=save_dir, seed=seed)
    print(f"          → actual vocab={meta['vocab_size_actual']} in {meta['train_time_s']:.2f}s")
    return meta


# ---------------------------------------------------------------------------
# Load a trained tokenizer as encode/decode callables
# ---------------------------------------------------------------------------

def load_tokenizer_fns(
    meta: dict[str, Any],
) -> tuple[callable, callable, int | None]:
    """Load a trained tokenizer and return (encode_fn, decode_fn, unk_id).

    Returns:
        encode_fn: str -> list[int]
        decode_fn: list[int] -> str
        unk_id: int or None (None if UNK is impossible, e.g. byte_bpe)
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(meta["tokenizer_path"])

    def encode_fn(text: str) -> list[int]:
        return tok.encode(text).ids

    def decode_fn(ids: list[int]) -> str:
        return tok.decode(ids)

    unk_id = meta.get("unk_id", None)
    if not meta.get("unk_possible", True):
        unk_id = None

    return encode_fn, decode_fn, unk_id


if __name__ == "__main__":
    # Smoke test: train a tiny BPE tokenizer on a few sentences
    from corpus import get_all_texts, CORPUS_VERSION

    texts = get_all_texts()
    print(f"Corpus: {len(texts)} samples, version={CORPUS_VERSION}")
    print(f"Library versions: {get_library_versions()}")

    for algo in SUPPORTED_ALGOS:
        meta = train_tokenizer(algo=algo, texts=texts, vocab_size=1000, force_retrain=True)
        encode_fn, decode_fn, unk_id = load_tokenizer_fns(meta)
        test = "Hello, Kiara!"
        ids = encode_fn(test)
        decoded = decode_fn(ids)
        ok = "✓" if decoded == test else "✗"
        print(f"  {algo:10s}  ids={ids[:5]}...  round-trip={ok}")
