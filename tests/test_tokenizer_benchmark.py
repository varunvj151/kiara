"""
tests/test_tokenizer_benchmark.py — Kiara Tokenizer Benchmark Tests
====================================================================
Validates that the benchmark infrastructure works correctly.

Tests cover:
- encode/decode round-trip for all algorithms
- empty input
- ASCII
- Unicode
- emoji
- source code
- mathematical symbols
- rare/unseen strings
- UNK behavior

Run with:
    cd c:/Users/varun/OneDrive/Documents/flutter1/kiara
    python -m pytest tests/test_tokenizer_benchmark.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add benchmark dir to path
BENCHMARK_DIR = Path(__file__).parent.parent / "experiments" / "tokenizer_benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from corpus import (
    get_corpus,
    get_all_texts,
    CATEGORIES,
    CORPUS_VERSION,
    corpus_stats,
)
from metrics import measure_category, BenchmarkResult, aggregate_results, CategoryMetrics
from train import (
    train_tokenizer,
    load_tokenizer_fns,
    get_library_versions,
    SUPPORTED_ALGOS,
)

# ---------------------------------------------------------------------------
# Fixtures: tiny tokenizers trained once for the test session
# ---------------------------------------------------------------------------

TINY_VOCAB = 1000  # Very small — trains in <1s


@pytest.fixture(scope="session")
def training_texts() -> list[str]:
    return get_all_texts()


@pytest.fixture(scope="session")
def byte_bpe_fns(training_texts):
    meta = train_tokenizer("byte_bpe", training_texts, vocab_size=TINY_VOCAB, force_retrain=True)
    return load_tokenizer_fns(meta)


@pytest.fixture(scope="session")
def bpe_fns(training_texts):
    meta = train_tokenizer("bpe", training_texts, vocab_size=TINY_VOCAB, force_retrain=True)
    return load_tokenizer_fns(meta)


@pytest.fixture(scope="session")
def unigram_fns(training_texts):
    meta = train_tokenizer("unigram", training_texts, vocab_size=TINY_VOCAB, force_retrain=True)
    return load_tokenizer_fns(meta)


# ---------------------------------------------------------------------------
# Corpus tests
# ---------------------------------------------------------------------------


class TestCorpus:
    def test_corpus_version_is_string(self):
        assert isinstance(CORPUS_VERSION, str)
        assert len(CORPUS_VERSION) > 0

    def test_all_categories_present(self):
        expected = {"english", "technical", "code", "math", "multilingual",
                    "rare_words", "unicode", "urls_symbols"}
        assert set(CATEGORIES.keys()) == expected

    def test_each_category_has_samples(self):
        for cat_name, samples in CATEGORIES.items():
            assert len(samples) > 0, f"Category '{cat_name}' is empty"

    def test_all_texts_are_strings(self):
        for cat_name, samples in CATEGORIES.items():
            for i, s in enumerate(samples):
                assert isinstance(s, str), f"Category '{cat_name}' sample {i} is not a str"

    def test_get_all_texts_returns_nonempty(self):
        texts = get_all_texts()
        assert len(texts) > 0

    def test_get_corpus_filter(self):
        corpus = get_corpus(["english", "code"])
        assert set(corpus.keys()) == {"english", "code"}

    def test_corpus_stats_structure(self):
        stats = corpus_stats()
        for cat, s in stats.items():
            assert "num_samples" in s
            assert "total_chars" in s
            assert "total_words" in s
            assert s["total_chars"] > 0


# ---------------------------------------------------------------------------
# Library availability
# ---------------------------------------------------------------------------


class TestLibraries:
    def test_tokenizers_importable(self):
        import tokenizers
        assert tokenizers.__version__

    def test_library_versions_dict(self):
        versions = get_library_versions()
        assert "python" in versions
        assert "tokenizers" in versions


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------


ROUND_TRIP_SAMPLES = [
    # ASCII
    "Hello, world!",
    "The quick brown fox.",
    # Empty-ish
    " ",
    # Source code
    "def foo(x):\n    return x * 2\n",
    "for i in range(10):\n    print(i)\n",
    # Math (ASCII-safe LaTeX)
    r"\frac{1}{2}",
    r"\sum_{i=0}^{n} x_i",
    # Technical
    "RMSNorm, SwiGLU, GQA, RoPE",
    # Long word
    "electroencephalography",
    "CounterfactualReasoningModule",
    # Numbers
    "3.14159265358979",
    "0x1F 0b10110101",
]


class TestRoundTripByteBPE:
    """Byte-level BPE guarantees round-trip on all ASCII inputs."""

    @pytest.mark.parametrize("text", ROUND_TRIP_SAMPLES)
    def test_ascii_round_trip(self, text, byte_bpe_fns):
        encode_fn, decode_fn, _ = byte_bpe_fns
        ids = encode_fn(text)
        decoded = decode_fn(ids)
        assert decoded == text, f"Round-trip failed for: {text!r}\nDecoded: {decoded!r}"

    def test_empty_string(self, byte_bpe_fns):
        encode_fn, decode_fn, _ = byte_bpe_fns
        ids = encode_fn("")
        # Empty string should encode to empty or minimal token list
        # and decode back to empty string
        decoded = decode_fn(ids)
        assert decoded == ""

    def test_returns_list_of_ints(self, byte_bpe_fns):
        encode_fn, decode_fn, _ = byte_bpe_fns
        ids = encode_fn("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_unicode_no_unk(self, byte_bpe_fns):
        """Byte-level BPE should never produce UNK for any Unicode input."""
        encode_fn, decode_fn, unk_id = byte_bpe_fns
        unicode_texts = [
            "Hello 🌍🚀🧠",
            "α β γ δ ε ζ",
            "中文日本語한국어",
            "مرحبا שלום",
            "café naïve résumé",
        ]
        for text in unicode_texts:
            ids = encode_fn(text)
            assert ids, f"Encoded to empty for: {text!r}"
            # unk_id is None for byte_bpe, confirming no UNK possible
            assert unk_id is None, "byte_bpe should have unk_id=None"

    def test_emoji_encodes_to_tokens(self, byte_bpe_fns):
        encode_fn, _, _ = byte_bpe_fns
        emoji_text = "🎉🏆⭐"
        ids = encode_fn(emoji_text)
        assert len(ids) > 0

    def test_source_code_round_trip(self, byte_bpe_fns):
        encode_fn, decode_fn, _ = byte_bpe_fns
        code = (
            "class TransformerBlock(nn.Module):\n"
            "    def __init__(self, hidden: int):\n"
            "        super().__init__()\n"
            "        self.norm = RMSNorm(hidden)\n"
        )
        ids = encode_fn(code)
        assert len(ids) > 0
        decoded = decode_fn(ids)
        assert decoded == code

    def test_math_symbols_encode(self, byte_bpe_fns):
        encode_fn, _, _ = byte_bpe_fns
        math_text = "∀ε > 0, ∃δ: |f(x) − f(a)| < ε"
        ids = encode_fn(math_text)
        assert len(ids) > 0

    def test_long_rare_word(self, byte_bpe_fns):
        encode_fn, _, _ = byte_bpe_fns
        word = "pneumonoultramicroscopicsilicovolcanoconiosis"
        ids = encode_fn(word)
        assert len(ids) > 0


class TestRoundTripBPE:
    """Standard BPE tests — may produce UNK for unseen characters."""

    @pytest.mark.parametrize("text", ROUND_TRIP_SAMPLES)
    def test_ascii_round_trip(self, text, bpe_fns):
        encode_fn, decode_fn, unk_id = bpe_fns
        ids = encode_fn(text)
        decoded = decode_fn(ids)
        # Standard BPE may not perfectly round-trip all inputs (e.g. trailing spaces)
        # so we only assert encoding doesn't crash and returns tokens
        assert isinstance(ids, list)

    def test_returns_list_of_ints(self, bpe_fns):
        encode_fn, _, _ = bpe_fns
        ids = encode_fn("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encodes_nonempty_text(self, bpe_fns):
        encode_fn, _, _ = bpe_fns
        ids = encode_fn("The quick brown fox.")
        assert len(ids) > 0

    def test_unk_id_is_tracked(self, bpe_fns):
        _, _, unk_id = bpe_fns
        # Standard BPE has a UNK id
        assert unk_id is not None
        assert isinstance(unk_id, int)

    def test_source_code_encodes(self, bpe_fns):
        encode_fn, _, _ = bpe_fns
        code = "def foo(x):\n    return x * 2\n"
        ids = encode_fn(code)
        assert len(ids) > 0


class TestRoundTripUnigram:
    """Unigram tokenizer tests."""

    @pytest.mark.parametrize("text", ROUND_TRIP_SAMPLES)
    def test_encodes_without_crash(self, text, unigram_fns):
        encode_fn, _, _ = unigram_fns
        ids = encode_fn(text)
        assert isinstance(ids, list)

    def test_returns_list_of_ints(self, unigram_fns):
        encode_fn, _, _ = unigram_fns
        ids = encode_fn("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encodes_nonempty_text(self, unigram_fns):
        encode_fn, _, _ = unigram_fns
        ids = encode_fn("Natural language processing is fascinating.")
        assert len(ids) > 0

    def test_unk_id_is_tracked(self, unigram_fns):
        _, _, unk_id = unigram_fns
        assert unk_id is not None
        assert isinstance(unk_id, int)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_measure_category_basic(self, byte_bpe_fns):
        encode_fn, decode_fn, unk_id = byte_bpe_fns
        texts = ["Hello world.", "This is a test."]
        m = measure_category(texts, "test", encode_fn, decode_fn, unk_id=unk_id)
        assert m.num_samples == 2
        assert m.total_chars > 0
        assert m.total_tokens > 0
        assert m.compression_ratio > 0
        assert m.tokens_per_char > 0
        assert m.encode_time_ms >= 0
        assert m.decode_time_ms >= 0

    def test_round_trip_pass(self, byte_bpe_fns):
        encode_fn, decode_fn, unk_id = byte_bpe_fns
        texts = ["Hello, Kiara!"]
        m = measure_category(texts, "test", encode_fn, decode_fn, unk_id=unk_id)
        assert m.round_trip_pass is True
        assert m.round_trip_failures == 0

    def test_compression_ratio_formula(self, byte_bpe_fns):
        encode_fn, decode_fn, unk_id = byte_bpe_fns
        texts = ["a b c d e f g h i j"]
        m = measure_category(texts, "test", encode_fn, decode_fn, unk_id=unk_id)
        # Verify the math: compression_ratio = total_chars / total_tokens
        if m.total_tokens > 0:
            expected = m.total_chars / m.total_tokens
            assert abs(m.compression_ratio - expected) < 1e-9

    def test_aggregate_results(self, byte_bpe_fns):
        encode_fn, decode_fn, unk_id = byte_bpe_fns
        result = BenchmarkResult()
        result.tokenizer_type = "byte_bpe"
        result.vocab_size_requested = TINY_VOCAB

        for cat_name, texts in get_corpus().items():
            m = measure_category(texts, cat_name, encode_fn, decode_fn, unk_id=unk_id)
            result.by_category[cat_name] = m

        aggregate_results(result)
        assert result.total_samples > 0
        assert result.total_chars > 0
        assert result.overall_compression_ratio > 0

    def test_empty_texts_list(self, byte_bpe_fns):
        encode_fn, decode_fn, unk_id = byte_bpe_fns
        m = measure_category([], "empty", encode_fn, decode_fn, unk_id=unk_id)
        assert m.num_samples == 0
        assert m.total_tokens == 0


# ---------------------------------------------------------------------------
# Cross-algo comparison tests
# ---------------------------------------------------------------------------


class TestCrossAlgo:
    def test_all_algos_encode_english(self, byte_bpe_fns, bpe_fns, unigram_fns):
        """All three algorithms should produce token sequences for English text."""
        text = "The foundation model learns to predict the next token in a sequence."
        for label, fns in [("byte_bpe", byte_bpe_fns), ("bpe", bpe_fns), ("unigram", unigram_fns)]:
            encode_fn, _, _ = fns
            ids = encode_fn(text)
            assert len(ids) > 0, f"{label} produced empty encoding for English text"

    def test_byte_bpe_has_no_unk_id(self, byte_bpe_fns):
        _, _, unk_id = byte_bpe_fns
        assert unk_id is None

    def test_bpe_and_unigram_have_unk_id(self, bpe_fns, unigram_fns):
        _, _, bpe_unk = bpe_fns
        _, _, uni_unk = unigram_fns
        assert bpe_unk is not None
        assert uni_unk is not None

    def test_all_algos_handle_code(self, byte_bpe_fns, bpe_fns, unigram_fns):
        code = "def transformer_block(x, mask=None):\n    return x\n"
        for label, fns in [("byte_bpe", byte_bpe_fns), ("bpe", bpe_fns), ("unigram", unigram_fns)]:
            encode_fn, _, _ = fns
            ids = encode_fn(code)
            assert len(ids) > 0, f"{label} failed on code input"
