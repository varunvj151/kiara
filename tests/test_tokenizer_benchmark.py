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
    get_eval_corpus,
    get_training_texts,
    get_eval_texts,
    get_all_texts,
    build_training_corpus,
    CATEGORIES,
    EVAL_CATEGORIES,
    CORPUS_VERSION,
    corpus_stats,
)
from metrics import (
    measure_category,
    measure_vocabulary_efficiency,
    BenchmarkResult,
    aggregate_results,
    CategoryMetrics,
    VocabularyEfficiency,
)
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
    return get_training_texts()


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
        assert set(EVAL_CATEGORIES.keys()) == expected

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
        assert "version" in stats
        assert "is_disjoint" in stats
        assert "training" in stats
        assert "evaluation" in stats
        assert "overall" in stats
        assert stats["is_disjoint"] is True
        for cat, s in stats["evaluation"]["categories"].items():
            assert "num_samples" in s
            assert "total_chars" in s
            assert "total_words" in s
            assert s["total_chars"] > 0

    def test_train_eval_corpora_disjoint(self):
        """Training corpus and held-out evaluation corpus must be strictly disjoint."""
        train_texts = get_training_texts()
        eval_texts = get_eval_texts()
        assert set(train_texts).isdisjoint(set(eval_texts)), "Training and eval corpora overlap!"

    def test_corpus_generation_deterministic(self):
        """Corpus generation with same seed must be 100% deterministic."""
        c1 = build_training_corpus(seed=1337, target_samples=500)
        c2 = build_training_corpus(seed=1337, target_samples=500)
        assert c1 == c2, "Corpus generator is not deterministic across runs!"

    def test_corpus_sufficiently_large(self):
        """Training corpus must have enough samples and unique vocabulary for scaling."""
        stats = corpus_stats()
        tr = stats["training"]
        assert tr["num_samples"] >= 50000, f"Expected >=50k training samples, got {tr['num_samples']}"
        assert tr["unique_words"] >= 50000, f"Expected >=50k unique words, got {tr['unique_words']}"
        assert tr["approx_mb"] >= 10.0, f"Expected >=10 MB training data, got {tr['approx_mb']}"

    def test_unicode_and_emoji_coverage(self):
        """Verify that training and eval corpora cover diverse Unicode scripts and emoji."""
        eval_all = " ".join(get_eval_texts())
        assert any(ord(c) > 0x1F600 for c in eval_all), "Missing emojis in eval corpus"
        assert any("\u4e00" <= c <= "\u9fff" for c in eval_all), "Missing CJK in eval corpus"
        assert any("\u0600" <= c <= "\u06ff" for c in eval_all), "Missing Arabic in eval corpus"
        assert any("\u0900" <= c <= "\u097f" for c in eval_all), "Missing Devanagari in eval corpus"

    def test_no_accidental_duplicated_blocks_dominate(self):
        """Verify that unique strings account for at least 99% of generated samples."""
        train_texts = get_training_texts()
        unique_ratio = len(set(train_texts)) / len(train_texts)
        assert unique_ratio >= 0.99, f"Duplicate strings dominate dataset (ratio: {unique_ratio:.4f})"



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
    """Byte-level BPE guarantees round-trip on all UTF-8 inputs."""

    @pytest.mark.parametrize("text", ROUND_TRIP_SAMPLES)
    def test_ascii_round_trip(self, text, byte_bpe_fns):
        encode_fn, decode_fn, _ = byte_bpe_fns
        ids = encode_fn(text)
        decoded = decode_fn(ids)
        assert decoded == text, f"Round-trip failed for: {text!r}\nDecoded: {decoded!r}"

    def test_empty_string(self, byte_bpe_fns):
        encode_fn, decode_fn, _ = byte_bpe_fns
        ids = encode_fn("")
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
            "https://github.com/kiara-project/kiara/blob/main/model.py#L1-L10",
            r"\frac{\partial L}{\partial \theta} = \int_{-\infty}^{+\infty} f(x) dx",
        ]
        for text in unicode_texts:
            ids = encode_fn(text)
            assert ids, f"Encoded to empty for: {text!r}"
            assert unk_id is None, "byte_bpe should have unk_id=None"
            decoded = decode_fn(ids)
            assert decoded == text, f"Round-trip failed for Unicode text: {text!r}"

    def test_emoji_encodes_to_tokens(self, byte_bpe_fns):
        encode_fn, decode_fn, _ = byte_bpe_fns
        emoji_text = "🎉🏆⭐"
        ids = encode_fn(emoji_text)
        assert len(ids) > 0
        assert decode_fn(ids) == emoji_text

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

    def test_math_symbols_round_trip(self, byte_bpe_fns):
        encode_fn, decode_fn, _ = byte_bpe_fns
        math_text = "∀ε > 0, ∃δ: |f(x) − f(a)| < ε"
        ids = encode_fn(math_text)
        assert len(ids) > 0
        assert decode_fn(ids) == math_text


class TestRoundTripBPE:
    """Standard character-level BPE with Metaspace — reversible for seen characters, UNK for unseen."""

    @pytest.mark.parametrize("text", ROUND_TRIP_SAMPLES)
    def test_ascii_round_trip(self, text, bpe_fns):
        encode_fn, decode_fn, unk_id = bpe_fns
        ids = encode_fn(text)
        decoded = decode_fn(ids)
        assert isinstance(ids, list)
        # Metaspace pre-tokenizer + decoder ensures exact round-trip for seen ASCII/English/code
        assert decoded == text, f"BPE round-trip failed on: {text!r}\nDecoded: {decoded!r}"

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
        assert unk_id is not None
        assert isinstance(unk_id, int)

    def test_source_code_round_trip(self, bpe_fns):
        encode_fn, decode_fn, _ = bpe_fns
        code = "def foo(x):\n    return x * 2\n"
        ids = encode_fn(code)
        assert len(ids) > 0
        assert decode_fn(ids) == code

    def test_punctuation_and_whitespace_preservation(self, bpe_fns):
        encode_fn, decode_fn, _ = bpe_fns
        text = "Hello, world! Punctuation: [1, 2, 3]; 'quoted'."
        ids = encode_fn(text)
        decoded = decode_fn(ids)
        assert decoded == text

    def test_unseen_character_produces_unk_and_roundtrip_mismatch(self, bpe_fns):
        """Standard BPE lacks byte fallback, so unseen characters produce UNK."""
        encode_fn, decode_fn, unk_id = bpe_fns
        text = "Hello 🌍🚀 (unseen emoji)"
        ids = encode_fn(text)
        assert unk_id in ids, "Expected UNK token for unseen emoji in standard BPE"
        decoded = decode_fn(ids)
        assert decoded != text, "Expected round-trip mismatch due to UNK substitution"


class TestRoundTripUnigram:
    """Unigram tokenizer with Byte Fallback — guaranteed zero UNK and exact round-trip."""

    @pytest.mark.parametrize("text", ROUND_TRIP_SAMPLES)
    def test_ascii_round_trip(self, text, unigram_fns):
        encode_fn, decode_fn, _ = unigram_fns
        ids = encode_fn(text)
        decoded = decode_fn(ids)
        assert isinstance(ids, list)
        assert decoded == text, f"Unigram round-trip failed on: {text!r}\nDecoded: {decoded!r}"

    def test_returns_list_of_ints(self, unigram_fns):
        encode_fn, _, _ = unigram_fns
        ids = encode_fn("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encodes_nonempty_text(self, unigram_fns):
        encode_fn, _, _ = unigram_fns
        ids = encode_fn("Natural language processing is fascinating.")
        assert len(ids) > 0

    def test_no_unk_id_tracked_with_byte_fallback(self, unigram_fns):
        _, _, unk_id = unigram_fns
        assert unk_id is None, "Unigram with byte fallback should have unk_id=None"

    def test_unicode_and_emoji_byte_fallback(self, unigram_fns):
        encode_fn, decode_fn, unk_id = unigram_fns
        unicode_samples = [
            "Hello 🌍🚀🧠",
            "Great work! 👏🎉🏆⭐🌟",
            "Weather: ☀️🌤️⛅🌧️⛈️🌩️❄️🌈",
            "∀ε > 0, ∃δ > 0: |x − a| < δ",
            "café naïve résumé façade coördinate Ångström",
            "機械学習は人工知能の一分野であり",
            "https://github.com/kiara-project/kiara/blob/main/transformer.py#L42",
            "$100.00 ± 0.05 (p < 0.001, n=1024)",
        ]
        for s in unicode_samples:
            ids = encode_fn(s)
            assert len(ids) > 0
            decoded = decode_fn(ids)
            assert decoded == s, f"Byte fallback round-trip failed for: {s!r}\nDecoded: {decoded!r}"


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
        assert m.round_trip_failures_due_to_unk == 0
        assert m.round_trip_failures_due_to_decoder == 0

    def test_diagnostic_failure_breakdown(self):
        """Verify that UNK vs decoder round-trip failures are correctly separated."""
        texts = ["known text", "text with unk", "corrupted text"]
        unk_id = 999

        # Mock encoder/decoder
        def mock_encode(t: str) -> list[int]:
            if "unk" in t:
                return [1, unk_id]
            if "corrupted" in t:
                return [1, 2]
            return [1, 3]

        def mock_decode(ids: list[int]) -> str:
            if unk_id in ids:
                return "text with [UNK]"
            if ids == [1, 2]:
                return "bad decode"  # Decoder issue without UNK
            return "known text"

        m = measure_category(texts, "test", mock_encode, mock_decode, unk_id=unk_id)
        assert m.round_trip_failures == 2
        assert m.round_trip_failures_due_to_unk == 1
        assert m.round_trip_failures_due_to_decoder == 1

    def test_compression_ratio_formula(self, byte_bpe_fns):
        encode_fn, decode_fn, unk_id = byte_bpe_fns
        texts = ["a b c d e f g h i j"]
        m = measure_category(texts, "test", encode_fn, decode_fn, unk_id=unk_id)
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

    def test_measure_vocabulary_efficiency(self, byte_bpe_fns):
        encode_fn, _, _ = byte_bpe_fns
        texts = ["The transformer architecture uses self-attention mechanisms.", "Kiara foundation model."]
        eff = measure_vocabulary_efficiency(texts, encode_fn, vocab_size=TINY_VOCAB, special_token_count=4)
        assert eff.vocab_size == TINY_VOCAB
        assert eff.usable_vocab_size == TINY_VOCAB - 4
        assert eff.unique_tokens_observed > 0
        assert 0.0 < eff.utilization_ratio <= 1.0
        assert eff.tokens_freq_ge_1 == eff.unique_tokens_observed
        assert 0.0 <= eff.top_10pct_active_token_occurrence_coverage <= 1.0
        assert 0.0 <= eff.top_25pct_active_token_occurrence_coverage <= 1.0
        assert 0.0 <= eff.top_50pct_active_token_occurrence_coverage <= 1.0
        assert 0.0 <= eff.top_100_token_occurrence_coverage <= 1.0
        assert 0.0 <= eff.top_500_token_occurrence_coverage <= 1.0
        assert 0.0 <= eff.top_1000_token_occurrence_coverage <= 1.0

    def test_measure_vocabulary_efficiency_empty(self):
        def dummy_encode(t: str) -> list[int]:
            return []

        eff = measure_vocabulary_efficiency([], dummy_encode, vocab_size=1000, special_token_count=4)
        assert eff.unique_tokens_observed == 0
        assert eff.utilization_ratio == 0.0
        assert eff.top_10pct_active_token_occurrence_coverage == 0.0
        assert eff.top_100_token_occurrence_coverage == 0.0


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

    def test_byte_bpe_and_unigram_have_no_unk_id(self, byte_bpe_fns, unigram_fns):
        _, _, byte_unk = byte_bpe_fns
        _, _, uni_unk = unigram_fns
        assert byte_unk is None
        assert uni_unk is None

    def test_bpe_has_unk_id(self, bpe_fns):
        _, _, bpe_unk = bpe_fns
        assert bpe_unk is not None

    def test_all_algos_handle_code(self, byte_bpe_fns, bpe_fns, unigram_fns):
        code = "def transformer_block(x, mask=None):\n    return x\n"
        for label, fns in [("byte_bpe", byte_bpe_fns), ("bpe", bpe_fns), ("unigram", unigram_fns)]:
            encode_fn, _, _ = fns
            ids = encode_fn(code)
            assert len(ids) > 0, f"{label} failed on code input"
