"""
metrics.py — Kiara Tokenizer Benchmark Metrics
===============================================
Computes all benchmark metrics for a single tokenizer on a list of texts.
Designed to be tokenizer-agnostic: accepts any callable that
encodes/decodes text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CategoryMetrics:
    """Metrics for one tokenizer on one corpus category."""

    category: str
    num_samples: int = 0
    total_chars: int = 0
    total_words: int = 0
    total_tokens: int = 0
    tokens_per_char: float = 0.0
    tokens_per_word: float = 0.0
    compression_ratio: float = 0.0  # chars per token
    encode_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    encode_throughput_tokens_per_sec: float = 0.0
    decode_throughput_tokens_per_sec: float = 0.0
    round_trip_pass: bool = True
    round_trip_failures: int = 0
    unk_token_count: int = 0
    unk_sample_count: int = 0  # samples that contained at least one UNK


@dataclass
class BenchmarkResult:
    """Full benchmark result for one tokenizer × vocabulary size combination."""

    tokenizer_type: str = ""
    vocab_size_requested: int = 0
    vocab_size_actual: int = 0
    library: str = ""
    python_version: str = ""
    library_version: str = ""
    random_seed: int = 0
    corpus_version: str = ""
    timestamp: str = ""

    # Per-category metrics
    by_category: dict[str, CategoryMetrics] = field(default_factory=dict)

    # Aggregate metrics across all categories
    total_samples: int = 0
    total_chars: int = 0
    total_words: int = 0
    total_tokens: int = 0
    overall_tokens_per_char: float = 0.0
    overall_tokens_per_word: float = 0.0
    overall_compression_ratio: float = 0.0
    overall_encode_time_ms: float = 0.0
    overall_decode_time_ms: float = 0.0
    overall_encode_throughput: float = 0.0
    overall_decode_throughput: float = 0.0
    overall_round_trip_pass: bool = True
    overall_round_trip_failures: int = 0
    overall_unk_count: int = 0
    overall_unk_sample_count: int = 0

    # Error tracking
    error: str = ""
    success: bool = True


# ---------------------------------------------------------------------------
# Core measurement function
# ---------------------------------------------------------------------------

UNK_TOKEN_ID = 0  # Conventional UNK id; overridden per tokenizer where possible


def _count_words(text: str) -> int:
    """Simple whitespace word count."""
    return len(text.split())


def measure_category(
    texts: list[str],
    category: str,
    encode_fn: Callable[[str], list[int]],
    decode_fn: Callable[[list[int]], str],
    unk_id: int | None = None,
    num_speed_repeats: int = 3,
) -> CategoryMetrics:
    """Measure all metrics for one category of texts.

    Args:
        texts: List of text strings to benchmark.
        category: Name of the category (for labeling).
        encode_fn: Function that takes a string and returns a list of token IDs.
        decode_fn: Function that takes a list of token IDs and returns a string.
        unk_id: Token ID treated as the unknown token (None = skip UNK counting).
        num_speed_repeats: Number of times to repeat speed measurement (median taken).

    Returns:
        CategoryMetrics with all computed metrics.
    """
    m = CategoryMetrics(category=category)

    if not texts:
        return m

    all_token_ids: list[list[int]] = []
    total_encode_ms = 0.0
    total_decode_ms = 0.0
    round_trip_failures = 0
    unk_token_count = 0
    unk_sample_count = 0

    # --- Encode all texts and measure timing ---
    encode_times_ms: list[float] = []
    for _ in range(num_speed_repeats):
        t0 = time.perf_counter()
        encoded_all = [encode_fn(text) for text in texts]
        encode_times_ms.append((time.perf_counter() - t0) * 1000.0)

    all_token_ids = encoded_all  # Use last run for correctness checks
    total_encode_ms = sorted(encode_times_ms)[num_speed_repeats // 2]  # median

    # --- Decode all token sequences and measure timing ---
    decode_times_ms: list[float] = []
    for _ in range(num_speed_repeats):
        t0 = time.perf_counter()
        decoded_all = [decode_fn(ids) for ids in all_token_ids]
        decode_times_ms.append((time.perf_counter() - t0) * 1000.0)

    total_decode_ms = sorted(decode_times_ms)[num_speed_repeats // 2]

    # --- Correctness and UNK analysis ---
    for i, (text, token_ids, decoded) in enumerate(zip(texts, all_token_ids, decoded_all)):
        m.num_samples += 1
        m.total_chars += len(text)
        m.total_words += _count_words(text)
        m.total_tokens += len(token_ids)

        # Round-trip check
        if decoded != text:
            round_trip_failures += 1

        # UNK check
        if unk_id is not None:
            n_unk = token_ids.count(unk_id)
            unk_token_count += n_unk
            if n_unk > 0:
                unk_sample_count += 1

    # --- Aggregate metrics ---
    m.round_trip_failures = round_trip_failures
    m.round_trip_pass = round_trip_failures == 0
    m.unk_token_count = unk_token_count
    m.unk_sample_count = unk_sample_count

    m.encode_time_ms = total_encode_ms
    m.decode_time_ms = total_decode_ms

    if m.total_chars > 0:
        m.tokens_per_char = m.total_tokens / m.total_chars
        m.compression_ratio = m.total_chars / m.total_tokens if m.total_tokens > 0 else 0.0

    if m.total_words > 0:
        m.tokens_per_word = m.total_tokens / m.total_words

    total_tokens_in_category = sum(len(ids) for ids in all_token_ids)
    if total_encode_ms > 0:
        m.encode_throughput_tokens_per_sec = total_tokens_in_category / (total_encode_ms / 1000.0)
    if total_decode_ms > 0:
        m.decode_throughput_tokens_per_sec = total_tokens_in_category / (total_decode_ms / 1000.0)

    return m


def aggregate_results(result: BenchmarkResult) -> None:
    """Compute aggregate metrics from per-category metrics in-place."""
    total_chars = total_words = total_tokens = 0
    total_encode_ms = total_decode_ms = 0.0
    round_trip_failures = 0
    unk_count = unk_sample_count = 0
    num_samples = 0

    for cat_metrics in result.by_category.values():
        total_chars += cat_metrics.total_chars
        total_words += cat_metrics.total_words
        total_tokens += cat_metrics.total_tokens
        total_encode_ms += cat_metrics.encode_time_ms
        total_decode_ms += cat_metrics.decode_time_ms
        round_trip_failures += cat_metrics.round_trip_failures
        unk_count += cat_metrics.unk_token_count
        unk_sample_count += cat_metrics.unk_sample_count
        num_samples += cat_metrics.num_samples

    result.total_samples = num_samples
    result.total_chars = total_chars
    result.total_words = total_words
    result.total_tokens = total_tokens
    result.overall_encode_time_ms = total_encode_ms
    result.overall_decode_time_ms = total_decode_ms
    result.overall_round_trip_failures = round_trip_failures
    result.overall_round_trip_pass = round_trip_failures == 0
    result.overall_unk_count = unk_count
    result.overall_unk_sample_count = unk_sample_count

    if total_chars > 0:
        result.overall_tokens_per_char = total_tokens / total_chars
        result.overall_compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0.0

    if total_words > 0:
        result.overall_tokens_per_word = total_tokens / total_words

    if total_encode_ms > 0:
        result.overall_encode_throughput = total_tokens / (total_encode_ms / 1000.0)

    if total_decode_ms > 0:
        result.overall_decode_throughput = total_tokens / (total_decode_ms / 1000.0)
