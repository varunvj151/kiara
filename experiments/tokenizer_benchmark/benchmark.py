"""
benchmark.py — Kiara Tokenizer Benchmark Runner
================================================
Orchestrates the full benchmark:
  1. Builds the corpus
  2. Trains all (algo × vocab_size) tokenizer combinations
  3. Runs all metrics
  4. Saves results to experiments/tokenizer_benchmark/results/

Usage:
    cd experiments/tokenizer_benchmark
    python benchmark.py

Or with options:
    python benchmark.py --algos byte_bpe bpe --vocab-sizes 16384 32768 --seed 1337

Results are saved to:
    results/results.json
    results/results.csv
    results/summary.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from the benchmark directory
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCHMARK_DIR))

from corpus import get_corpus, get_eval_corpus, get_training_texts, corpus_stats, CORPUS_VERSION
from metrics import (
    BenchmarkResult,
    measure_category,
    measure_vocabulary_efficiency,
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

RESULTS_DIR = BENCHMARK_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

DEFAULT_ALGOS = ["byte_bpe", "bpe", "unigram"]
DEFAULT_VOCAB_SIZES = [16384, 32768, 50000, 65536, 100352]
DEFAULT_SEED = 1337

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_single(
    algo: str,
    vocab_size: int,
    training_texts: list[str],
    eval_corpus: dict[str, list[str]],
    seed: int,
    force_retrain: bool,
) -> BenchmarkResult:
    """Run benchmark for one (algo, vocab_size) combination.

    Trains exclusively on `training_texts` and evaluates exclusively
    on the held-out `eval_corpus`.

    Returns a BenchmarkResult (success or failure recorded).
    """
    result = BenchmarkResult()
    result.tokenizer_type = algo
    result.vocab_size_requested = vocab_size
    result.random_seed = seed
    result.corpus_version = CORPUS_VERSION
    result.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    lib_versions = get_library_versions()
    result.python_version = lib_versions.get("python", "unknown")
    result.library = "tokenizers"
    result.library_version = lib_versions.get("tokenizers", "unknown")

    # --- Train tokenizer on training corpus only ---
    try:
        meta = train_tokenizer(
            algo=algo,
            texts=training_texts,
            vocab_size=vocab_size,
            seed=seed,
            force_retrain=force_retrain,
        )
    except Exception as exc:
        result.success = False
        result.error = f"Training failed: {exc}"
        print(f"  ERROR training {algo} vocab={vocab_size}: {exc}")
        return result

    result.vocab_size_actual = meta.get("vocab_size_actual", 0)

    # --- Load encode/decode functions ---
    try:
        encode_fn, decode_fn, unk_id = load_tokenizer_fns(meta)
    except Exception as exc:
        result.success = False
        result.error = f"Loading tokenizer failed: {exc}"
        print(f"  ERROR loading {algo} vocab={vocab_size}: {exc}")
        return result

    # --- Run metrics per held-out category ---
    for cat_name, texts in eval_corpus.items():
        try:
            cat_metrics = measure_category(
                texts=texts,
                category=cat_name,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                unk_id=unk_id,
            )
            result.by_category[cat_name] = cat_metrics
        except Exception as exc:
            # Record partial failure for this category rather than crashing
            cat_metrics = CategoryMetrics(category=cat_name)
            cat_metrics.round_trip_pass = False
            cat_metrics.round_trip_failures = len(texts)
            result.by_category[cat_name] = cat_metrics
            print(f"  WARNING: category '{cat_name}' failed for {algo} vocab={vocab_size}: {exc}")

    # --- Aggregate ---
    aggregate_results(result)

    # --- Vocabulary efficiency ---
    #
    # Use a deterministic representative sample from the training corpus.
    # The tokenizer was trained on the full training corpus, but this metric
    # intentionally measures vocabulary usage on a smaller sample.
    sample_size = min(10_000, len(training_texts))
    vocabulary_sample = training_texts[:sample_size]

    try:
        vocab_efficiency = measure_vocabulary_efficiency(
            texts=vocabulary_sample,
            encode_fn=encode_fn,
            vocab_size=result.vocab_size_actual,
            special_token_count=4,
        )
        result.vocabulary_efficiency = vocab_efficiency
    except Exception as exc:
        print(
            f"  WARNING: vocabulary efficiency failed for "
            f"{algo} vocab={vocab_size}: {exc}"
        )

    return result


def run_benchmark(
    algos: list[str],
    vocab_sizes: list[int],
    seed: int = DEFAULT_SEED,
    force_retrain: bool = False,
) -> list[BenchmarkResult]:
    """Run the full benchmark matrix.

    Args:
        algos: List of algorithm names.
        vocab_sizes: List of vocabulary sizes.
        seed: Random seed.
        force_retrain: If True, retrain all tokenizers from scratch.

    Returns:
        List of BenchmarkResult objects, one per (algo, vocab_size) pair.
    """
    training_texts = get_training_texts(seed=seed)
    eval_corpus = get_eval_corpus()
    results: list[BenchmarkResult] = []

    total = len(algos) * len(vocab_sizes)
    n = 0

    for algo in algos:
        for vocab_size in vocab_sizes:
            n += 1
            print(f"\n[{n}/{total}] {algo.upper()} vocab={vocab_size:,}")
            result = run_single(
                algo=algo,
                vocab_size=vocab_size,
                training_texts=training_texts,
                eval_corpus=eval_corpus,
                seed=seed,
                force_retrain=force_retrain,
            )
            results.append(result)


            if result.success:
                status = "PASS" if result.overall_round_trip_pass else "FAIL (round-trip)"
                print(
                    f"  actual_vocab={result.vocab_size_actual:,}  "
                    f"compression={result.overall_compression_ratio:.2f} chars/tok  "
                    f"tok/char={result.overall_tokens_per_char:.3f}  "
                    f"UNK={result.overall_unk_count}  [{status}]"
                )
            else:
                print(f"  FAILED: {result.error}")

    return results


# ---------------------------------------------------------------------------
# Results serialisation
# ---------------------------------------------------------------------------


def result_to_dict(r: BenchmarkResult) -> dict:
    """Convert a BenchmarkResult to a JSON-serialisable dict."""

    def cat_to_dict(m: CategoryMetrics) -> dict:
        return {
            "category": m.category,
            "num_samples": m.num_samples,
            "total_chars": m.total_chars,
            "total_words": m.total_words,
            "total_tokens": m.total_tokens,
            "tokens_per_char": round(m.tokens_per_char, 5),
            "tokens_per_word": round(m.tokens_per_word, 3),
            "compression_ratio": round(m.compression_ratio, 3),
            "encode_time_ms": round(m.encode_time_ms, 3),
            "decode_time_ms": round(m.decode_time_ms, 3),
            "encode_throughput_tps": round(m.encode_throughput_tokens_per_sec),
            "decode_throughput_tps": round(m.decode_throughput_tokens_per_sec),
            "round_trip_pass": m.round_trip_pass,
            "round_trip_failures": m.round_trip_failures,
            "round_trip_failures_due_to_unk": m.round_trip_failures_due_to_unk,
            "round_trip_failures_due_to_decoder": m.round_trip_failures_due_to_decoder,
            "unk_token_count": m.unk_token_count,
            "unk_sample_count": m.unk_sample_count,
        }

    return {
        "tokenizer_type": r.tokenizer_type,
        "vocab_size_requested": r.vocab_size_requested,
        "vocab_size_actual": r.vocab_size_actual,
        "library": r.library,
        "library_version": r.library_version,
        "python_version": r.python_version,
        "random_seed": r.random_seed,
        "corpus_version": r.corpus_version,
        "timestamp": r.timestamp,
        "success": r.success,
        "error": r.error,
        "aggregate": {
            "total_samples": r.total_samples,
            "total_chars": r.total_chars,
            "total_words": r.total_words,
            "total_tokens": r.total_tokens,
            "tokens_per_char": round(r.overall_tokens_per_char, 5),
            "tokens_per_word": round(r.overall_tokens_per_word, 3),
            "compression_ratio": round(r.overall_compression_ratio, 3),
            "encode_time_ms": round(r.overall_encode_time_ms, 3),
            "decode_time_ms": round(r.overall_decode_time_ms, 3),
            "encode_throughput_tps": round(r.overall_encode_throughput),
            "decode_throughput_tps": round(r.overall_decode_throughput),
            "round_trip_pass": r.overall_round_trip_pass,
            "round_trip_failures": r.overall_round_trip_failures,
            "round_trip_failures_due_to_unk": r.overall_round_trip_failures_due_to_unk,
            "round_trip_failures_due_to_decoder": r.overall_round_trip_failures_due_to_decoder,
            "unk_count": r.overall_unk_count,
            "unk_sample_count": r.overall_unk_sample_count,
        },
        "by_category": {k: cat_to_dict(v) for k, v in r.by_category.items()},
        "vocabulary_efficiency": None
        if r.vocabulary_efficiency is None
        else {
            "vocab_size": r.vocabulary_efficiency.vocab_size,
            "special_token_count": r.vocabulary_efficiency.special_token_count,
            "usable_vocab_size": r.vocabulary_efficiency.usable_vocab_size,
            "unique_tokens_observed": r.vocabulary_efficiency.unique_tokens_observed,
            "utilization_ratio": round(
                r.vocabulary_efficiency.utilization_ratio, 5
            ),
            "tokens_freq_ge_1": r.vocabulary_efficiency.tokens_freq_ge_1,
            "tokens_freq_ge_10": r.vocabulary_efficiency.tokens_freq_ge_10,
            "tokens_freq_ge_100": r.vocabulary_efficiency.tokens_freq_ge_100,
            "well_represented_ratio_ge_10": round(
                r.vocabulary_efficiency.well_represented_ratio_ge_10,
                5,
            ),
            "well_represented_ratio_ge_100": round(
                r.vocabulary_efficiency.well_represented_ratio_ge_100,
                5,
            ),
            "top_10pct_active_token_occurrence_coverage": round(
                r.vocabulary_efficiency.top_10pct_active_token_occurrence_coverage,
                5,
            ),
            "top_25pct_active_token_occurrence_coverage": round(
                r.vocabulary_efficiency.top_25pct_active_token_occurrence_coverage,
                5,
            ),
            "top_50pct_active_token_occurrence_coverage": round(
                r.vocabulary_efficiency.top_50pct_active_token_occurrence_coverage,
                5,
            ),
            "top_100_token_occurrence_coverage": round(
                r.vocabulary_efficiency.top_100_token_occurrence_coverage,
                5,
            ),
            "top_500_token_occurrence_coverage": round(
                r.vocabulary_efficiency.top_500_token_occurrence_coverage,
                5,
            ),
            "top_1000_token_occurrence_coverage": round(
                r.vocabulary_efficiency.top_1000_token_occurrence_coverage,
                5,
            ),
        },
    }


def save_json(results: list[BenchmarkResult]) -> Path:
    data = [result_to_dict(r) for r in results]
    path = RESULTS_DIR / "results.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_csv(results: list[BenchmarkResult]) -> Path:
    import csv

    path = RESULTS_DIR / "results.csv"
    fieldnames = [
        "tokenizer_type",
        "vocab_size_requested",
        "vocab_size_actual",
        "success",
        "total_samples",
        "total_chars",
        "total_tokens",
        "tokens_per_char",
        "tokens_per_word",
        "compression_ratio",
        "encode_time_ms",
        "decode_time_ms",
        "round_trip_pass",
        "round_trip_failures",
        "round_trip_failures_due_to_unk",
        "round_trip_failures_due_to_decoder",
        "unk_count",
        "vocab_utilization_ratio",
        "well_rep_ge_10_ratio",
        "well_rep_ge_100_ratio",
        "top_10pct_active_cov",
        "top_25pct_active_cov",
        "top_50pct_active_cov",
        "top_100_cov",
        "top_500_cov",
        "top_1000_cov",
        "library_version",
        "python_version",
        "timestamp",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            ve = r.vocabulary_efficiency
            writer.writerow(
                {
                    "tokenizer_type": r.tokenizer_type,
                    "vocab_size_requested": r.vocab_size_requested,
                    "vocab_size_actual": r.vocab_size_actual,
                    "success": r.success,
                    "total_samples": r.total_samples,
                    "total_chars": r.total_chars,
                    "total_tokens": r.total_tokens,
                    "tokens_per_char": round(r.overall_tokens_per_char, 5),
                    "tokens_per_word": round(r.overall_tokens_per_word, 3),
                    "compression_ratio": round(r.overall_compression_ratio, 3),
                    "encode_time_ms": round(r.overall_encode_time_ms, 3),
                    "decode_time_ms": round(r.overall_decode_time_ms, 3),
                    "round_trip_pass": r.overall_round_trip_pass,
                    "round_trip_failures": r.overall_round_trip_failures,
                    "round_trip_failures_due_to_unk": r.overall_round_trip_failures_due_to_unk,
                    "round_trip_failures_due_to_decoder": r.overall_round_trip_failures_due_to_decoder,
                    "unk_count": r.overall_unk_count,
                    "vocab_utilization_ratio": round(ve.utilization_ratio, 5) if ve else None,
                    "well_rep_ge_10_ratio": round(ve.well_represented_ratio_ge_10, 5) if ve else None,
                    "well_rep_ge_100_ratio": round(ve.well_represented_ratio_ge_100, 5) if ve else None,
                    "top_10pct_active_cov": round(ve.top_10pct_active_token_occurrence_coverage, 5) if ve else None,
                    "top_25pct_active_cov": round(ve.top_25pct_active_token_occurrence_coverage, 5) if ve else None,
                    "top_50pct_active_cov": round(ve.top_50pct_active_token_occurrence_coverage, 5) if ve else None,
                    "top_100_cov": round(ve.top_100_token_occurrence_coverage, 5) if ve else None,
                    "top_500_cov": round(ve.top_500_token_occurrence_coverage, 5) if ve else None,
                    "top_1000_cov": round(ve.top_1000_token_occurrence_coverage, 5) if ve else None,
                    "library_version": r.library_version,
                    "python_version": r.python_version,
                    "timestamp": r.timestamp,
                    "error": r.error,
                }
            )
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kiara Tokenizer Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--algos",
        nargs="+",
        default=DEFAULT_ALGOS,
        choices=SUPPORTED_ALGOS,
        help="Tokenizer algorithms to benchmark",
    )
    p.add_argument(
        "--vocab-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_VOCAB_SIZES,
        help="Vocabulary sizes to benchmark",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    p.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain tokenizers even if cached versions exist",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test: 3 algos × 1 small vocab size only",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke:
        algos = ["byte_bpe", "bpe", "unigram"]
        vocab_sizes = [1000]
        print("=== SMOKE TEST MODE (vocab=1000, 3 algos) ===")
    else:
        algos = args.algos
        vocab_sizes = args.vocab_sizes

    stats = corpus_stats()
    print("=" * 60)
    print("Kiara Tokenizer Benchmark")
    print(f"  Algorithms : {algos}")
    print(f"  Vocab sizes: {vocab_sizes}")
    print(f"  Seed       : {args.seed}")
    print(f"  Versions   : {get_library_versions()}")
    print(f"  Corpus ver : {CORPUS_VERSION}")
    print(f"  Train data : {stats['training']['num_samples']:,} samples, {stats['training']['approx_mb']} MB, {stats['training']['unique_words']:,} unique words")
    print(f"  Eval data  : {stats['evaluation']['num_samples']} samples across {len(stats['evaluation']['categories'])} categories (Disjoint: {stats['is_disjoint']})")
    print("=" * 60)

    results = run_benchmark(
        algos=algos,
        vocab_sizes=vocab_sizes,
        seed=args.seed,
        force_retrain=args.force_retrain,
    )

    # Save results
    json_path = save_json(results)
    csv_path = save_csv(results)

    print(f"\nResults saved to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")

    # Print quick summary table
    print("\n" + "=" * 90)
    print(f"{'Algorithm':<12} {'VocabReq':>9} {'VocabAct':>9} {'Compress':>9} {'tok/char':>9} "
          f"{'UNK':>6} {'RoundTrip':>10} {'Success':>8}")
    print("-" * 90)
    for r in results:
        rt = "PASS" if r.overall_round_trip_pass else "FAIL"
        succ = "OK" if r.success else "ERROR"
        print(
            f"{r.tokenizer_type:<12} {r.vocab_size_requested:>9,} {r.vocab_size_actual:>9,} "
            f"{r.overall_compression_ratio:>9.3f} {r.overall_tokens_per_char:>9.4f} "
            f"{r.overall_unk_count:>6} {rt:>10} {succ:>8}"
        )
    print("=" * 90)

    print("\nTokenizer benchmark implementation complete.")
    print("Run the benchmark before making the tokenizer decision.")


if __name__ == "__main__":
    main()
