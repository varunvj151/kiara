"""
experiments/corpus_v3_report.py — Per-category statistics for Corpus v3.0
==========================================================================
Run: python experiments/corpus_v3_report.py
"""

from __future__ import annotations

import io
import random
import statistics
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")

from experiments.tokenizer_benchmark.corpus_v3 import (
    _generate_code,
    _generate_multilingual,
    _generate_natural_language,
    _generate_structured,
    _generate_technical,
    _get_eval_b_texts,
    _CATEGORY_FRACTIONS,
    build_v3_training_corpus,
    get_v3_eval_a_texts,
    verify_disjoint,
    DEFAULT_SEED,
)
from kiara.tokenizer import KiaraTokenizer

# ---------------------------------------------------------------------------

tok = KiaraTokenizer.load()

TARGET = 25000
targets = {k: int(TARGET * v) for k, v in _CATEGORY_FRACTIONS.items()}
targets["natural"] += TARGET - sum(targets.values())

GEN = {
    "natural": _generate_natural_language,
    "technical": _generate_technical,
    "code": _generate_code,
    "multilingual": _generate_multilingual,
    "structured": _generate_structured,
}

# ---------------------------------------------------------------------------
# Per-category statistics
# ---------------------------------------------------------------------------

print(f"\nCorpus v3.0 — Per-Category Statistics at {TARGET:,} total samples")
print(f"Tokenizer : KiaraTokenizer vocab={tok.vocab_size}")
print()

rows = []
for cat, n in targets.items():
    print(f"  Analysing [{cat}] ({n:,} samples)...", flush=True)
    samples = GEN[cat](random.Random(DEFAULT_SEED), n)
    unique_texts = set(samples)
    dup_count = n - len(unique_texts)

    char_lens = [len(s) for s in samples]
    tok_seqs = [tok.encode(s) for s in samples]
    tok_lens = [len(ids) for ids in tok_seqs]
    all_toks = [t for ids in tok_seqs for t in ids]
    unique_tok_types = len(set(all_toks))
    unk_count = all_toks.count(tok.unk_token_id)
    tok_lens_sorted = sorted(tok_lens)

    rows.append(
        {
            "cat": cat,
            "n": n,
            "unique_n": len(unique_texts),
            "dup_count": dup_count,
            "dup_rate": dup_count / n * 100,
            "total_chars": sum(char_lens),
            "total_toks": len(all_toks),
            "char_mean": statistics.mean(char_lens),
            "char_std": statistics.stdev(char_lens) if n > 1 else 0.0,
            "tok_mean": statistics.mean(tok_lens),
            "tok_std": statistics.stdev(tok_lens) if n > 1 else 0.0,
            "tok_p10": tok_lens_sorted[int(n * 0.10)],
            "tok_p25": tok_lens_sorted[int(n * 0.25)],
            "tok_p50": tok_lens_sorted[int(n * 0.50)],
            "tok_p75": tok_lens_sorted[int(n * 0.75)],
            "tok_p90": tok_lens_sorted[int(n * 0.90)],
            "unique_tok_types": unique_tok_types,
            "tok_diversity": unique_tok_types / max(1, len(all_toks)),
            "unk_count": unk_count,
        }
    )

# ---------------------------------------------------------------------------
# Main table
# ---------------------------------------------------------------------------

print()
W = 130
print("=" * W)
print(
    f"  {'Cat':12s}  {'N':>6}  {'Unique':>6}  {'Dup%':>5}  "
    f"{'Tokens':>8}  {'Tok/Smp':>7}  {'mean':>6}  {'std':>5}  "
    f"{'P10':>4}  {'P25':>4}  {'P50':>4}  {'P75':>4}  {'P90':>4}  "
    f"{'TokTypes':>8}  {'Div%':>5}  {'UNK':>4}"
)
print("-" * W)
for r in rows:
    print(
        f"  {r['cat']:12s}  {r['n']:>6,}  {r['unique_n']:>6,}  {r['dup_rate']:>5.2f}  "
        f"{r['total_toks']:>8,}  {r['tok_mean']:>7.1f}  {r['tok_mean']:>6.1f}  {r['tok_std']:>5.1f}  "
        f"{r['tok_p10']:>4}  {r['tok_p25']:>4}  {r['tok_p50']:>4}  {r['tok_p75']:>4}  {r['tok_p90']:>4}  "
        f"{r['unique_tok_types']:>8,}  {r['tok_diversity']*100:>5.2f}  {r['unk_count']:>4}"
    )

total_toks = sum(r["total_toks"] for r in rows)
total_n = sum(r["n"] for r in rows)
total_uniq = sum(r["unique_n"] for r in rows)
total_dup = sum(r["dup_count"] for r in rows)
print("-" * W)
print(
    f"  {'TOTAL':12s}  {total_n:>6,}  {total_uniq:>6,}  {total_dup/total_n*100:>5.2f}  "
    f"{total_toks:>8,}"
)
print("=" * W)

# ---------------------------------------------------------------------------
# Code deep-dive: unique structural patterns (first 80 chars of each unique)
# ---------------------------------------------------------------------------

print()
print("CODE CATEGORY — Unique structural prefixes (first 80 chars, first 15 unique):")
print("-" * 80)
code_samples = _generate_code(random.Random(DEFAULT_SEED), targets["code"])
seen_prefix = []
for s in code_samples:
    prefix = s.strip()[:80].replace("\n", " / ")
    if prefix not in seen_prefix:
        seen_prefix.append(prefix)
    if len(seen_prefix) >= 15:
        break
for i, p in enumerate(seen_prefix, 1):
    print(f"  {i:2d}. {p}")

# ---------------------------------------------------------------------------
# Overlap verification at full 25K scale
# ---------------------------------------------------------------------------

print()
print("OVERLAP VERIFICATION at 25K:")
print("-" * 40)
full_train = build_v3_training_corpus(seed=DEFAULT_SEED, target_samples=25000)
eval_a = get_v3_eval_a_texts()
eval_b = _get_eval_b_texts()
report = verify_disjoint(full_train, eval_a, eval_b)
print(f"  is_clean               : {report.is_clean}")
print(f"  train/eval_a (exact)   : {report.train_eval_a_exact}")
print(f"  train/eval_b (exact)   : {report.train_eval_b_exact}")
print(f"  eval_a/eval_b (exact)  : {report.eval_a_eval_b_exact}")
print(f"  train/eval_a (norm)    : {report.train_eval_a_normalized}")
print(f"  train/eval_b (norm)    : {report.train_eval_b_normalized}")
print(f"  train/eval_a (sha256)  : {report.train_eval_a_sha256}")
print(f"  train/eval_b (sha256)  : {report.train_eval_b_sha256}")

# ---------------------------------------------------------------------------
# Sample previews (first unique from each category)
# ---------------------------------------------------------------------------

print()
print("SAMPLE PREVIEW (first sample, 150 chars):")
print("-" * 80)
for cat, n in targets.items():
    samples = GEN[cat](random.Random(DEFAULT_SEED), n)
    preview = samples[0][:150].replace("\n", " / ")
    print(f"  [{cat:12s}] {preview}")
    print()

# ---------------------------------------------------------------------------
# Eval A and B statistics
# ---------------------------------------------------------------------------

print()
print("EVAL A (Generalization) — per-category sample count:")
from experiments.tokenizer_benchmark.corpus_v3 import get_v3_eval_a_corpus
eval_a_cat = get_v3_eval_a_corpus()
for cat, samps in eval_a_cat.items():
    toks = sum(len(tok.encode(s)) for s in samps)
    print(f"  {cat:12s}: {len(samps):>3} samples  {toks:>5} tokens")
print(f"  {'TOTAL':12s}: {len(eval_a):>3} samples  {sum(len(tok.encode(s)) for s in eval_a):>5} tokens")

print()
print("EVAL B (Robustness) — summary:")
from experiments.tokenizer_benchmark.corpus import get_eval_corpus
eval_b_cat = get_eval_corpus()
for cat, samps in eval_b_cat.items():
    toks = sum(len(tok.encode(s)) for s in samps)
    print(f"  {cat:12s}: {len(samps):>3} samples  {toks:>5} tokens")
print(f"  {'TOTAL':12s}: {len(eval_b):>3} samples  {sum(len(tok.encode(s)) for s in eval_b):>5} tokens")
