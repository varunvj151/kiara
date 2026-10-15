"""
experiments/tokenizer_transfer_report.py — Phase 26A.2: Tokenizer Transfer Benchmark
======================================================================================
Audits how the frozen 32,768 Byte-Level BPE tokenizer (trained on synthetic data)
transfers to the real-world text corpus (Kiara-Real-v1.0) BEFORE training.

Measures:
  1. Token efficiency (chars/token, tokens/word, compression ratio)
  2. Vocabulary utilization & coverage out of 32,768
  3. Byte fallback frequency (raw single-byte tokens)
  4. Sequence length distribution (P10..P99)
  5. Cross-comparison: Synthetic Corpus v3 vs Real Corpus v1
"""

from __future__ import annotations

import io
import json
import statistics
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kiara.tokenizer import KiaraTokenizer

DATA_DIR = REPO_ROOT / "data" / "real"


def run_tokenizer_transfer_audit():
    print(f"\n{'='*90}")
    print("  PHASE 26A.2: TOKENIZER TRANSFER BENCHMARK & REAL CORPUS AUDIT")
    print(f"{'='*90}\n")

    tokenizer = KiaraTokenizer.load()
    print(f"  Tokenizer Algorithm : Byte-Level BPE (Frozen Decision 003)")
    print(f"  Tokenizer Vocab Size: {tokenizer.vocab_size:,}\n")

    # Load real corpus
    train_file = DATA_DIR / "train.jsonl"
    eval_file = DATA_DIR / "eval_real.jsonl"

    if not train_file.exists() or not eval_file.exists():
        raise FileNotFoundError("Missing real corpus files in data/real/. Run acquire.py first.")

    train_docs = [json.loads(line)["text"] for line in train_file.read_text(encoding="utf-8").strip().split("\n")]
    eval_docs = [json.loads(line)["text"] for line in eval_file.read_text(encoding="utf-8").strip().split("\n")]

    print(f"  Loaded Train Docs   : {len(train_docs):,}")
    print(f"  Loaded Eval Docs    : {len(eval_docs):,}\n")

    # Audit Function
    def audit_split(docs: list[str], split_name: str) -> dict:
        print(f"  Auditing [{split_name}] ({len(docs):,} documents)...", flush=True)

        char_lengths = [len(d) for d in docs]
        word_counts = [len(d.split()) for d in docs]

        all_token_seqs = [tokenizer.encode(d) for d in docs]
        token_lengths = [len(seq) for seq in all_token_seqs]

        all_tokens = [tok_id for seq in all_token_seqs for tok_id in seq]
        total_tokens = len(all_tokens)
        unique_tokens = set(all_tokens)

        # Byte fallback analysis: byte tokens occupy IDs 0..255 (or single character byte tokens)
        # In byte-BPE, single-byte fallback tokens are decoded to raw bytes
        byte_fallback_count = sum(1 for tok_id in all_tokens if tok_id < 256)
        unk_count = all_tokens.count(tokenizer.unk_token_id)

        sorted_char = sorted(char_lengths)
        sorted_tok = sorted(token_lengths)
        n = len(docs)

        stats = {
            "num_docs": n,
            "total_chars": sum(char_lengths),
            "total_words": sum(word_counts),
            "total_tokens": total_tokens,
            "chars_per_token": sum(char_lengths) / max(1, total_tokens),
            "tokens_per_word": total_tokens / max(1, sum(word_counts)),
            "active_vocab": len(unique_tokens),
            "vocab_coverage_pct": (len(unique_tokens) / tokenizer.vocab_size) * 100,
            "byte_fallback_count": byte_fallback_count,
            "byte_fallback_pct": (byte_fallback_count / max(1, total_tokens)) * 100,
            "unk_count": unk_count,
            "char_mean": statistics.mean(char_lengths),
            "char_std": statistics.stdev(char_lengths) if n > 1 else 0.0,
            "tok_mean": statistics.mean(token_lengths),
            "tok_std": statistics.stdev(token_lengths) if n > 1 else 0.0,
            "tok_p10": sorted_tok[int(n * 0.10)],
            "tok_p25": sorted_tok[int(n * 0.25)],
            "tok_p50": sorted_tok[int(n * 0.50)],
            "tok_p75": sorted_tok[int(n * 0.75)],
            "tok_p90": sorted_tok[int(n * 0.90)],
            "tok_p99": sorted_tok[int(n * 0.99)],
        }
        return stats

    train_stats = audit_split(train_docs, "train")
    eval_stats = audit_split(eval_docs, "eval_held_out")

    # Display Report Table
    print("\n" + "=" * 90)
    print("  CORPUS AUDIT & TOKENIZER TRANSFER SUMMARY")
    print("=" * 90)
    print(f"  {'Metric':<32}  {'Real Train':>18}  {'Real Eval (Held-out)':>22}")
    print("-" * 90)
    print(f"  {'Total Documents':<32}  {train_stats['num_docs']:>18,}  {eval_stats['num_docs']:>22,}")
    print(f"  {'Total Characters':<32}  {train_stats['total_chars']:>18,}  {eval_stats['total_chars']:>22,}")
    print(f"  {'Total Words':<32}  {train_stats['total_words']:>18,}  {eval_stats['total_words']:>22,}")
    print(f"  {'Total Tokens':<32}  {train_stats['total_tokens']:>18,}  {eval_stats['total_tokens']:>22,}")
    print(f"  {'Chars / Token (Compression)':<32}  {train_stats['chars_per_token']:>18.3f}  {eval_stats['chars_per_token']:>22.3f}")
    print(f"  {'Tokens / Word':<32}  {train_stats['tokens_per_word']:>18.3f}  {eval_stats['tokens_per_word']:>22.3f}")
    print(f"  {'Active Vocabulary Size':<32}  {train_stats['active_vocab']:>18,}  {eval_stats['active_vocab']:>22,}")
    print(f"  {'Vocab Utilization (%)':<32}  {train_stats['vocab_coverage_pct']:>17.2f}%  {eval_stats['vocab_coverage_pct']:>21.2f}%")
    print(f"  {'Byte Fallback Rate (%)':<32}  {train_stats['byte_fallback_pct']:>17.3f}%  {eval_stats['byte_fallback_pct']:>21.3f}%")
    print(f"  {'Unknown Tokens ([UNK])':<32}  {train_stats['unk_count']:>18}  {eval_stats['unk_count']:>22}")
    print("-" * 90)
    print(f"  {'Mean Tokens / Document':<32}  {train_stats['tok_mean']:>18.1f}  {eval_stats['tok_mean']:>22.1f}")
    print(f"  {'Token Length Std':<32}  {train_stats['tok_std']:>18.1f}  {eval_stats['tok_std']:>22.1f}")
    print(f"  {'Token Length P10':<32}  {train_stats['tok_p10']:>18}  {eval_stats['tok_p10']:>22}")
    print(f"  {'Token Length P50 (Median)':<32}  {train_stats['tok_p50']:>18}  {eval_stats['tok_p50']:>22}")
    print(f"  {'Token Length P90':<32}  {train_stats['tok_p90']:>18}  {eval_stats['tok_p90']:>22}")
    print(f"  {'Token Length P99':<32}  {train_stats['tok_p99']:>18}  {eval_stats['tok_p99']:>22}")
    print("=" * 90)

    # Cross Comparison with Synthetic Corpus v3
    print("\n" + "=" * 90)
    print("  CROSS-DOMAIN BENCHMARK: REAL (Stage 26) vs SYNTHETIC (Stage 25)")
    print("=" * 90)
    print(f"  {'Feature':<30}  {'Synthetic Corpus v3':>24}  {'Real Corpus v1':>26}")
    print("-" * 90)
    print(f"  {'Data Provenance':<30}  {'Parametric Synthetic':>24}  {'TinyStories + Wikitext':>26}")
    train_tok_str = f"{train_stats['total_tokens']:,}"
    train_cpt_str = f"{train_stats['chars_per_token']:.3f}"
    train_tpw_str = f"{train_stats['tokens_per_word']:.3f}"
    train_vocab_str = f"{train_stats['active_vocab']:,} ({train_stats['vocab_coverage_pct']:.1f}%)"

    print(f"  {'Training Tokens':<30}  {'3,078,117':>24}  {train_tok_str:>26}")
    print(f"  {'Chars / Token':<30}  {'~3.95':>24}  {train_cpt_str:>26}")
    print(f"  {'Tokens / Word':<30}  {'~1.45':>24}  {train_tpw_str:>26}")
    print(f"  {'Active Vocab':<30}  {'10,834 (33.1%)':>24}  {train_vocab_str:>26}")
    print(f"  {'UNK Tokens':<30}  {'0':>24}  {train_stats['unk_count']:>26}")
    print(f"  {'Domain Variety':<30}  {'5 Generator Templates':>24}  {'Authentic Human/GPT-4 Prose':>26}")
    print("=" * 90 + "\n")

    # Save audit results to JSON
    audit_output = DATA_DIR / "tokenizer_transfer_audit.json"
    with open(audit_output, "w", encoding="utf-8") as f:
        json.dump({"train": train_stats, "eval_held_out": eval_stats}, f, indent=2)
    print(f"  [Audit Saved] Saved benchmark results to {audit_output}\n")


if __name__ == "__main__":
    run_tokenizer_transfer_audit()
