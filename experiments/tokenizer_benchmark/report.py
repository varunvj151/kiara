"""
report.py — Kiara Tokenizer Benchmark Report Generator
=======================================================
Reads results/results.json and generates a human-readable
Markdown report: results/summary.md

Usage:
    cd experiments/tokenizer_benchmark
    python report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"


def _star(condition: bool, true: str = "PASS", false: str = "FAIL") -> str:
    return true if condition else false


def _fmt(value, fmt: str = ".3f") -> str:
    if value is None:
        return "-"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def generate_report(results_path: Path | None = None) -> str:
    """Generate a Markdown summary report from results.json."""
    if results_path is None:
        results_path = RESULTS_DIR / "results.json"

    if not results_path.exists():
        return (
            "# Kiara Tokenizer Benchmark -- Results Not Found\n\n"
            f"Expected: {results_path}\n\n"
            "Run `python benchmark.py` first to generate results.\n"
        )

    data = json.loads(results_path.read_text(encoding="utf-8"))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append("# Kiara Tokenizer Benchmark Results\n")
    lines.append(f"*Generated: {generated_at}*\n")
    lines.append(
        "> **Important:** These are empirical measurements. "
        "No tokenizer has been selected yet. "
        "This report is evidence -- not a decision.\n"
    )
    lines.append("---\n")

    # --- Environment ---
    if data:
        r0 = data[0]
        lines.append("## Environment\n")
        lines.append(f"- **Python version:** {r0.get('python_version', '?')}")
        lines.append(f"- **Tokenizers library:** {r0.get('library_version', '?')}")
        lines.append(f"- **Corpus version:** {r0.get('corpus_version', '?')}")
        lines.append(f"- **Random seed:** {r0.get('random_seed', '?')}")
        lines.append("")

    # --- Aggregate comparison table ---
    lines.append("## Aggregate Comparison\n")
    lines.append(
        "| Algorithm | Vocab Req | Vocab Actual | Compress (chars/tok) | tok/char | "
        "tok/word | UNK tokens | Round-trip | Status |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for r in data:
        agg = r.get("aggregate", {})
        rt = _star(agg.get("round_trip_pass", False))
        status = "OK" if r.get("success") else f"ERROR: {r.get('error', 'error')[:40]}"
        lines.append(
            f"| {r['tokenizer_type']} "
            f"| {r['vocab_size_requested']:,} "
            f"| {r.get('vocab_size_actual', '?'):,} "
            f"| {_fmt(agg.get('compression_ratio'))} "
            f"| {_fmt(agg.get('tokens_per_char'), '.4f')} "
            f"| {_fmt(agg.get('tokens_per_word'), '.3f')} "
            f"| {agg.get('unk_count', '?')} "
            f"| {rt} "
            f"| {status} |"
        )

    lines.append("")

    # --- Per-category breakdown ---
    lines.append("## Per-Category Compression Ratio (chars/token)\n")
    categories = []
    if data:
        for r in data:
            for cat in r.get("by_category", {}):
                if cat not in categories:
                    categories.append(cat)

    header = "| Algorithm | Vocab |" + "".join(f" {c} |" for c in categories)
    separator = "|---|---|" + "---|" * len(categories)
    lines.append(header)
    lines.append(separator)

    for r in data:
        if not r.get("success"):
            continue
        by_cat = r.get("by_category", {})
        row = f"| {r['tokenizer_type']} | {r['vocab_size_requested']:,} |"
        for cat in categories:
            cat_data = by_cat.get(cat, {})
            row += f" {_fmt(cat_data.get('compression_ratio'))} |"
        lines.append(row)

    lines.append("")

    # --- UNK analysis ---
    lines.append("## Unknown Token Analysis\n")
    lines.append("| Algorithm | Vocab | UNK tokens | UNK samples | Round-trip failures |")
    lines.append("|---|---|---|---|---|")
    for r in data:
        if not r.get("success"):
            continue
        agg = r.get("aggregate", {})
        lines.append(
            f"| {r['tokenizer_type']} "
            f"| {r['vocab_size_requested']:,} "
            f"| {agg.get('unk_count', '?')} "
            f"| {agg.get('unk_sample_count', '?')} "
            f"| {agg.get('round_trip_failures', '?')} |"
        )

    lines.append("")

    # --- Speed comparison ---
    lines.append("## Encoding / Decoding Speed\n")
    lines.append("| Algorithm | Vocab | Encode (ms) | Decode (ms) | Encode tok/s | Decode tok/s |")
    lines.append("|---|---|---|---|---|---|")
    for r in data:
        if not r.get("success"):
            continue
        agg = r.get("aggregate", {})
        lines.append(
            f"| {r['tokenizer_type']} "
            f"| {r['vocab_size_requested']:,} "
            f"| {_fmt(agg.get('encode_time_ms'), '.1f')} "
            f"| {_fmt(agg.get('decode_time_ms'), '.1f')} "
            f"| {int(agg.get('encode_throughput_tps', 0)):,} "
            f"| {int(agg.get('decode_throughput_tps', 0)):,} |"
        )

    lines.append("")

    # --- Observations ---
    lines.append("## Observations\n")
    lines.append(
        "The following observations are derived from the measurements above. "
        "They are empirical summaries, not design decisions.\n"
    )

    # Find best compression
    best_compress = None
    for r in data:
        if not r.get("success"):
            continue
        agg = r.get("aggregate", {})
        cr = agg.get("compression_ratio", 0)
        if best_compress is None or cr > best_compress[0]:
            best_compress = (cr, r["tokenizer_type"], r["vocab_size_requested"])

    if best_compress:
        lines.append(
            f"- **Best compression:** {best_compress[1]} at vocab={best_compress[2]:,} "
            f"({best_compress[0]:.3f} chars/token)"
        )

    # UNK safety
    zero_unk = [
        f"{r['tokenizer_type']}@{r['vocab_size_requested']:,}"
        for r in data
        if r.get("success") and r.get("aggregate", {}).get("unk_count", 1) == 0
    ]
    if zero_unk:
        lines.append(f"- **Zero UNK tokens:** {', '.join(zero_unk)}")
    else:
        lines.append("- **Zero UNK tokens:** None (all algorithms produced at least one UNK)")

    # Round-trip failures
    rt_fail = [
        f"{r['tokenizer_type']}@{r['vocab_size_requested']:,}"
        for r in data
        if r.get("success") and not r.get("aggregate", {}).get("round_trip_pass", True)
    ]
    if rt_fail:
        lines.append(f"- **Round-trip failures:** {', '.join(rt_fail)}")
    else:
        lines.append("- **Round-trip correctness:** All successful runs passed.")

    lines.append("")
    lines.append("---\n")
    lines.append(
        "**Next step:** Review these results against the criteria in "
        "`docs/research/tokenizer.md` Section 11 before updating `docs/decisions.md`.\n"
    )
    lines.append(
        "**The tokenizer decision has NOT been made. "
        "Do not freeze a tokenizer until the full benchmark has been reviewed.**\n"
    )

    return "\n".join(lines)


def main() -> None:
    report = generate_report()
    out_path = RESULTS_DIR / "summary.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report saved to {out_path}]")


if __name__ == "__main__":
    main()
