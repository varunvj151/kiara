# Kiara Tokenizer Benchmark

This directory contains the tokenizer benchmark for Kiara, implementing Step 19 (Tokenizer Research) of the Phase 2 Core Infrastructure plan.

> **Status:** Research benchmark. No tokenizer has been selected yet.
> Run the benchmark and review `results/summary.md` before updating `docs/decisions.md`.

---

## Structure

```
experiments/tokenizer_benchmark/
├── README.md          # This file
├── corpus.py          # Benchmark corpus builder (8 categories, no downloads)
├── train.py           # Tokenizer trainer (byte_bpe, bpe, unigram)
├── benchmark.py       # Main benchmark orchestrator
├── metrics.py         # Metrics computation (15 metrics)
├── report.py          # Markdown report generator
├── data/
│   └── models/        # Trained tokenizer model files (auto-created)
└── results/           # Output files (auto-created)
    ├── results.json
    ├── results.csv
    └── summary.md
```

---

## Quick Start

### 1. Smoke test (fastest — verifies everything works)

```bash
cd experiments/tokenizer_benchmark
python benchmark.py --smoke
```

This trains all 3 algorithms at `vocab_size=1000` and runs the full metric suite. Completes in ~30 seconds on CPU.

### 2. Subset benchmark

```bash
python benchmark.py --algos byte_bpe bpe --vocab-sizes 16384 32768
```

### 3. Full benchmark (3 algos × 5 vocab sizes = 15 runs)

```bash
python benchmark.py
```

Estimated time on CPU: 5–20 minutes depending on hardware.

### 4. Generate the Markdown report

```bash
python report.py
```

Report is written to `results/summary.md`.

---

## Running Tests

From the project root:

```bash
python -m pytest tests/test_tokenizer_benchmark.py -v
```

The tests use session-scoped fixtures — tokenizers are trained once at `vocab_size=1000` and reused across all tests.

---

## Algorithms

| ID | Algorithm | Library | UNK possible |
|---|---|---|---|
| `byte_bpe` | Byte-level BPE | HuggingFace tokenizers | No — zero UNK guaranteed |
| `bpe` | Standard BPE (character-level) | HuggingFace tokenizers | Yes |
| `unigram` | Unigram | HuggingFace tokenizers | Yes |

## Vocabulary Sizes

| Size | Notes |
|---|---|
| 16,384 | Minimal; suitable for tiny/small models |
| 32,768 | LLaMA 1/2 baseline |
| 50,000 | GPT-2/3 reference point |
| 65,536 | Extended multilingual coverage |
| 100,352 | GPT-4 cl100k_base reference point |

## Corpus Categories

| Category | Description |
|---|---|
| english | Standard English prose |
| technical | Technical/scientific English |
| code | Python, C++, JavaScript source code |
| math | LaTeX and Unicode math notation |
| multilingual | 14 languages and scripts |
| rare_words | Long and morphologically complex words |
| unicode | Emoji, symbols, mixed scripts |
| urls_symbols | URLs, file paths, operators, JSON |

## Metrics

All metrics are measured and recorded. No results are fabricated.

| Metric | Description |
|---|---|
| vocab_size_actual | Real vocabulary size after training |
| total_tokens | Token count across full corpus |
| tokens_per_char | Lower = better compression |
| tokens_per_word | Tokens needed per whitespace word |
| compression_ratio | Characters per token (higher = more efficient) |
| encode_time_ms | Median encoding time across 3 repeats |
| decode_time_ms | Median decoding time across 3 repeats |
| round_trip_pass | Whether decode(encode(text)) == text |
| unk_count | Total UNK tokens produced |
| by_category | All metrics broken down per corpus category |

---

## Reproducibility

Every benchmark run records:

- Random seed (default: 1337)
- Tokenizer algorithm and library version
- Python version
- Corpus version
- Timestamp
- Actual vocabulary size achieved

These are included in `results/results.json`.

---

## Decision Gate

The tokenizer decision must NOT be made before:

1. Running the full benchmark.
2. Reviewing `results/summary.md`.
3. Verifying round-trip correctness for Unicode edge cases.
4. Comparing compression ratios across all corpus categories.
5. Confirming UNK behavior for each algorithm.

When ready, document the decision in `docs/decisions.md` as Decision 003.
