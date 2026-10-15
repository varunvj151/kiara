# Kiara Tokenizer Benchmark Results

*Generated: 2026-08-21 10:11 UTC*

> **Important:** These are empirical measurements. No tokenizer has been selected yet. This report is evidence -- not a decision.

---

## Environment & Corpus

- **Python version:** 3.14.3
- **Tokenizers library:** 0.23.1
- **Corpus version:** 2.0.0
- **Random seed:** 1337
- **Training dataset:** 100,000 samples (18.75 MB, 146,429 unique words)
- **Evaluation dataset:** 173 samples across 8 categories (Disjoint: True)

## Aggregate Comparison

| Algorithm | Vocab Req | Vocab Actual | Compress (chars/tok) | tok/char | tok/word | UNK tokens | Round-trip | Status |
|---|---|---|---|---|---|---|---|---|
| byte_bpe | 16,384 | 16,384 | 1.863 | 0.5369 | 4.238 | 0 | PASS | OK |
| byte_bpe | 32,768 | 32,768 | 1.867 | 0.5355 | 4.227 | 0 | PASS | OK |
| byte_bpe | 50,000 | 50,000 | 1.870 | 0.5348 | 4.222 | 0 | PASS | OK |
| byte_bpe | 65,536 | 65,536 | 1.870 | 0.5348 | 4.222 | 0 | PASS | OK |
| byte_bpe | 100,352 | 100,352 | 1.870 | 0.5348 | 4.222 | 0 | PASS | OK |
| bpe | 16,384 | 16,384 | 2.090 | 0.4784 | 3.776 | 414 | FAIL | OK |
| bpe | 32,768 | 32,768 | 2.102 | 0.4758 | 3.756 | 414 | FAIL | OK |
| bpe | 50,000 | 50,000 | 2.108 | 0.4745 | 3.746 | 414 | FAIL | OK |
| bpe | 65,536 | 65,536 | 2.110 | 0.4740 | 3.742 | 414 | FAIL | OK |
| bpe | 100,352 | 100,352 | 2.112 | 0.4734 | 3.737 | 414 | FAIL | OK |
| unigram | 16,384 | 16,384 | 1.492 | 0.6702 | 5.290 | 0 | PASS | OK |
| unigram | 32,768 | 32,768 | 1.498 | 0.6677 | 5.271 | 0 | PASS | OK |
| unigram | 50,000 | 50,000 | 1.506 | 0.6639 | 5.241 | 0 | PASS | OK |
| unigram | 65,536 | 65,536 | 1.513 | 0.6611 | 5.219 | 0 | PASS | OK |
| unigram | 100,352 | 100,352 | 1.525 | 0.6557 | 5.176 | 0 | PASS | OK |

## Per-Category Compression Ratio (chars/token)

| Algorithm | Vocab | english | technical | code | math | multilingual | rare_words | unicode | urls_symbols |
|---|---|---|---|---|---|---|---|---|---|
| byte_bpe | 16,384 | 2.477 | 2.576 | 2.265 | 1.612 | 1.212 | 2.888 | 0.760 | 1.830 |
| byte_bpe | 32,768 | 2.481 | 2.578 | 2.271 | 1.617 | 1.214 | 2.888 | 0.760 | 1.850 |
| byte_bpe | 50,000 | 2.484 | 2.581 | 2.278 | 1.619 | 1.216 | 2.888 | 0.760 | 1.850 |
| byte_bpe | 65,536 | 2.484 | 2.581 | 2.278 | 1.619 | 1.216 | 2.888 | 0.760 | 1.850 |
| byte_bpe | 100,352 | 2.484 | 2.581 | 2.278 | 1.619 | 1.216 | 2.888 | 0.760 | 1.850 |
| bpe | 16,384 | 2.459 | 2.578 | 2.086 | 1.987 | 1.583 | 2.880 | 1.310 | 2.019 |
| bpe | 32,768 | 2.474 | 2.584 | 2.099 | 1.995 | 1.585 | 2.936 | 1.315 | 2.037 |
| bpe | 50,000 | 2.484 | 2.587 | 2.105 | 2.002 | 1.585 | 2.953 | 1.315 | 2.052 |
| bpe | 65,536 | 2.484 | 2.590 | 2.106 | 2.005 | 1.585 | 2.953 | 1.315 | 2.061 |
| bpe | 100,352 | 2.484 | 2.596 | 2.110 | 2.005 | 1.588 | 2.961 | 1.315 | 2.061 |
| unigram | 16,384 | 1.827 | 1.891 | 1.622 | 1.402 | 1.031 | 2.325 | 0.704 | 1.579 |
| unigram | 32,768 | 1.837 | 1.927 | 1.637 | 1.364 | 1.024 | 2.382 | 0.704 | 1.590 |
| unigram | 50,000 | 1.860 | 1.943 | 1.645 | 1.397 | 1.045 | 2.377 | 0.688 | 1.581 |
| unigram | 65,536 | 1.867 | 1.956 | 1.649 | 1.397 | 1.048 | 2.398 | 0.691 | 1.592 |
| unigram | 100,352 | 1.881 | 1.971 | 1.671 | 1.400 | 1.051 | 2.409 | 0.693 | 1.631 |

## Unknown Token & Round-Trip Diagnostics

| Algorithm | Vocab | UNK tokens | UNK samples | Total RT Failures | Failures (UNK) | Failures (Decoder) |
|---|---|---|---|---|---|---|
| byte_bpe | 16,384 | 0 | 0 | 0 | 0 | 0 |
| byte_bpe | 32,768 | 0 | 0 | 0 | 0 | 0 |
| byte_bpe | 50,000 | 0 | 0 | 0 | 0 | 0 |
| byte_bpe | 65,536 | 0 | 0 | 0 | 0 | 0 |
| byte_bpe | 100,352 | 0 | 0 | 0 | 0 | 0 |
| bpe | 16,384 | 414 | 47 | 47 | 47 | 0 |
| bpe | 32,768 | 414 | 47 | 47 | 47 | 0 |
| bpe | 50,000 | 414 | 47 | 47 | 47 | 0 |
| bpe | 65,536 | 414 | 47 | 47 | 47 | 0 |
| bpe | 100,352 | 414 | 47 | 47 | 47 | 0 |
| unigram | 16,384 | 0 | 0 | 0 | 0 | 0 |
| unigram | 32,768 | 0 | 0 | 0 | 0 | 0 |
| unigram | 50,000 | 0 | 0 | 0 | 0 | 0 |
| unigram | 65,536 | 0 | 0 | 0 | 0 | 0 |
| unigram | 100,352 | 0 | 0 | 0 | 0 | 0 |

## Vocabulary Efficiency (10k Training Sample)

| Algorithm | Vocab | Utilized | Util % | Freq >= 10 | Well-Rep (>=10) % | Freq >= 100 | Top 10% Active | Top 25% Active | Top 50% Active | Top 100 | Top 500 | Top 1k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| byte_bpe | 16,384 | 1,612 | 9.8% | 639 | 3.9% | 274 | 86.1% | 96.8% | 99.0% | 79.2% | 97.6% | 99.5% |
| byte_bpe | 32,768 | 2,743 | 8.4% | 527 | 1.6% | 274 | 94.9% | 98.4% | 99.3% | 79.9% | 97.7% | 98.9% |
| byte_bpe | 50,000 | 2,993 | 6.0% | 516 | 1.0% | 274 | 95.9% | 98.5% | 99.2% | 80.1% | 97.9% | 98.8% |
| byte_bpe | 65,536 | 3,341 | 5.1% | 493 | 0.8% | 274 | 96.8% | 98.6% | 99.1% | 80.5% | 98.0% | 98.8% |
| byte_bpe | 100,352 | 3,426 | 3.4% | 466 | 0.5% | 274 | 97.2% | 98.6% | 99.1% | 80.9% | 98.1% | 98.7% |
| bpe | 16,384 | 1,659 | 10.1% | 584 | 3.6% | 254 | 85.5% | 96.1% | 98.6% | 78.0% | 96.9% | 99.1% |
| bpe | 32,768 | 1,987 | 6.1% | 532 | 1.6% | 253 | 89.1% | 97.1% | 98.9% | 78.5% | 97.1% | 98.9% |
| bpe | 50,000 | 2,515 | 5.0% | 466 | 0.9% | 253 | 93.5% | 97.7% | 99.0% | 78.9% | 97.1% | 98.6% |
| bpe | 65,536 | 2,822 | 4.3% | 446 | 0.7% | 253 | 94.8% | 97.9% | 99.0% | 79.1% | 97.1% | 98.5% |
| bpe | 100,352 | 3,420 | 3.4% | 412 | 0.4% | 253 | 96.1% | 98.0% | 99.0% | 79.5% | 97.2% | 98.2% |
| unigram | 16,384 | 1,071 | 6.5% | 536 | 3.3% | 352 | 77.6% | 95.9% | 99.7% | 75.5% | 99.6% | 100.0% |
| unigram | 32,768 | 1,250 | 3.8% | 527 | 1.6% | 352 | 83.0% | 97.2% | 99.8% | 75.0% | 99.6% | 99.9% |
| unigram | 50,000 | 1,510 | 3.0% | 509 | 1.0% | 340 | 88.4% | 98.6% | 99.8% | 74.4% | 99.5% | 99.9% |
| unigram | 65,536 | 1,853 | 2.8% | 499 | 0.8% | 342 | 91.6% | 99.3% | 99.8% | 74.5% | 99.4% | 99.8% |
| unigram | 100,352 | 2,364 | 2.4% | 491 | 0.5% | 341 | 94.2% | 99.4% | 99.7% | 74.6% | 99.3% | 99.6% |

## Encoding / Decoding Speed

| Algorithm | Vocab | Encode (ms) | Decode (ms) | Encode tok/s | Decode tok/s |
|---|---|---|---|---|---|
| byte_bpe | 16,384 | 13.4 | 2.8 | 525,582 | 2,515,885 |
| byte_bpe | 32,768 | 15.8 | 3.0 | 445,656 | 2,324,044 |
| byte_bpe | 50,000 | 13.6 | 2.8 | 518,056 | 2,530,911 |
| byte_bpe | 65,536 | 13.3 | 3.5 | 526,166 | 2,010,480 |
| byte_bpe | 100,352 | 21.6 | 4.5 | 324,813 | 1,573,651 |
| bpe | 16,384 | 7.6 | 2.5 | 826,969 | 2,496,522 |
| bpe | 32,768 | 8.7 | 2.4 | 714,204 | 2,619,307 |
| bpe | 50,000 | 8.0 | 2.5 | 781,213 | 2,508,255 |
| bpe | 65,536 | 7.6 | 2.6 | 821,009 | 2,426,499 |
| bpe | 100,352 | 8.0 | 2.6 | 772,107 | 2,347,498 |
| unigram | 16,384 | 9.7 | 3.0 | 904,353 | 2,898,751 |
| unigram | 32,768 | 9.6 | 2.9 | 913,753 | 2,981,227 |
| unigram | 50,000 | 9.4 | 3.2 | 928,748 | 2,764,648 |
| unigram | 65,536 | 9.5 | 3.0 | 911,439 | 2,862,373 |
| unigram | 100,352 | 16.7 | 4.5 | 515,369 | 1,914,250 |

## Observations

The following observations are derived from the measurements above. They are empirical summaries, not design decisions.

- **Best compression:** bpe at vocab=100,352 (2.112 chars/token)
- **Zero UNK tokens:** byte_bpe@16,384, byte_bpe@32,768, byte_bpe@50,000, byte_bpe@65,536, byte_bpe@100,352, unigram@16,384, unigram@32,768, unigram@50,000, unigram@65,536, unigram@100,352
- **Round-trip failures:** bpe@16,384, bpe@32,768, bpe@50,000, bpe@65,536, bpe@100,352

---

**Next step:** Review these results against the criteria in `docs/research/tokenizer.md` Section 11 before updating `docs/decisions.md`.

**The tokenizer decision has NOT been made. Do not freeze a tokenizer until the full benchmark has been reviewed.**
