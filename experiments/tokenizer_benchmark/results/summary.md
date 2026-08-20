# Kiara Tokenizer Benchmark Results

*Generated: 2026-08-20 17:05 UTC*

> **Important:** These are empirical measurements. No tokenizer has been selected yet. This report is evidence -- not a decision.

---

## Environment

- **Python version:** 3.14.3
- **Tokenizers library:** 0.23.1
- **Corpus version:** 1.0.0
- **Random seed:** 1337

## Aggregate Comparison

| Algorithm | Vocab Req | Vocab Actual | Compress (chars/tok) | tok/char | tok/word | UNK tokens | Round-trip | Status |
|---|---|---|---|---|---|---|---|---|
| byte_bpe | 1,000 | 1,000 | 2.037 | 0.4910 | 3.859 | 0 | PASS | OK |
| bpe | 1,000 | 1,000 | 2.312 | 0.4325 | 3.399 | 0 | FAIL | OK |
| unigram | 1,000 | 1,000 | 2.087 | 0.4792 | 3.766 | 0 | PASS | OK |

## Per-Category Compression Ratio (chars/token)

| Algorithm | Vocab | english | technical | code | math | multilingual | rare_words | unicode | urls_symbols |
|---|---|---|---|---|---|---|---|---|---|
| byte_bpe | 1,000 | 2.676 | 2.605 | 2.595 | 1.652 | 1.210 | 2.799 | 1.129 | 1.855 |
| bpe | 1,000 | 2.795 | 2.659 | 3.017 | 1.725 | 1.602 | 2.711 | 1.684 | 1.952 |
| unigram | 1,000 | 2.495 | 2.315 | 2.490 | 1.845 | 1.540 | 2.743 | 1.229 | 1.871 |

## Unknown Token Analysis

| Algorithm | Vocab | UNK tokens | UNK samples | Round-trip failures |
|---|---|---|---|---|
| byte_bpe | 1,000 | 0 | 0 | 0 |
| bpe | 1,000 | 0 | 0 | 145 |
| unigram | 1,000 | 0 | 0 | 0 |

## Encoding / Decoding Speed

| Algorithm | Vocab | Encode (ms) | Decode (ms) | Encode tok/s | Decode tok/s |
|---|---|---|---|---|---|
| byte_bpe | 1,000 | 11.2 | 2.1 | 497,173 | 2,624,197 |
| bpe | 1,000 | 6.4 | 1.3 | 765,873 | 3,891,406 |
| unigram | 1,000 | 8.7 | 2.1 | 626,487 | 2,615,258 |

## Observations

The following observations are derived from the measurements above. They are empirical summaries, not design decisions.

- **Best compression:** bpe at vocab=1,000 (2.312 chars/token)
- **Zero UNK tokens:** byte_bpe@1,000, bpe@1,000, unigram@1,000
- **Round-trip failures:** bpe@1,000

---

**Next step:** Review these results against the criteria in `docs/research/tokenizer.md` Section 11 before updating `docs/decisions.md`.

**The tokenizer decision has NOT been made. Do not freeze a tokenizer until the full benchmark has been reviewed.**
