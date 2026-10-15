# Kiara Engineering Decisions

Every important design decision should be recorded so that the reasoning behind Kiara remains transparent over time.

## Decision 001

### Tokenizer

Chosen: BPE

Reason:
- Efficient
- Widely used
- Good compression
- Fast

Alternatives:
- SentencePiece
- WordPiece
- Character

Status: Accepted

## Decision 002

### Model Family

Chosen: Decoder-only Transformer

Reason:
- Strong language modeling baseline
- Scales well
- Fits modern foundation model research

Alternatives:
- Encoder-only
- Encoder-decoder

Status: Accepted


## Decision 003: Tokenizer Algorithm and Vocabulary Size

**Status:** Accepted

**Date:** 2026-08-21 (Supersedes preliminary placeholder in Decision 001)

**Decision:**
Kiara will use a **Byte-Level BPE tokenizer with a vocabulary size of 32,768 (32K)**.

### Evidence

The decision is based on the comprehensive tokenizer benchmark at:

`experiments/tokenizer_benchmark/results/`

The benchmark evaluated:
- Byte-Level BPE
- Standard BPE
- Unigram (with byte fallback)

across vocabulary sizes:
- 16,384
- 32,768
- 50,000
- 65,536
- 100,352

using the fixed benchmark corpus (100,000 training samples, 18.75 MB), a held-out evaluation corpus (173 samples across 8 domains), and deterministic benchmark configurations.

### Decision Rationale

Byte-Level BPE was selected because it provides the best overall balance of robustness, compression efficiency, vocabulary efficiency, and inference speed.

#### 1. Robustness & Round-Trip Fidelity
Byte-Level BPE produced:
- **0 UNK tokens** (0.0% UNK rate)
- **0 UNK-affected evaluation samples**
- **0 round-trip failures** (100.0% exact text reconstruction)

across all tested vocabulary sizes. This satisfies the requirement for arbitrary Unicode, emoji, source code, mathematical notation, and multilingual inputs.

#### 2. Compression
At 32K vocabulary:
- **Byte-BPE:** 1.867 chars/token
- **Unigram:** 1.498 chars/token
- **Standard BPE:** 2.102 chars/token (distorted by UNK token substitutions)

Byte-BPE provides +24.6% higher compression efficiency than Unigram while maintaining zero UNK tokens.

#### 3. Code Handling
At 32K vocabulary:
- **Byte-BPE:** 2.271 chars/token
- **Unigram:** 1.637 chars/token
- **Standard BPE:** 2.099 chars/token

Byte-BPE provides the strongest compression among viable zero-UNK candidates on programming code (+38.7% over Unigram).

#### 4. Multilingual Handling
At 32K vocabulary:
- **Byte-BPE:** 1.214 chars/token
- **Unigram:** 1.024 chars/token
- **Standard BPE:** 1.585 chars/token

Byte-BPE provides better compression than Unigram while retaining byte-level coverage and exact round-trip behavior.

#### 5. Vocabulary Efficiency
At 32K on the representative 10k training sample:
- **Active Byte-BPE tokens:** 2,743
- **Active vocabulary utilization:** 8.4%
- **Tokens occurring $\ge 10$ times:** 527
- **Top 100 active tokens coverage:** 79.9% of all token occurrences
- **Top 500 active tokens coverage:** 97.7% of all token occurrences

Increasing vocabulary capacity introduces increasingly rare tokens while providing negligible additional compression.

#### 6. Vocabulary Scaling
Byte-BPE compression across vocabulary sizes:
- **16K:** 1.863 chars/token (9.8% utilization)
- **32K:** 1.867 chars/token (8.4% utilization)
- **50K:** 1.870 chars/token (6.0% utilization)
- **65K:** 1.870 chars/token (5.1% utilization)
- **100K:** 1.870 chars/token (3.4% utilization)

The benchmark demonstrates that 32K is the smallest tested vocabulary that captures the observed compression plateau for Byte-BPE on the current corpus; larger vocabularies provide negligible additional compression while inflating embedding and LM head parameter overhead (e.g. ~67M parameters at 32K vs ~205M at 100K with $d_{\text{model}}=1024$).

#### 7. Standard BPE Disqualification
Standard BPE was disqualified because it produced:
- **414 UNK tokens**
- **47 affected evaluation samples**
- **47 round-trip failures**

These failures persisted across all tested vocabulary sizes, confirming that scaling character-level BPE vocabulary does not resolve its fundamental lack of byte fallback for out-of-vocabulary Unicode.

### Alternatives Considered

#### Unigram
Rejected as primary tokenizer.
- *Advantages:* 0 UNK, 0 round-trip failures, strong core concentration.
- *Disadvantages:* Substantially worse compression than Byte-BPE across all 8 evaluated domains; lower active vocabulary utilization.

#### Standard BPE
Disqualified due to non-zero UNK rate and persistent round-trip failures on Unicode inputs.

### Final Configuration

```yaml
tokenizer:
  type: byte_bpe
  vocab_size: 32768
  special_tokens:
    - "[UNK]"
    - "[PAD]"
    - "[BOS]"
    - "[EOS]"
```

### Reproducibility & Benchmark Environment

- Python: 3.14.3
- HuggingFace tokenizers: 0.23.1
- Corpus version: 2.0.0
- Random seed: 1337
- Training samples: 100,000
- Held-out evaluation samples: 173
- Evaluation categories: 8

*Note on Unigram implementation:* In HuggingFace `tokenizers==0.23.1`, Unigram with byte fallback was evaluated with 256 byte fallback tokens (`<0x00>`..<0xFF>) injected into the model vocabulary and `byte_fallback=True` enabled.

### Scope of Decision

This decision is based on the current Kiara tokenizer benchmark corpus and evaluation methodology. It freezes the tokenizer configuration for initial Kiara model pre-training and downstream infrastructure.

