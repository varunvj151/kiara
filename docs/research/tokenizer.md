# Kiara Tokenizer Research

**Status:** Research Stage — Step 19
**Stage:** Pre-implementation. No tokenizer has been selected yet.
**Goal:** Understand all major tokenization strategies, their tradeoffs, and define a rigorous benchmark before making a final, evidence-based decision for Kiara.

---

## Table of Contents

1. What Tokenization Does in an LLM
2. Candidate Tokenizer Families
   - 2.1 Character Tokenization
   - 2.2 Word Tokenization
   - 2.3 WordPiece
   - 2.4 Unigram (SentencePiece)
   - 2.5 Byte-Pair Encoding (BPE)
   - 2.6 Byte-Level BPE
3. Cross-Domain Implications
4. Vocabulary Size Tradeoffs
5. Unknown-Token Behavior
6. Tokenizer Training Considerations
7. Licensing and Reproducibility Considerations
8. Tokenizer Benchmark Specification
9. Benchmark Datasets and Sample Inputs
10. Candidate Vocabulary Sizes
11. Recommendation Framework

---

## 1. What Tokenization Does in an LLM

A language model does not operate on raw text. It operates on a sequence of integer IDs. The tokenizer is the module responsible for mapping raw text into that integer sequence, and converting that sequence back into text.

This conversion is fundamental. Every design decision about the tokenizer propagates through the entire model:

- **Vocabulary size** directly determines the size of the embedding table and the final linear projection (LM head). For a model with hidden dimension H and vocabulary size V, the LM head is a matrix of shape [H, V].
- **Compression ratio** (how many characters map to one token) determines how many tokens a document consumes. A poor tokenizer wastes context window budget, reducing effective context length.
- **Granularity** determines whether the model must learn character-level patterns explicitly or whether subword structure is encoded implicitly.
- **Coverage** determines how the model handles rare or novel inputs: unknown words, foreign scripts, new programming languages, scientific notation.

The tokenizer is one of the few frozen components of a foundation model. Once training begins with a fixed tokenizer and vocabulary, changing either requires starting over from scratch. This makes the pre-training tokenizer decision one of the highest-stakes design choices in the project.

---

## 2. Candidate Tokenizer Families

### 2.1 Character Tokenization

#### Description

Every Unicode character is its own token. The vocabulary is the set of all unique characters in the training corpus.

#### How it works

```
Input:  "Hello, world!"
Tokens: ['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']
IDs:    [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]
```

#### Advantages

- Vocabulary is extremely small (a few hundred to a few thousand tokens).
- Handles all Unicode characters that appear in the training corpus.
- No unknown tokens for seen characters.
- No segmentation ambiguity.
- Complete linguistic transparency.

#### Disadvantages

- Sequence lengths explode. A 4096-token context window at the character level covers only a few hundred words, compared to several thousand at the subword level.
- The model must learn to reconstruct meaning from individual characters -- a much harder inductive task.
- All linguistic structure (morphology, words, phrases) must be learned purely from position and attention patterns, not baked into the tokenization.
- Practically unusable at foundation model scale due to extreme sequence lengths.

#### Verdict

Character tokenization is primarily useful for ultra-low-vocabulary studies and linguistic analysis. It is not a practical candidate for a foundation model.

---

### 2.2 Word Tokenization

#### Description

Text is split on whitespace and punctuation. Each word form (surface form) is its own token.

#### How it works

```
Input:  "The cats sat on the mat."
Tokens: ['The', 'cats', 'sat', 'on', 'the', 'mat', '.']
IDs:    [1432, 9847, 5523, 88, 201, 6103, 4]
```

#### Advantages

- Tokens correspond to human-recognizable units.
- Short sequences for common English text.
- Interpretable token boundaries.

#### Disadvantages

- Vocabulary explodes: English alone has hundreds of thousands of word forms. Extending to code, math, and multilingual text requires millions of tokens.
- Almost guaranteed to produce UNK for rare words, proper nouns, technical terms, and any text not seen during tokenizer training.
- Morphologically rich languages (German, Finnish, Turkish, Arabic) produce enormous vocabularies because each inflected form is a distinct token.
- Code identifiers like get_transformer_block_hidden_state become one token or fail entirely.
- Numbers are all unique: 3.14159, 2.71828, 1000000 are all separate tokens.
- No subword sharing between related forms: run, runs, running, runner share no tokens, so the embedding table learns them independently.

#### Verdict

Word tokenization is not a viable candidate for a modern foundation model. It is retained here only for educational completeness.

---

### 2.3 WordPiece

#### Description

WordPiece is a subword algorithm developed by Google and used in BERT. It is similar to BPE but optimizes a different objective during vocabulary construction.

#### Training algorithm (simplified)

1. Initialize vocabulary with individual characters.
2. At each step, find the pair of tokens whose merge maximizes the likelihood of the training corpus under a unigram language model over the current vocabulary.
3. Merge that pair and add it to the vocabulary.
4. Repeat until the vocabulary reaches the target size.

Unlike BPE (which merges based on raw frequency), WordPiece merges based on the ratio of the pair's joint count to the product of the individual token counts. It prefers merges that provide the most information gain.

#### How it works

```
Input:   "unbelievable"
Tokens:  ['un', '##believe', '##able']
```

Continuation pieces are prefixed with ##, distinguishing them from word-initial pieces.

#### Advantages

- Well-studied, widely used, and heavily optimized.
- Generally produces more linguistically coherent splits than raw BPE.
- Performs well on English morphology.

#### Disadvantages

- The ## prefix convention introduces an asymmetry: word-initial and word-continuation pieces are distinct tokens, doubling the effective vocabulary for some word forms.
- The training objective is more expensive than BPE's greedy frequency count.
- Poor at handling languages without whitespace delimiters (Chinese, Japanese, Thai) unless combined with additional pre-tokenization.
- Rare words and code identifiers still produce long sequences.

#### Primary Users

BERT, DistilBERT, ELECTRA, mBERT.

---

### 2.4 Unigram (SentencePiece)

#### Description

Unigram is a probabilistic subword algorithm implemented by Google's SentencePiece library. Unlike BPE and WordPiece (which are constructive -- they build the vocabulary by merging), Unigram is destructive: it starts with a large candidate vocabulary and iteratively removes tokens that contribute least to corpus likelihood.

#### Training algorithm (simplified)

1. Initialize a large candidate vocabulary (all substrings up to a length limit, or a BPE vocabulary as a seed).
2. Assign each candidate a probability based on a unigram language model.
3. Compute the loss (negative log-likelihood of the corpus).
4. Remove the tokens whose removal causes the smallest increase in loss.
5. Repeat until the target vocabulary size is reached.

Crucially, Unigram supports stochastic segmentation: the same string can be tokenized in multiple ways during training, where the probability of each segmentation is sampled from the unigram model. This regularization improves downstream model robustness.

#### Advantages

- Probabilistic foundation -- theoretically well-motivated.
- Stochastic segmentation during training provides regularization and can improve generalization.
- SentencePiece treats text as a raw byte sequence; no pre-tokenization or whitespace splitting is required. Critical for languages without space delimiters.
- Fully language-agnostic by design.
- Naturally handles multilingual corpora without script-specific rules.
- Vocabulary is pruned by principled loss minimization rather than greedy frequency counting.

#### Disadvantages

- Training is more computationally expensive than BPE.
- Vocabulary construction is less transparent.
- The stochastic segmentation property requires that the inference-time tokenizer use the most probable (Viterbi) segmentation, introducing a distinction between training and inference behavior.

#### Primary Users

T5, mT5, AlBERT, XLNet, Gemma.

---

### 2.5 Byte-Pair Encoding (BPE)

#### Description

BPE was originally a data compression algorithm. Its application to NLP tokenization was introduced in "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016). It is currently the most widely deployed subword tokenization algorithm in production foundation models.

#### Training algorithm

1. Pre-tokenize the corpus into initial units (typically characters, or character + space marker).
2. Count all consecutive symbol pairs in the corpus.
3. Merge the most frequent pair into a new token.
4. Update all pair counts.
5. Repeat until the vocabulary reaches the target size.

This is a deterministic, greedy, frequency-driven algorithm.

#### How it works

```
Initial corpus (after character split):
  "l o w </w>", "l o w e r </w>", "n e w e s t </w>", "w i d e s t </w>"

Step 1: most frequent pair: ('e', 's') -> merge to 'es'
Step 2: most frequent pair: ('es', 't') -> merge to 'est'
Step 3: most frequent pair: ('l', 'o') -> merge to 'lo'
...
Final tokens for "lowest": ['lo', 'w', 'est', '</w>']
```

#### Advantages

- Simple, fast, and deterministic.
- Well-studied with decades of literature.
- Extremely efficient implementations (HuggingFace tokenizers uses Rust under the hood).
- Vocabulary size is directly and precisely controllable.
- Widely used: GPT-2, GPT-3, GPT-4, LLaMA 1/2/3, Mistral, and many others use BPE or close variants.

#### Disadvantages

- Standard BPE operates on Unicode characters, not bytes. Rare Unicode characters or emoji that are not in the training corpus can produce UNK or fail entirely.
- The vocabulary is corpus-dependent: BPE trained on English-heavy corpora may tokenize other languages very inefficiently.
- No stochastic regularization during training (unlike Unigram).
- The greedy frequency-based merge strategy does not optimize any global objective; it is a heuristic.

#### Primary Users

GPT-2, GPT-3, LLaMA 1/2, Mistral, Falcon, and many open-weight models.

---

### 2.6 Byte-Level BPE

#### Description

Byte-level BPE is a variant of BPE where the initial alphabet is the 256 byte values (0-255) rather than Unicode characters. This was introduced in GPT-2 (Radford et al., 2019).

Because every valid string of text is a sequence of bytes, byte-level BPE is guaranteed to tokenize any input without ever producing an unknown token.

#### Training algorithm

Identical to BPE, except:
- The initial vocabulary is 256 byte values.
- Pre-tokenization often uses a regex to split text into word-like chunks before BPE merges are applied within each chunk.

#### Advantages

- Zero unknown tokens: every possible input -- any Unicode character, emoji, binary data -- can be tokenized.
- Excellent Unicode and multilingual handling by construction.
- Consistent with the view that text is fundamentally a byte stream.
- Well-established at scale: GPT-2, GPT-3, GPT-4, RoBERTa, BLOOM, Falcon, Qwen, and many others.
- tiktoken and HuggingFace tokenizers both have fast, production-ready byte-level BPE implementations.

#### Disadvantages

- For heavily multilingual corpora, rare scripts are represented by long byte sequences before merges, potentially resulting in less efficient compression than character-level BPE trained explicitly on those scripts.
- The 256-byte initial vocabulary means the first ~256 tokens must be allocated to single bytes, slightly inflating the vocabulary.
- Token boundaries may be less linguistically interpretable.

#### Primary Users

GPT-2, GPT-3, GPT-4 (tiktoken cl100k_base, o200k_base), RoBERTa, BLOOM, Falcon, Qwen2.

---

## 3. Cross-Domain Implications

### 3.1 English Text

| Approach | Behavior |
|---|---|
| Character | ~5x longer sequences than subword; model must learn all word structure |
| Word | Efficient for common words; catastrophic OOV for rare or technical words |
| WordPiece | Good performance; commonly used in English BERT-family models |
| Unigram | Good performance; slightly more linguistically principled on morphology |
| BPE | Good compression; common English words often become single tokens |
| Byte-level BPE | Equivalent to BPE on ASCII text; no practical difference for English |

**Preferred candidates for English:** BPE or Byte-level BPE.

---

### 3.2 Multilingual Text

| Approach | Behavior |
|---|---|
| Character | Language-agnostic; but impractically long sequences |
| Word | Catastrophic -- vocabulary explodes across scripts and morphologies |
| WordPiece | Requires per-language tuning; mBERT works but shows script imbalance |
| Unigram (SentencePiece) | Best-designed for multilingual; stochastic segmentation helps balance |
| BPE | Works but distributes vocabulary unevenly across languages |
| Byte-level BPE | Handles all scripts by construction; no OOV; may be less efficient per character for low-resource scripts |

**Preferred candidates for multilingual:** Unigram or Byte-level BPE.

---

### 3.3 Programming Code

Code has distinct properties from natural language:
- Identifiers are concatenated with no spaces: TransformerBlock, get_hidden_state.
- Common patterns repeat: def, return, import, self., ->.
- Indentation is semantically significant.
- Numeric literals appear in many forms: 0x1F, 3.14e-5, 0b10110.

| Approach | Behavior |
|---|---|
| Character | Every character is a token; indentation is expensive |
| Word | Splits on _ inconsistently; long identifiers are OOV |
| WordPiece | Reasonable but continuation markers add overhead |
| Unigram | Reasonable; stochastic splits can help with identifier diversity |
| BPE | Common patterns (def, return) become single tokens naturally via frequency |
| Byte-level BPE | Same as BPE; no OOV on any binary or hex literal or identifier |

**Preferred candidates for code:** BPE or Byte-level BPE with a code-balanced training corpus.

---

### 3.4 Mathematical Notation

Mathematical text is particularly challenging:
- Symbols: Greek letters, arrows, operators, integrals.
- Equations mix numbers, operators, and variables densely.
- LaTeX markup: \frac{d}{dx}, \sum_{i=0}^{n}, \mathbb{R}.
- Numbers: 1.618033, 2.71828182845, 6.022e23.

| Approach | Behavior |
|---|---|
| Character | Handles all symbols but enormous sequence length |
| Word | Splits on operators; number forms are all OOV |
| WordPiece | Rare symbols are individual characters or UNK |
| Unigram | Better handling of LaTeX patterns if math is in training data |
| BPE | Common math tokens emerge if training corpus includes math |
| Byte-level BPE | No OOV; rare Unicode symbols are byte-decomposed |

**Key insight:** The tokenizer alone cannot solve math. The training corpus must contain substantial mathematical text. However, a tokenizer that handles rare Unicode symbols without OOV is safer for math-heavy domains.

---

### 3.5 Technical and Scientific Text

Technical text includes: units (m/s, kg*m^2/s^2), chemical formulas (C6H12O6), citations ([1], et al.), and abbreviations.

BPE and byte-level BPE both handle these domains adequately if the training corpus is balanced. The critical factor is corpus composition, not just tokenizer algorithm.

---

### 3.6 Rare Words and Unseen Inputs

| Approach | Handles Rare Words | Handles Unseen Words |
|---|---|---|
| Character | Yes (always) | Yes (always) |
| Word | No (UNK) | No (UNK) |
| WordPiece | Partially (character fallback) | Partially (character fallback) |
| Unigram | Yes (character fallback) | Yes (character fallback) |
| BPE (char-level) | Yes (character fallback) | Yes (character fallback) |
| Byte-level BPE | Yes (byte fallback) | Yes (always -- any byte sequence) |

Only byte-level BPE provides a hard guarantee of zero unknown tokens for any input.

---

### 3.7 Unicode and Emoji

| Approach | Unicode Handling | Emoji Handling |
|---|---|---|
| Character | Full coverage (if character in vocab) | Full if seen during training |
| Word | UNK for most non-ASCII | UNK |
| WordPiece | UNK for unseen chars | UNK or char fallback |
| Unigram | Depends on training corpus | Partial |
| BPE (char-level) | UNK if char not in initial vocab | UNK |
| Byte-level BPE | Full coverage (all Unicode = UTF-8 bytes) | Full coverage always |

Byte-level BPE is the only approach that guarantees correct handling of all valid Unicode inputs, including all emoji, all scripts, and any future Unicode extensions.

---

### 3.8 Context Window Efficiency

Compression ratio (characters per token) is a direct measure of context window efficiency. A model with a 4096-token context window covers more content when each token represents more characters.

| Approach | Typical chars/token (English prose) |
|---|---|
| Character | ~1.0 |
| Word | ~4-6 (but high OOV rate) |
| WordPiece 32K | ~3.5-4.5 |
| Unigram 32K | ~3.5-4.5 |
| BPE 32K | ~4.0-5.0 |
| Byte-level BPE 50K | ~4.0-5.0 |
| tiktoken cl100k_base (100K) | ~4.5-6.0 |

These are rough estimates from the literature. The benchmark section specifies how to measure this precisely on Kiara's actual training corpus.

---

## 4. Vocabulary Size Tradeoffs

Vocabulary size is an independent axis from tokenizer algorithm. A BPE tokenizer can be trained with 16K, 32K, 50K, or 100K vocabulary.

### Effect on Model Size

The vocabulary directly affects two parameter groups:
1. Embedding table: vocab_size x hidden_size. At hidden_size=1024 (Kiara base), a 32K vocab = 32M embedding parameters; a 100K vocab = 100M parameters.
2. LM head: another hidden_size x vocab_size matrix. Together, embedding + LM head can represent 20-40% of total model parameters at large vocabulary sizes.

### Effect on Compression

Larger vocabularies generally produce better compression (fewer tokens per character) because more common multi-character sequences are represented as single tokens.

### Effect on the Rare-Token Problem

Larger vocabularies contain more rare tokens. Tokens that appear very few times in the training corpus have undertrained embeddings. This is the token degeneration problem. Models may not learn reliable representations for vocabulary items seen fewer than ~100 times during training.

### Practical Guidance

| Vocab Size | Best For | Drawback |
|---|---|---|
| 16K | Tiny/small models; CPU-only training | Poorer compression; more tokens per sequence |
| 32K | Standard English-heavy models | Mediocre multilingual performance |
| 50K | Balanced English + code + some multilingual | Moderate parameter overhead |
| 64K | Strong multilingual + code coverage | Larger embedding table |
| 100K+ | Full multilingual coverage; maximum compression | Significant parameter cost; many rare tokens |

---

## 5. Unknown-Token Behavior

The UNK token represents a silent failure mode: the tokenizer has encountered a string it cannot represent. Any UNK output means:

1. Information is irreversibly lost.
2. Round-trip correctness (decode(encode(text)) == text) fails.
3. The model has no way to recover the original input.

### By Algorithm

| Algorithm | Can produce UNK? | Fallback strategy |
|---|---|---|
| Character | Yes (for unseen characters) | None |
| Word | Yes (very frequently) | None |
| WordPiece | Yes (for unseen characters not in vocab) | Character-level fallback |
| Unigram | Rarely (if character is in initial vocab) | Character-level fallback |
| BPE (char-level) | Yes (for characters not in initial char vocab) | None if char not in initial set |
| Byte-level BPE | Never | Byte decomposition always succeeds |

### Kiara Requirement

Given Kiara's intended scope (English + code + math + multilingual), producing UNK tokens for any valid UTF-8 input is unacceptable. The tokenizer must be able to encode any input losslessly.

---

## 6. Tokenizer Training Considerations

### Training Corpus Composition

The tokenizer vocabulary will reflect the frequency distribution of the training corpus. Key considerations:

- Balance English, code, and math in the tokenizer training corpus, even if the pretraining corpus has a different balance.
- Include Python, Java, C++, JavaScript, and mathematical notation in the tokenizer training data.
- Include a sample of non-English text if any multilingual capability is desired.
- A tokenizer training corpus of 1-10 GB of representative text is generally sufficient.

### Training Corpus Size vs. Pretraining Corpus Size

Tokenizer training is a one-time, fast computation. It does not need to run on the full pretraining corpus. A representative 1-10 GB sample is sufficient to produce a high-quality vocabulary.

### Normalization and Pre-tokenization

The tokenizer may apply text normalization steps before tokenization:
- Unicode normalization (NFC, NFD, NFKC, NFKD): affects how accented characters and compatibility forms are treated.
- Lowercasing: reduces effective vocabulary but loses information.
- Whitespace handling: whether leading spaces are part of tokens (GPT-2 style) or not.

For Kiara, case-sensitive tokenization is preferred to avoid information loss, especially for code where MyClass and myclass are different identifiers.

### Reproducibility

The tokenizer is a frozen artifact that defines the model forever. It must be:
- Version-controlled: stored in the repository as a JSON or model file with a hash.
- Documented: training corpus, normalization settings, pre-tokenization rules, and vocabulary size must be recorded in docs/decisions.md.
- Reproducible: given the same training corpus and settings, another researcher should be able to reproduce the vocabulary.

---

## 7. Licensing and Reproducibility Considerations

| Tokenizer Library | License | Notes |
|---|---|---|
| HuggingFace tokenizers | Apache 2.0 | Fast Rust implementation; supports BPE, WordPiece, Unigram |
| SentencePiece | Apache 2.0 | Google's library; supports BPE and Unigram; best for multilingual |
| tiktoken (OpenAI) | MIT | Extremely fast; primarily designed for GPT-4 vocabularies; read-only |
| sentencepiece Python | Apache 2.0 | Standard Python bindings for SentencePiece |

Key decision: The tokenizer library is not the same as the tokenizer algorithm. HuggingFace tokenizers supports both BPE and Unigram. SentencePiece is the reference implementation for Unigram.

For Kiara, the preferred libraries are:
- HuggingFace tokenizers for BPE or byte-level BPE experiments.
- SentencePiece if Unigram is selected.

Both are open-source and Apache 2.0 licensed.

---

## 8. Tokenizer Benchmark Specification

This section specifies the benchmark that must be run before selecting Kiara's final tokenizer.

### 8.1 Metrics

The following metrics must be measured for each candidate tokenizer:

| Metric | Definition | Unit |
|---|---|---|
| Tokens per character | len(tokens) / len(text) for a fixed corpus | ratio |
| Tokens per word | len(tokens) / word_count | ratio |
| Compression ratio | Characters per token: 1 / (tokens per character) | chars/token |
| Vocabulary coverage | Fraction of corpus words that map to a single token | % |
| Vocabulary efficiency | Fraction of vocabulary tokens appearing at least K times | % |
| Encoding speed | Tokens encoded per second | tokens/s |
| Decoding speed | Tokens decoded per second | tokens/s |
| Round-trip correctness | decode(encode(text)) == text for all test inputs | pass/fail |
| Unknown token rate | Fraction of inputs that produce at least one UNK | % |
| Code compression ratio | Chars per token on source code corpus | chars/token |
| Math compression ratio | Chars per token on mathematical text | chars/token |
| Multilingual compression ratio | Chars per token across language samples | chars/token |
| Unicode round-trip | Correct encode/decode of emoji, CJK, Arabic, Devanagari | pass/fail |
| Long-word behavior | Avg tokens for words of length >15 characters | tokens/word |
| URL tokenization | Tokens per URL for a sample of URLs | tokens/URL |

### 8.2 Benchmark Dimensions

For each candidate configuration (algorithm + vocabulary size), run the full metric suite over:
- All dataset categories (see Section 9)
- All proposed vocabulary sizes (see Section 10)
- Consistently with a fixed random seed for stochastic methods

### 8.3 Proposed Experiment Matrix

```
Algorithms x Vocab Sizes x Datasets = benchmark cells

Algorithms: [BPE, Byte-BPE, WordPiece, Unigram]
Vocab Sizes: [16K, 32K, 50K, 64K, 100K]
Datasets:    [english, code, math, multilingual, unicode]

Total cells: 4 x 5 x 5 = 100 benchmark configurations
```

Each configuration should produce a row in a benchmark table stored in:
  experiments/tokenizer_benchmark/results.csv

### 8.4 Statistical Notes

- Run encoding speed benchmarks 5 times and report the median to reduce variance.
- Measure on text of consistent length (e.g., fixed 10,000 characters per benchmark sample) for fair comparison.
- Report vocabulary size as the exact number of tokens in the trained vocabulary, which may differ slightly from the target due to minimum frequency cutoffs.

---

## 9. Benchmark Datasets and Sample Inputs

Important: No copyrighted datasets are downloaded. All sample data is either synthetic, excerpted for fair use research purposes, or sourced from open-licensed materials.

### 9.1 Normal English Prose

Source: Project Gutenberg (public domain) or synthetic generated prose.

Sample texts:
- Short sentences (avg 15 words)
- Long paragraphs (avg 200 words)
- Mixed punctuation and contractions

Example sample:
```
The foundation model learns to predict the next token in a sequence.
Training requires massive amounts of compute and carefully curated data.
Kiara is being designed from first principles to understand each component.
```

### 9.2 Technical English

Sample texts:
- API documentation style
- Academic abstract style
- Numerical reports

Example sample:
```
The RMSNorm layer normalizes activations by dividing by the root mean square.
GQA reduces the memory bandwidth requirement of multi-head attention by grouping queries.
SwiGLU activation introduces a gating mechanism: FFN(x) = SiLU(W1 x) times (W2 x).
```

### 9.3 Source Code

Sample texts (Python, JavaScript, C++):

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    return torch.matmul(F.softmax(scores, dim=-1), v)
```

```cpp
template<typename T>
class TransformerBlock {
    int hidden_size_;
    float dropout_rate_;
    std::vector<AttentionHead> heads_;
};
```

### 9.4 Mathematical Notation

Sample texts (LaTeX and Unicode math):

```
For all epsilon > 0, there exists delta > 0 such that |x - a| < delta implies |f(x) - f(a)| < epsilon
\frac{\partial L}{\partial \theta} = -\frac{1}{N} \sum_{i=1}^{N} y_i \log \hat{y}_i
E = mc^2, F = ma
\mathbb{E}_{x ~ p}[f(x)] = \int f(x) p(x) dx
```

### 9.5 Multilingual Text

Sample texts:
- Spanish: El aprendizaje automatico es una rama de la inteligencia artificial.
- French: Les modeles de fondation necessitent des corpus d'entrainement massifs.
- German: Sprachmodelle werden auf grossen Textmengen vortrainiert.
- Japanese: Transformer moderu wa daikiboona koopasu de jizen gakushu saremasu.
- Arabic: (Arabic text for: Machine learning is a branch of artificial intelligence.)
- Hindi: (Hindi text for: Language models are trained at a large scale.)

### 9.6 Rare and Long Words

Sample: Words with length > 12 characters and technical jargon.
```
electroencephalography
deoxyribonucleic acid
superconductivity
immunohistochemistry
counterfactual
anthropomorphization
```

### 9.7 URLs and Symbols

```
https://github.com/kiara-project/kiara/blob/main/kiara/model/transformer.py
https://arxiv.org/abs/1706.03762
https://huggingface.co/datasets/openwebtext
$100.00 +/- 0.05 (p < 0.001)
```

### 9.8 Emoji and Unicode Edge Cases

```
Hello world (with emoji: globe, rocket, brain)
Cafe, naive, resume (with diacritics)
Korean, Chinese, Arabic script samples
Mathematical script letters
Unicode superscript digits: superscript 0-9
Integration symbols: single, double, triple, contour
```

---

## 10. Candidate Vocabulary Sizes

The following vocabulary sizes are proposed for benchmarking. Note: these are proposed experiments. No results have been generated yet.

### 16,384 (16K)

Rationale: Minimal viable vocabulary for a small/tiny model. Reduces embedding table size significantly. Appropriate for the Kiara tiny configuration (tiny.yaml). Compression ratio will be lower than larger vocabularies, producing longer token sequences. Suitable for initial CPU-scale experiments.

Expected tradeoff: Poorer compression -> longer sequences -> less effective context utilization.

Models that use this range: Older generation small models; some character-level hybrid approaches.

---

### 32,768 (32K)

Rationale: The most common choice for modern English-dominant foundation models. LLaMA 1 and LLaMA 2 use 32K. Provides good compression for English and code. Manageable embedding table size at Kiara's hidden dimension (32K x 1024 = 32M parameters).

Expected tradeoff: Good English/code compression. Weaker multilingual coverage.

Models that use this range: LLaMA 1, LLaMA 2, Mistral 7B.

---

### 50,257 (approx. 50K)

Rationale: The vocabulary size of GPT-2 and GPT-3 (specifically 50,257 = 50,000 BPE merges + 256 byte tokens + 1 special token). This is a well-studied vocabulary size with extensive published compression data available for comparison. It represents a natural reference point.

Expected tradeoff: Good English compression; reasonable code coverage.

Models that use this range: GPT-2, GPT-3.

---

### 65,536 (64K)

Rationale: Larger vocabulary for improved compression and multilingual coverage. Gemma uses 256K; Qwen uses 150K; LLaMA 3 uses 128K -- indicating a trend toward larger vocabularies. 64K is a reasonable intermediate that balances parameter cost and coverage.

Expected tradeoff: Better compression ratio; better multilingual handling; larger embedding table.

---

### 100,352 (approx. 100K)

Rationale: GPT-4's tiktoken cl100k_base vocabulary is 100,277 tokens. This vocabulary has been shown to provide excellent compression ratios for English, code, and many other languages. Testing this vocabulary size allows direct comparison with GPT-4-class tokenization efficiency.

Expected tradeoff: Best compression ratio. Largest parameter overhead. Many rare tokens with sparse training signal.

Models that use this range: GPT-4 (cl100k_base, o200k_base).

---

## 11. Recommendation Framework

### Current State

This section records the state of the research decision as of Step 19.

- No tokenizer has been selected.
- No benchmark experiments have been run.
- The architectural decisions recorded in docs/decisions.md include a preliminary BPE entry. This entry must be reviewed after benchmark results are available.

### Current Working Hypothesis

Based on the analysis in this document, byte-level BPE is the current leading hypothesis for the following reasons:

1. Zero unknown tokens: Critical for a model expected to handle code, math, and non-English text.
2. Proven at scale: Used by GPT-2, GPT-3, GPT-4, RoBERTa, BLOOM, Falcon, and Qwen.
3. Strong tooling: HuggingFace tokenizers provides a fast, production-ready implementation.
4. Context window efficiency: At 32K-100K vocabulary, byte-level BPE provides good compression.

However, this hypothesis must not be treated as a final decision. Unigram (SentencePiece) has theoretical advantages for multilingual coverage and stochastic regularization that may prove significant in practice.

### Evidence Required Before Final Decision

The following evidence must be gathered before locking in the tokenizer:

1. Compression benchmark results for byte-level BPE and Unigram across all dataset categories in Section 9.
2. Round-trip correctness confirmed across all Unicode edge cases.
3. Unknown token rate confirmed to be 0% for byte-level BPE.
4. Code compression ratio comparison between BPE and Unigram.
5. Multilingual compression ratio comparison.
6. Encoding/decoding speed benchmarks.
7. Vocabulary efficiency analysis: what fraction of each vocabulary is well-represented in a sample training corpus.

### Preferred Initial Vocabulary Sizes to Benchmark

For the first benchmark run, focus on:

1. 32K -- covers the LLaMA-class baseline; comparable to the current base.yaml vocab_size: 32000.
2. 64K -- reasonable extension; tests whether larger vocabulary materially improves compression.
3. 100K -- tests GPT-4-class compression; provides ceiling reference.

The 16K run is recommended only for Kiara tiny experiments and can be deprioritized in the first round.

### Preferred Tokenizer Families for First Benchmark

1. Byte-level BPE (primary candidate)
2. Unigram / SentencePiece (primary alternative)
3. Standard BPE (for comparison baseline against byte-level BPE)

WordPiece and character tokenization are low priority for the first benchmark round given their known limitations at foundation model scale.

### Decision Gate

The final tokenizer decision should be made in a dedicated entry in docs/decisions.md with the following structure:

```
Decision 003: Tokenizer Algorithm
Status: Pending
Required: Benchmark results from experiments/tokenizer_benchmark/
```

The tokenizer must be frozen before any model weights are initialized. Once training begins, changing the tokenizer requires restarting from scratch.

---

## References

The following documents and resources informed this research:

- Sennrich, R., Haddow, B., Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. ACL 2016. (Original BPE for NLP paper)
- Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI. (GPT-2; byte-level BPE)
- Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. ACL 2018. (Unigram / SentencePiece)
- Kudo, T., Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.
- Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. (WordPiece)
- HuggingFace Tokenizers documentation: https://huggingface.co/docs/tokenizers
- OpenAI tiktoken: https://github.com/openai/tiktoken
- Touvron, H. et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. (BPE at 32K)
- Existing Kiara research documents: docs/research/ (Claude, GPT, LLaMA, Mistral, Gemini, DeepSeek, Phi, Qwen, Grok)
- Kiara model specification: docs/model_spec.md
- Kiara architectural decisions: docs/decisions.md
- Kiara Phase 2 plan: docs/phase2_core_infrastructure.md
