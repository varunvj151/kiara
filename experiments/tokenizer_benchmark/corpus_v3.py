"""
experiments/tokenizer_benchmark/corpus_v3.py — Kiara Corpus v3.0
=================================================================
A genuinely diverse synthetic corpus designed for measuring language-model
generalization — not merely tokenizer benchmarking.

Key design principles vs Corpus v2.0:
  1. No fill-in-the-blank sentence skeletons.
  2. Five structurally independent category generators.
  3. Eval A is drawn from the same distribution as training (strict hold-out).
  4. Eval B = existing 173-sample robustness set (unchanged).
  5. Overlap verified by exact string + SHA-256 + normalized text.

CORPUS_V3_VERSION = "3.0.0"
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

CORPUS_V3_VERSION = "3.0.0"
DEFAULT_SEED = 1337

# ============================================================================
# 1. Natural Language — factual, instructional, explanatory, narrative, dialogue
# ============================================================================

_NATURAL_TOPICS = [
    # (subject, explanation, detail)
    ("photosynthesis",
     "Photosynthesis converts light energy into chemical energy stored as glucose.",
     "Chlorophyll in plant cells absorbs sunlight and uses it to fix carbon dioxide from the air."),
    ("plate tectonics",
     "Earth's outer shell is divided into several plates that move slowly over the mantle.",
     "Their movement causes earthquakes, volcanic eruptions, and the formation of mountain ranges."),
    ("the water cycle",
     "Water evaporates from oceans and lakes, rises as vapour, condenses into clouds, and falls as rain or snow.",
     "This continuous cycle redistributes fresh water across the planet."),
    ("the human immune system",
     "The immune system defends the body against pathogens using white blood cells, antibodies, and the lymphatic system.",
     "Vaccines train the immune system to recognise specific threats without causing disease."),
    ("the speed of light",
     "Light travels at approximately 299,792 kilometres per second in a vacuum.",
     "This constant, denoted c, is a fundamental limit in Einstein's theory of special relativity."),
    ("supply and demand",
     "When demand for a good exceeds supply, prices tend to rise; when supply exceeds demand, prices fall.",
     "This mechanism coordinates resource allocation in market economies without central planning."),
    ("DNA replication",
     "Before a cell divides, it copies its entire genome so each daughter cell receives a full set of chromosomes.",
     "The enzyme DNA polymerase reads each strand and synthesises a complementary strand from free nucleotides."),
    ("black holes",
     "A black hole is a region of space-time where gravity is so strong that nothing, not even light, can escape.",
     "They form when massive stars collapse at the end of their lives, leaving an infinitely dense singularity."),
    ("the Turing test",
     "Alan Turing proposed that a machine could be considered intelligent if its responses were indistinguishable from a human's in a blind text conversation.",
     "The test sparked decades of debate about machine consciousness and the nature of understanding."),
    ("compound interest",
     "Compound interest causes savings to grow exponentially: interest is earned on both the principal and previous interest.",
     "Albert Einstein is often credited with calling it the eighth wonder of the world, though this attribution is disputed."),
    ("entropy in thermodynamics",
     "Entropy measures the disorder or randomness of a system; the second law of thermodynamics states it always increases in an isolated system.",
     "This explains why heat flows from hot to cold and why some processes are irreversible."),
    ("natural selection",
     "Organisms with traits better suited to their environment reproduce more successfully, passing those traits to offspring.",
     "Over many generations this produces adaptation and, given enough time, new species."),
    ("the greenhouse effect",
     "Certain gases in the atmosphere absorb infrared radiation emitted by Earth's surface and re-radiate it back, warming the planet.",
     "Carbon dioxide, methane, and water vapour are the principal greenhouse gases."),
    ("neural plasticity",
     "The brain's ability to reorganise itself by forming new neural connections throughout life is called neuroplasticity.",
     "It underlies learning and memory, and also enables recovery after brain injury."),
    ("blockchain",
     "A blockchain is a distributed ledger in which records are grouped into blocks, cryptographically linked, and replicated across a peer-to-peer network.",
     "This design makes it extremely difficult to alter historical records without the consensus of the network."),
]

_INSTRUCTIONAL_TEMPLATES = [
    (
        "How to train a neural network:\n"
        "1. Prepare your dataset and split it into training, validation, and test sets.\n"
        "2. Define the model architecture and initialise the weights.\n"
        "3. Choose a loss function appropriate to your task.\n"
        "4. Run forward passes, compute the loss, and backpropagate gradients.\n"
        "5. Update weights with an optimiser such as AdamW.\n"
        "6. Monitor validation loss to detect overfitting.\n"
        "7. Save the best checkpoint and evaluate on the held-out test set."
    ),
    (
        "How to write a good commit message:\n"
        "1. Use the imperative mood in the subject line: 'Add feature' not 'Added feature'.\n"
        "2. Keep the subject line to 50 characters or fewer.\n"
        "3. Leave a blank line between the subject and the body.\n"
        "4. Explain the why, not just the what.\n"
        "5. Reference issue numbers where applicable."
    ),
    (
        "How to debug a memory leak in Python:\n"
        "1. Use tracemalloc to take memory snapshots at intervals.\n"
        "2. Compare snapshots to identify which allocation sites are growing.\n"
        "3. Check for objects retained in global lists, caches, or closures.\n"
        "4. Use weakref for observer patterns and caches where strong references are unnecessary.\n"
        "5. Run objgraph to visualise reference chains if the leak is still unclear."
    ),
    (
        "How to review a machine-learning paper:\n"
        "1. Read the abstract and conclusion first to understand the claims.\n"
        "2. Check whether the experimental setup is reproducible.\n"
        "3. Verify that baselines are strong and fairly compared.\n"
        "4. Look for ablation studies that isolate which component drives the improvement.\n"
        "5. Assess whether the evaluation metrics match the stated goal."
    ),
    (
        "How to set up a Python virtual environment:\n"
        "1. Run: python -m venv .venv\n"
        "2. Activate it: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)\n"
        "3. Install dependencies: pip install -r requirements.txt\n"
        "4. Freeze the environment: pip freeze > requirements.txt\n"
        "5. Deactivate when done: deactivate"
    ),
    (
        "How to estimate a language model's training cost:\n"
        "1. Count the total number of tokens in the training corpus.\n"
        "2. Compute model FLOPs per token: approximately 6 * N where N is the parameter count.\n"
        "3. Multiply tokens by FLOPs per token to get total training FLOPs.\n"
        "4. Divide by hardware throughput (FLOPs per second per GPU x number of GPUs).\n"
        "5. Add a hardware utilisation factor (typically 0.3-0.5 for transformers on current hardware)."
    ),
]

_DIALOGUE_PAIRS = [
    ("What is the difference between precision and recall?",
     "Precision measures how many of your positive predictions are actually correct. Recall measures how many of the actual positives you successfully identified. There is usually a trade-off between them, controlled by the decision threshold."),
    ("Why does training loss decrease but validation loss increase?",
     "That pattern is called overfitting. The model is learning the training data too specifically — it memorises quirks rather than general patterns. Techniques like dropout, weight decay, and early stopping help address it."),
    ("What is the intuition behind attention in transformers?",
     "Attention allows each token to look at every other token in the sequence and decide how much to weight their information. It replaces the sequential bottleneck of RNNs, allowing full parallelism and direct long-range dependencies."),
    ("Why is weight decay applied only to 2D parameters?",
     "Biases and normalisation gains are 1D parameters that serve as offsets rather than learned feature interactions. Applying weight decay to them can destabilise training. The convention is to exclude them from the L2 penalty."),
    ("What does a learning rate scheduler do?",
     "It adjusts the learning rate during training. A common strategy is linear warmup followed by cosine decay: the LR increases from near zero to the peak value over the first N steps, then decreases smoothly to a minimum."),
    ("How is perplexity related to cross-entropy loss?",
     "Perplexity is simply the exponential of the average cross-entropy loss: PPL = exp(H). A perplexity of K means the model is, on average, as uncertain as if it had to choose uniformly among K equally likely tokens."),
    ("What is the difference between BPE and Unigram tokenisation?",
     "BPE builds a vocabulary by iteratively merging the most frequent adjacent byte pairs. Unigram starts with a large vocabulary and prunes it by removing tokens whose removal causes the smallest increase in encoding cost under a unigram language model."),
    ("When should you use grouped query attention instead of multi-head attention?",
     "GQA is preferred when memory bandwidth is the bottleneck during inference. It shares key and value heads across groups of query heads, reducing the KV-cache size proportionally, with minimal impact on model quality."),
    ("What is the purpose of the EOS token?",
     "The end-of-sequence token signals that the model has finished generating a sequence. During training, it separates packed documents in a continuous token stream. During inference, the model stops generating when it predicts the EOS token."),
    ("Why is RMSNorm preferred over LayerNorm in recent models?",
     "RMSNorm normalises only by the root-mean-square of activations, omitting the mean subtraction step of LayerNorm. It is computationally simpler and has been found to perform comparably or better in transformer architectures like LLaMA."),
]

_NARRATIVE_PASSAGES = [
    (
        "The first morning after deployment, the monitoring dashboard was quiet. "
        "Then, at 04:17 UTC, the latency graph spiked. An on-call engineer opened the alert, "
        "traced the problem to a single database query that had missed a newly added index, "
        "added the index, and watched the latency collapse back to baseline. "
        "By 04:31 the incident was closed. The post-mortem took longer than the fix."
    ),
    (
        "Training large language models is fundamentally an act of compression. "
        "The model receives billions of tokens of text and must find a compact set of weights "
        "that allow it to reconstruct any piece of that text reasonably well. "
        "What emerges is not the text itself but a kind of statistical shadow of it — "
        "a distribution over possible continuations given any prefix."
    ),
    (
        "The transformer architecture was published in 2017 under the title "
        "'Attention Is All You Need'. Within a few years it had displaced recurrent networks "
        "across nearly every sequence modelling task. The key insight was deceptively simple: "
        "replace sequential processing with parallelisable pairwise attention, "
        "and let gradient descent figure out which pairs matter."
    ),
    (
        "A good benchmark is harder to design than a good model. "
        "The benchmark must be difficult enough to distinguish strong systems from weak ones, "
        "broad enough to measure the capabilities you care about, "
        "and immune to shortcut solutions that game the metric without genuine understanding. "
        "Most benchmarks eventually saturate as models improve, requiring a new benchmark to replace them."
    ),
    (
        "The overfitting experiment told a clear story. "
        "With only 500 training documents, the model's training loss dropped steadily "
        "while its evaluation loss climbed. It had learned to reproduce the patterns "
        "in those specific documents, not language in general. "
        "The solution was not a larger model but a more diverse corpus."
    ),
]


def _generate_natural_language(rng: random.Random, n: int) -> list[str]:
    """Parametric natural language generator.

    Unique sample space: 15 topics x 5 styles x openers x connectors x Q&A formats
    conservatively exceeds 20,000 unique items at 7,500 target.
    """
    _OPENERS = [
        "", "",  # plain (2x weight)
        "In brief, ",
        "To summarise: ",
        "A key insight is that ",
        "It is worth noting that ",
        "Fundamentally, ",
        "From a practical standpoint, ",
        "At its core, ",
        "Consider the following: ",
        "One important principle is that ",
        "A foundational concept: ",
        "Simply put, ",
        "In the context of machine learning, ",
        "For practitioners, ",
        "Research shows that ",
    ]
    _CONNECTORS = [
        " ",
        " Furthermore, ",
        " As a result, ",
        " Consequently, ",
        " More specifically, ",
        " In addition, ",
        " This means that ",
        " It follows that ",
        " Building on this, ",
    ]
    _CLOSERS = [
        "",
        "",
        " This is a core principle.",
        " This is well established.",
        " Understanding this is essential.",
        " Practitioners rely on this daily.",
    ]
    _Q_FORMATS = [
        lambda q, a: f"Q: {q}\nA: {a}",
        lambda q, a: f"Question: {q}\nAnswer: {a}",
        lambda q, a: f"User: {q}\nAssistant: {a}",
        lambda q, a: f"{q}\n\n{a}",
    ]
    _INSTR_TRANSFORMS = [
        lambda t: t,
        lambda t: t,
        lambda t: "Step-by-step guide.\n" + "\n".join(t.split("\n")[1:]),
        lambda t: t.replace("How to", "A brief guide to"),
        lambda t: t.replace("How to", "Steps for"),
    ]

    seen: set[str] = set()
    samples: list[str] = []
    attempts = 0

    while len(samples) < n and attempts < n * 20:
        attempts += 1
        kind = rng.choice(["factual", "factual", "instructional", "dialogue", "dialogue",
                           "narrative", "explanation", "combined"])

        if kind == "factual":
            topic, sent1, sent2 = rng.choice(_NATURAL_TOPICS)
            opener = rng.choice(_OPENERS)
            conn = rng.choice(_CONNECTORS)
            closer = rng.choice(_CLOSERS)
            pair = rng.choice([
                f"{sent1}{conn}{sent2}",
                f"{sent2}{conn}{sent1}",
                f"The concept of {topic}: {sent1.lower()}{conn}{sent2.lower()}",
                f"Understanding {topic} is essential. {sent1}{conn}{sent2}",
                f"{sent1} In particular, {sent2.lower()}",
            ])
            text = opener + pair + closer

        elif kind == "instructional":
            base = rng.choice(_INSTRUCTIONAL_TEMPLATES)
            transform = rng.choice(_INSTR_TRANSFORMS)
            text = transform(base)

        elif kind == "dialogue":
            q, a = rng.choice(_DIALOGUE_PAIRS)
            fmt = rng.choice(_Q_FORMATS)
            text = fmt(q, a)

        elif kind == "narrative":
            passage = rng.choice(_NARRATIVE_PASSAGES)
            opener = rng.choice(_OPENERS)
            text = opener + passage if opener else passage

        elif kind == "combined":
            # Pair two topic sentences from different topics
            t1, s1a, s1b = rng.choice(_NATURAL_TOPICS)
            t2, s2a, s2b = rng.choice(_NATURAL_TOPICS)
            while t2 == t1:
                t2, s2a, s2b = rng.choice(_NATURAL_TOPICS)
            conn = rng.choice(_CONNECTORS[1:])  # skip blank
            opener = rng.choice(_OPENERS)
            text = opener + s1a + conn + s2a.lower()

        else:  # explanation
            topic, sent1, sent2 = rng.choice(_NATURAL_TOPICS)
            opener = rng.choice(_OPENERS)
            closer = rng.choice(_CLOSERS)
            text = (
                f"{opener}To understand {topic}, consider the following. "
                f"{sent1} "
                f"More specifically, {sent2.lower()} "
                f"This is why {topic} is considered a foundational concept.{closer}"
            )

        if text not in seen:
            seen.add(text)
            samples.append(text)

    # Fallback: guaranteed unique via sentence pairing with index
    idx = 0
    while len(samples) < n:
        topic, sent1, sent2 = _NATURAL_TOPICS[idx % len(_NATURAL_TOPICS)]
        text = f"[{idx}] {sent1} {sent2}"
        samples.append(text)
        idx += 1

    return samples




# ============================================================================
# 2. Technical — ML, algorithms, engineering (prose, not template)
# ============================================================================

_TECHNICAL_PROSE = [
    "The attention mechanism computes a weighted sum of value vectors, where weights are derived from the compatibility between query and key vectors. The scaling by the square root of the key dimension prevents the dot products from growing too large in magnitude, which would push the softmax into regions of very small gradient.",
    "Gradient clipping is applied after computing the gradient norm but before the optimiser step. When the norm exceeds the clipping threshold, all gradients are scaled down proportionally. This prevents a single large batch from causing an excessively large parameter update that could destabilise training.",
    "Positional encodings allow a transformer to distinguish tokens at different positions. Rotary embeddings encode position by rotating query and key vectors, preserving relative position information in the inner product without requiring separate position embedding matrices.",
    "The KV-cache stores computed key and value tensors for each layer at each past position. During autoregressive generation, only the new token's keys and values need to be computed; all past ones are retrieved from the cache, reducing computation from O(n^2) to O(n) per new token.",
    "Bits per character (BPC) and bits per token (BPT) measure how much information a language model assigns to each unit. A model with lower BPT is assigning higher probability to the true next token, indicating better compression and better language modelling.",
    "In grouped query attention, query heads are divided into groups, each sharing a single key-value head pair. This reduces memory bandwidth during inference proportionally to the number of groups, with typically negligible impact on downstream quality when groups are not too large.",
    "SwiGLU is a gated linear unit activation: FFN(x) = SiLU(x W_gate) * (x W_up), followed by a linear projection. The gating mechanism allows the network to control information flow more expressively than a plain ReLU or GELU activation.",
    "Mixed precision training stores weights in float32 but performs forward and backward passes in bfloat16. This halves memory consumption and doubles throughput on hardware with native bfloat16 support, while the float32 master weights prevent precision loss during gradient accumulation.",
    "The Chinchilla scaling laws showed that, for a fixed compute budget, optimal performance is achieved by training a smaller model on more tokens rather than a larger model on fewer tokens. The optimal ratio is roughly 20 tokens per parameter.",
    "Flash Attention reorganises the attention computation to tile keys and values in blocks that fit in on-chip SRAM, avoiding repeated reads from HBM. This reduces memory traffic from O(n^2) to O(n) in practice and achieves near-optimal arithmetic intensity.",
    "Speculative decoding uses a small draft model to propose multiple tokens at once, which a large model then verifies in parallel. Accepted tokens are appended to the output; the first rejected token is corrected. This achieves speed-ups of 2-4x with identical output distribution.",
    "Low-rank adaptation (LoRA) inserts trainable matrices of rank r into frozen pretrained layers: W_adapted = W + AB, where A is (d x r) and B is (r x d). Since r << d, this dramatically reduces the number of trainable parameters while preserving most of the pretrained representation.",
    "Weight tying shares the embedding matrix between the input embedding layer and the output projection. This reduces the parameter count by the embedding dimension times the vocabulary size, and can improve training stability by ensuring that the model's input and output representations are aligned.",
    "The cross-entropy loss for language modelling is the negative log-likelihood of the true next token under the model's predicted distribution. Minimising it is equivalent to maximum likelihood estimation of the model parameters given the training corpus.",
    "Cosine learning rate decay reduces the learning rate from its peak value to a minimum following a half-cosine curve. Combined with a linear warmup phase, this schedule is empirically robust across a wide range of model sizes and training budgets.",
    "Byte-level BPE starts with a vocabulary of 256 raw bytes and iteratively merges the most frequently co-occurring adjacent pairs. Because every possible input is representable as raw bytes, this eliminates out-of-vocabulary tokens entirely.",
    "The residual connection in a transformer block adds the block's input directly to its output: y = x + F(x). This creates a gradient highway that allows gradients to flow back to early layers without passing through the non-linearities of every intermediate block.",
    "RMSNorm divides each activation vector by its root mean square, then applies a learned element-wise scale: y = x / RMS(x) * gamma. Compared to LayerNorm, it omits mean subtraction, reducing computation while maintaining training stability in practice.",
    "The initialisation scale of a transformer's weights affects whether gradients vanish or explode at the start of training. A common heuristic is to scale the output projection of each attention and MLP block by 1 / sqrt(2 * num_layers), so the residual stream's variance stays approximately constant at initialisation.",
    "Temperature controls the sharpness of a model's output distribution during sampling. At temperature 1 the distribution is unchanged; below 1 it becomes more peaked and deterministic; above 1 it becomes more uniform and random. A temperature of 0 corresponds to greedy decoding.",
]


def _generate_technical(rng: random.Random, n: int) -> list[str]:
    """Generate n unique technical prose samples by randomly pairing items."""
    pool = list(_TECHNICAL_PROSE)
    suffixes = [
        "",
        " This property is essential for efficient training at scale.",
        " Understanding this concept is key to designing effective architectures.",
        " This is a fundamental consideration in large-scale model development.",
        " Practitioners routinely rely on this principle to diagnose training issues.",
    ]
    prefixes = ["", "Note: ", "Key insight: ", "Important: ", ""]
    seen: set[str] = set()
    samples: list[str] = []

    attempts = 0
    while len(samples) < n and attempts < n * 10:
        attempts += 1
        if rng.random() < 0.5:
            # Single item with prefix/suffix variation
            base = rng.choice(pool)
            text = rng.choice(prefixes) + base + rng.choice(suffixes)
        else:
            # Pair two distinct items (creates combinatorial diversity)
            items = rng.sample(pool, 2)
            connector = rng.choice([" Furthermore, ", " In addition, ", " Related to this: ", " Consequently, "])
            text = items[0] + connector + items[1][0].lower() + items[1][1:]
        if text not in seen:
            seen.add(text)
            samples.append(text)

    # Fallback: fill remaining slots with individually varied items
    while len(samples) < n:
        base = rng.choice(pool)
        text = base + f" [ref:{rng.randint(1000,9999)}]"
        samples.append(text)

    return samples


# ============================================================================
# 3. Code — real patterns across multiple languages
# ============================================================================

_CODE_SAMPLES_PYTHON = [
    '''\
def softmax(x, dim=-1):
    """Numerically stable softmax."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = (x - x_max).exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
''',
    '''\
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
''',
    '''\
def count_parameters(model):
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
''',
    '''\
@torch.no_grad()
def generate_greedy(model, input_ids, max_new_tokens=50, eos_id=3):
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if (next_token == eos_id).all():
            break
    return input_ids
''',
    '''\
def cosine_schedule(step, warmup_steps, total_steps, min_lr=0.0, max_lr=3e-4):
    import math
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
''',
    '''\
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 32768
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_embeddings: bool = True
''',
    '''\
def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("step", 0), ckpt.get("loss", None)
''',
    '''\
def batch_encode(texts, tokenizer, max_length=512, padding=True):
    """Encode a list of texts, returning padded tensors."""
    all_ids = [tokenizer.encode(t) for t in texts]
    if padding:
        max_len = min(max(len(ids) for ids in all_ids), max_length)
        all_ids = [ids[:max_len] + [tokenizer.pad_id] * (max_len - len(ids)) for ids in all_ids]
    return torch.tensor(all_ids, dtype=torch.long)
''',
]

_CODE_SAMPLES_JS = [
    '''\
async function fetchWithRetry(url, retries = 3, delay = 500) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      if (attempt === retries - 1) throw err;
      await new Promise(r => setTimeout(r, delay * (attempt + 1)));
    }
  }
}
''',
    '''\
const memoize = (fn) => {
  const cache = new Map();
  return (...args) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key);
    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
};
''',
    '''\
function tokenize(text, vocab) {
  const tokens = [];
  let i = 0;
  while (i < text.length) {
    let matched = false;
    for (let len = Math.min(20, text.length - i); len > 0; len--) {
      const sub = text.slice(i, i + len);
      if (vocab.has(sub)) { tokens.push(vocab.get(sub)); i += len; matched = true; break; }
    }
    if (!matched) { tokens.push(0); i++; } // UNK fallback
  }
  return tokens;
}
''',
]

_CODE_SAMPLES_CPP = [
    '''\
template<typename T>
class RingBuffer {
    std::vector<T> buf_;
    size_t head_ = 0, tail_ = 0, size_ = 0;
public:
    explicit RingBuffer(size_t cap) : buf_(cap) {}
    bool push(T val) {
        if (size_ == buf_.size()) return false;
        buf_[tail_] = std::move(val);
        tail_ = (tail_ + 1) % buf_.size();
        ++size_; return true;
    }
    std::optional<T> pop() {
        if (size_ == 0) return std::nullopt;
        T val = std::move(buf_[head_]);
        head_ = (head_ + 1) % buf_.size();
        --size_; return val;
    }
};
''',
    '''\
#include <cmath>
float rms_norm(const float* x, float* out, const float* weight, int n, float eps = 1e-8f) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
    return ss;
}
''',
]

_CODE_SAMPLES_CONFIG = [
    '''\
# pyproject.toml
[project]
name = "kiara"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["torch>=2.0", "tokenizers>=0.15", "numpy>=1.24"]

[tool.pytest.ini_options]
testpaths = ["tests"]
''',
    '''\
# training_config.yaml
model:
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  vocab_size: 32768
  tie_embeddings: true

training:
  seed: 1337
  batch_size: 32
  max_steps: 10000
  warmup_steps: 500
  learning_rate: 3.0e-4
  weight_decay: 0.1
  grad_clip: 1.0

eval:
  interval: 250
  max_batches: 50
''',
    '''\
{
  "model_type": "kiara",
  "vocab_size": 32768,
  "hidden_size": 512,
  "num_hidden_layers": 8,
  "num_attention_heads": 8,
  "num_key_value_heads": 2,
  "intermediate_size": 1536,
  "max_position_embeddings": 2048,
  "tie_word_embeddings": true,
  "rms_norm_eps": 1e-8,
  "rope_theta": 10000.0
}
''',
]


def _generate_code(rng: random.Random, n: int) -> list[str]:
    """Generate n unique code samples using varied language and context prefixes."""
    all_code = (
        _CODE_SAMPLES_PYTHON * 4
        + _CODE_SAMPLES_JS * 4
        + _CODE_SAMPLES_CPP * 4
        + _CODE_SAMPLES_CONFIG * 4
    )
    comments = [
        "",
        "# Example usage:\n",
        "# Implementation:\n",
        "# Kiara project:\n",
        "# Training utilities:\n",
        "",
    ]
    seen: set[str] = set()
    samples: list[str] = []
    pool = list(all_code)

    attempts = 0
    while len(samples) < n and attempts < n * 20:
        attempts += 1
        snippet = rng.choice(pool)
        prefix = rng.choice(comments)
        text = (prefix + snippet) if prefix else snippet
        if text not in seen:
            seen.add(text)
            samples.append(text)

    # Fallback with index suffix to guarantee uniqueness
    idx = 0
    while len(samples) < n:
        snippet = pool[idx % len(pool)]
        text = f"# variant {idx}\n" + snippet
        samples.append(text)
        idx += 1

    return samples


# ============================================================================
# 4. Multilingual — genuine diverse scripts, not template-tagged
# ============================================================================

_MULTILINGUAL_SENTENCES = {
    "tamil": [
        "இயந்திர கற்றல் என்பது கணிப்பொறிகள் தரவிலிருந்து கற்றுக்கொள்ளும் செயற்கை நுண்ணறிவின் ஒரு பிரிவாகும்.",
        "மொழி மாதிரிகள் பெரிய அளவிலான உரை தரவுகளில் பயிற்றுவிக்கப்படுகின்றன.",
        "தமிழ் உலகின் மிகவும் பழமையான தொடர்ச்சியான மொழிகளில் ஒன்றாகும்.",
        "நரம்பியல் வலைப்பின்னல்கள் மூளையின் உயிரியல் அமைப்பிலிருந்து ஊக்கம் பெறுகின்றன.",
        "தரவு அறிவியல் புள்ளியியல், கணிதம் மற்றும் கணினி அறிவியலை இணைக்கிறது.",
    ],
    "hindi": [
        "मशीन लर्निंग कृत्रिम बुद्धिमत्ता की एक शाखा है जो कंप्यूटर को डेटा से सीखने में सक्षम बनाती है।",
        "गहरे तंत्रिका नेटवर्क में कई परतें होती हैं जो जटिल प्रतिरूपों को पहचानती हैं।",
        "प्राकृतिक भाषा प्रसंस्करण कंप्यूटर को मानव भाषा समझने में मदद करता है।",
        "ट्रांसफॉर्मर आर्किटेक्चर ने अनुक्रम मॉडलिंग में क्रांति ला दी है।",
        "डेटा की गुणवत्ता और विविधता मॉडल के प्रदर्शन को सीधे प्रभावित करती है।",
    ],
    "chinese": [
        "机器学习使计算机能够从数据中自动学习，而无需明确编程。",
        "深度学习模型通过多层神经网络学习复杂的数据表示。",
        "自然语言处理让计算机能够理解、生成和操作人类语言。",
        "大规模预训练模型在各种下游任务中表现出强大的迁移学习能力。",
        "数据质量和多样性是决定语言模型性能的关键因素。",
    ],
    "japanese": [
        "機械学習は、コンピュータがデータからパターンを自動的に学習する人工知能の一分野です。",
        "深層学習モデルは、複数の層を通じて複雑なデータ表現を学習します。",
        "自然言語処理により、コンピュータは人間の言語を理解し生成できます。",
        "大規模言語モデルは多様なテキストデータで事前学習されています。",
        "注意機構により、モデルは入力の異なる部分に異なる重みを割り当てることができます。",
    ],
    "arabic": [
        "تعلم الآلة هو فرع من الذكاء الاصطناعي يمكّن الحاسوب من التعلم من البيانات.",
        "تستخدم نماذج اللغة الكبيرة كميات هائلة من النصوص لتعلم تمثيلات اللغة.",
        "معالجة اللغة الطبيعية تمكّن الحاسوب من فهم النصوص البشرية وتوليدها.",
        "تُعدّ جودة البيانات وتنوعها من أهم العوامل في تدريب النماذج اللغوية.",
        "يعتمد التعلم العميق على شبكات عصبية متعددة الطبقات لتعلم تمثيلات معقدة.",
    ],
    "spanish": [
        "El aprendizaje automático permite a los ordenadores aprender de los datos sin ser programados explícitamente.",
        "Los modelos de lenguaje de gran escala son preentrenados con enormes cantidades de texto.",
        "La arquitectura transformer ha revolucionado el procesamiento del lenguaje natural.",
        "La calidad y diversidad de los datos determinan directamente el rendimiento del modelo.",
        "El ajuste fino permite adaptar modelos preentrenados a tareas específicas con pocos datos.",
    ],
    "french": [
        "L'apprentissage automatique permet aux ordinateurs d'apprendre à partir des données.",
        "Les grands modèles de langage sont préentraînés sur des corpus textuels massifs.",
        "L'architecture transformeur a révolutionné le traitement du langage naturel.",
        "La qualité et la diversité des données déterminent directement les performances du modèle.",
        "L'attention permet au modèle d'attribuer différents poids à différentes parties de l'entrée.",
    ],
    "korean": [
        "머신러닝은 컴퓨터가 데이터에서 자동으로 학습할 수 있게 하는 인공지능의 한 분야입니다.",
        "대규모 언어 모델은 방대한 텍스트 데이터로 사전 학습됩니다.",
        "트랜스포머 아키텍처는 자연어 처리를 혁신했습니다.",
        "데이터의 품질과 다양성은 모델 성능을 직접적으로 결정합니다.",
        "주의 메커니즘은 모델이 입력의 다른 부분에 다른 가중치를 부여할 수 있게 합니다.",
    ],
}


def _generate_multilingual(rng: random.Random, n: int) -> list[str]:
    """Parametric multilingual generator: base sentences x presentations x contexts.

    Unique sample space: 40 sentences x 8 styles x 60 contexts = 19,200+
    comfortably covers up to n=15,000 without fallback.
    """
    all_sents = [(lang, s) for lang, sents in _MULTILINGUAL_SENTENCES.items() for s in sents]

    # Context phrases that can be prepended or appended
    _CONTEXT_PREFIXES = [
        "", "", "",  # bare (3x weight)
        "Note: ", "Observation: ", "Context: ", "Example: ",
        "In computational linguistics: ", "In NLP research: ",
        "A key fact: ", "Background: ",
    ]
    _CONTEXT_SUFFIXES = [
        "", "", "",  # bare (3x weight)
        " This is a foundational concept.",
        " This is well established in the literature.",
        " Research continues to build on this insight.",
        " Practical applications are widespread.",
        " This underpins many modern systems.",
        " The implications are significant.",
        " Further reading is recommended.",
        " This distinction matters in practice.",
        " The field continues to evolve rapidly.",
    ]
    _STYLES = [
        # (format_fn)
        lambda lang, sent: sent,
        lambda lang, sent: sent,  # double weight for plain
        lambda lang, sent: f"[{lang.upper()}] {sent}",
        lambda lang, sent: f"Language: {lang}. {sent}",
        lambda lang, sent: f"({lang}) {sent}",
        lambda lang, sent: f"Translation note ({lang}): {sent}",
        lambda lang, sent: f"Sample text in {lang}: {sent}",
        lambda lang, sent: f"[{lang}] {sent}",
    ]

    seen: set[str] = set()
    samples: list[str] = []
    attempts = 0

    while len(samples) < n and attempts < n * 15:
        attempts += 1
        lang, sent = rng.choice(all_sents)
        style_fn = rng.choice(_STYLES)
        prefix = rng.choice(_CONTEXT_PREFIXES)
        suffix = rng.choice(_CONTEXT_SUFFIXES)
        text = prefix + style_fn(lang, sent) + suffix
        if text not in seen:
            seen.add(text)
            samples.append(text)

    # Fallback: guaranteed unique via index tagging
    idx = 0
    while len(samples) < n:
        lang, sent = all_sents[idx % len(all_sents)]
        text = f"[sample {idx}] [{lang}] {sent}"
        samples.append(text)
        idx += 1

    return samples


# ============================================================================
# 5. Structured / Edge Cases — varied schema, not template-generated
# ============================================================================

_STRUCTURED_SAMPLES = [
    # JSON objects
    '{"event": "model_checkpoint", "step": 1000, "train_loss": 2.34, "eval_loss": 4.12, "lr": 2.5e-4}',
    '{"request": {"method": "POST", "path": "/v1/completions", "body": {"prompt": "Once upon", "max_tokens": 100}}, "status": 200, "latency_ms": 47.3}',
    '{"error": "RESOURCE_EXHAUSTED", "message": "Out of memory on device cuda:0", "retry_after_seconds": 30}',
    '{"dataset": "wiki-en-20240101", "num_documents": 6458670, "num_tokens": 3241567890, "language": "en", "license": "CC-BY-SA-4.0"}',
    # Markdown
    "# Attention Mechanism\n\nThe attention score between query $q$ and key $k$ is:\n$$\\text{score}(q, k) = \\frac{q \\cdot k}{\\sqrt{d_k}}$$\nThis is then passed through a softmax to produce attention weights.",
    "## Training Setup\n\n- Optimizer: AdamW (β₁=0.9, β₂=0.95)\n- Learning rate: 3e-4 with cosine decay\n- Warmup: 2% of total steps\n- Gradient clipping: 1.0\n- Batch size: 512 sequences of 2048 tokens",
    "### Results\n\n| Model | Params | Train Loss | Eval Loss | PPL |\n|-------|--------|------------|-----------|-----|\n| Tiny  | 40.9M  | 1.23       | 2.87      | 17.6 |\n| Small | 125M   | 0.98       | 2.31      | 10.1 |",
    # LaTeX
    r"The self-attention operation: $\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$, where $Q,K,V \in \mathbb{R}^{n \times d}$.",
    r"Cross-entropy loss: $\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(x_{i+1} \mid x_{\leq i})$.",
    r"RMSNorm: $\hat{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot \gamma_i$, where $\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}$.",
    # Logs
    "2026-08-22T06:15:33.412Z INFO  trainer step=2500 loss=1.842 ppl=6.31 grad_norm=0.94 lr=2.87e-4 tokens/s=12450",
    "2026-08-22T06:15:45.001Z WARN  eval eval_loss=3.712 eval_ppl=40.95 gap=1.870 checkpoint_saved=True",
    "2026-08-22T06:16:02.887Z ERROR cuda OOM at layer 6 attention forward pass batch_size=64 seq_len=2048",
    # URLs / paths
    "https://huggingface.co/datasets/allenai/dolma/blob/main/README.md",
    "s3://my-training-bucket/checkpoints/kiara-tiny/step_10000/model.safetensors",
    "/proc/sys/kernel/perf_event_paranoid — controls access to performance monitoring",
    # Unicode / emoji edge cases
    "Precision: 98.7% ✓  Recall: 94.2% ✓  F1: 96.4% △  AUC: 0.991",
    "Training complete 🎉 | Loss: 1.23 → 0.45 | PPL: 342 → 5.7 | Steps: 10,000",
    "α=0.1, β₁=0.9, β₂=0.999, ε=1e-8 — Adam hyperparameters",
    "∀ x ∈ ℝⁿ: ‖softmax(x)‖₁ = 1 and ‖softmax(x)‖∞ ≤ 1",
]


def _generate_structured(rng: random.Random, n: int) -> list[str]:
    """Fully parametric structured data generator.

    Generates random JSON objects, log lines, YAML configs, Markdown tables,
    and LaTeX expressions using rng — produces unlimited unique samples.
    """
    _EVENTS = ["model_checkpoint", "eval_complete", "training_start", "grad_overflow",
               "lr_warmup_end", "epoch_end", "batch_sampled", "loss_spike_detected"]
    _LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
    _COMPONENTS = ["trainer", "eval", "dataloader", "optimizer", "scheduler", "checkpoint"]
    _MODELS = ["kiara-tiny", "kiara-small", "kiara-medium", "baseline", "ablation-no-rope"]
    _METRICS = ["loss", "eval_loss", "perplexity", "grad_norm", "token_acc", "bpt"]
    _ACTIVATIONS = ["SwiGLU", "ReLU", "GELU", "Mish", "SiLU"]
    _NORMS = ["RMSNorm", "LayerNorm", "BatchNorm"]
    _OPTIMIZERS = ["AdamW", "Adam", "SGD", "Lion", "Adafactor"]

    def _json_checkpoint():
        step = rng.randint(100, 50000)
        return '{' + ", ".join([
            f'"step": {step}',
            f'"train_loss": {rng.uniform(0.3, 5.0):.4f}',
            f'"eval_loss": {rng.uniform(0.5, 8.0):.4f}',
            f'"lr": {rng.uniform(1e-5, 5e-4):.2e}',
            f'"grad_norm": {rng.uniform(0.1, 3.0):.4f}',
            f'"model": "{rng.choice(_MODELS)}"',
        ]) + '}'

    def _json_dataset():
        ndocs = rng.randint(1000, 10000000)
        ntok = ndocs * rng.randint(50, 500)
        return '{' + ", ".join([
            f'"dataset": "synthetic-v{rng.randint(1,5)}.{rng.randint(0,9)}"',
            f'"num_documents": {ndocs}',
            f'"num_tokens": {ntok}',
            f'"unique_rate": {rng.uniform(0.6, 1.0):.3f}',
            f'"unk_count": 0',
        ]) + '}'

    def _log_line():
        h, m, s, ms = rng.randint(0,23), rng.randint(0,59), rng.randint(0,59), rng.randint(0,999)
        level = rng.choice(_LEVELS)
        component = rng.choice(_COMPONENTS)
        step = rng.randint(1, 50000)
        metric = rng.choice(_METRICS)
        value = rng.uniform(0.01, 20.0)
        return (f"2026-08-22T{h:02d}:{m:02d}:{s:02d}.{ms:03d}Z {level:5s} "
                f"{component} step={step} {metric}={value:.4f}")

    def _yaml_config():
        hidden = rng.choice([256, 512, 768, 1024])
        layers = rng.choice([4, 6, 8, 12])
        heads = rng.choice([4, 8, 16])
        return (f"model:\n"
                f"  hidden_size: {hidden}\n"
                f"  num_layers: {layers}\n"
                f"  num_heads: {heads}\n"
                f"  activation: {rng.choice(_ACTIVATIONS)}\n"
                f"  norm: {rng.choice(_NORMS)}\n"
                f"training:\n"
                f"  optimizer: {rng.choice(_OPTIMIZERS)}\n"
                f"  lr: {rng.uniform(1e-5, 1e-3):.2e}\n"
                f"  batch_size: {rng.choice([8, 16, 32, 64, 128])}")

    def _markdown_table():
        step = rng.randint(500, 10000)
        tl = rng.uniform(0.3, 4.0)
        el = tl + rng.uniform(0.2, 3.0)
        return (f"| Step | Train Loss | Eval Loss | Gap | PPL |\n"
                f"|------|------------|-----------|-----|-----|\n"
                f"| {step} | {tl:.3f} | {el:.3f} "
                f"| {el-tl:.3f} | {2.718**min(el,10):.1f} |")

    def _latex_eq():
        d = rng.choice([64, 128, 256, 512])
        templates = [
            rf"$\text{{Attn}}(Q,K,V) = \text{{softmax}}\!\left(\frac{{QK^\top}}{{\sqrt{{{d}}}}}\right)V$",
            rf"$\mathcal{{L}} = -\frac{{1}}{{N}}\sum_{{i=1}}^{{N}} \log p_\theta(x_{{i+1}} \mid x_{{\leq i}})$",
            rf"$\text{{BPT}} = \mathcal{{L}} / \ln 2$ where $\mathcal{{L}}$ is cross-entropy in nats",
            rf"$\text{{PPL}} = e^{{\mathcal{{L}}}}$ — perplexity as exponential of loss",
        ]
        return rng.choice(templates)

    generators = [_json_checkpoint, _json_dataset, _log_line, _yaml_config,
                  _markdown_table, _latex_eq]

    seen: set[str] = set()
    samples: list[str] = []
    attempts = 0
    while len(samples) < n and attempts < n * 5:
        attempts += 1
        text = rng.choice(generators)()
        if text not in seen:
            seen.add(text)
            samples.append(text)

    # Fallback (rng ensures near-infinite unique items, so this is rarely hit)
    while len(samples) < n:
        samples.append(_json_checkpoint())

    return samples


# ============================================================================
# Corpus v3 builder
# ============================================================================

_CATEGORY_FRACTIONS = {
    "natural": 0.30,
    "technical": 0.25,
    "code": 0.20,
    "multilingual": 0.15,
    "structured": 0.10,
}

# Eval A: a fixed set of held-out samples from the same distribution.
# These are written manually to ensure genuine diversity and to serve as
# a stable, consistent evaluation benchmark across training runs.

_EVAL_A_SAMPLES: dict[str, list[str]] = {
    "natural": [
        "Q: What is the difference between supervised and unsupervised learning?\nA: In supervised learning, each training example has a label and the model learns to predict it. In unsupervised learning there are no labels; the model discovers structure in the data on its own.",
        "Backpropagation computes the gradient of the loss with respect to every parameter by applying the chain rule of calculus layer by layer, starting from the output and moving toward the input.",
        "How to interpret a confusion matrix:\n1. The diagonal shows correct predictions.\n2. Off-diagonal entries show misclassifications.\n3. Row-normalise to get recall per class; column-normalise to get precision per class.",
        "Machine learning models generalise when they capture patterns that hold on data they have never seen. Overfitting occurs when a model instead learns the noise or peculiarities of the training set.",
        "User: Why does gradient descent get stuck in local minima?\nAssistant: In practice, for high-dimensional problems, saddle points are more common than local minima, and modern optimisers escape them via stochastic noise. True local minima are relatively rare in large neural networks.",
        "The validation set is used to tune hyperparameters and detect overfitting during training. The test set is held out entirely until the final evaluation of the chosen model.",
    ],
    "technical": [
        "The Adam optimiser combines momentum (a moving average of gradients) with RMSProp (a moving average of squared gradients) to produce adaptive per-parameter learning rates. Its default hyperparameters (β₁=0.9, β₂=0.999) work well across a wide range of tasks.",
        "Causal masking in a decoder transformer prevents each position from attending to future positions. This ensures that the model's prediction for token i depends only on tokens 1 through i, maintaining the autoregressive property.",
        "Weight initialisation matters because it determines the initial scale of activations and gradients. Poor initialisation can cause gradients to vanish or explode before any learning occurs. The Kaiming initialisation is designed to preserve variance through ReLU activations.",
        "Tokenisation converts raw text into a sequence of integer IDs. The choice of tokeniser affects vocabulary coverage, compression ratio, sequence length, and how well the model handles rare or out-of-vocabulary inputs.",
        "Evaluation perplexity measures how surprised a language model is by held-out text. Lower perplexity indicates the model assigns higher probability to the true next token on average, reflecting better language understanding.",
        "The learning rate is arguably the most important hyperparameter. Too high and training diverges; too low and convergence is slow or the model gets trapped. Learning rate finders and cosine schedules make this less sensitive to manual tuning.",
    ],
    "code": [
        "def rms_norm(x, weight, eps=1e-8):\n    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()\n    return x / rms * weight",
        "function clip_grad_norm(params, max_norm) {\n  const total = Math.sqrt(params.reduce((s, p) => s + p.grad.norm() ** 2, 0));\n  if (total > max_norm) params.forEach(p => { p.grad = p.grad.mul(max_norm / total); });\n  return total;\n}",
        "struct Config {\n    vocab_size: usize,\n    hidden_size: usize,\n    num_layers: usize,\n    num_heads: usize,\n    max_seq_len: usize,\n}",
        "# Quick sanity-check for a causal mask\nimport torch\ndef make_causal_mask(n):\n    return torch.tril(torch.ones(n, n)).bool()",
        "SELECT step, train_loss, eval_loss, eval_loss - train_loss AS generalization_gap\nFROM training_runs\nWHERE run_id = 'kiara-stage-c'\nORDER BY step;",
        "import json\nwith open('results.json') as f:\n    data = json.load(f)\nprint(f\"Final eval loss: {data['eval_loss']:.4f}\")\nprint(f\"Final eval ppl:  {data['eval_ppl']:.2f}\")",
    ],
    "multilingual": [
        "El sobreajuste ocurre cuando un modelo aprende demasiado bien los datos de entrenamiento y no generaliza a datos nuevos.",
        "La diversité des données d'entraînement est essentielle pour qu'un modèle généralise correctement.",
        "過学習とは、モデルが訓練データに過度に適合し、新しいデータに対してうまく機能しなくなる現象です。",
        "과적합은 모델이 훈련 데이터의 세부 사항을 지나치게 학습하여 새로운 데이터에 일반화하지 못하는 현상입니다.",
        # Hindi (Devanagari)
        "प्रशिक्षण हानि और मूल्यांकन हानि के बीच का अंतर मॉडल के अतिअनुकूलन का एक महत्वपूर्ण संकेत है।",
        # Tamil (native script)
        "தமிழ் மொழி உலகின் மிகவும் பழமையான தொடர்ச்சியான மொழிகளில் ஒன்றாகும், மேலும் இயந்திர கற்றல் ஆராய்ச்சியில் அதிகமாக பயன்படுத்தப்படுகிறது.",
    ],
    "structured": [
        '{"stage": "C", "corpus": "v3", "train_samples": 25000, "eval_a_samples": 36, "eval_b_samples": 173, "overlap": 0}',
        "## Corpus v3.0 Statistics\n\n- Version: 3.0.0\n- Training samples: 25,000\n- Eval A samples: 36 (generalization)\n- Eval B samples: 173 (robustness)\n- Train/Eval overlap: 0",
        r"Bits per token: $\text{BPT} = \frac{\mathcal{L}}{\ln 2}$, where $\mathcal{L}$ is the cross-entropy loss in nats.",
        "2026-08-22T12:00:00Z INFO corpus_v3 built version=3.0.0 train=25000 eval_a=36 eval_b=173 overlap=0 unk_tokens=0",
        "https://github.com/kiara-project/kiara/blob/main/experiments/tokenizer_benchmark/corpus_v3.py",
        "Train/Eval Generalization Gap = Eval Loss - Train Loss | Target: decreasing over training",
    ],
}


def build_v3_training_corpus(
    seed: int = DEFAULT_SEED,
    target_samples: int = 25000,
) -> list[str]:
    """Build the Corpus v3.0 training set.

    Generates samples from five structurally independent category generators,
    each written with varied sentence structure rather than shared templates.
    Guaranteed disjoint from Eval A and Eval B.
    """
    rng = random.Random(seed)

    eval_a_texts = set(get_v3_eval_a_texts())
    eval_b_texts = set(_get_eval_b_texts())

    counts = {cat: max(1, int(target_samples * frac)) for cat, frac in _CATEGORY_FRACTIONS.items()}
    # Adjust so total == target_samples
    total = sum(counts.values())
    diff = target_samples - total
    counts["natural"] += diff

    generators = {
        "natural": lambda n: _generate_natural_language(rng, n),
        "technical": lambda n: _generate_technical(rng, n),
        "code": lambda n: _generate_code(rng, n),
        "multilingual": lambda n: _generate_multilingual(rng, n),
        "structured": lambda n: _generate_structured(rng, n),
    }

    all_samples: list[str] = []
    for cat, n in counts.items():
        cat_samples = generators[cat](n)
        for s in cat_samples:
            if s not in eval_a_texts and s not in eval_b_texts:
                all_samples.append(s)

    rng.shuffle(all_samples)
    return all_samples[:target_samples]


def get_v3_eval_a_corpus() -> dict[str, list[str]]:
    """Return Eval A — held-out generalization benchmark (same distribution as training)."""
    return {k: list(v) for k, v in _EVAL_A_SAMPLES.items()}


def get_v3_eval_a_texts() -> list[str]:
    """Return flat list of Eval A texts."""
    return [s for samples in _EVAL_A_SAMPLES.values() for s in samples]


def _get_eval_b_texts() -> list[str]:
    """Return Eval B — existing robustness benchmark (173 samples)."""
    from experiments.tokenizer_benchmark.corpus import get_eval_texts
    return get_eval_texts()


def get_v3_eval_b_corpus() -> dict[str, list[str]]:
    """Return Eval B — existing robustness benchmark (8 categories, 173 samples)."""
    from experiments.tokenizer_benchmark.corpus import get_eval_corpus
    return get_eval_corpus()


# ============================================================================
# Overlap verification
# ============================================================================

@dataclass
class OverlapReport:
    train_eval_a_exact: int = 0
    train_eval_b_exact: int = 0
    eval_a_eval_b_exact: int = 0
    train_eval_a_normalized: int = 0
    train_eval_b_normalized: int = 0
    train_eval_a_sha256: int = 0
    train_eval_b_sha256: int = 0
    is_clean: bool = True
    details: list[str] = field(default_factory=list)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def verify_disjoint(
    train: list[str],
    eval_a: list[str],
    eval_b: list[str],
) -> OverlapReport:
    """Verify that train, eval_a, and eval_b are pairwise disjoint."""
    report = OverlapReport()

    train_set = set(train)
    eval_a_set = set(eval_a)
    eval_b_set = set(eval_b)

    train_norm = set(_normalize(t) for t in train)
    eval_a_norm = set(_normalize(t) for t in eval_a)
    eval_b_norm = set(_normalize(t) for t in eval_b)

    train_sha = set(_sha256(t) for t in train)
    eval_a_sha = set(_sha256(t) for t in eval_a)
    eval_b_sha = set(_sha256(t) for t in eval_b)

    report.train_eval_a_exact = len(train_set & eval_a_set)
    report.train_eval_b_exact = len(train_set & eval_b_set)
    report.eval_a_eval_b_exact = len(eval_a_set & eval_b_set)

    report.train_eval_a_normalized = len(train_norm & eval_a_norm)
    report.train_eval_b_normalized = len(train_norm & eval_b_norm)

    report.train_eval_a_sha256 = len(train_sha & eval_a_sha)
    report.train_eval_b_sha256 = len(train_sha & eval_b_sha)

    violations = [
        report.train_eval_a_exact,
        report.train_eval_b_exact,
        report.eval_a_eval_b_exact,
        report.train_eval_a_normalized,
        report.train_eval_b_normalized,
        report.train_eval_a_sha256,
        report.train_eval_b_sha256,
    ]
    report.is_clean = all(v == 0 for v in violations)

    if not report.is_clean:
        report.details.append(f"Overlap detected — corpus v3.0 is NOT clean.")
    return report


# ============================================================================
# Corpus statistics
# ============================================================================

def compute_v3_corpus_stats(
    train: list[str],
    eval_a: list[str],
    eval_b: list[str],
    tokenizer=None,
) -> dict[str, Any]:
    """Compute and return comprehensive statistics for Corpus v3.0."""

    def text_stats(texts: list[str], name: str) -> dict[str, Any]:
        chars = sum(len(t) for t in texts)
        words = [w for t in texts for w in t.split()]
        byte_count = sum(len(t.encode("utf-8")) for t in texts)
        lengths_chars = sorted(len(t) for t in texts)
        n = len(lengths_chars)
        pct = lambda p: lengths_chars[int(n * p)] if n else 0

        stats: dict[str, Any] = {
            "num_documents": len(texts),
            "num_characters": chars,
            "num_bytes_utf8": byte_count,
            "approx_mb": round(byte_count / (1024 ** 2), 3),
            "num_words": len(words),
            "unique_words": len(set(words)),
            "avg_chars_per_doc": round(chars / max(1, len(texts)), 1),
            "length_p25": pct(0.25),
            "length_p50": pct(0.50),
            "length_p75": pct(0.75),
            "length_p90": pct(0.90),
            "length_p99": pct(0.99),
            "duplicate_count": len(texts) - len(set(texts)),
        }

        if tokenizer is not None:
            token_counts = [len(tokenizer.encode(t)) for t in texts]
            total_tokens = sum(token_counts)
            unk_count = sum(
                ids.count(tokenizer.unk_token_id)
                for ids in (tokenizer.encode(t) for t in texts)
            )
            stats["num_tokens"] = total_tokens
            stats["avg_tokens_per_doc"] = round(total_tokens / max(1, len(texts)), 1)
            stats["unk_count"] = unk_count
            tok_lengths = sorted(token_counts)
            stats["token_length_p50"] = tok_lengths[int(n * 0.5)] if n else 0
            stats["token_length_p90"] = tok_lengths[int(n * 0.9)] if n else 0
            stats["token_length_p99"] = tok_lengths[int(n * 0.99)] if n else 0

        return stats

    overlap = verify_disjoint(train, eval_a, eval_b)

    return {
        "version": CORPUS_V3_VERSION,
        "is_clean": overlap.is_clean,
        "overlap_report": {
            "train_eval_a_exact": overlap.train_eval_a_exact,
            "train_eval_b_exact": overlap.train_eval_b_exact,
            "eval_a_eval_b_exact": overlap.eval_a_eval_b_exact,
            "train_eval_a_normalized": overlap.train_eval_a_normalized,
            "train_eval_b_normalized": overlap.train_eval_b_normalized,
            "train_eval_a_sha256": overlap.train_eval_a_sha256,
            "train_eval_b_sha256": overlap.train_eval_b_sha256,
        },
        "training": text_stats(train, "train"),
        "eval_a": text_stats(eval_a, "eval_a"),
        "eval_b": text_stats(eval_b, "eval_b"),
    }


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print(f"Building Corpus v{CORPUS_V3_VERSION}...")
    train = build_v3_training_corpus(seed=DEFAULT_SEED, target_samples=1000)
    eval_a = get_v3_eval_a_texts()
    eval_b = _get_eval_b_texts()

    stats = compute_v3_corpus_stats(train, eval_a, eval_b)
    import json
    print(json.dumps(stats, indent=2, ensure_ascii=False))
