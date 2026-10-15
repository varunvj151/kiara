"""
corpus.py — Kiara Tokenizer Benchmark Corpus Builder (v2.0.0)
============================================================
Builds a controlled, reproducible, local benchmark corpus:
1. Training Corpus: Large, diverse, deterministic synthetic multi-domain dataset
   (~100,000 samples, ~20+ MB) used exclusively for tokenizer training.
2. Evaluation Corpus: Held-out benchmark dataset across 8 distinct categories,
   strictly disjoint from the training data.

All generation is fully deterministic (seed=1337), local, and reproducible.
No copyrighted or external datasets are downloaded.
"""

from __future__ import annotations

import random
from typing import Any

CORPUS_VERSION = "2.0.0"
DEFAULT_SEED = 1337

# ---------------------------------------------------------------------------
# Held-Out Evaluation Corpus (8 Canonical Categories)
# ---------------------------------------------------------------------------


def _eval_english_prose() -> list[str]:
    """Standard English prose: sentences and short paragraphs."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "A language model learns to predict the next token in a sequence.",
        "Training requires large amounts of compute and carefully curated data.",
        "The attention mechanism allows the model to relate every position to every other position.",
        "Backpropagation computes gradients by applying the chain rule of calculus.",
        "Overfitting occurs when a model memorizes training examples rather than learning general patterns.",
        "Regularization techniques such as dropout reduce overfitting by randomly zeroing activations.",
        "The learning rate controls the step size during gradient descent optimization.",
        "Transformer models have largely replaced recurrent architectures for sequence modeling tasks.",
        "Pre-training on large corpora followed by fine-tuning enables strong transfer learning.",
        (
            "Natural language processing has advanced dramatically since the introduction of the "
            "transformer architecture in 2017. Models trained on vast amounts of text data have "
            "demonstrated the ability to answer questions, write code, summarize documents, and "
            "perform reasoning tasks that were previously considered very difficult for machines."
        ),
        (
            "The foundation model paradigm represents a fundamental shift in how AI systems are built. "
            "Rather than training task-specific models from scratch, a single large model is pretrained "
            "on diverse data and then adapted to specific tasks through fine-tuning or prompting."
        ),
        "She sells seashells by the seashore.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm, but the second mouse gets the cheese.",
        "In the beginning was the Word, and the Word was with God.",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael.",
        "Happy families are all alike; every unhappy family is unhappy in its own way.",
        "A journey of a thousand miles begins with a single step.",
        "Knowledge is power, but enthusiasm pulls the switch.",
        "Simplicity is the prerequisite for reliability.",
        "The only way to do great work is to love what you do.",
    ]


def _eval_technical_english() -> list[str]:
    """Technical and scientific English text."""
    return [
        "The RMSNorm layer normalizes activations by dividing by the root mean square of the vector.",
        "Grouped Query Attention reduces memory bandwidth by sharing key and value heads across query groups.",
        "SwiGLU activation: FFN(x) = SiLU(W1 x) * (W2 x), where * denotes element-wise multiplication.",
        "Rotary Position Embeddings (RoPE) encode position information by rotating query and key vectors.",
        "Byte-Pair Encoding starts with a character-level vocabulary and iteratively merges frequent pairs.",
        "Gradient clipping prevents exploding gradients by rescaling the gradient norm when it exceeds a threshold.",
        "Causal masking ensures that attention at position i cannot attend to positions j > i.",
        "The cross-entropy loss for language modeling is: L = -sum(y_i * log(p_i)) over all positions.",
        "Learning rate warmup gradually increases the learning rate from zero to its target value.",
        "bfloat16 format uses 8 exponent bits and 7 mantissa bits, offering the same dynamic range as float32.",
        "KV-cache stores computed key and value tensors during autoregressive generation to avoid recomputation.",
        "Perplexity is defined as exp(H(p, q)) where H is the cross-entropy between the true and model distributions.",
        "The embedding matrix of size V x d maps each token index to a dense vector of dimension d.",
        "Flash Attention reorders operations to reduce HBM reads/writes, achieving O(N) memory complexity.",
        "Weight decay adds an L2 penalty lambda * ||w||^2 to the loss function to regularize model parameters.",
        "MMLU (Massive Multitask Language Understanding) evaluates knowledge across 57 subject areas.",
        "HumanEval measures functional code correctness by checking whether generated code passes unit tests.",
        "The AIME competition tests high-school level mathematical competition problem-solving ability.",
        "GSM8K contains 8500 grade-school math word problems requiring multi-step arithmetic reasoning.",
        "SWE-bench evaluates the ability of language models to resolve real GitHub issues in open-source repos.",
        "Direct Preference Optimization (DPO) optimizes policy models directly on human preference pairs.",
        "Quantized Low-Rank Adaptation (QLoRA) backpropagates gradients through 4-bit quantized base weights.",
        "Mixture of Experts (MoE) routes each token to a subset of feed-forward networks via a gating function.",
        "Speculative decoding accelerates autoregressive generation using a smaller draft model.",
    ]


def _eval_source_code() -> list[str]:
    """Source code samples: Python, C++, JavaScript, Rust, Go, SQL."""
    return [
        "import torch\nimport torch.nn as nn\n",
        "def scaled_dot_product_attention(q, k, v, mask=None):\n    d_k = q.size(-1)\n    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)\n    if mask is not None:\n        scores = scores.masked_fill(mask == 0, float('-inf'))\n    return torch.matmul(torch.softmax(scores, dim=-1), v)\n",
        "class RMSNorm(nn.Module):\n    def __init__(self, dim: int, eps: float = 1e-8):\n        super().__init__()\n        self.eps = eps\n        self.weight = nn.Parameter(torch.ones(dim))\n\n    def forward(self, x):\n        norm = x.norm(2, dim=-1, keepdim=True)\n        return self.weight * x / (norm + self.eps)\n",
        "for epoch in range(num_epochs):\n    optimizer.zero_grad()\n    logits = model(input_ids)\n    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))\n    loss.backward()\n    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n    optimizer.step()\n",
        "def get_cosine_schedule(optimizer, warmup_steps, total_steps):\n    def lr_lambda(current_step):\n        if current_step < warmup_steps:\n            return float(current_step) / float(max(1, warmup_steps))\n        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))\n        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))\n    return LambdaLR(optimizer, lr_lambda)\n",
        "const transformer = new TransformerModel({ hiddenSize: 1024, numLayers: 24, numHeads: 16 });\n",
        "async function generateText(prompt, maxTokens = 200) {\n    const tokens = tokenizer.encode(prompt);\n    for (let i = 0; i < maxTokens; i++) {\n        const logits = await model.forward(tokens);\n        const nextToken = sampleTopK(logits, k=50);\n        tokens.push(nextToken);\n        if (nextToken === tokenizer.eosId) break;\n    }\n    return tokenizer.decode(tokens);\n}\n",
        "template<typename T>\nclass AttentionLayer {\npublic:\n    AttentionLayer(int hidden_size, int num_heads)\n        : hidden_size_(hidden_size), num_heads_(num_heads),\n          head_dim_(hidden_size / num_heads) {}\n\n    torch::Tensor forward(torch::Tensor x, torch::optional<torch::Tensor> mask = torch::nullopt);\nprivate:\n    int hidden_size_, num_heads_, head_dim_;\n};\n",
        "#!/bin/bash\npython train.py \\\n    --config configs/tiny.yaml \\\n    --seed 1337 \\\n    --dtype bfloat16 \\\n    --max_steps 10000 \\\n    --checkpoint_interval 1000\n",
        "get_transformer_block_hidden_state_representation",
        "CounterfactualReasoningModuleForMultiHopQuestionAnswering",
        "loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-1)",
        "self._attn_weights = torch.zeros(batch, heads, seq_len, seq_len, device=device, dtype=dtype)",
        "0x1F 0xFF 0b10110101 3.14159265358979 2.718281828459 1e-8 6.022e23",
        "fn main() -> Result<(), Box<dyn std::error::Error>> {\n    let config = Config::from_file(\"kiara.toml\")?;\n    let engine = InferenceEngine::new(&config)?;\n    println!(\"Kiara engine initialized: {:?}\", engine);\n    Ok(())\n}\n",
        "SELECT model_name, avg(eval_loss) as mean_loss FROM benchmark_runs WHERE status = 'COMPLETED' GROUP BY model_name HAVING count(*) >= 5 ORDER BY mean_loss ASC LIMIT 10;",
    ]


def _eval_mathematics() -> list[str]:
    """Mathematical notation: LaTeX, Unicode symbols, equations."""
    return [
        r"\frac{\partial L}{\partial \theta} = -\frac{1}{N} \sum_{i=1}^{N} y_i \log \hat{y}_i",
        r"\mathbb{E}_{x \sim p(x)}[f(x)] = \int_{-\infty}^{+\infty} f(x) p(x) \, dx",
        r"\hat{\theta}_{MLE} = \arg\max_{\theta} \sum_{i=1}^{N} \log p(x_i | \theta)",
        r"P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}",
        r"\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right]^T",
        r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V",
        r"e^{i\pi} + 1 = 0",
        r"\sum_{n=0}^{\infty} \frac{x^n}{n!} = e^x",
        "∀ε > 0, ∃δ > 0: |x − a| < δ ⟹ |f(x) − f(a)| < ε",
        "∇·E = ρ/ε₀, ∇×B = μ₀J + μ₀ε₀ ∂E/∂t",
        "ℝ, ℂ, ℕ, ℤ, ℚ — standard number set notation",
        "∫₀^∞ e^{−x²} dx = √π / 2",
        "λ₁ + λ₂ + ⋯ + λₙ = tr(A)",
        "‖x‖₂ = √(x₁² + x₂² + ⋯ + xₙ²)",
        "σ(x) = 1 / (1 + e^{−x})",
        "3.14159265358979323846",
        "2.71828182845904523536",
        "1.41421356237309504880",
        "1.61803398874989484820",
        "0.57721566490153286060",
        r"\mathcal{L}_{ELBO}(\theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z))",
        r"\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}",
    ]


def _eval_multilingual() -> list[str]:
    """Text samples from multiple languages and scripts."""
    return [
        "El aprendizaje automático es una rama de la inteligencia artificial que permite a los sistemas aprender.",
        "Les modèles de fondation nécessitent des corpus d'entraînement massifs et diversifiés.",
        "L'intelligenza artificiale sta rivoluzionando molti settori dell'economia moderna.",
        "A inteligência artificial está transformando a maneira como interagimos com a tecnologia.",
        "Sprachmodelle werden auf großen Textmengen aus dem Internet vortrainiert und dann feinabgestimmt.",
        "Kunstmatige intelligentie heeft grote vooruitgang geboekt op het gebied van taalbegrip.",
        "機械学習は人工知能の一分野であり、コンピューターがデータからパターンを学習することを可能にします。",
        "机器学习是人工智能的一个子领域，使计算机能够从数据中自动学习和改进。",
        "머신러닝은 컴퓨터가 데이터에서 패턴을 학습하여 예측을 수행하는 인공지능의 한 분야입니다.",
        "التعلم الآلي هو فرع من فروع الذكاء الاصطناعي يمكن الحاسوب من التعلم من البيانات.",
        "یادگیری ماشین شاخه‌ای از هوش مصنوعی است که به رایانه‌ها امکان یادگیری از داده‌ها را می‌دهد.",
        "मशीन लर्निंग कृत्रिम बुद्धिमत्ता का एक उप-क्षेत्र है जो कंप्यूटर को डेटा से सीखने में सक्षम बनाता है।",
        "যন্ত্র शिक्षा কৃত্রিম बुद्धिमत्ता का एक उप-क्षेत्र है जो कंप्यूटर को डेटा से सीखने में सक्षम बनाता है।",
        "Makine öğrenmesi, bilgisayarların verilerden öğrenmesini sağlayan yapay zekanın bir dalıdır.",
        "Uczenie maszynowe to dziedzina sztucznej inteligencji umożliwiająca komputerom uczenie się z danych.",
        "Maskininlärning är ett område inom artificiell intelligens som gör det möjligt för datorer att lära sig.",
        "Машинное обучение позволяет алгоритмам выявлять скрытые закономерности в сложных массивах данных.",
        "תחום הבינה המלאכותית מאפשר למחשבים לנתח מידע ולהסיק מסקנות באופן אוטונומי.",
    ]


def _eval_rare_and_long_words() -> list[str]:
    """Rare, long, and morphologically complex words."""
    return [
        "electroencephalography",
        "deoxyribonucleic acid",
        "immunohistochemistry",
        "pneumonoultramicroscopicsilicovolcanoconiosis",
        "hippopotomonstrosesquipedaliophobia",
        "antidisestablishmentarianism",
        "floccinaucinihilipilification",
        "supercalifragilisticexpialidocious",
        "CounterfactualReasoningModuleForMultiHopQuestionAnswering",
        "TransformerBasedSequenceToSequenceModelWithGroupedQueryAttention",
        "get_scaled_dot_product_attention_weights_for_causal_mask",
        "MultiHeadSelfAttentionWithRotaryPositionalEmbeddingsAndRMSNorm",
        "running ran runs runner runners runnable",
        "tokenize tokenizer tokenization tokenized tokenizable detokenize",
        "unbelievable unbelievably unbelievableness counterintuitively",
        "anthropomorphization anthropomorphized anthropomorphizing",
        "eigenvector eigenvalue eigendecomposition pseudoinverse",
        "backpropagation vanishing-gradient exploding-gradient hyperparameter",
        "quantization dequantization int8 bfloat16 float32 mixed-precision",
        "LongFormer BigBird FlashAttention GroupedQueryAttention LoRA QLoRA",
        "pseudopseudohypoparathyroidism",
        "dichlorodiphenyltrichloroethane",
        "incomprehensibilities spectrophotofluorometrically",
    ]


def _eval_unicode_and_emoji() -> list[str]:
    """Unicode edge cases, emoji, and special symbols."""
    return [
        "Hello 🌍🚀🧠🔥💡✨🎯📊",
        "Great work! 👏🎉🏆⭐🌟",
        "Weather: ☀️🌤️⛅🌧️⛈️🌩️❄️🌈",
        "👋🏻👋🏼👋🏽👋🏾👋🏿",
        "👨‍💻👩‍🔬👨‍🚀👩‍🎨🧑‍🏫",
        "α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω",
        "Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω",
        "∀ ∃ ∄ ∈ ∉ ∋ ∌ ∩ ∪ ⊂ ⊃ ⊄ ⊅ ⊆ ⊇",
        "∫ ∬ ∭ ∮ ∯ ∰ ∱ ∲ ∳",
        "← → ↑ ↓ ↔ ↕ ⇐ ⇒ ⇑ ⇓ ⇔",
        "✓ ✗ ✘ ✔ ✕ ✖ ✙ ✚ ✛ ✜ ✝ ✞",
        "café naïve résumé façade coördinate Ångström",
        "über über über — German umlaut",
        "Ñoño, señor, jalapeño — Spanish tilde-n",
        "ＡＢＣＤＥＦＧ — fullwidth Latin",
        "｢｣ 【】 『』 「」 — CJK brackets",
        "The word مرحبا means 'hello' in Arabic.",
        "שלום means 'peace' or 'hello' in Hebrew.",
        "x⁰ x¹ x² x³ x⁴ x⁵ H₂O CO₂ C₆H₁₂O₆",
        "𝒜𝓁𝓰𝑒𝒷𝓇𝒶 — Mathematical Script",
        "𝔄𝔅ℭ𝔇𝔈𝔉𝔊 — Mathematical Fraktur",
        "\u0000 — null byte (edge case)",
        "🏴‍☠️ 🏳️‍🌈 🇺🇸 🇬🇧 🇩🇪 🇯🇵 🇮🇳 — National and composite flags",
        "﷽ — Arabic Basmala ligature (U+FDFD)",
    ]


def _eval_urls_and_symbols() -> list[str]:
    """URLs, file paths, code symbols, and punctuation-heavy text."""
    return [
        "https://github.com/kiara-project/kiara/blob/main/kiara/model/transformer.py#L42-L87",
        "https://arxiv.org/abs/1706.03762",
        "https://huggingface.co/docs/tokenizers/python/latest/",
        "https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html",
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        "C:\\Users\\varun\\OneDrive\\Documents\\flutter1\\kiara\\configs\\base.yaml",
        "/home/user/.local/lib/python3.12/site-packages/tokenizers/__init__.py",
        "SELECT id, text, label FROM training_data WHERE split = 'train' ORDER BY id LIMIT 1000;",
        "$100.00 ± 0.05 (p < 0.001, n=1024, 95% CI: [99.95, 100.05])",
        "API_KEY=sk-1234567890abcdef TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "git commit -m 'feat(tokenizer): add byte-level BPE benchmark'",
        "chmod +x ./scripts/run_benchmark.sh && ./scripts/run_benchmark.sh --config configs/tiny.yaml",
        "cat /proc/cpuinfo | grep 'model name' | head -1",
        "E=mc², F=ma, P=IV, v=d/t, a=(v₂-v₁)/Δt",
        "1+1=2, 2²=4, √16=4, ³√27=3, log₂(1024)=10",
        "/* TODO: implement causal masking */ // FIXME: memory leak in attention",
        "<!-- HTML comment --> <div class='container'> </div>",
        "{ \"model\": \"kiara-tiny\", \"vocab_size\": 16384, \"hidden\": 256, \"layers\": 8 }",
        "key=value&another=thing&encoded=%20%21%40%23",
        "---\nlayout: post\ntitle: 'Kiara: Building a Foundation Model'\ndate: 2026-08-20\n---\n",
        "urn:isbn:978-0-13-468599-1 uuid:f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
        "IPv6: [2001:0db8:85a3:0000:0000:8a2e:0370:7334]:8080/metrics",
    ]


# Base evaluation registry
EVAL_CATEGORIES: dict[str, list[str]] = {
    "english": _eval_english_prose(),
    "technical": _eval_technical_english(),
    "code": _eval_source_code(),
    "math": _eval_mathematics(),
    "multilingual": _eval_multilingual(),
    "rare_words": _eval_rare_and_long_words(),
    "unicode": _eval_unicode_and_emoji(),
    "urls_symbols": _eval_urls_and_symbols(),
}

# Alias for backward compatibility
CATEGORIES = EVAL_CATEGORIES


def get_eval_corpus(categories: list[str] | None = None) -> dict[str, list[str]]:
    """Return held-out evaluation corpus filtered to requested categories."""
    if categories is None:
        return {k: list(v) for k, v in EVAL_CATEGORIES.items()}
    return {k: list(EVAL_CATEGORIES[k]) for k in categories if k in EVAL_CATEGORIES}


def get_corpus(categories: list[str] | None = None) -> dict[str, list[str]]:
    """Return the evaluation corpus (alias for get_eval_corpus)."""
    return get_eval_corpus(categories)


def get_eval_texts(categories: list[str] | None = None) -> list[str]:
    """Return flat list of all evaluation corpus texts."""
    corpus = get_eval_corpus(categories)
    texts: list[str] = []
    for samples in corpus.values():
        texts.extend(samples)
    return texts


# ---------------------------------------------------------------------------
# Training Corpus Generator (Deterministic, Scalable, Multi-Domain)
# ---------------------------------------------------------------------------

_LEXICON_DOMAINS: dict[str, list[str]] = {
    "deep_learning": [
        "transformer", "attention", "gradient", "backpropagation", "optimization", "regularization",
        "generalization", "hyperparameter", "activation", "convolution", "recurrent", "feedforward",
        "autoregressive", "encoder", "decoder", "latent", "representation", "embedding", "dimensionality",
        "softmax", "cross-entropy", "normalization", "residual", "stochastic", "convergence", "checkpoint",
        "rotary-embedding", "positional-encoding", "grouped-query", "multi-head", "flash-attention",
        "swiglu", "gelu", "silu", "rmsnorm", "layernorm", "adamw", "cosine-annealing", "weight-decay",
        "dropout", "gradient-clipping", "mixed-precision", "bfloat16", "quantization", "dequantization",
        "lora", "qlora", "perplexity", "cross-entropy", "logits", "tokenization", "detokenization"
    ],
    "systems_architecture": [
        "concurrency", "distributed", "synchronization", "deadlock", "semaphore", "mutex", "pipeline",
        "throughput", "latency", "cache-coherence", "virtual-memory", "paging", "kernel", "scheduler",
        "asynchronous", "coroutine", "prefetch", "virtualization", "hypervisor", "orchestration",
        "microservice", "monolith", "idempotent", "serialization", "deserialization", "protobuf",
        "marshaling", "consensus", "raft", "paxos", "sharding", "replication", "partitioning"
    ],
    "mathematics_statistics": [
        "eigenvalue", "eigenvector", "determinant", "singular-value", "decomposition", "orthonormal",
        "isometric", "homomorphism", "isomorphism", "diffeomorphism", "manifold", "topological",
        "compactness", "differentiability", "holomorphic", "asymptotic", "stochastic", "martingale",
        "markov-chain", "ergodicity", "poisson", "gaussian", "dirichlet", "laplace", "fourier",
        "hessian", "jacobian", "gradient", "divergence", "curl", "laplacian", "curvature", "riemannian"
    ],
    "biology_biochemistry": [
        "deoxyribonucleic", "ribonucleic", "polymerase", "nucleotide", "mitochondria", "ribosome",
        "endoplasmic", "reticulum", "phosphorylation", "transcription", "translation", "methylation",
        "epigenetics", "histone", "chromatin", "telomere", "cytokine", "interferon", "monoclonal",
        "immunoglobulin", "neurogenesis", "synaptogenesis", "neurotransmitter", "myelination",
        "oligodendrocyte", "astrocyte", "microglia", "synaptic-plasticity", "long-term-potentiation"
    ],
    "physics_astronomy": [
        "thermodynamics", "electrodynamics", "hydrodynamics", "superconductivity", "semiconductor",
        "ferromagnetism", "paramagnetism", "crystallography", "spectroscopy", "interferometry",
        "chromatography", "stoichiometry", "catalysis", "enthalpy", "entropy", "superposition",
        "entanglement", "qubit", "hamiltonian", "lagrangian", "spacetime", "gravitational-wave"
    ],
}

_PREFIXES = [
    "anti", "auto", "bi", "co", "counter", "de", "dis", "extra", "hetero", "homo", "hyper", "hypo",
    "in", "inter", "intra", "macro", "micro", "mis", "mono", "multi", "neo", "non", "omni", "over",
    "pan", "para", "poly", "post", "pre", "pro", "pseudo", "quasi", "re", "retro", "semi", "sub",
    "super", "syn", "trans", "tri", "ultra", "un", "under", "uni", "iso", "aniso", "ortho"
]

_SUFFIXES = [
    "ability", "able", "action", "al", "ance", "ant", "ary", "ate", "ation", "ative", "ator",
    "centric", "cracy", "cy", "dom", "ed", "ee", "ence", "ent", "er", "es", "esque", "est",
    "etic", "ful", "fy", "genesis", "graph", "graphy", "hood", "ia", "ial", "ian", "ic", "ical",
    "ide", "ier", "iferous", "ify", "ing", "ion", "ious", "isation", "ise", "ish", "ism", "ist",
    "ite", "itis", "ity", "ium", "ive", "ization", "ize", "less", "let", "like", "ling", "logist",
    "logy", "ly", "lysis", "ment", "metry", "morph", "morphism", "ness", "oid", "ology", "on",
    "or", "ory", "ose", "osis", "ous", "path", "pathy", "phile", "philia", "phobe", "phobia",
    "phone", "phony", "plasm", "scope", "scopy", "ship", "sion", "sphere", "stasis", "stat",
    "tion", "tomy", "trophy", "tude", "ty", "ular", "ure", "ward", "wise", "y"
]

_MULTILINGUAL_PATTERNS = [
    ("de", "Die mathematische Formalisierung neuronaler Netze ermöglicht präzise Konvergenzbeweise in der Optimierungstheorie."),
    ("fr", "L'intelligence artificielle repose sur des modèles statistiques capables de généraliser à partir de données massives."),
    ("es", "El aprendizaje profundo transforma la comprensión del procesamiento del lenguaje natural mediante redes neuronales."),
    ("it", "La rappresentazione distribuita delle parole cattura relazioni semantiche complesse nello spazio vettoriale."),
    ("pt", "Modelos de linguagem de grande escala demonstram capacidades emergentes em tarefas de raciocínio lógico."),
    ("nl", "Machinaal leren stelt algoritmen in staat om patronen in complexe data te ontdekken zonder expliciete instructies."),
    ("ru", "Машинное обучение основывается на статистических методах анализа многомерных данных и градиентной оптимизации."),
    ("zh", "大规模语言模型通过在海量多源语料上进行自监督预训练来获得通用的语言表征能力。"),
    ("ja", "自己注意機構は入力系列内の全要素間の依存関係を並列に計算することを可能にします。"),
    ("ko", "인공신경망의 가중치 초기화와 학습률 스케줄링은 모델의 수렴 속도와 최종 성능을 결정합니다."),
    ("ar", "تعتمد خوارزميات معالجة اللغة الطبيعية على تحويل النصوص إلى متجهات عددية كثيفة في فضاء دلالي."),
    ("hi", "गहरे तंत्रिका नेटवर्क जटिल प्रतिमानों को सीखने के लिए बहु-स्तरीय प्रतिनिधित्व का उपयोग करते हैं।"),
    ("pl", "Przetwarzanie języka naturalnego wykorzystuje zaawansowane modele statystyczne do analizy semantyki tekstu."),
    ("tr", "Büyük dil modelleri metin tamamlama ve anlamsal çıkarım görevlerinde yüksek başarı göstermektedir."),
    ("he", "אלגוריתמים של למידת מכונה מסוגלים לזהות תבניות מורכבות בנתונים רב-ממדיים ללא הנחיות מפורשות."),
    ("sv", "Djupinlärningsmodeller förlitar sig på optimeringsalgoritmer för att minimera förlustfunktionen effektivt."),
    ("el", "Η μηχανική μάθηση επιτρέπει σε υπολογιστικά συστήματα να εξάγουν γνώση από σύνθετα πολυδιάστατα δεδομένα."),
    ("vi", "Mô hình ngôn ngữ lớn học cách biểu diễn ngữ nghĩa của các từ trong không gian vector nhiều chiều.")
]


def build_training_corpus(
    seed: int = DEFAULT_SEED,
    target_samples: int = 100000,
) -> list[str]:
    """Generate a diverse, rich, deterministic synthetic training corpus.

    Ensures the training data covers:
    - Natural English prose and deep scientific/technical texts
    - 8 programming languages with diverse syntax (Python, C++, Rust, JS/TS, Go, Java, SQL, Shell)
    - Mathematical formulations and LaTeX notation
    - 18+ natural languages with Latin, Cyrillic, Greek, CJK, Arabic, Hebrew, and Indic scripts
    - Synthetic morphology (prefixes, roots, suffixes), compound rare terms, long identifiers
    - Structured data (JSON, YAML, TOML, Markdown, CSV, XML, logs)
    - Numbers (floats, scientific notation, hex, binary, decimals, IP addresses, dates, timestamps)
    - URLs, file paths (Windows & POSIX), query parameters, UUIDs, hashes

    Guaranteed to be strictly disjoint from the evaluation corpus.
    """
    rng = random.Random(seed)
    eval_text_set = set(get_eval_texts())

    all_domain_terms = [t for sub in _LEXICON_DOMAINS.values() for t in sub]
    samples: list[str] = []

    def add_if_disjoint(s: str) -> None:
        if s not in eval_text_set:
            samples.append(s)

    # 1. Scientific & Technical English Prose (~30%)
    prose_templates = [
        "In modern {domain}, the relationship between {t1} and {t2} establishes a {adj} framework for {t3} analysis.",
        "Recent investigations into {t1} demonstrate that {adj} regularizations mitigate {t2} degradation under {t3} constraints.",
        "The mathematical convergence of {t1} relies upon {adj} approximations of {t2} across high-dimensional {t3} manifolds.",
        "Empirical benchmarks indicate that scaling {t1} by a factor of {num} yields substantial improvements in {t2} efficiency.",
        "We formalize the {rare} property in {domain} by proving that {t1} remains invariant under {adj} {t2} transformations.",
        "A critical trade-off exists between {t1} throughput and {t2} latency when deploying {adj} {t3} architectures.",
        "By applying {rare} principles to {domain}, practitioners can systematically optimize {t1} while bounding {t2} variance."
    ]

    adjectives = [
        "stochastic", "deterministic", "asymptotic", "hierarchical", "orthogonal", "non-linear",
        "differentiable", "isometric", "equivariant", "compositional", "spectral", "multimodal",
        "topological", "probabilistic", "autoregressive", "hyperbolic", "ergodic", "symplectic"
    ]

    target_prose = int(target_samples * 0.30)
    for _ in range(target_prose):
        d_name = rng.choice(list(_LEXICON_DOMAINS.keys()))
        t1 = rng.choice(_LEXICON_DOMAINS[d_name])
        t2 = rng.choice(_LEXICON_DOMAINS[d_name])
        t3 = rng.choice(all_domain_terms)
        adj = rng.choice(adjectives)
        pref = rng.choice(_PREFIXES)
        root = rng.choice(all_domain_terms).replace("-", "")
        suff = rng.choice(_SUFFIXES)
        rare_term = f"{pref}{root}{suff}"
        num = rng.choice([f"{rng.randint(2, 64)}x", f"{rng.uniform(0.01, 99.99):.2f}", f"{rng.uniform(1e-6, 1e-2):.2e}"])
        tmpl = rng.choice(prose_templates)
        sentence = tmpl.format(domain=d_name.replace("_", " "), t1=t1, t2=t2, t3=t3, adj=adj, rare=rare_term, num=num)
        add_if_disjoint(sentence)

    # 2. Multi-Language Source Code (~25%)
    target_code = int(target_samples * 0.25)
    languages = ["python", "cpp", "rust", "js", "go", "java", "sql", "bash"]
    for _ in range(target_code):
        lang = rng.choice(languages)
        t_base = rng.choice(all_domain_terms).replace("-", "_")
        pref = rng.choice(_PREFIXES)
        suff = rng.choice(_SUFFIXES)
        ident = f"{pref}_{t_base}_{suff}_{rng.randint(10, 999)}"
        dim = rng.choice([64, 128, 256, 512, 1024, 2048, 4096, 8192])
        lr = rng.choice([1e-4, 3e-4, 5e-5, 1e-3, 2.5e-4])

        if lang == "python":
            code = (
                f"@torch.no_grad()\n"
                f"def compute_{ident}(tensor: torch.Tensor, scale: float = {rng.uniform(0.1, 10.0):.4f}) -> dict[str, Any]:\n"
                f"    dim_h = tensor.size(-1)\n"
                f"    norm_val = tensor.norm(p=2, dim=-1, keepdim=True)\n"
                f"    return {{'normalized': tensor / (norm_val + 1e-8), 'dim': dim_h, 'lr': {lr}}}\n"
            )
        elif lang == "cpp":
            code = (
                f"template <typename T, size_t BlockSize = {dim}>\n"
                f"class {ident.title().replace('_', '')}Engine {{\n"
                f"public:\n"
                f"    explicit {ident.title().replace('_', '')}Engine(double alpha) : alpha_(alpha) {{}}\n"
                f"    T process(const T* input, size_t len) noexcept;\n"
                f"private:\n"
                f"    double alpha_;\n"
                f"    std::array<T, BlockSize> cache_;\n"
                f"}};\n"
            )
        elif lang == "rust":
            code = (
                f"pub struct {ident.title().replace('_', '')}Config {{\n"
                f"    pub max_capacity: usize,\n"
                f"    pub timeout_ms: u64,\n"
                f"    pub features: Vec<String>,\n"
                f"}}\n"
                f"impl {ident.title().replace('_', '')}Config {{\n"
                f"    pub fn default_preset() -> Self {{\n"
                f"        Self {{ max_capacity: {dim}, timeout_ms: {rng.randint(50, 5000)}, features: vec![\"async\".into()] }}\n"
                f"    }}\n"
                f"}}\n"
            )
        elif lang == "js":
            code = (
                f"export async function handle{ident.title().replace('_', '')}(req, res) {{\n"
                f"    const {{ queryId, batchSize = {dim} }} = req.body;\n"
                f"    const timestamp = Date.now();\n"
                f"    const result = await db.collection('{ident}').find({{ id: queryId }}).limit(batchSize).toArray();\n"
                f"    res.status(200).json({{ status: 'success', latencyMs: Date.now() - timestamp, data: result }});\n"
                f"}}\n"
            )
        elif lang == "go":
            code = (
                f"type {ident.title().replace('_', '')}Worker struct {{\n"
                f"    ID        int64         `json:\"id\"`\n"
                f"    Threshold float64       `json:\"threshold\"`\n"
                f"    Interval  time.Duration `json:\"interval\"`\n"
                f"}}\n"
                f"func (w *{ident.title().replace('_', '')}Worker) Execute(ctx context.Context) error {{\n"
                f"    return nil\n"
                f"}}\n"
            )
        elif lang == "java":
            code = (
                f"package com.kiara.core.{ident.lower()};\n\n"
                f"public final class {ident.title().replace('_', '')}Manager implements AutoCloseable {{\n"
                f"    private final int bufferSize = {dim};\n"
                f"    public int getBufferSize() {{ return this.bufferSize; }}\n"
                f"    @Override public void close() {{ /* cleanup */ }}\n"
                f"}}\n"
            )
        elif lang == "sql":
            code = (
                f"WITH telemetry_agg AS (\n"
                f"    SELECT device_id, count(*) as event_count, avg(latency_ms) as avg_lat\n"
                f"    FROM events_{ident}\n"
                f"    WHERE created_at >= '2026-01-01T00:00:00Z' AND partition_key = {rng.randint(1, 128)}\n"
                f"    GROUP BY device_id HAVING count(*) > {rng.randint(10, 100)}\n"
                f")\n"
                f"SELECT * FROM telemetry_agg ORDER BY avg_lat ASC LIMIT 25;\n"
            )
        else: # bash
            code = (
                f"#!/usr/bin/env bash\n"
                f"set -euo pipefail\n"
                f"export {ident.upper()}_MAX_THREADS={rng.randint(4, 64)}\n"
                f"mkdir -p /tmp/kiara/{ident}\n"
                f"curl -fsSL \"https://api.kiara.local/{ident}/status\" | jq '.healthy'\n"
            )
        add_if_disjoint(code)

    # 3. Mathematical Notation & LaTeX (~15%)
    target_math = int(target_samples * 0.15)
    greek_syms = [
        r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\zeta", r"\eta", r"\theta",
        r"\iota", r"\kappa", r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi", r"\rho", r"\sigma",
        r"\tau", r"\upsilon", r"\phi", r"\chi", r"\psi", r"\omega", r"\Gamma", r"\Delta",
        r"\Theta", r"\Lambda", r"\Sigma", r"\Phi", r"\Psi", r"\Omega"
    ]
    math_ops = [r"\int", r"\sum", r"\prod", r"\lim", r"\sup", r"\inf", r"\nabla", r"\partial"]
    for _ in range(target_math):
        s1 = rng.choice(greek_syms)
        s2 = rng.choice(greek_syms)
        op = rng.choice(math_ops)
        var = rng.choice(["x", "y", "z", "w", "t", r"\theta", r"\xi"])
        eq_type = rng.choice(["gaussian", "fourier", "entropy", "gradient", "expectation"])

        if eq_type == "gaussian":
            eq = rf"\mathcal{{N}}({var} \mid {s1}, {s2}^2) = \frac{{1}}{{\sqrt{{2\pi {s2}^2}}}} \exp\left(-\frac{{({var} - {s1})^2}}{{2{s2}^2}}\right)"
        elif eq_type == "fourier":
            eq = rf"\hat{{f}}(\omega) = \int_{{-\infty}}^{{+\infty}} f(t) e^{{-i \omega t}} dt \iff f(t) = \frac{{1}}{{2\pi}} \int_{{-\infty}}^{{+\infty}} \hat{{f}}(\omega) e^{{i \omega t}} d\omega"
        elif eq_type == "entropy":
            eq = rf"D_{{KL}}(P \parallel Q) = {op}_{{{var}}} P({var}) \log \frac{{P({var})}}{{Q({var})}} \ge 0 \quad (\text{{Gibbs' Inequality}})"
        elif eq_type == "gradient":
            eq = rf"\nabla_{{{s1}}} \mathcal{{L}}({s1}, {s2}) = \mathbb{{E}}_{{q}}\left[ \nabla_{{{s1}}} \log p({var} \mid {s1}) \right] + \lambda {s1}"
        else:
            eq = rf"\mathbb{{E}}_{{x \sim \mathcal{{D}}}}\left[ \| f_\theta(x) - y \|_2^2 \right] \le \frac{{\text{{Tr}}(\Sigma)}}{{N}} + \mathcal{{O}}\left(\frac{{1}}{{\sqrt{{N}}}}\right)"
        meta_math = f"Equation #{rng.randint(1000, 99999)}: {eq} [Condition: {s1} > 0, tolerance={rng.uniform(1e-9, 1e-4):.1e}]"
        add_if_disjoint(meta_math)

    # 4. Multilingual Text & Cross-Script Corpus (~15%)
    target_multi = int(target_samples * 0.15)
    for _ in range(target_multi):
        lang_code, base_sentence = rng.choice(_MULTILINGUAL_PATTERNS)
        ref_id = rng.randint(10000, 999999)
        val = rng.uniform(0.001, 0.999)
        augmented = f"[{lang_code.upper()}] {base_sentence} [Index: {ref_id}, Weight: {val:.4f}]"
        add_if_disjoint(augmented)

    # 5. Structured Data, URLs, Configs, Logs, Identifiers & Numbers (~15%)
    target_structured = target_samples - len(samples)
    for _ in range(target_structured):
        choice = rng.choice(["json", "yaml", "url_path", "log_event", "rare_identifier"])
        pref = rng.choice(_PREFIXES)
        root = rng.choice(all_domain_terms).replace("-", "_")
        suff = rng.choice(_SUFFIXES)
        uuid_a = f"{rng.randint(0x1000, 0xFFFF):04x}"
        uuid_b = f"{rng.randint(0x1000, 0xFFFF):04x}"
        uuid_c = f"{rng.randint(0x1000, 0xFFFF):04x}"
        full_uuid = f"{uuid_a}{uuid_b}-{uuid_c}-4{uuid_a[1:]}-a{uuid_b[1:]}-{uuid_c}{uuid_a}"

        if choice == "json":
            entry = (
                f'{{"id": "{full_uuid}", "service": "{pref}_{root}", "status_code": {rng.choice([200, 201, 204, 400, 404, 500])}, '
                f'"latency_ms": {rng.uniform(0.5, 250.0):.2f}, "active": {rng.choice(["true", "false"])}, '
                f'"memory_bytes": {rng.randint(1024, 1073741824)}, "tags": ["{root}", "{suff}", "v{rng.randint(1, 9)}"]}}'
            )
        elif choice == "yaml":
            entry = (
                f"model:\n"
                f"  name: 'kiara-{pref}-{root}'\n"
                f"  hidden_size: {rng.choice([512, 1024, 2048, 4096])}\n"
                f"  num_heads: {rng.choice([8, 16, 32, 64])}\n"
                f"  dropout_prob: {rng.choice([0.0, 0.05, 0.1])}\n"
                f"  checkpoint_uri: 's3://kiara-models/{root}/{full_uuid}.safetensors'\n"
            )
        elif choice == "url_path":
            domain = rng.choice(["kiara.ai", "huggingface.co", "github.com", "arxiv.org", "registry.npmjs.org", "crates.io"])
            ext = rng.choice(["tar.gz", "parquet", "safetensors", "json", "arrow", "bin", "whl"])
            entry = (
                f"https://{domain}/packages/{pref}_{root}/v{rng.randint(1, 10)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}/"
                f"download/{root}_{uuid_a}.{ext}?auth=bearer_token_{rng.randint(100000, 999999)}&expires=2026-12-31T23:59:59Z"
            )
        elif choice == "log_event":
            ip = f"{rng.randint(10, 220)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
            port = rng.randint(1024, 65535)
            entry = (
                f"2026-08-21T11:{rng.randint(10, 59)}:{rng.randint(10, 59)}.{rng.randint(100, 999)}Z "
                f"[LEVEL={rng.choice(['INFO', 'WARN', 'DEBUG'])}] cluster-node-{rng.randint(1, 64)} "
                f"({ip}:{port}) connection_id={full_uuid} processed_items={rng.randint(1, 10000)} "
                f"throughput={rng.uniform(10.0, 5000.0):.2f}req/s memory_used={rng.uniform(1.0, 64.0):.1f}GB"
            )
        else: # rare_identifier
            rare_camel = f"{pref.capitalize()}{root.title()}{suff.capitalize()}Strategy"
            rare_snake = f"{pref}_{root}_{suff}_pipeline_v{rng.randint(1, 9)}"
            rare_screaming = f"MAX_{pref.upper()}_{root.upper()}_{suff.upper()}_THRESHOLD"
            hex_val = f"0x{rng.randint(0x10000000, 0xFFFFFFFF):08X}"
            bin_val = f"0b{rng.randint(0, 255):08b}"
            entry = f"IdentifierSet: {rare_camel} | {rare_snake} | {rare_screaming} = {hex_val} (bitmask: {bin_val})"

        add_if_disjoint(entry)

    return samples


# Cache for generated training texts
_CACHED_TRAINING_CORPUS: list[str] | None = None


def get_training_texts(seed: int = DEFAULT_SEED, target_samples: int = 100000) -> list[str]:
    """Return the generated training corpus (cached in memory)."""
    global _CACHED_TRAINING_CORPUS
    if _CACHED_TRAINING_CORPUS is None:
        _CACHED_TRAINING_CORPUS = build_training_corpus(seed=seed, target_samples=target_samples)
    return _CACHED_TRAINING_CORPUS


def get_all_texts(categories: list[str] | None = None) -> list[str]:
    """Return text list: training texts if categories is None, else evaluation texts."""
    if categories is None:
        return get_training_texts()
    return get_eval_texts(categories)


# ---------------------------------------------------------------------------
# Corpus Statistics
# ---------------------------------------------------------------------------


def corpus_stats() -> dict[str, Any]:
    """Compute and return comprehensive statistics for training and evaluation corpora."""
    train_texts = get_training_texts()
    eval_corpus = get_eval_corpus()
    eval_texts = get_eval_texts()

    # Training stats
    train_chars = sum(len(t) for t in train_texts)
    train_words_list = [w for t in train_texts for w in t.split()]
    train_words_count = len(train_words_list)
    train_unique_words = len(set(train_words_list))
    train_unique_strings = len(set(train_texts))
    train_bytes = sum(len(t.encode("utf-8")) for t in train_texts)
    train_codepoints = len({ord(c) for t in train_texts for c in t})

    # Evaluation stats per category
    eval_by_cat: dict[str, dict[str, int]] = {}
    eval_total_chars = 0
    eval_total_words = 0
    all_eval_words: list[str] = []

    for cat_name, samples in eval_corpus.items():
        cat_words = [w for s in samples for w in s.split()]
        cat_chars = sum(len(s) for s in samples)
        eval_total_chars += cat_chars
        eval_total_words += len(cat_words)
        all_eval_words.extend(cat_words)
        eval_by_cat[cat_name] = {
            "num_samples": len(samples),
            "total_chars": cat_chars,
            "total_words": len(cat_words),
            "unique_words": len(set(cat_words)),
        }

    eval_unique_words = len(set(all_eval_words))
    eval_unique_strings = len(set(eval_texts))
    eval_bytes = sum(len(s.encode("utf-8")) for s in eval_texts)
    eval_codepoints = len({ord(c) for s in eval_texts for c in s})

    # Combined stats
    combined_codepoints = len({ord(c) for t in train_texts for c in t} | {ord(c) for s in eval_texts for c in s})
    is_disjoint = set(train_texts).isdisjoint(set(eval_texts))

    return {
        "version": CORPUS_VERSION,
        "is_disjoint": is_disjoint,
        "training": {
            "num_samples": len(train_texts),
            "total_chars": train_chars,
            "total_words": train_words_count,
            "unique_words": train_unique_words,
            "unique_strings": train_unique_strings,
            "approx_byte_count": train_bytes,
            "approx_mb": round(train_bytes / (1024 * 1024), 2),
            "unique_unicode_codepoints": train_codepoints,
        },
        "evaluation": {
            "num_samples": len(eval_texts),
            "total_chars": eval_total_chars,
            "total_words": eval_total_words,
            "unique_words": eval_unique_words,
            "unique_strings": eval_unique_strings,
            "approx_byte_count": eval_bytes,
            "unique_unicode_codepoints": eval_codepoints,
            "categories": eval_by_cat,
        },
        "overall": {
            "total_samples": len(train_texts) + len(eval_texts),
            "total_chars": train_chars + eval_total_chars,
            "total_words": train_words_count + eval_total_words,
            "unique_unicode_codepoints": combined_codepoints,
            "total_approx_mb": round((train_bytes + eval_bytes) / (1024 * 1024), 2),
        },
    }


if __name__ == "__main__":
    stats = corpus_stats()
    print(f"=== Kiara Tokenizer Benchmark Corpus Stats (v{stats['version']}) ===")
    print(f"Disjoint train/eval: {stats['is_disjoint']}")
    print("\nTraining Corpus:")
    for k, v in stats["training"].items():
        print(f"  {k:28s}: {v:,}" if isinstance(v, int) else f"  {k:28s}: {v}")

    print("\nEvaluation Corpus:")
    for k, v in stats["evaluation"].items():
        if k != "categories":
            print(f"  {k:28s}: {v:,}" if isinstance(v, int) else f"  {k:28s}: {v}")

    print("\nEvaluation Categories:")
    for cat, cat_s in stats["evaluation"]["categories"].items():
        print(f"  {cat:15s} -> samples={cat_s['num_samples']:3d}, chars={cat_s['total_chars']:6d}, words={cat_s['total_words']:5d}, unique_words={cat_s['unique_words']:5d}")
