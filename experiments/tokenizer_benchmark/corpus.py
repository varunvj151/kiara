"""
corpus.py — Kiara Tokenizer Benchmark Corpus Builder
=====================================================
Builds a controlled, reproducible, local benchmark corpus.
No external downloads. All text is synthetic or public-domain.

Each category returns a list of strings.
The corpus is intentionally small so it runs quickly on CPU.
"""

from __future__ import annotations

CORPUS_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------


def english_prose() -> list[str]:
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
    ]


def technical_english() -> list[str]:
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
    ]


def source_code() -> list[str]:
    """Source code samples: Python, C++, JavaScript."""
    return [
        # Python
        "import torch\nimport torch.nn as nn\n",
        "def scaled_dot_product_attention(q, k, v, mask=None):\n    d_k = q.size(-1)\n    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)\n    if mask is not None:\n        scores = scores.masked_fill(mask == 0, float('-inf'))\n    return torch.matmul(torch.softmax(scores, dim=-1), v)\n",
        "class RMSNorm(nn.Module):\n    def __init__(self, dim: int, eps: float = 1e-8):\n        super().__init__()\n        self.eps = eps\n        self.weight = nn.Parameter(torch.ones(dim))\n\n    def forward(self, x):\n        norm = x.norm(2, dim=-1, keepdim=True)\n        return self.weight * x / (norm + self.eps)\n",
        "for epoch in range(num_epochs):\n    optimizer.zero_grad()\n    logits = model(input_ids)\n    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))\n    loss.backward()\n    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n    optimizer.step()\n",
        "def get_cosine_schedule(optimizer, warmup_steps, total_steps):\n    def lr_lambda(current_step):\n        if current_step < warmup_steps:\n            return float(current_step) / float(max(1, warmup_steps))\n        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))\n        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))\n    return LambdaLR(optimizer, lr_lambda)\n",
        # JavaScript
        "const transformer = new TransformerModel({ hiddenSize: 1024, numLayers: 24, numHeads: 16 });\n",
        "async function generateText(prompt, maxTokens = 200) {\n    const tokens = tokenizer.encode(prompt);\n    for (let i = 0; i < maxTokens; i++) {\n        const logits = await model.forward(tokens);\n        const nextToken = sampleTopK(logits, k=50);\n        tokens.push(nextToken);\n        if (nextToken === tokenizer.eosId) break;\n    }\n    return tokenizer.decode(tokens);\n}\n",
        # C++
        "template<typename T>\nclass AttentionLayer {\npublic:\n    AttentionLayer(int hidden_size, int num_heads)\n        : hidden_size_(hidden_size), num_heads_(num_heads),\n          head_dim_(hidden_size / num_heads) {}\n\n    torch::Tensor forward(torch::Tensor x, torch::optional<torch::Tensor> mask = torch::nullopt);\nprivate:\n    int hidden_size_, num_heads_, head_dim_;\n};\n",
        # Shell / config
        "#!/bin/bash\npython train.py \\\n    --config configs/tiny.yaml \\\n    --seed 1337 \\\n    --dtype bfloat16 \\\n    --max_steps 10000 \\\n    --checkpoint_interval 1000\n",
        # Identifiers and operators
        "get_transformer_block_hidden_state_representation",
        "CounterfactualReasoningModuleForMultiHopQuestionAnswering",
        "loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-1)",
        "self._attn_weights = torch.zeros(batch, heads, seq_len, seq_len, device=device, dtype=dtype)",
        "0x1F 0xFF 0b10110101 3.14159265358979 2.718281828459 1e-8 6.022e23",
    ]


def mathematics() -> list[str]:
    """Mathematical notation: LaTeX, Unicode symbols, equations."""
    return [
        # LaTeX
        r"\frac{\partial L}{\partial \theta} = -\frac{1}{N} \sum_{i=1}^{N} y_i \log \hat{y}_i",
        r"\mathbb{E}_{x \sim p(x)}[f(x)] = \int_{-\infty}^{+\infty} f(x) p(x) \, dx",
        r"\hat{\theta}_{MLE} = \arg\max_{\theta} \sum_{i=1}^{N} \log p(x_i | \theta)",
        r"P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}",
        r"\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right]^T",
        r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V",
        r"e^{i\pi} + 1 = 0",
        r"\sum_{n=0}^{\infty} \frac{x^n}{n!} = e^x",
        # Unicode math symbols
        "∀ε > 0, ∃δ > 0: |x − a| < δ ⟹ |f(x) − f(a)| < ε",
        "∇·E = ρ/ε₀, ∇×B = μ₀J + μ₀ε₀ ∂E/∂t",
        "ℝ, ℂ, ℕ, ℤ, ℚ — standard number set notation",
        "∫₀^∞ e^{−x²} dx = √π / 2",
        "λ₁ + λ₂ + ⋯ + λₙ = tr(A)",
        "‖x‖₂ = √(x₁² + x₂² + ⋯ + xₙ²)",
        "σ(x) = 1 / (1 + e^{−x})",
        # Number sequences
        "3.14159265358979323846",
        "2.71828182845904523536",
        "1.41421356237309504880",
        "1.61803398874989484820",
        "0.57721566490153286060",
    ]


def multilingual() -> list[str]:
    """Text samples from multiple languages and scripts."""
    return [
        # Romance languages
        "El aprendizaje automático es una rama de la inteligencia artificial que permite a los sistemas aprender.",
        "Les modèles de fondation nécessitent des corpus d'entraînement massifs et diversifiés.",
        "L'intelligenza artificiale sta rivoluzionando molti settori dell'economia moderna.",
        "A inteligência artificial está transformando a maneira como interagimos com a tecnologia.",
        # Germanic
        "Sprachmodelle werden auf großen Textmengen aus dem Internet vortrainiert und dann feinabgestimmt.",
        "Kunstmatige intelligentie heeft grote vooruitgang geboekt op het gebied van taalbegrip.",
        # CJK
        "機械学習は人工知能の一分野であり、コンピューターがデータからパターンを学習することを可能にします。",
        "机器学习是人工智能的一个子领域，使计算机能够从数据中自动学习和改进。",
        "머신러닝은 컴퓨터가 데이터에서 패턴을 학습하여 예측을 수행하는 인공지능의 한 분야입니다.",
        # Right-to-left scripts
        "التعلم الآلي هو فرع من فروع الذكاء الاصطناعي يمكن الحاسوب من التعلم من البيانات.",
        "یادگیری ماشین شاخه‌ای از هوش مصنوعی است که به رایانه‌ها امکان یادگیری از داده‌ها را می‌دهد.",
        # Indic
        "मशीन लर्निंग कृत्रिम बुद्धिमत्ता का एक उप-क्षेत्र है जो कंप्यूटर को डेटा से सीखने में सक्षम बनाता है।",
        "যন্ত্র শিক্ষা কৃত্রিম বুদ্ধিমত্তার একটি শাখা যা কম্পিউটারকে ডেটা থেকে শিখতে সক্ষম করে।",
        # Other
        "Makine öğrenmesi, bilgisayarların verilerden öğrenmesini sağlayan yapay zekanın bir dalıdır.",
        "Uczenie maszynowe to dziedzina sztucznej inteligencji umożliwiająca komputerom uczenie się z danych.",
    ]


def rare_and_long_words() -> list[str]:
    """Rare, long, and morphologically complex words."""
    return [
        # Long medical/scientific words
        "electroencephalography",
        "deoxyribonucleic acid",
        "immunohistochemistry",
        "pneumonoultramicroscopicsilicovolcanoconiosis",
        "hippopotomonstrosesquipedaliophobia",
        "antidisestablishmentarianism",
        "floccinaucinihilipilification",
        "supercalifragilisticexpialidocious",
        # Technical compound words
        "CounterfactualReasoningModuleForMultiHopQuestionAnswering",
        "TransformerBasedSequenceToSequenceModelWithGroupedQueryAttention",
        "get_scaled_dot_product_attention_weights_for_causal_mask",
        "MultiHeadSelfAttentionWithRotaryPositionalEmbeddingsAndRMSNorm",
        # Inflected forms (English morphology)
        "running ran runs runner runners runnable",
        "tokenize tokenizer tokenization tokenized tokenizable detokenize",
        "unbelievable unbelievably unbelievableness counterintuitively",
        "anthropomorphization anthropomorphized anthropomorphizing",
        # Domain-specific jargon
        "eigenvector eigenvalue eigendecomposition pseudoinverse",
        "backpropagation vanishing-gradient exploding-gradient hyperparameter",
        "quantization dequantization int8 bfloat16 float32 mixed-precision",
        "LongFormer BigBird FlashAttention GroupedQueryAttention LoRA QLoRA",
    ]


def unicode_and_emoji() -> list[str]:
    """Unicode edge cases, emoji, and special symbols."""
    return [
        # Basic emoji
        "Hello 🌍🚀🧠🔥💡✨🎯📊",
        "Great work! 👏🎉🏆⭐🌟",
        "Weather: ☀️🌤️⛅🌧️⛈️🌩️❄️🌈",
        # Emoji with skin tone modifiers
        "👋🏻👋🏼👋🏽👋🏾👋🏿",
        # Compound emoji (ZWJ sequences)
        "👨‍💻👩‍🔬👨‍🚀👩‍🎨🧑‍🏫",
        # Unicode math and symbols
        "α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω",
        "Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω",
        "∀ ∃ ∄ ∈ ∉ ∋ ∌ ∩ ∪ ⊂ ⊃ ⊄ ⊅ ⊆ ⊇",
        "∫ ∬ ∭ ∮ ∯ ∰ ∱ ∲ ∳",
        "← → ↑ ↓ ↔ ↕ ⇐ ⇒ ⇑ ⇓ ⇔",
        "✓ ✗ ✘ ✔ ✕ ✖ ✙ ✚ ✛ ✜ ✝ ✞",
        # Diacritics and accents
        "café naïve résumé façade coördinate Ångström",
        "über über über — German umlaut",
        "Ñoño, señor, jalapeño — Spanish tilde-n",
        # East Asian width
        "ＡＢＣＤＥＦＧ — fullwidth Latin",
        "｢｣ 【】 『』 「」 — CJK brackets",
        # RTL mixed with LTR
        "The word مرحبا means 'hello' in Arabic.",
        "שלום means 'peace' or 'hello' in Hebrew.",
        # Superscripts and subscripts
        "x⁰ x¹ x² x³ x⁴ x⁵ H₂O CO₂ C₆H₁₂O₆",
        # Rare Unicode planes
        "𝒜𝓁𝓰𝑒𝒷𝓇𝒶 — Mathematical Script",
        "𝔄𝔅ℭ𝔇𝔈𝔉𝔊 — Mathematical Fraktur",
        "\u0000 — null byte (edge case)",
    ]


def urls_and_symbols() -> list[str]:
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
    ]


# ---------------------------------------------------------------------------
# Corpus registry
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, list[str]] = {
    "english": english_prose(),
    "technical": technical_english(),
    "code": source_code(),
    "math": mathematics(),
    "multilingual": multilingual(),
    "rare_words": rare_and_long_words(),
    "unicode": unicode_and_emoji(),
    "urls_symbols": urls_and_symbols(),
}


def get_corpus(categories: list[str] | None = None) -> dict[str, list[str]]:
    """Return corpus dict filtered to requested categories.

    Args:
        categories: List of category names to include. None = all categories.

    Returns:
        Dict mapping category name -> list of text strings.
    """
    if categories is None:
        return dict(CATEGORIES)
    return {k: CATEGORIES[k] for k in categories if k in CATEGORIES}


def get_all_texts(categories: list[str] | None = None) -> list[str]:
    """Return all text strings concatenated across categories."""
    corpus = get_corpus(categories)
    texts: list[str] = []
    for samples in corpus.values():
        texts.extend(samples)
    return texts


def corpus_stats() -> dict[str, dict]:
    """Return basic statistics about each corpus category."""
    stats = {}
    for name, samples in CATEGORIES.items():
        all_text = " ".join(samples)
        stats[name] = {
            "num_samples": len(samples),
            "total_chars": len(all_text),
            "total_words": len(all_text.split()),
        }
    return stats


if __name__ == "__main__":
    print(f"Corpus version: {CORPUS_VERSION}\n")
    for cat, s in corpus_stats().items():
        print(f"  {cat:15s}  samples={s['num_samples']:3d}  chars={s['total_chars']:6d}  words={s['total_words']:5d}")
