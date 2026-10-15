# Kiara Production Roadmap

A hybrid development strategy (**Local Laptop for engineering & smoke tests + Free Google Colab / Kaggle GPUs for training**) to bring Kiara from research checkpoint to a production-ready conversational and reasoning model.

---

## 1. Hybrid Development & Training Loop
* **Local Laptop**: Code architecture, dataset pipelines, unit tests, and 50–100 step CPU smoke tests (under 60s, zero laptop strain).
* **Free Cloud GPUs (Colab / Kaggle T4/A100)**: Full 3,000–10,000 step training runs (15–25 mins with mixed precision), auto-saving checkpoints to Google Drive / Hugging Face Hub.
* **Deployment**: Local lightweight quantized CPU/ONNX runtime or cloud container.

---

## 2. Roadmap Phases

### Phase 1: Colab Integration & Stage 26D Scaling
- [ ] Create `experiments/smoke_test.py` for ultra-fast local shape/loss sanity checks.
- [ ] Create `notebooks/kiara_colab_trainer.ipynb` for push-button Colab T4 GPU execution.
- [ ] Train **Stage 26D.1** (Kiara-Small ~125M params) on Colab.
- [ ] Train **Stage 26D.2** (Context Window scaling to 256 and 512 tokens).

### Phase 2: Autoregressive Text Generation & Inference Engine
- [ ] Implement Key-Value (KV) cache in `KiaraAttention` for $O(1)$ token latency.
- [ ] Implement sampling module in `kiara/inference/generator.py` (Temperature, Top-p, Top-k, Repetition Penalty).
- [ ] Streaming token generation interface (`yield token`).
- [ ] Terminal chat REPL (`scripts/chat.py`).

### Phase 3: Automated Benchmark & Evaluation Suite
- [ ] Standardized multi-domain perplexity test on held-out datasets.
- [ ] Token completion accuracy, ROUGE-L, and repetition loop detection.
- [ ] Automated benchmark reporting CLI.

### Phase 4: Supervised Instruction Fine-Tuning (SFT)
- [ ] Define ChatML format (`<|im_start|>user ... <|im_end|><|im_start|>assistant ... <|im_end|>`).
- [ ] Curate 50k–100k sample multi-turn reasoning and instruction dataset.
- [ ] Masked cross-entropy loss (backprop only on assistant completion tokens).
- [ ] Fine-tune on Colab to produce `Kiara-Instruct`.

### Phase 5: Alignment & Safety
- [ ] Direct Preference Optimization (DPO) or rejection sampling for helpfulness and conciseness.
- [ ] Adversarial prompt testing and system persona verification.

### Phase 6: Production Serving & Deployment
- [ ] OpenAI-compatible FastAPI server (`/v1/chat/completions`, `/v1/models`).
- [ ] Export to Hugging Face `safetensors` format with full config and tokenizer files.
- [ ] INT8 / INT4 weight quantization for fast, instant local CPU inference.
- [ ] Modern web chat interface.
- [ ] Production Dockerfile.
