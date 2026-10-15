"""
experiments/pretrain.py — Step 25: Controlled Pre-Training Experiment
======================================================================
Staged diagnostic language model pre-training for Kiara.

Stages:
  25A: Tiny smoke test    — 500 samples, 200 steps  (Corpus v2)
  25B: Small validation   — 5,000 samples, 1,000 steps (Corpus v2)
  25C: Corpus v3 run     — 25,000 samples, 3,000 steps (Corpus v3, dual eval)

Primary metrics (C):
  train_loss | eval_a_loss | eval_a_ppl | eval_b_loss | eval_b_ppl
  bits_per_token | generalization_gap

Secondary diagnostics (logged to JSON, not primary table):
  token_accuracy | grad_norm | lr

Scientific constraint:
  Stage C uses the SAME model, tokenizer, and optimizer as Stage B.
  Only the corpus changes. This makes it a controlled experiment.

Usage:
  python experiments/pretrain.py --stage A
  python experiments/pretrain.py --stage B
  python experiments/pretrain.py --stage C
  python experiments/pretrain.py --stage C --resume checkpoints/stage_C/checkpoint_step_500.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Ensure project root is on the import path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.tokenizer_benchmark.corpus import (
    build_training_corpus,
    get_eval_corpus,
    get_eval_texts,
)
from experiments.tokenizer_benchmark.corpus_v3 import (
    build_v3_training_corpus,
    get_v3_eval_a_corpus,
    get_v3_eval_a_texts,
    get_v3_eval_b_corpus,
    _get_eval_b_texts,
    DEFAULT_SEED as V3_SEED,
)
from kiara.data import DataCollatorForCausalLM, PackedDataset
from kiara.model import KiaraConfig, KiaraForCausalLM
from kiara.tokenizer import KiaraTokenizer
from kiara.trainer import (
    build_optimizer,
    get_cosine_schedule_with_warmup,
    load_checkpoint,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Stage configurations
# ---------------------------------------------------------------------------

STAGE_CONFIGS: dict[str, dict] = {
    "A": {
        "description": "Tiny smoke test — 500 training samples, 200 steps (Corpus v2)",
        "corpus": "v2",
        "train_samples": 500,
        "max_steps": 200,
        "warmup_steps": 20,
        "log_interval": 10,
        "eval_interval": 50,
        "save_interval": 100,
        "seq_len": 64,
        "batch_size": 8,
        "learning_rate": 3e-4,
    },
    "B": {
        "description": "Small validation — 5,000 training samples, 1,000 steps (Corpus v2)",
        "corpus": "v2",
        "train_samples": 5000,
        "max_steps": 1000,
        "warmup_steps": 50,
        "log_interval": 25,
        "eval_interval": 100,
        "save_interval": 250,
        "seq_len": 64,
        "batch_size": 16,
        "learning_rate": 3e-4,
    },
    "C": {
        "description": "Corpus v3.0 — 25,000 samples, 3,000 steps (controlled generalization experiment)",
        "corpus": "v3",
        "train_samples": 25000,
        "max_steps": 3000,
        "warmup_steps": 150,
        "log_interval": 50,
        "eval_interval": 250,
        "save_interval": 500,
        "seq_len": 64,
        "batch_size": 32,
        "learning_rate": 3e-4,
    },
}

# ---------------------------------------------------------------------------
# Model architecture — FROZEN. Do not change between stages.
# ---------------------------------------------------------------------------
MODEL_CONFIG = KiaraConfig(
    vocab_size=32768,
    hidden_size=512,
    intermediate_size=1536,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=128,
    tie_word_embeddings=True,
)

SEED = 1337
LN2 = math.log(2.0)


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def format_number(n: float | int) -> str:
    if n >= 1e6:
        return f"{n/1e6:.3f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


@torch.no_grad()
def generate_samples(
    model: KiaraForCausalLM,
    tokenizer: KiaraTokenizer,
    prompts: list[str],
    max_new_tokens: int = 40,
    device: torch.device = torch.device("cpu"),
) -> list[dict[str, str]]:
    model.eval()
    results = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.0)
        decoded = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        results.append({"prompt": prompt, "generated": decoded})
    model.train()
    return results


@torch.no_grad()
def compute_token_accuracy(model, input_ids, labels, device):
    """Diagnostic only — not a primary metric."""
    model.eval()
    out = model(input_ids=input_ids.to(device))
    shift_logits = out.logits[..., :-1, :]
    shift_labels = labels[..., 1:].to(device)
    mask = shift_labels != -100
    if not mask.any():
        model.train()
        return 0.0
    preds = torch.argmax(shift_logits, dim=-1)
    acc = (preds[mask] == shift_labels[mask]).float().mean().item()
    model.train()
    return acc


@torch.no_grad()
def run_eval(
    model: KiaraForCausalLM,
    loader: DataLoader,
    device: torch.device,
    name: str = "eval",
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                labels=batch["labels"].to(device),
            )
            total_loss += out.loss.item()
            n_batches += 1
    model.train()
    avg_loss = total_loss / max(1, n_batches)
    ppl = math.exp(min(avg_loss, 20.0))
    bpt = avg_loss / LN2
    return {
        f"{name}_loss": round(avg_loss, 6),
        f"{name}_ppl": round(ppl, 4),
        f"{name}_bpt": round(bpt, 6),
    }


def make_loader(
    texts: list[str],
    tokenizer: KiaraTokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ds = PackedDataset.from_texts(texts, tokenizer, seq_len=seq_len, bos=False, eos=True)
    collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        drop_last=shuffle,
    )


def build_datasets_v2(stage_cfg, tokenizer):
    """Corpus v2 — used for Stage A and B (regression only)."""
    print(f"  [corpus v2] Generating {stage_cfg['train_samples']:,} training texts...", flush=True)
    t0 = time.time()
    all_train = build_training_corpus(seed=SEED, target_samples=100000)
    train_texts = all_train[: stage_cfg["train_samples"]]
    eval_texts = get_eval_texts()
    print(f"  [corpus v2] Done in {time.time()-t0:.1f}s | train={len(train_texts):,}  eval={len(eval_texts):,}", flush=True)

    seq_len = stage_cfg["seq_len"]
    train_loader = make_loader(train_texts, tokenizer, seq_len, stage_cfg["batch_size"], True, SEED)
    # v2 has only one eval set; map it to eval_b slot, eval_a=None
    eval_b_loader = make_loader(eval_texts, tokenizer, seq_len, stage_cfg["batch_size"], False, SEED)
    train_ds = train_loader.dataset
    print(
        f"  [dataset] train_chunks={len(train_ds):,} "
        f"({format_number(train_ds.num_tokens)} tokens) | "
        f"eval_chunks={len(eval_b_loader.dataset):,}",
        flush=True,
    )
    return train_loader, None, eval_b_loader  # (train, eval_a=None, eval_b)


def build_datasets_v3(stage_cfg, tokenizer):
    """Corpus v3 — used for Stage C (controlled generalization experiment)."""
    print(f"  [corpus v3] Building {stage_cfg['train_samples']:,} training samples...", flush=True)
    t0 = time.time()
    train_texts = build_v3_training_corpus(seed=V3_SEED, target_samples=stage_cfg["train_samples"])
    eval_a_texts = get_v3_eval_a_texts()          # 30 held-out generalization samples
    eval_b_texts = _get_eval_b_texts()            # 173 robustness samples
    elapsed = time.time() - t0
    print(
        f"  [corpus v3] Done in {elapsed:.1f}s | "
        f"train={len(train_texts):,}  eval_a={len(eval_a_texts)}  eval_b={len(eval_b_texts)}",
        flush=True,
    )

    seq_len = stage_cfg["seq_len"]
    train_loader = make_loader(train_texts, tokenizer, seq_len, stage_cfg["batch_size"], True, V3_SEED)
    eval_a_loader = make_loader(eval_a_texts, tokenizer, seq_len, stage_cfg["batch_size"], False, V3_SEED)
    eval_b_loader = make_loader(eval_b_texts, tokenizer, seq_len, stage_cfg["batch_size"], False, V3_SEED)

    train_ds = train_loader.dataset
    print(
        f"  [dataset] train_chunks={len(train_ds):,} "
        f"({format_number(train_ds.num_tokens)} tokens) | "
        f"eval_a_chunks={len(eval_a_loader.dataset):,} | "
        f"eval_b_chunks={len(eval_b_loader.dataset):,}",
        flush=True,
    )
    return train_loader, eval_a_loader, eval_b_loader


SAMPLE_PROMPTS = [
    "The quick brown fox",
    "def attention(",
    "Machine learning is",
]


def train_stage(stage: str, resume_path: str | None = None) -> dict:
    stage_cfg = STAGE_CONFIGS[stage]
    corpus_ver = stage_cfg["corpus"]
    dual_eval = corpus_ver == "v3"  # Stage C uses dual Eval A + Eval B

    print(f"\n{'='*80}", flush=True)
    print(f"  KIARA PRE-TRAINING — Stage {stage}", flush=True)
    print(f"  {stage_cfg['description']}", flush=True)
    print(f"  Corpus: {corpus_ver}  |  Dual eval: {dual_eval}", flush=True)
    print(f"{'='*80}\n", flush=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device   : {device}", flush=True)

    output_dir = Path(f"checkpoints/stage_{stage}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training_log.jsonl"

    # Tokenizer
    tokenizer = KiaraTokenizer.load()
    print(f"  Tokenizer: vocab_size={tokenizer.vocab_size}", flush=True)

    # Model — identical across all stages
    total_params = MODEL_CONFIG.calculate_expected_parameters()
    print(
        f"  Model    : Kiara-Tiny | "
        f"{format_number(total_params['total'])} params | "
        f"{format_number(total_params['non_embedding'])} non-embedding",
        flush=True,
    )
    model = KiaraForCausalLM(MODEL_CONFIG)
    model.to(device)

    # Datasets
    if corpus_ver == "v3":
        train_loader, eval_a_loader, eval_b_loader = build_datasets_v3(stage_cfg, tokenizer)
    else:
        train_loader, eval_a_loader, eval_b_loader = build_datasets_v2(stage_cfg, tokenizer)

    # Optimizer + scheduler — identical across all stages
    optimizer = build_optimizer(model, lr=stage_cfg["learning_rate"], weight_decay=0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=stage_cfg["warmup_steps"],
        num_training_steps=stage_cfg["max_steps"],
        min_lr_ratio=0.1,
    )

    # Resume
    global_step = 0
    if resume_path:
        meta = load_checkpoint(resume_path, model=model, optimizer=optimizer, scheduler=scheduler)
        global_step = meta.get("global_step", 0)
        print(f"  Resumed from: {resume_path} (step {global_step})", flush=True)

    history: list[dict] = []
    accum_loss = accum_acc = 0.0
    accum_steps = 0
    train_start = time.time()

    # ---------- header ----------
    if dual_eval:
        print(
            f"\n  {'Step':>6}  {'TrainLoss':>9}  {'TrainPPL':>8}  "
            f"{'EvalA_Loss':>10}  {'EvalA_PPL':>9}  "
            f"{'EvalB_Loss':>10}  {'EvalB_PPL':>9}  "
            f"{'BPT':>6}  {'Gap':>7}  {'LR':>10}",
            flush=True,
        )
        print(f"  {'-'*120}", flush=True)
    else:
        print(
            f"\n  {'Step':>6}  {'TrainLoss':>9}  {'TrainPPL':>8}  "
            f"{'EvalLoss':>10}  {'EvalPPL':>9}  "
            f"{'GradNorm':>8}  {'LR':>10}",
            flush=True,
        )
        print(f"  {'-'*90}", flush=True)

    # ---------- initial eval ----------
    model.eval()
    if eval_a_loader is not None:
        init_a = run_eval(model, eval_a_loader, device, "eval_a")
        init_b = run_eval(model, eval_b_loader, device, "eval_b")
        print(
            f"  {'0':>6}  {'--':>9}  {'--':>8}  "
            f"{init_a['eval_a_loss']:>10.4f}  {init_a['eval_a_ppl']:>9.2f}  "
            f"{init_b['eval_b_loss']:>10.4f}  {init_b['eval_b_ppl']:>9.2f}  "
            f"{'--':>6}  {'--':>7}  {'--':>10}",
            flush=True,
        )
    else:
        init_b = run_eval(model, eval_b_loader, device, "eval_b")
        print(
            f"  {'0':>6}  {'--':>9}  {'--':>8}  "
            f"{init_b['eval_b_loss']:>10.4f}  {init_b['eval_b_ppl']:>9.2f}  "
            f"{'--':>8}  {'--':>10}",
            flush=True,
        )
    model.train()

    # ---------- training loop ----------
    while global_step < stage_cfg["max_steps"]:
        for batch in train_loader:
            if global_step >= stage_cfg["max_steps"]:
                break

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step  += 1
            accum_loss   += loss.item()
            accum_acc    += compute_token_accuracy(model, input_ids, labels, device)
            accum_steps  += 1

            if global_step % stage_cfg["log_interval"] == 0:
                avg_loss = accum_loss / accum_steps
                avg_acc  = accum_acc  / accum_steps
                avg_ppl  = math.exp(min(avg_loss, 20.0))
                current_lr = scheduler.get_last_lr()[0]

                log_entry: dict = {
                    "step": global_step,
                    "train_loss": round(avg_loss, 6),
                    "train_ppl":  round(avg_ppl,  4),
                    "grad_norm":  round(float(grad_norm), 4),
                    "lr":         current_lr,
                    "token_accuracy": round(avg_acc * 100, 2),  # diagnostic
                    "elapsed_s":  round(time.time() - train_start, 1),
                }

                # ---------- eval ----------
                if global_step % stage_cfg["eval_interval"] == 0:
                    if eval_a_loader is not None:
                        ea = run_eval(model, eval_a_loader, device, "eval_a")
                        eb = run_eval(model, eval_b_loader, device, "eval_b")
                        gap = ea["eval_a_loss"] - avg_loss
                        log_entry.update(ea)
                        log_entry.update(eb)
                        log_entry["generalization_gap"] = round(gap, 6)
                        log_entry["bits_per_token"]     = ea["eval_a_bpt"]
                        print(
                            f"  {global_step:>6}  {avg_loss:>9.4f}  {avg_ppl:>8.2f}  "
                            f"{ea['eval_a_loss']:>10.4f}  {ea['eval_a_ppl']:>9.2f}  "
                            f"{eb['eval_b_loss']:>10.4f}  {eb['eval_b_ppl']:>9.2f}  "
                            f"{ea['eval_a_bpt']:>6.3f}  {gap:>+7.3f}  {current_lr:>10.6f}",
                            flush=True,
                        )
                    else:
                        eb = run_eval(model, eval_b_loader, device, "eval_b")
                        log_entry.update(eb)
                        print(
                            f"  {global_step:>6}  {avg_loss:>9.4f}  {avg_ppl:>8.2f}  "
                            f"{eb['eval_b_loss']:>10.4f}  {eb['eval_b_ppl']:>9.2f}  "
                            f"{float(grad_norm):>8.4f}  {current_lr:>10.6f}",
                            flush=True,
                        )
                else:
                    if dual_eval:
                        print(
                            f"  {global_step:>6}  {avg_loss:>9.4f}  {avg_ppl:>8.2f}  "
                            f"{'--':>10}  {'--':>9}  {'--':>10}  {'--':>9}  "
                            f"{'--':>6}  {'--':>7}  {current_lr:>10.6f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"  {global_step:>6}  {avg_loss:>9.4f}  {avg_ppl:>8.2f}  "
                            f"{'--':>10}  {'--':>9}  "
                            f"{float(grad_norm):>8.4f}  {current_lr:>10.6f}",
                            flush=True,
                        )

                history.append(log_entry)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

                accum_loss = accum_acc = 0.0
                accum_steps = 0

            # ---------- checkpoint ----------
            if global_step % stage_cfg["save_interval"] == 0:
                ckpt_path = output_dir / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(
                    target_path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=0,
                    loss=history[-1]["train_loss"] if history else None,
                    config=MODEL_CONFIG,
                    tokenizer_metadata={"vocab_size": tokenizer.vocab_size},
                )
                print(f"\n  [ckpt] Saved checkpoint -> {ckpt_path}", flush=True)
                samples = generate_samples(model, tokenizer, SAMPLE_PROMPTS, max_new_tokens=30, device=device)
                print(f"\n  [samples @ step {global_step}]", flush=True)
                for s in samples:
                    print(f"    Prompt   : {s['prompt']}", flush=True)
                    print(f"    Generated: {s['generated']}", flush=True)
                print(flush=True)

    # ---------- final eval ----------
    final_ckpt = output_dir / "checkpoint_final.pt"
    save_checkpoint(
        target_path=final_ckpt, model=model, optimizer=optimizer, scheduler=scheduler,
        global_step=global_step, epoch=0,
        loss=history[-1]["train_loss"] if history else None,
        config=MODEL_CONFIG, tokenizer_metadata={"vocab_size": tokenizer.vocab_size},
    )

    if eval_a_loader is not None:
        final_a = run_eval(model, eval_a_loader, device, "eval_a")
        final_b = run_eval(model, eval_b_loader, device, "eval_b")
        final_gap = final_a["eval_a_loss"] - (history[-1]["train_loss"] if history else 0.0)
    else:
        final_a = {}
        final_b = run_eval(model, eval_b_loader, device, "eval_b")
        final_gap = None

    print(f"\n{'='*80}", flush=True)
    print(f"  Stage {stage} complete — {global_step} steps", flush=True)
    print(f"  Final train loss     : {history[-1]['train_loss']:.6f}", flush=True)
    print(f"  Final train PPL      : {history[-1]['train_ppl']:.4f}", flush=True)
    if final_a:
        print(f"  Final Eval A loss    : {final_a['eval_a_loss']:.6f}", flush=True)
        print(f"  Final Eval A PPL     : {final_a['eval_a_ppl']:.4f}", flush=True)
        print(f"  Final Eval A BPT     : {final_a['eval_a_bpt']:.4f}", flush=True)
        print(f"  Final Gen. gap       : {final_gap:+.4f}", flush=True)
    print(f"  Final Eval B loss    : {final_b['eval_b_loss']:.6f}", flush=True)
    print(f"  Final Eval B PPL     : {final_b['eval_b_ppl']:.4f}", flush=True)
    print(f"  Log file             : {log_file}", flush=True)
    print(f"  Final checkpoint     : {final_ckpt}", flush=True)
    print(f"{'='*80}\n", flush=True)

    # NaN/Inf health check
    clean = True
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"  [CRITICAL] NaN/Inf in: {name}", flush=True)
            clean = False
            break
    if clean:
        print(f"  [health] No NaN/Inf in any parameter. [OK]", flush=True)

    # Final generated samples
    final_samples = generate_samples(model, tokenizer, SAMPLE_PROMPTS, max_new_tokens=40, device=device)
    print(f"\n  [final samples]", flush=True)
    for s in final_samples:
        print(f"    Prompt   : {s['prompt']}", flush=True)
        print(f"    Generated: {s['generated']}", flush=True)
    print(flush=True)

    result = {
        "stage": stage,
        "corpus": corpus_ver,
        "global_step": global_step,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_train_ppl":  history[-1]["train_ppl"]  if history else None,
        "final_eval_a_loss": final_a.get("eval_a_loss"),
        "final_eval_a_ppl":  final_a.get("eval_a_ppl"),
        "final_eval_a_bpt":  final_a.get("eval_a_bpt"),
        "final_eval_b_loss": final_b["eval_b_loss"],
        "final_eval_b_ppl":  final_b["eval_b_ppl"],
        "final_generalization_gap": final_gap,
        "checkpoint": str(final_ckpt),
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved -> {results_file}", flush=True)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiara Step 25: Controlled Pre-Training")
    parser.add_argument(
        "--stage", choices=["A", "B", "C"], default="C",
        help="Training stage (A=tiny smoke, B=small val, C=corpus v3 generalization)",
    )
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train_stage(args.stage, resume_path=args.resume)
