"""
experiments/pretrain_real.py — Step 26A: Controlled Real-Data Pre-Training Experiment
=======================================================================================
Executes Phase 26A.3: Controlled pre-training of the 40.9M Kiara-Tiny Transformer
on the real-world text corpus (Kiara-Real-v1.0: TinyStories + Wikitext).

Evaluates simultaneously on:
  1. Eval Real (Held-out same-distribution natural text: 1,535 documents / ~578K tokens)
  2. Eval Robustness (Standard 173-sample benchmark suite)

Primary Metrics:
  - train_loss | train_ppl
  - eval_real_loss | eval_real_ppl | eval_real_bpt
  - eval_robust_loss | eval_robust_ppl
  - generalization_gap (eval_real_loss - train_loss)

Invariants (FROZEN):
  - Tokenizer: Byte-Level BPE 32K (Decision 003)
  - Architecture: Kiara-Tiny 40,903,168 parameters
  - Optimizer: AdamW (lr=3e-4, wd=0.1, clip=1.0)
  - Schedule: 150-step linear warmup + cosine decay to 10%
  - Total steps: 3,000 steps (matching Stage 25C)
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.tokenizer_benchmark.corpus_v3 import _get_eval_b_texts
from kiara.data import DataCollatorForCausalLM, PackedDataset
from kiara.model import KiaraConfig, KiaraForCausalLM
from kiara.tokenizer import KiaraTokenizer
from kiara.trainer import (
    build_optimizer,
    get_cosine_schedule_with_warmup,
    load_checkpoint,
    save_checkpoint,
)

DATA_DIR = REPO_ROOT / "data" / "real"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "stage_26A"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Architecture Configuration — 100% FROZEN from Stage 25
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

TRAIN_CONFIG = {
    "train_steps": 3000,
    "warmup_steps": 150,
    "log_interval": 50,
    "eval_interval": 250,
    "save_interval": 500,
    "seq_len": 64,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "seed": 1337,
}

SAMPLE_PROMPTS = [
    "Once upon a time",
    "The quick brown fox",
    "def attention(",
    "In scientific research,",
]

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


def train_real_stage(resume_path: str | None = None) -> dict:
    print(f"\n{'='*80}", flush=True)
    print(f"  KIARA PRE-TRAINING — STAGE 26A (REAL-WORLD TEXT PRE-TRAINING)", flush=True)
    print(f"  Corpus: Kiara-Real-v1.0 (TinyStories + Wikitext) | 3,000 steps", flush=True)
    print(f"  Model : Kiara-Tiny (40.903M params) | Dual Evaluation (Real + Robustness)", flush=True)
    print(f"{'='*80}\n", flush=True)

    set_seed(TRAIN_CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device   : {device}", flush=True)

    log_file = CHECKPOINT_DIR / "training_log.jsonl"
    tokenizer = KiaraTokenizer.load()
    print(f"  Tokenizer: Byte-Level BPE (vocab_size={tokenizer.vocab_size:,})", flush=True)

    total_params = MODEL_CONFIG.calculate_expected_parameters()
    print(
        f"  Model    : Kiara-Tiny | "
        f"{format_number(total_params['total'])} params | "
        f"{format_number(total_params['non_embedding'])} non-embedding",
        flush=True,
    )
    model = KiaraForCausalLM(MODEL_CONFIG)
    model.to(device)

    # Load Real Corpus Files
    train_file = DATA_DIR / "train.jsonl"
    eval_file = DATA_DIR / "eval_real.jsonl"

    if not train_file.exists() or not eval_file.exists():
        raise FileNotFoundError("Missing real corpus files. Run experiments/data_real/acquire.py first.")

    train_texts = [json.loads(l)["text"] for l in train_file.read_text(encoding="utf-8").strip().split("\n")]
    eval_real_texts = [json.loads(l)["text"] for l in eval_file.read_text(encoding="utf-8").strip().split("\n")][:200]
    eval_robust_texts = _get_eval_b_texts()

    seq_len = TRAIN_CONFIG["seq_len"]
    batch_size = TRAIN_CONFIG["batch_size"]

    print(f"  [Dataset] Building DataLoaders...", flush=True)
    train_loader = make_loader(train_texts, tokenizer, seq_len, batch_size, True, TRAIN_CONFIG["seed"])
    eval_real_loader = make_loader(eval_real_texts, tokenizer, seq_len, batch_size, False, TRAIN_CONFIG["seed"])
    eval_robust_loader = make_loader(eval_robust_texts, tokenizer, seq_len, batch_size, False, TRAIN_CONFIG["seed"])

    train_ds = train_loader.dataset
    print(
        f"  [Dataset] train_chunks={len(train_ds):,} ({format_number(train_ds.num_tokens)} tokens) | "
        f"eval_real_chunks={len(eval_real_loader.dataset):,} | "
        f"eval_robust_chunks={len(eval_robust_loader.dataset):,}",
        flush=True,
    )

    optimizer = build_optimizer(model, lr=TRAIN_CONFIG["learning_rate"], weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAIN_CONFIG["warmup_steps"],
        num_training_steps=TRAIN_CONFIG["train_steps"],
        min_lr_ratio=0.1,
    )

    global_step = 0
    if resume_path:
        meta = load_checkpoint(resume_path, model=model, optimizer=optimizer, scheduler=scheduler)
        global_step = meta.get("global_step", 0)
        print(f"  Resumed from: {resume_path} (step {global_step})", flush=True)

    history: list[dict] = []
    accum_loss = 0.0
    accum_steps = 0
    train_start = time.time()

    # Header
    print(
        f"\n  {'Step':>6}  {'TrainLoss':>9}  {'TrainPPL':>8}  "
        f"{'EvalReal_L':>10}  {'EvalReal_PPL':>12}  "
        f"{'EvalRob_L':>10}  {'EvalRob_PPL':>11}  "
        f"{'BPT':>6}  {'Gap':>7}  {'LR':>10}",
        flush=True,
    )
    print(f"  {'-'*125}", flush=True)

    # Initial Evaluation (Step 0)
    if global_step == 0:
        model.eval()
        init_real = run_eval(model, eval_real_loader, device, "eval_real")
        init_robust = run_eval(model, eval_robust_loader, device, "eval_robust")
        print(
            f"  {'0':>6}  {'--':>9}  {'--':>8}  "
            f"{init_real['eval_real_loss']:>10.4f}  {init_real['eval_real_ppl']:>12.2f}  "
            f"{init_robust['eval_robust_loss']:>10.4f}  {init_robust['eval_robust_ppl']:>11.2f}  "
            f"{'--':>6}  {'--':>7}  {'--':>10}",
            flush=True,
        )
        model.train()

    # Training Loop
    while global_step < TRAIN_CONFIG["train_steps"]:
        for batch in train_loader:
            if global_step >= TRAIN_CONFIG["train_steps"]:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
            optimizer.step()
            scheduler.step()

            global_step += 1
            accum_loss += loss.item()
            accum_steps += 1

            if global_step % TRAIN_CONFIG["log_interval"] == 0:
                avg_loss = accum_loss / accum_steps
                avg_ppl = math.exp(min(avg_loss, 20.0))
                current_lr = scheduler.get_last_lr()[0]

                log_entry: dict = {
                    "step": global_step,
                    "train_loss": round(avg_loss, 6),
                    "train_ppl": round(avg_ppl, 4),
                    "grad_norm": round(float(grad_norm), 4),
                    "lr": current_lr,
                    "elapsed_s": round(time.time() - train_start, 1),
                }

                if global_step % TRAIN_CONFIG["eval_interval"] == 0:
                    er = run_eval(model, eval_real_loader, device, "eval_real")
                    eb = run_eval(model, eval_robust_loader, device, "eval_robust")
                    gap = er["eval_real_loss"] - avg_loss
                    log_entry.update(er)
                    log_entry.update(eb)
                    log_entry["generalization_gap"] = round(gap, 6)
                    log_entry["bits_per_token"] = er["eval_real_bpt"]

                    print(
                        f"  {global_step:>6}  {avg_loss:>9.4f}  {avg_ppl:>8.2f}  "
                        f"{er['eval_real_loss']:>10.4f}  {er['eval_real_ppl']:>12.2f}  "
                        f"{eb['eval_robust_loss']:>10.4f}  {eb['eval_robust_ppl']:>11.2f}  "
                        f"{er['eval_real_bpt']:>6.3f}  {gap:>+7.3f}  {current_lr:>10.6f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  {global_step:>6}  {avg_loss:>9.4f}  {avg_ppl:>8.2f}  "
                        f"{'--':>10}  {'--':>12}  {'--':>10}  {'--':>11}  "
                        f"{'--':>6}  {'--':>7}  {current_lr:>10.6f}",
                        flush=True,
                    )

                history.append(log_entry)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

                accum_loss = 0.0
                accum_steps = 0

            # Save checkpoint
            if global_step % TRAIN_CONFIG["save_interval"] == 0:
                ckpt_path = CHECKPOINT_DIR / f"checkpoint_step_{global_step}.pt"
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
                samples = generate_samples(model, tokenizer, SAMPLE_PROMPTS, max_new_tokens=35, device=device)
                print(f"\n  [samples @ step {global_step}]", flush=True)
                for s in samples:
                    print(f"    Prompt   : {s['prompt']}", flush=True)
                    print(f"    Generated: {s['generated']}", flush=True)
                print(flush=True)

    # Final Evaluation & Checkpoint
    final_ckpt = CHECKPOINT_DIR / "checkpoint_final.pt"
    save_checkpoint(
        target_path=final_ckpt,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        epoch=0,
        loss=history[-1]["train_loss"] if history else None,
        config=MODEL_CONFIG,
        tokenizer_metadata={"vocab_size": tokenizer.vocab_size},
    )

    final_real = run_eval(model, eval_real_loader, device, "eval_real")
    final_robust = run_eval(model, eval_robust_loader, device, "eval_robust")
    final_gap = final_real["eval_real_loss"] - (history[-1]["train_loss"] if history else 0.0)

    print(f"\n{'='*80}", flush=True)
    print(f"  Stage 26A Complete — {global_step} steps", flush=True)
    print(f"  Final Train Loss     : {history[-1]['train_loss']:.6f}", flush=True)
    print(f"  Final Train PPL      : {history[-1]['train_ppl']:.4f}", flush=True)
    print(f"  Final Eval Real Loss : {final_real['eval_real_loss']:.6f}", flush=True)
    print(f"  Final Eval Real PPL  : {final_real['eval_real_ppl']:.4f}", flush=True)
    print(f"  Final Eval Real BPT  : {final_real['eval_real_bpt']:.4f}", flush=True)
    print(f"  Final Gen. Gap       : {final_gap:+.4f}", flush=True)
    print(f"  Final Eval Robust Loss: {final_robust['eval_robust_loss']:.6f}", flush=True)
    print(f"  Final Eval Robust PPL : {final_robust['eval_robust_ppl']:.4f}", flush=True)
    print(f"  Final Checkpoint     : {final_ckpt}", flush=True)
    print(f"{'='*80}\n", flush=True)

    clean = True
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"  [CRITICAL] NaN/Inf in parameter: {name}", flush=True)
            clean = False
            break
    if clean:
        print(f"  [Health] No NaN/Inf in any parameter. [OK]", flush=True)

    result = {
        "stage": "26A",
        "corpus": "Kiara-Real-v1.0",
        "global_step": global_step,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_train_ppl": history[-1]["train_ppl"] if history else None,
        "final_eval_real_loss": final_real["eval_real_loss"],
        "final_eval_real_ppl": final_real["eval_real_ppl"],
        "final_eval_real_bpt": final_real["eval_real_bpt"],
        "final_eval_robust_loss": final_robust["eval_robust_loss"],
        "final_eval_robust_ppl": final_robust["eval_robust_ppl"],
        "final_generalization_gap": final_gap,
        "checkpoint": str(final_ckpt),
    }

    results_file = CHECKPOINT_DIR / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved -> {results_file}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiara Stage 26A: Real-Data Pretraining")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train_real_stage(resume_path=args.resume)
