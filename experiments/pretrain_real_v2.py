"""
experiments/pretrain_real_v2.py — Stage 26C: Controlled Multi-Domain Pre-Training
===================================================================================
Executes 3,000 optimization steps of pre-training on the Kiara-Real-v2.0 multi-domain
mixture (Natural, Technical, Code, Math, Multilingual, Structured).

Strict Invariants:
  - Architecture: Kiara-Tiny (40,903,168 params) [FROZEN]
  - Tokenizer: 32K Byte-BPE [FROZEN]
  - Context Length: 64 tokens [FROZEN]
  - Optimizer: AdamW, lr=3e-4, wd=0.1, clip=1.0, warmup 150, cosine decay [FROZEN]
  - Compute: 3,000 steps x 32 batch_size x 64 tokens = 6,144,000 tokens [FROZEN]

Outputs:
  - checkpoints/stage_26C/checkpoint_final.pt
  - checkpoints/stage_26C/training_log.jsonl
  - checkpoints/stage_26C/results.json
"""

from __future__ import annotations

import io
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kiara.data import DataCollatorForCausalLM, PackedDataset
from kiara.model import KiaraConfig, KiaraForCausalLM
from kiara.tokenizer import KiaraTokenizer
from kiara.trainer import (
    build_optimizer,
    get_cosine_schedule_with_warmup,
    save_checkpoint,
)

DATA_DIR = REPO_ROOT / "data" / "real_v2"
OUTPUT_DIR = REPO_ROOT / "checkpoints" / "stage_26C"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1337
TOTAL_STEPS = 3000
EVAL_INTERVAL = 250
LOG_INTERVAL = 50
SAVE_INTERVAL = 500
BATCH_SIZE = 32
SEQ_LEN = 64
LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 150
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model_and_tokenizer() -> tuple[KiaraForCausalLM, KiaraTokenizer]:
    tokenizer = KiaraTokenizer.load()
    config = KiaraConfig(
        vocab_size=32768,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = KiaraForCausalLM(config)
    return model, tokenizer


def load_datasets(tokenizer: KiaraTokenizer) -> tuple[PackedDataset, PackedDataset, list[str]]:
    train_file = DATA_DIR / "train.jsonl"
    eval_file = DATA_DIR / "eval_real.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Missing {train_file}. Run acquire_v2.py first.")

    train_lines = train_file.read_text(encoding="utf-8").strip().split("\n")
    eval_lines = eval_file.read_text(encoding="utf-8").strip().split("\n")

    train_texts = [json.loads(l)["text"] for l in train_lines]
    eval_texts = [json.loads(l)["text"] for l in eval_lines]

    print(f"  Packaging {len(train_texts):,} training docs into PackedDataset (seq_len={SEQ_LEN})...", flush=True)
    train_ds = PackedDataset.from_texts(train_texts, tokenizer, seq_len=SEQ_LEN, bos=False, eos=True)
    eval_ds = PackedDataset.from_texts(eval_texts, tokenizer, seq_len=SEQ_LEN, bos=False, eos=True)

    print(f"  -> Train chunks: {len(train_ds):,} | Eval chunks: {len(eval_ds):,}", flush=True)
    return train_ds, eval_ds, eval_texts


@torch.no_grad()
def evaluate_loss(model: KiaraForCausalLM, dataset: PackedDataset, tokenizer: KiaraTokenizer, device: torch.device, max_batches: int = 40) -> float:
    model.eval()
    collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, labels=labels)

        mask = labels != -100
        n_tok = mask.sum().item()
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok

    return total_loss / max(1, total_tokens)


def run_stage_26c_pretraining():
    print(f"\n{'='*85}")
    print("  STAGE 26C: CONTROLLED MULTI-DOMAIN REAL-TEXT PRE-TRAINING")
    print(f"  Model: Kiara-Tiny (40.903M params) | Corpus: Kiara-Real-v2.0 | Steps: {TOTAL_STEPS:,}")
    print(f"{'='*85}\n")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training Device: {device}")

    model, tokenizer = create_model_and_tokenizer()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Verified Parameter Count: {total_params:,} parameters (FROZEN 40.9M)")

    train_ds, eval_ds, eval_texts = load_datasets(tokenizer)
    collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    optimizer = build_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=TOTAL_STEPS, min_lr_ratio=0.1)

    # Step 0 Baseline
    init_eval_loss = evaluate_loss(model, eval_ds, tokenizer, device)
    init_eval_ppl = math.exp(min(init_eval_loss, 20.0))
    print(f"\n  [Step 0 Baseline] Multi-Domain Eval Loss: {init_eval_loss:.4f} (PPL: {init_eval_ppl:.1f})\n")

    log_file = OUTPUT_DIR / "training_log.jsonl"
    log_fp = open(log_file, "w", encoding="utf-8")

    global_step = 0
    running_loss = 0.0
    running_tokens = 0
    t0 = time.time()

    print(f"  {'Step':>6}  {'Train Loss':>11}  {'Train PPL':>10}  {'Eval Loss':>10}  {'Eval PPL':>10}  {'LR':>10}  {'Tok/s':>9}  {'ETA':>8}")
    print("-" * 85)

    data_iter = iter(train_loader)

    while global_step < TOTAL_STEPS:
        model.train()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"NaN/Inf loss at step {global_step}!")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        global_step += 1
        n_tok = (labels != -100).sum().item()
        running_loss += loss.item() * n_tok
        running_tokens += n_tok

        # Evaluation Interval
        if global_step % EVAL_INTERVAL == 0 or global_step == TOTAL_STEPS:
            avg_train_loss = running_loss / max(1, running_tokens)
            train_ppl = math.exp(min(avg_train_loss, 20.0))
            eval_loss = evaluate_loss(model, eval_ds, tokenizer, device)
            eval_ppl = math.exp(min(eval_loss, 20.0))

            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            tok_per_sec = running_tokens / max(0.1, elapsed)
            steps_left = TOTAL_STEPS - global_step
            sec_left = (steps_left / global_step) * elapsed
            eta_str = time.strftime("%H:%M:%S", time.gmtime(sec_left))

            print(f"  {global_step:>6}  {avg_train_loss:>11.4f}  {train_ppl:>10.2f}  {eval_loss:>10.4f}  {eval_ppl:>10.2f}  {current_lr:>10.2e}  {tok_per_sec:>9.1f}  {eta_str:>8}", flush=True)

            log_entry = {
                "step": global_step,
                "train_loss": round(avg_train_loss, 4),
                "train_ppl": round(train_ppl, 2),
                "eval_loss": round(eval_loss, 4),
                "eval_ppl": round(eval_ppl, 2),
                "lr": current_lr,
                "tokens_seen": global_step * BATCH_SIZE * SEQ_LEN,
                "elapsed_sec": round(elapsed, 1),
            }
            log_fp.write(json.dumps(log_entry) + "\n")
            log_fp.flush()

        # Checkpoint Saving
        if global_step % SAVE_INTERVAL == 0 or global_step == TOTAL_STEPS:
            ckpt_name = f"checkpoint_step_{global_step}.pt" if global_step < TOTAL_STEPS else "checkpoint_final.pt"
            save_checkpoint(
                OUTPUT_DIR / ckpt_name,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                epoch=0,
                loss=loss.item(),
                config=model.config,
            )
            print(f"  [ckpt] Saved {ckpt_name}", flush=True)

    log_fp.close()
    total_time = round(time.time() - t0, 1)
    print("=" * 85)
    print(f"\n  Pre-training finished in {total_time}s.")
    print(f"  Final Multi-Domain Checkpoint: {OUTPUT_DIR / 'checkpoint_final.pt'}\n")

    results = {
        "stage": "26C",
        "dataset": "Kiara-Real-v2.0",
        "steps": TOTAL_STEPS,
        "final_train_loss": round(avg_train_loss, 4),
        "final_train_ppl": round(train_ppl, 2),
        "final_eval_loss": round(eval_loss, 4),
        "final_eval_ppl": round(eval_ppl, 2),
        "generalization_gap": round(eval_loss - avg_train_loss, 4),
        "total_training_time_sec": total_time,
        "checkpoint_final": str(OUTPUT_DIR / "checkpoint_final.pt"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_stage_26c_pretraining()
