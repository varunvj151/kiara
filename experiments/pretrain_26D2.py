"""
experiments/pretrain_26D2.py — Stage 26D.2: Context Window Scaling
====================================================================
Controlled experiment: scale sequence length T from 64 to 256 and 512.
ALL other variables are FROZEN identical to Stage 26C:

  Strict Invariants (FROZEN from 26C):
    - Architecture: Kiara-Tiny (40,903,168 params, d_model=512, 8 layers, 8Q/2KV) [FROZEN]
    - Tokenizer:    32K Byte-BPE [FROZEN]
    - Corpus:       Kiara-Real-v2.0 [FROZEN]
    - Optimizer:    AdamW, lr=3e-4, wd=0.1, clip=1.0 [FROZEN]
    - Schedule:     warmup=150, cosine decay to min_lr=3e-5 [FROZEN]
    - Tokens/step:  2,048 tokens per step (batch_size x seq_len = 2,048) [FROZEN]
    - Total tokens: 3,000 steps x 2,048 = 6,144,000 tokens [FROZEN]

  Single Variable Changed:
    - 26D.2a: seq_len = 256, batch_size = 8  (4x context window)
    - 26D.2b: seq_len = 512, batch_size = 4  (8x context window)

Outputs:
  - checkpoints/stage_26D2_256/checkpoint_final.pt
  - checkpoints/stage_26D2_512/checkpoint_final.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
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

SEED = 1337
TOTAL_STEPS = 3000
EVAL_INTERVAL = 250
SAVE_INTERVAL = 500
LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 150
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Kiara-Tiny Architecture (FROZEN)
KIARA_TINY_CONFIG = dict(
    vocab_size=32768,
    hidden_size=512,
    intermediate_size=1536,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=512,  # supports up to 512 context
    tie_word_embeddings=True,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model_and_tokenizer() -> tuple[KiaraForCausalLM, KiaraTokenizer]:
    tokenizer = KiaraTokenizer.load()
    config = KiaraConfig(**KIARA_TINY_CONFIG)
    model = KiaraForCausalLM(config)
    return model, tokenizer


def load_datasets(tokenizer: KiaraTokenizer, seq_len: int) -> tuple[PackedDataset, PackedDataset]:
    train_file = DATA_DIR / "train.jsonl"
    eval_file = DATA_DIR / "eval_real.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Missing {train_file}. Run acquire_v2.py first.")

    train_texts = [json.loads(l)["text"] for l in train_file.read_text(encoding="utf-8").strip().split("\n")]
    eval_texts = [json.loads(l)["text"] for l in eval_file.read_text(encoding="utf-8").strip().split("\n")]

    print(f"  Packaging {len(train_texts):,} training docs (seq_len={seq_len})...", flush=True)
    train_ds = PackedDataset.from_texts(train_texts, tokenizer, seq_len=seq_len, bos=False, eos=True)
    eval_ds = PackedDataset.from_texts(eval_texts, tokenizer, seq_len=seq_len, bos=False, eos=True)

    print(f"  Train chunks: {len(train_ds):,} | Eval chunks: {len(eval_ds):,}", flush=True)
    return train_ds, eval_ds


@torch.no_grad()
def evaluate_loss(
    model: KiaraForCausalLM,
    dataset: PackedDataset,
    tokenizer: KiaraTokenizer,
    device: torch.device,
    batch_size: int,
    max_batches: int = 40,
) -> float:
    model.eval()
    collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

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


def run_stage_26d2(seq_len: int = 256):
    batch_size = 2048 // seq_len  # Maintains exact 2,048 tokens/step
    tag = f"stage_26D2_{seq_len}"
    output_dir = REPO_ROOT / "checkpoints" / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*90}")
    print(f"  STAGE 26D.2: CONTEXT WINDOW SCALING (T = {seq_len} tokens, Batch = {batch_size})")
    print(f"  Model: Kiara-Tiny (40.9M) [FROZEN] | Corpus: Kiara-Real-v2.0 [FROZEN] | Steps: {TOTAL_STEPS:,}")
    print(f"  Tokens/step: {seq_len * batch_size:,} | Total Tokens: {TOTAL_STEPS * seq_len * batch_size:,} [FROZEN]")
    print(f"{'='*90}\n")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training Device: {device}")

    model, tokenizer = create_model_and_tokenizer()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Verified Model Params: {total_params:,} (FROZEN 40.9M)")

    model.config.save_pretrained(output_dir)

    train_ds, eval_ds = load_datasets(tokenizer, seq_len=seq_len)
    collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)

    optimizer = build_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TOTAL_STEPS,
        min_lr_ratio=0.1,
    )

    init_eval_loss = evaluate_loss(model, eval_ds, tokenizer, device, batch_size=batch_size)
    print(f"\n  [Step 0 Baseline] Multi-Domain Eval Loss: {init_eval_loss:.4f} (PPL: {math.exp(min(init_eval_loss, 20.0)):.1f})\n")

    log_fp = open(output_dir / "training_log.jsonl", "w", encoding="utf-8")
    global_step = running_loss = running_tokens = 0
    t0 = time.time()

    print(f"  {'Step':>6}  {'Train Loss':>11}  {'Train PPL':>10}  {'Eval Loss':>10}  {'Eval PPL':>10}  {'LR':>10}  {'Tok/s':>9}  {'ETA':>8}")
    print("-" * 90)

    data_iter = iter(train_loader)
    avg_train_loss = train_ppl = eval_loss = eval_ppl = 0.0

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

        if global_step % EVAL_INTERVAL == 0 or global_step == TOTAL_STEPS:
            avg_train_loss = running_loss / max(1, running_tokens)
            train_ppl = math.exp(min(avg_train_loss, 20.0))
            eval_loss = evaluate_loss(model, eval_ds, tokenizer, device, batch_size=batch_size)
            eval_ppl = math.exp(min(eval_loss, 20.0))

            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            tok_s = running_tokens / max(0.1, elapsed)
            sec_left = ((TOTAL_STEPS - global_step) / global_step) * elapsed if global_step > 0 else 0
            eta = time.strftime("%H:%M:%S", time.gmtime(sec_left))

            print(f"  {global_step:>6}  {avg_train_loss:>11.4f}  {train_ppl:>10.2f}  "
                  f"{eval_loss:>10.4f}  {eval_ppl:>10.2f}  {current_lr:>10.2e}  {tok_s:>9.1f}  {eta:>8}", flush=True)

            log_entry = {
                "step": global_step,
                "train_loss": round(avg_train_loss, 4),
                "train_ppl": round(train_ppl, 2),
                "eval_loss": round(eval_loss, 4),
                "eval_ppl": round(eval_ppl, 2),
                "lr": current_lr,
                "tokens_seen": global_step * batch_size * seq_len,
                "elapsed_sec": round(elapsed, 1),
            }
            log_fp.write(json.dumps(log_entry) + "\n")
            log_fp.flush()

        if global_step % SAVE_INTERVAL == 0 or global_step == TOTAL_STEPS:
            ckpt_name = f"checkpoint_step_{global_step}.pt" if global_step < TOTAL_STEPS else "checkpoint_final.pt"
            save_checkpoint(
                output_dir / ckpt_name,
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
    print("=" * 90)
    print(f"\n  Stage 26D.2 (T={seq_len}) complete in {total_time}s.")
    print(f"  Final Checkpoint: {output_dir / 'checkpoint_final.pt'}\n")

    results = {
        "stage": f"26D.2_{seq_len}",
        "experiment": "context_window_scaling",
        "model": "Kiara-Tiny",
        "total_params": total_params,
        "context_len": seq_len,
        "batch_size": batch_size,
        "steps": TOTAL_STEPS,
        "tokens_seen": TOTAL_STEPS * batch_size * seq_len,
        "final_train_loss": round(avg_train_loss, 4),
        "final_train_ppl": round(train_ppl, 2),
        "final_eval_loss": round(eval_loss, 4),
        "final_eval_ppl": round(eval_ppl, 2),
        "generalization_gap": round(eval_loss - avg_train_loss, 4),
        "total_training_time_sec": total_time,
        "checkpoint_final": str(output_dir / "checkpoint_final.pt"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256, choices=[256, 512])
    args = parser.parse_args()
    run_stage_26d2(seq_len=args.seq_len)
