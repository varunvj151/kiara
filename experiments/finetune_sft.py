"""
experiments/finetune_sft.py — Supervised Instruction Fine-Tuning (SFT)
======================================================================
Transforms Kiara from raw text-completion into an interactive AI assistant.
- Implements Response-Only Loss Masking (labels = -100 on prompt tokens).
- Fast, low learning-rate fine-tuning with AdamW and Cosine decay.
- Automatically enables GPU Mixed Precision (AMP FP16) on CUDA.
- Saves checkpoints to checkpoints/kiara_instruct/
"""

from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from kiara.model import KiaraConfig, KiaraForCausalLM
from kiara.tokenizer import KiaraTokenizer
from kiara.trainer import build_optimizer, get_cosine_schedule_with_warmup, load_checkpoint, save_checkpoint

DATA_DIR = REPO_ROOT / "data" / "instructions"
DEFAULT_BASE_CKPT = REPO_ROOT / "checkpoints" / "stage_26D2_256" / "checkpoint_final.pt"
OUTPUT_DIR = REPO_ROOT / "checkpoints" / "kiara_instruct"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


class SFTDataset(Dataset):
    """Dataset with Response-Only Loss Masking."""

    def __init__(self, data_path: Path, tokenizer: KiaraTokenizer, seq_len: int = 256):
        self.samples = []
        self.seq_len = seq_len
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt_str = PROMPT_TEMPLATE.format(instruction=item["instruction"].strip())
                resp_str = item["response"].strip()

                prompt_ids = tokenizer.encode(prompt_str, bos=False, eos=False)
                resp_ids = tokenizer.encode(resp_str, bos=False, eos=False) + [eos_id]

                input_ids = prompt_ids + resp_ids
                # Mask prompt tokens with -100 so loss is only calculated on assistant answer
                labels = [-100] * len(prompt_ids) + resp_ids

                # Truncate if exceeds max seq_len
                if len(input_ids) > seq_len:
                    input_ids = input_ids[:seq_len]
                    labels = labels[:seq_len]

                # Pad to seq_len
                pad_len = seq_len - len(input_ids)
                if pad_len > 0:
                    input_ids = input_ids + [pad_id] * pad_len
                    labels = labels + [-100] * pad_len

                attention_mask = [1 if tok != pad_id else 0 for tok in input_ids]

                self.samples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def sft_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


@torch.no_grad()
def evaluate_sft_loss(model, dataloader, device, max_batches: int = 30):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = labels != -100
        n_tok = mask.sum().item()
        if n_tok == 0:
            continue
        out = model(input_ids=input_ids, labels=labels)
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def sample_test_prompt(model, tokenizer, device, prompt: str = "Who are you?"):
    model.eval()
    full_prompt = PROMPT_TEMPLATE.format(instruction=prompt)
    input_ids = tokenizer.encode(full_prompt, bos=True)
    t_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    out = model.generate(
        input_ids=t_ids,
        max_new_tokens=60,
        temperature=0.3,
        top_k=20,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    decoded = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    continuation = decoded[len(full_prompt):].strip()
    return continuation


def run_instruction_finetuning(
    base_checkpoint: Path,
    steps: int = 800,
    batch_size: int = 8,
    lr: float = 1e-4,
    seq_len: int = 256,
):
    print("=" * 90)
    print("  KIARA SUPERVISED INSTRUCTION FINE-TUNING (SFT)")
    print(f"  Base Model Checkpoint: {base_checkpoint}")
    print(f"  Target Steps: {steps} | Batch Size: {batch_size} | Seq Len: {seq_len} | LR: {lr}")
    print("=" * 90)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    print(f"  Device: {device} | Mixed Precision (AMP): {use_cuda}\n")

    tokenizer = KiaraTokenizer.load()

    # Load base config
    config_file = base_checkpoint.parent / "config.json"
    if config_file.exists():
        config = KiaraConfig.from_pretrained(base_checkpoint.parent)
    else:
        # Fallback to tiny config with 256/512 max pos
        config = KiaraConfig(
            vocab_size=32768, hidden_size=512, intermediate_size=1536,
            num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=2,
            max_position_embeddings=512, tie_word_embeddings=True
        )

    model = KiaraForCausalLM(config)
    print(f"Loading base weights from {base_checkpoint}...")
    load_checkpoint(base_checkpoint, model=model)
    model.to(device)
    config.save_pretrained(OUTPUT_DIR)

    # Load SFT datasets
    train_file = DATA_DIR / "train_sft.jsonl"
    eval_file = DATA_DIR / "eval_sft.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Missing {train_file}. Run generate_sft_dataset.py first.")

    print("  Tokenizing and masking instruction datasets...")
    train_ds = SFTDataset(train_file, tokenizer, seq_len=seq_len)
    eval_ds = SFTDataset(eval_file, tokenizer, seq_len=seq_len)
    print(f"  Train samples: {len(train_ds):,} | Eval samples: {len(eval_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=sft_collate_fn)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=sft_collate_fn)

    optimizer = build_optimizer(model, lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=steps, min_lr_ratio=0.1)

    init_eval_loss = evaluate_sft_loss(model, eval_loader, device)
    print(f"\n  [Step 0 Pre-SFT Baseline] Eval Loss: {init_eval_loss:.4f} (PPL: {math.exp(min(init_eval_loss, 20.0)):.2f})")
    print(f"  [Step 0 Baseline Answer to 'Who are you?']: '{sample_test_prompt(model, tokenizer, device, 'Who are you?')}'\n")

    t0 = time.time()
    global_step = 0
    running_loss = 0.0
    running_tokens = 0

    print(f"  {'Step':>6}  {'Train Loss':>11}  {'Train PPL':>10}  {'Eval Loss':>10}  {'Eval PPL':>10}  {'LR':>10}  {'ETA':>8}")
    print("-" * 80)

    data_iter = iter(train_loader)
    while global_step < steps:
        model.train()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        if use_cuda:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        global_step += 1

        n_tok = (labels != -100).sum().item()
        running_loss += loss.item() * n_tok
        running_tokens += n_tok

        # Evaluation & check step
        if global_step % 100 == 0 or global_step == steps:
            avg_train_loss = running_loss / max(1, running_tokens)
            train_ppl = math.exp(min(avg_train_loss, 20.0))
            eval_loss = evaluate_sft_loss(model, eval_loader, device)
            eval_ppl = math.exp(min(eval_loss, 20.0))
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            sec_left = ((steps - global_step) / global_step) * elapsed
            eta = time.strftime("%H:%M:%S", time.gmtime(sec_left))

            print(f"  {global_step:>6}  {avg_train_loss:>11.4f}  {train_ppl:>10.2f}  "
                  f"{eval_loss:>10.4f}  {eval_ppl:>10.2f}  {current_lr:>10.2e}  {eta:>8}", flush=True)

            # Sample interactive output
            test_ans = sample_test_prompt(model, tokenizer, device, "Who are you?")
            print(f"      -> Sample Q: 'Who are you?' | A: '{test_ans}'\n", flush=True)

        if global_step % 400 == 0 or global_step == steps:
            ckpt_name = "checkpoint_final.pt" if global_step == steps else f"checkpoint_step_{global_step}.pt"
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

    total_time = round(time.time() - t0, 1)
    print("=" * 90)
    print(f"  INSTRUCTION FINE-TUNING COMPLETE in {total_time}s!")
    print(f"  Instruct Model Checkpoint: {OUTPUT_DIR / 'checkpoint_final.pt'}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", type=str, default=str(DEFAULT_BASE_CKPT), help="Path to base pretrained checkpoint")
    parser.add_argument("--steps", type=int, default=800, help="Number of SFT steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    args = parser.parse_args()

    ckpt_p = Path(args.base_ckpt)
    if not ckpt_p.exists():
        # Fallback to stage_26C if 26D2 not locally available
        fallback = REPO_ROOT / "checkpoints" / "stage_26C" / "checkpoint_final.pt"
        if fallback.exists():
            ckpt_p = fallback

    run_instruction_finetuning(
        base_checkpoint=ckpt_p,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
    )
