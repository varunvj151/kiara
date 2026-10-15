"""
experiments/eval_stage_26C.py — Phase 26C.5: Multi-Domain Comparative Evaluation
================================================================================
Executes the exact 5-pillar evaluation suite from Stage 26B on the newly trained
Stage 26C checkpoint (Kiara-Tiny 40.9M, trained on Kiara-Real-v2.0 multi-domain mixture).

Pillars Evaluated:
  26C.5.1: Sub-Domain Analysis (TinyStories & Wikitext-2)
  26C.5.2: Cross-Domain Transfer (Technical ML, Code, Structured Data, Math/LaTeX)
  26C.5.3: Multilingual Transfer (Tamil, Hindi, Chinese, Japanese, Arabic, Korean, Spanish, French)
  26C.5.4: Memorization & Data Leakage Audit (Exact match, ROUGE-L, Edit sim, Likelihood)
  26C.5.5: Qualitative Prompt Diagnostics & Generation

Generates direct side-by-side comparison with Stage 26A Baseline to measure:
  - Absolute PPL & Loss per domain
  - Perplexity fold-reduction (26A PPL / 26C PPL)
  - Bits-per-token (BPT) delta
  - Transfer gradient curve analysis
"""

from __future__ import annotations

import difflib
import io
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.tokenizer_benchmark.corpus import get_eval_corpus
from experiments.tokenizer_benchmark.corpus_v3 import (
    _MULTILINGUAL_SENTENCES,
    get_v3_eval_a_corpus,
)
from kiara.data import DataCollatorForCausalLM, PackedDataset
from kiara.model import KiaraConfig, KiaraForCausalLM
from kiara.tokenizer import KiaraTokenizer
from kiara.trainer import load_checkpoint

DATA_DIR = REPO_ROOT / "data" / "real_v2"
CHECKPOINT_26C_PATH = REPO_ROOT / "checkpoints" / "stage_26C" / "checkpoint_final.pt"
STAGE_26B_AUDIT_PATH = REPO_ROOT / "checkpoints" / "stage_26B" / "capability_audit.json"
OUTPUT_DIR = REPO_ROOT / "checkpoints" / "stage_26C"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1337
LN2 = math.log(2.0)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Core Metric Computation (Teacher-Forced)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_teacher_forced_metrics(
    model: KiaraForCausalLM,
    texts: list[str],
    tokenizer: KiaraTokenizer,
    device: torch.device,
    seq_len: int = 64,
    batch_size: int = 32,
) -> dict[str, float]:
    if not texts:
        return {"loss": 0.0, "ppl": 1.0, "bpt": 0.0, "num_tokens": 0, "chars_per_token": 0.0}

    model.eval()
    ds = PackedDataset.from_texts(texts, tokenizer, seq_len=seq_len, bos=False, eos=True)
    if len(ds) == 0:
        return {"loss": 0.0, "ppl": 1.0, "bpt": 0.0, "num_tokens": 0, "chars_per_token": 0.0}

    collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    total_loss = 0.0
    total_tokens = 0
    total_chars = sum(len(t) for t in texts)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, labels=labels)

        mask = labels != -100
        n_tok = mask.sum().item()
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    bpt = avg_loss / LN2
    cpt = total_chars / max(1, total_tokens)

    return {
        "loss": round(avg_loss, 4),
        "ppl": round(ppl, 2),
        "bpt": round(bpt, 4),
        "num_tokens": total_tokens,
        "chars_per_token": round(cpt, 3),
    }


# ---------------------------------------------------------------------------
# 26C.5.1 Sub-Domain Analysis
# ---------------------------------------------------------------------------

def run_subdomain_audit(model, tokenizer, device) -> dict[str, Any]:
    print("\n" + "=" * 95)
    print("  PILLAR 26C.5.1: SUB-DOMAIN ANALYSIS (IN-DISTRIBUTION ENGLISH)")
    print("=" * 95)

    print("  [1/2] Loading TinyStories validation set...", flush=True)
    ts_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
    r = requests.get(ts_url, headers={"Range": "bytes=0-300000"}, timeout=30)
    raw_ts = [s.strip() for s in r.text.split("<|endoftext|>") if len(s.strip()) >= 80][:300]

    print("  [2/2] Loading Wikitext validation set...", flush=True)
    wiki_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
    r_wiki = requests.get(wiki_url, timeout=30)
    raw_wiki = [line.strip() for line in r_wiki.text.split("\n") if len(line.strip()) >= 100 and "<unk>" not in line][:300]

    ts_metrics = compute_teacher_forced_metrics(model, raw_ts, tokenizer, device)
    wiki_metrics = compute_teacher_forced_metrics(model, raw_wiki, tokenizer, device)

    # Multi-domain held-out eval set from disk
    eval_file = DATA_DIR / "eval_real.jsonl"
    eval_docs = [json.loads(l)["text"] for l in eval_file.read_text(encoding="utf-8").strip().split("\n")][:500]
    multi_eval_metrics = compute_teacher_forced_metrics(model, eval_docs, tokenizer, device)

    print(f"\n  {'Sub-Domain':<32}  {'Docs':>6}  {'Tokens':>10}  {'26C Loss':>10}  {'26C PPL':>10}  {'26C BPT':>10}")
    print("-" * 95)
    print(f"  {'TinyStories (Narrative/Dialogue)':<32}  {len(raw_ts):>6}  {ts_metrics['num_tokens']:>10,}  {ts_metrics['loss']:>10.4f}  {ts_metrics['ppl']:>10.2f}  {ts_metrics['bpt']:>10.4f}")
    print(f"  {'Wikitext-2 (Expository/Formal)':<32}  {len(raw_wiki):>6}  {wiki_metrics['num_tokens']:>10,}  {wiki_metrics['loss']:>10.4f}  {wiki_metrics['ppl']:>10.2f}  {wiki_metrics['bpt']:>10.4f}")
    print(f"  {'Multi-Domain Held-Out (v2.0)':<32}  {len(eval_docs):>6}  {multi_eval_metrics['num_tokens']:>10,}  {multi_eval_metrics['loss']:>10.4f}  {multi_eval_metrics['ppl']:>10.2f}  {multi_eval_metrics['bpt']:>10.4f}")
    print("=" * 95)

    return {
        "tinystories": ts_metrics,
        "wikitext": wiki_metrics,
        "multi_domain_held_out": multi_eval_metrics,
    }


# ---------------------------------------------------------------------------
# 26C.5.2 Cross-Domain Transfer
# ---------------------------------------------------------------------------

def run_crossdomain_audit(model, tokenizer, device) -> dict[str, Any]:
    print("\n" + "=" * 95)
    print("  PILLAR 26C.5.2: CROSS-DOMAIN TRANSFER EVALUATION")
    print("=" * 95)

    v3_eval = get_v3_eval_a_corpus()
    std_eval = get_eval_corpus()

    categories = {
        "Technical ML Prose": v3_eval.get("technical", []) + std_eval.get("technical", []),
        "Source Code (Py/JS/C++)": v3_eval.get("code", []) + std_eval.get("code", []),
        "Structured / JSON / Tables": v3_eval.get("structured", []),
        "Mathematics / LaTeX": std_eval.get("math", []),
        "Rare Words / Vocabulary": std_eval.get("rare_words", []),
        "URLs / Symbols / Paths": std_eval.get("urls_symbols", []),
    }

    results = {}
    print(f"  {'Target Domain':<32}  {'Samples':>7}  {'Tokens':>10}  {'26C Loss':>10}  {'26C PPL':>10}  {'26C BPT':>10}")
    print("-" * 95)

    for domain_name, texts in categories.items():
        metrics = compute_teacher_forced_metrics(model, texts, tokenizer, device)
        results[domain_name] = metrics
        print(f"  {domain_name:<32}  {len(texts):>7}  {metrics['num_tokens']:>10,}  {metrics['loss']:>10.4f}  {metrics['ppl']:>10.2f}  {metrics['bpt']:>10.4f}")

    print("=" * 95)
    return results


# ---------------------------------------------------------------------------
# 26C.5.3 Multilingual Transfer by Language
# ---------------------------------------------------------------------------

def run_multilingual_audit(model, tokenizer, device) -> dict[str, Any]:
    print("\n" + "=" * 95)
    print("  PILLAR 26C.5.3: MULTILINGUAL TRANSFER BY LANGUAGE")
    print("=" * 95)

    languages = _MULTILINGUAL_SENTENCES
    results = {}

    print(f"  {'Language':<18}  {'Sentences':>10}  {'Tokens':>10}  {'26C Loss':>10}  {'26C PPL':>12}  {'26C BPT':>10}  {'Byte-Fall%':>12}")
    print("-" * 95)

    for lang, sentences in languages.items():
        metrics = compute_teacher_forced_metrics(model, sentences, tokenizer, device)

        all_ids = [tok_id for s in sentences for tok_id in tokenizer.encode(s)]
        byte_fallback = sum(1 for tid in all_ids if tid < 256)
        byte_pct = round((byte_fallback / max(1, len(all_ids))) * 100, 2)
        metrics["byte_fallback_pct"] = byte_pct

        results[lang] = metrics
        print(f"  {lang.title():<18}  {len(sentences):>10}  {metrics['num_tokens']:>10,}  {metrics['loss']:>10.4f}  {metrics['ppl']:>12.2f}  {metrics['bpt']:>10.4f}  {byte_pct:>11.1f}%")

    print("=" * 95)
    return results


# ---------------------------------------------------------------------------
# 26C.5.4 Memorization & Leakage Audit
# ---------------------------------------------------------------------------

def compute_rouge_l(pred_tokens: list[str], target_tokens: list[str]) -> float:
    if not pred_tokens or not target_tokens:
        return 0.0
    matcher = difflib.SequenceMatcher(None, pred_tokens, target_tokens)
    match_len = sum(triple.size for triple in matcher.get_matching_blocks())
    prec = match_len / len(pred_tokens)
    rec = match_len / len(target_tokens)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def compute_edit_similarity(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


@torch.no_grad()
def run_memorization_audit(model, tokenizer, device, num_samples: int = 60) -> dict[str, Any]:
    print("\n" + "=" * 95)
    print("  PILLAR 26C.5.4: MEMORIZATION & DATA LEAKAGE AUDIT")
    print("=" * 95)

    train_file = DATA_DIR / "train.jsonl"
    eval_file = DATA_DIR / "eval_real.jsonl"

    rng = random.Random(SEED)
    all_train = [json.loads(l)["text"] for l in train_file.read_text(encoding="utf-8").strip().split("\n")]
    all_eval = [json.loads(l)["text"] for l in eval_file.read_text(encoding="utf-8").strip().split("\n")]

    train_sample = rng.sample(all_train, min(num_samples, len(all_train)))
    eval_sample = rng.sample(all_eval, min(num_samples, len(all_eval)))

    model.eval()

    def audit_subset(docs: list[str], name: str) -> dict[str, float]:
        exact_matches = 0
        rouge_l_scores = []
        edit_sim_scores = []
        target_losses = []

        for doc in docs:
            tok_ids = tokenizer.encode(doc, bos=True)
            if len(tok_ids) < 70:
                continue

            prefix_ids = tok_ids[:40]
            true_continuation_ids = tok_ids[40:70]
            true_continuation_text = tokenizer.decode(true_continuation_ids, skip_special_tokens=True).strip()

            input_t = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            gen_out = model.generate(input_t, max_new_tokens=30, temperature=0.0)
            gen_ids = gen_out[0, len(prefix_ids):].tolist()
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            if gen_ids == true_continuation_ids or gen_text == true_continuation_text:
                exact_matches += 1

            gen_words = gen_text.split()
            true_words = true_continuation_text.split()
            rouge_l_scores.append(compute_rouge_l(gen_words, true_words))
            edit_sim_scores.append(compute_edit_similarity(gen_text, true_continuation_text))

            full_seq = prefix_ids + true_continuation_ids
            full_t = torch.tensor([full_seq], dtype=torch.long, device=device)
            labels_t = full_t.clone()
            labels_t[0, :len(prefix_ids)] = -100
            out = model(input_ids=full_t, labels=labels_t)
            target_losses.append(out.loss.item())

        n = len(rouge_l_scores)
        return {
            "num_tested": n,
            "exact_match_pct": round((exact_matches / max(1, n)) * 100, 2),
            "mean_rouge_l": round(sum(rouge_l_scores) / max(1, n), 4),
            "mean_edit_similarity": round(sum(edit_sim_scores) / max(1, n), 4),
            "continuation_loss": round(sum(target_losses) / max(1, n), 4),
            "continuation_ppl": round(math.exp(min(sum(target_losses) / max(1, n), 20.0)), 2),
        }

    train_audit = audit_subset(train_sample, "train")
    eval_audit = audit_subset(eval_sample, "eval")

    print(f"\n  {'Metric':<35}  {'Training Prompts':>20}  {'Held-Out Prompts':>20}  {'Ratio (Tr/Ev)':>15}")
    print("-" * 95)
    print(f"  {'Exact Verbatim Match (%)':<35}  {train_audit['exact_match_pct']:>19.2f}%  {eval_audit['exact_match_pct']:>19.2f}%  {'-':>15}")
    print(f"  {'ROUGE-L Overlap (0.0 to 1.0)':<35}  {train_audit['mean_rouge_l']:>20.4f}  {eval_audit['mean_rouge_l']:>20.4f}  {train_audit['mean_rouge_l']/max(1e-5, eval_audit['mean_rouge_l']):>15.2f}x")
    print(f"  {'Edit Similarity (0.0 to 1.0)':<35}  {train_audit['mean_edit_similarity']:>20.4f}  {eval_audit['mean_edit_similarity']:>20.4f}  {train_audit['mean_edit_similarity']/max(1e-5, eval_audit['mean_edit_similarity']):>15.2f}x")
    print(f"  {'Continuation Cross-Entropy Loss':<35}  {train_audit['continuation_loss']:>20.4f}  {eval_audit['continuation_loss']:>20.4f}  {'-':>15}")
    print(f"  {'Continuation Perplexity (PPL)':<35}  {train_audit['continuation_ppl']:>20.2f}  {eval_audit['continuation_ppl']:>20.2f}  {'-':>15}")
    print("=" * 95)

    return {
        "train_audit": train_audit,
        "eval_audit": eval_audit,
    }


# ---------------------------------------------------------------------------
# 26C.5.5 Qualitative Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_qualitative_diagnostic(model, tokenizer, device) -> list[dict[str, str]]:
    print("\n" + "=" * 95)
    print("  PILLAR 26C.5.5: QUALITATIVE PROMPT DIAGNOSTICS (GREEDY INFERENCE)")
    print("=" * 95)

    test_prompts = [
        ("Narrative", "Once upon a time, in a small quiet village,"),
        ("Expository", "The solar system consists of the Sun and the objects that"),
        ("Dialogue", 'Lily looked at the big box and asked, "What is inside?'),
        ("Reasoning", "Because it was raining heavily outside, Tom decided to"),
        ("Technical", "In computer science, a neural network is defined as"),
        ("Code", "def bubble_sort(arr):"),
        ("Math", "The Cauchy-Schwarz inequality states that"),
    ]

    model.eval()
    samples = []
    for domain, prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, bos=True)
        t_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        out = model.generate(t_ids, max_new_tokens=40, temperature=0.0)
        decoded = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        continuation = decoded[len(prompt):].strip()
        samples.append({"domain": domain, "prompt": prompt, "completion": continuation})

        print(f"\n  [{domain.upper()}]")
        print(f"  Prompt     : {prompt}")
        print(f"  Continuation: {continuation}")

    print("\n" + "=" * 95 + "\n")
    return samples


# ---------------------------------------------------------------------------
# Master Runner & Side-by-Side Comparison Generator
# ---------------------------------------------------------------------------

def run_stage_26c_full_evaluation():
    print(f"\n{'='*95}")
    print("  STAGE 26C.5: MULTI-DOMAIN CAPABILITY & TRANSFER BENCHMARK")
    print(f"  Model: Kiara-Tiny (40.903M params) | Checkpoint: stage_26C/checkpoint_final.pt")
    print(f"{'='*95}")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Evaluation Device: {device}")

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

    print(f"  Loading Stage 26C Checkpoint: {CHECKPOINT_26C_PATH}...", flush=True)
    load_checkpoint(CHECKPOINT_26C_PATH, model=model)
    model.to(device)
    model.eval()

    t0 = time.time()

    p1 = run_subdomain_audit(model, tokenizer, device)
    p2 = run_crossdomain_audit(model, tokenizer, device)
    p3 = run_multilingual_audit(model, tokenizer, device)
    p4 = run_memorization_audit(model, tokenizer, device)
    p5 = run_qualitative_diagnostic(model, tokenizer, device)

    elapsed = round(time.time() - t0, 1)

    # -----------------------------------------------------------------------
    # Comparative 26A vs 26C Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("  STAGE 26A (Narrative Baseline) vs STAGE 26C (Multi-Domain Mixture) COMPARATIVE AUDIT")
    print("=" * 105)

    comp_table = []
    if STAGE_26B_AUDIT_PATH.exists():
        audit_26a = json.loads(STAGE_26B_AUDIT_PATH.read_text(encoding="utf-8"))

        print(f"\n  {'Domain / Capability Slice':<32}  {'26A Baseline PPL':>18}  {'26C Target PPL':>18}  {'Delta (26C vs 26A)':>20}  {'Fold Reduction':>15}")
        print("-" * 105)

        # 1. Sub-domains
        for sub_k, label in [("tinystories", "TinyStories (Narrative)"), ("wikitext", "WikiText-2 (Formal Prose)")]:
            ppl_a = audit_26a["pillar_1_subdomain"][sub_k]["ppl"]
            ppl_c = p1[sub_k]["ppl"]
            diff = round(ppl_c - ppl_a, 2)
            fold = round(ppl_a / max(1e-4, ppl_c), 2)
            fold_str = f"{fold}x faster" if fold >= 1.0 else f"{round(1/fold, 2)}x slower"
            print(f"  {label:<32}  {ppl_a:>18.2f}  {ppl_c:>18.2f}  {diff:>+20.2f}  {fold_str:>15}")
            comp_table.append({"domain": label, "ppl_26a": ppl_a, "ppl_26c": ppl_c, "delta": diff, "fold_improvement": fold})

        # 2. Cross-domains
        for dom_k in ["Technical ML Prose", "Source Code (Py/JS/C++)", "Structured / JSON / Tables", "Mathematics / LaTeX", "Rare Words / Vocabulary", "URLs / Symbols / Paths"]:
            if dom_k in audit_26a["pillar_2_crossdomain"] and dom_k in p2:
                ppl_a = audit_26a["pillar_2_crossdomain"][dom_k]["ppl"]
                ppl_c = p2[dom_k]["ppl"]
                diff = round(ppl_c - ppl_a, 2)
                fold = round(ppl_a / max(1e-4, ppl_c), 2)
                fold_str = f"{fold}x reduction" if fold >= 1.0 else f"{round(1/fold, 2)}x increase"
                print(f"  {dom_k:<32}  {ppl_a:>18.2f}  {ppl_c:>18.2f}  {diff:>+20.2f}  {fold_str:>15}")
                comp_table.append({"domain": dom_k, "ppl_26a": ppl_a, "ppl_26c": ppl_c, "delta": diff, "fold_improvement": fold})

        # 3. Multilingual
        for lang_k in ["tamil", "hindi", "chinese", "japanese", "arabic", "spanish", "french", "korean"]:
            if lang_k in audit_26a["pillar_3_multilingual"] and lang_k in p3:
                ppl_a = audit_26a["pillar_3_multilingual"][lang_k]["ppl"]
                ppl_c = p3[lang_k]["ppl"]
                diff = round(ppl_c - ppl_a, 2)
                fold = round(ppl_a / max(1e-4, ppl_c), 2)
                fold_str = f"{fold}x reduction" if fold >= 1.0 else f"{round(1/fold, 2)}x increase"
                print(f"  {lang_k.title() + ' (Multilingual)':<32}  {ppl_a:>18.2f}  {ppl_c:>18.2f}  {diff:>+20.2f}  {fold_str:>15}")
                comp_table.append({"domain": lang_k.title(), "ppl_26a": ppl_a, "ppl_26c": ppl_c, "delta": diff, "fold_improvement": fold})

        print("=" * 105 + "\n")

    # Save reports
    report_26c = {
        "stage": "26C",
        "checkpoint": str(CHECKPOINT_26C_PATH),
        "device": str(device),
        "audit_duration_seconds": elapsed,
        "pillar_1_subdomain": p1,
        "pillar_2_crossdomain": p2,
        "pillar_3_multilingual": p3,
        "pillar_4_memorization": p4,
        "pillar_5_qualitative": p5,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(OUTPUT_DIR / "capability_audit_26c.json", "w", encoding="utf-8") as f:
        json.dump(report_26c, f, indent=2)

    with open(OUTPUT_DIR / "comparative_26a_vs_26c.json", "w", encoding="utf-8") as f:
        json.dump(comp_table, f, indent=2)

    print(f"  Master 26C Audit Report: {OUTPUT_DIR / 'capability_audit_26c.json'}")
    print(f"  Comparative Report    : {OUTPUT_DIR / 'comparative_26a_vs_26c.json'}\n")
    return report_26c


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_stage_26c_full_evaluation()
