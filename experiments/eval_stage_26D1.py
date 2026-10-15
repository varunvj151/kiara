"""
experiments/eval_stage_26D1.py - Stage 26D.1 Evaluation
=========================================================
Identical 5-pillar evaluation as 26C.5, but comparing:
  - 26C Baseline:  Kiara-Tiny  40.9M  (stage_26C/checkpoint_final.pt)
  - 26D.1 Target:  Kiara-Small ~125M  (stage_26D1/checkpoint_final.pt)

All other variables frozen. The only difference is model parameter count.
This isolates the scaling law: does 3x parameter count improve representations
on the same corpus, same compute budget?

Outputs:
  checkpoints/stage_26D1/capability_audit_26D1.json
  checkpoints/stage_26D1/comparative_26C_vs_26D1.json
"""
from __future__ import annotations
import difflib, json, math, random, sys, time
from pathlib import Path
import requests, torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.tokenizer_benchmark.corpus import get_eval_corpus
from experiments.tokenizer_benchmark.corpus_v3 import _MULTILINGUAL_SENTENCES, get_v3_eval_a_corpus
from kiara.data import DataCollatorForCausalLM, PackedDataset
from kiara.model import KiaraConfig, KiaraForCausalLM
from kiara.tokenizer import KiaraTokenizer
from kiara.trainer import load_checkpoint

DATA_DIR = REPO_ROOT / "data" / "real_v2"
CKPT_26C  = REPO_ROOT / "checkpoints" / "stage_26C"  / "checkpoint_final.pt"
CKPT_26D1 = REPO_ROOT / "checkpoints" / "stage_26D1" / "checkpoint_final.pt"
AUDIT_26C = REPO_ROOT / "checkpoints" / "stage_26C"  / "capability_audit_26c.json"
OUTPUT_DIR = REPO_ROOT / "checkpoints" / "stage_26D1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1337
LN2  = math.log(2.0)
BATCH_SIZE = 32
SEQ_LEN = 64

TINY_CFG = dict(vocab_size=32768, hidden_size=512, intermediate_size=1536,
                num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=2,
                max_position_embeddings=128, tie_word_embeddings=True)
SMALL_CFG = dict(vocab_size=32768, hidden_size=768, intermediate_size=2048,
                 num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=3,
                 max_position_embeddings=128, tie_word_embeddings=True)

def set_seed(s): torch.manual_seed(s); random.seed(s)

@torch.no_grad()
def compute_metrics(model, texts, tokenizer, device, seq_len=SEQ_LEN, bs=BATCH_SIZE):
    if not texts: return {"loss": 0.0, "ppl": 1.0, "bpt": 0.0, "num_tokens": 0}
    model.eval()
    ds = PackedDataset.from_texts(texts, tokenizer, seq_len=seq_len, bos=False, eos=True)
    if len(ds) == 0: return {"loss": 0.0, "ppl": 1.0, "bpt": 0.0, "num_tokens": 0}
    collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collator)
    total_loss = total_tokens = 0
    for batch in loader:
        ids = batch["input_ids"].to(device); lbl = batch["labels"].to(device)
        out = model(input_ids=ids, labels=lbl)
        mask = lbl != -100; n = mask.sum().item()
        total_loss += out.loss.item() * n; total_tokens += n
    avg = total_loss / max(1, total_tokens)
    return {"loss": round(avg, 4), "ppl": round(math.exp(min(avg, 20.0)), 2),
            "bpt": round(avg / LN2, 4), "num_tokens": total_tokens}

def load_model(cfg_dict, ckpt_path, device):
    config = KiaraConfig(**cfg_dict)
    model  = KiaraForCausalLM(config)
    load_checkpoint(ckpt_path, model=model)
    model.to(device); model.eval()
    return model

def eval_model(model, tokenizer, device, tag):
    print(f"\n  Evaluating {tag}...")
    # P1 Sub-domain
    ts_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
    r = requests.get(ts_url, headers={"Range": "bytes=0-300000"}, timeout=30)
    raw_ts   = [s.strip() for s in r.text.split("<|endoftext|>") if len(s.strip()) >= 80][:300]
    wiki_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
    r_wiki   = requests.get(wiki_url, timeout=30)
    raw_wiki = [l.strip() for l in r_wiki.text.split("\n") if len(l.strip()) >= 100 and "<unk>" not in l][:300]
    eval_docs = [json.loads(l)["text"] for l in
                 (DATA_DIR / "eval_real.jsonl").read_text(encoding="utf-8").strip().split("\n")][:500]
    p1 = {"tinystories": compute_metrics(model, raw_ts, tokenizer, device),
          "wikitext":     compute_metrics(model, raw_wiki, tokenizer, device),
          "multi_held_out": compute_metrics(model, eval_docs, tokenizer, device)}
    # P2 Cross-domain
    v3_eval = get_v3_eval_a_corpus(); std_eval = get_eval_corpus()
    cats = {"Technical ML Prose":        v3_eval.get("technical", []) + std_eval.get("technical", []),
            "Source Code (Py/JS/C++)":   v3_eval.get("code", []) + std_eval.get("code", []),
            "Structured / JSON / Tables": v3_eval.get("structured", []),
            "Mathematics / LaTeX":        std_eval.get("math", []),
            "Rare Words / Vocabulary":    std_eval.get("rare_words", []),
            "URLs / Symbols / Paths":     std_eval.get("urls_symbols", [])}
    p2 = {k: compute_metrics(model, v, tokenizer, device) for k, v in cats.items()}
    # P3 Multilingual
    p3 = {}
    for lang, sents in _MULTILINGUAL_SENTENCES.items():
        m = compute_metrics(model, sents, tokenizer, device)
        all_ids = [tid for s in sents for tid in tokenizer.encode(s)]
        m["byte_fallback_pct"] = round(sum(1 for t in all_ids if t < 256) / max(1, len(all_ids)) * 100, 2)
        p3[lang] = m
    return {"p1": p1, "p2": p2, "p3": p3}

def run_stage_26d1_evaluation():
    print(f"\n{'='*95}")
    print("  STAGE 26D.1: KIARA-TINY (40.9M) vs KIARA-SMALL (~125M) CONTROLLED COMPARISON")
    print(f"  Frozen: Corpus=Kiara-Real-v2.0 | Context=64T | Steps=3000 | Tokenizer=32K Byte-BPE")
    print(f"{'='*95}")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    tokenizer = KiaraTokenizer.load()

    # Try to load 26C audit from disk first (avoids re-running 26C)
    if AUDIT_26C.exists():
        print(f"\n  Loading 26C audit from {AUDIT_26C}...", flush=True)
        audit_26c = json.loads(AUDIT_26C.read_text(encoding="utf-8"))
        r26c = {"p1": audit_26c["pillar_1_subdomain"],
                "p2": audit_26c["pillar_2_crossdomain"],
                "p3": audit_26c["pillar_3_multilingual"]}
    else:
        print("\n  26C audit not found — evaluating 26C checkpoint live...", flush=True)
        m26c = load_model(TINY_CFG, CKPT_26C, device)
        r26c = eval_model(m26c, tokenizer, device, "Kiara-Tiny 26C")
        del m26c; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    m26d1 = load_model(SMALL_CFG, CKPT_26D1, device)
    r26d1 = eval_model(m26d1, tokenizer, device, "Kiara-Small 26D.1")

    # Print comparison table
    print(f"\n{'='*105}")
    print("  26C (Kiara-Tiny 40.9M) vs 26D.1 (Kiara-Small ~125M) — SAME CORPUS, SAME STEPS")
    print(f"{'='*105}")
    print(f"\n  {'Domain / Capability':<32}  {'26C PPL':>10}  {'26D.1 PPL':>10}  {'Delta':>10}  {'Fold':>8}")
    print("-" * 75)

    comp = []
    def compare_row(label, ppl_c, ppl_d1):
        d = round(ppl_d1 - ppl_c, 2)
        fold = round(ppl_c / max(1e-4, ppl_d1), 2)
        flag = "BETTER" if fold > 1.0 else ("same" if abs(d) < 0.05 else "WORSE")
        print(f"  {label:<32}  {ppl_c:>10.2f}  {ppl_d1:>10.2f}  {d:>+10.2f}  {fold:>6.2f}x  {flag}")
        comp.append({"domain": label, "ppl_26c": ppl_c, "ppl_26d1": ppl_d1, "delta": d, "fold": fold})

    # P1
    for k, lbl in [("tinystories", "TinyStories"), ("wikitext", "WikiText-2")]:
        compare_row(lbl, r26c["p1"][k]["ppl"], r26d1["p1"][k]["ppl"])
    # P2
    for k in ["Technical ML Prose", "Source Code (Py/JS/C++)", "Structured / JSON / Tables",
              "Mathematics / LaTeX", "Rare Words / Vocabulary", "URLs / Symbols / Paths"]:
        if k in r26c["p2"] and k in r26d1["p2"]:
            compare_row(k, r26c["p2"][k]["ppl"], r26d1["p2"][k]["ppl"])
    # P3
    for lang in ["tamil", "hindi", "chinese", "japanese", "arabic", "korean", "spanish", "french"]:
        if lang in r26c["p3"] and lang in r26d1["p3"]:
            compare_row(f"{lang.title()} (Multilingual)", r26c["p3"][lang]["ppl"], r26d1["p3"][lang]["ppl"])

    print(f"{'='*105}\n")

    report = {"stage": "26D.1", "experiment": "parameter_capacity_scaling",
              "device": str(device), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "kiara_tiny_26c": r26c, "kiara_small_26d1": r26d1}
    with open(OUTPUT_DIR / "capability_audit_26D1.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(OUTPUT_DIR / "comparative_26C_vs_26D1.json", "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2)
    print(f"  Audit report  : {OUTPUT_DIR / 'capability_audit_26D1.json'}")
    print(f"  Comparison CSV: {OUTPUT_DIR / 'comparative_26C_vs_26D1.json'}\n")
    return report

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    run_stage_26d1_evaluation()
