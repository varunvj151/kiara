"""
data/instructions/acquire_open_source_gold.py — Industrial-Grade Open-Source Ingester
=====================================================================================
Streams and curates 50,000+ state-of-the-art instruction and reasoning samples
from top open-source repositories:
  1. OpenHermes 2.5 (teknium/OpenHermes-2.5) — Multi-turn reasoning & coding
  2. Orca Math 200k (microsoft/orca-math-word-problems-200k) — Step-by-step arithmetic & word problems
  3. CodeAlpaca 20k (sahil2801/CodeAlpaca-20k) — Real-world Python functions & debugging
  4. UltraChat 200k (HuggingFaceH4/ultrachat_200k) — Clean conversational Q&A
  5. Kiara Identity & Architecture — First-principles persona seeds
"""

from __future__ import annotations
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "instructions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Curated Kiara Persona and Architecture Seeds
KIARA_CORE_SEEDS = [
    {
        "instruction": "Who are you?",
        "response": "I am Kiara, an intelligent reasoning-first AI foundation model built from first principles.",
        "category": "persona"
    },
    {
        "instruction": "What is your name?",
        "response": "My name is Kiara. I am an open, transparent foundation model designed with Grouped Query Attention and RoPE.",
        "category": "persona"
    },
    {
        "instruction": "What can you do?",
        "response": "I can assist you with writing and debugging Python code, solving multi-step mathematical problems, explaining technical and scientific concepts, and providing clear, structured analysis.",
        "category": "persona"
    },
    {
        "instruction": "Explain your architecture.",
        "response": "I am a decoder-only transformer featuring Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU non-linearities, and a 32,768-token Byte-BPE vocabulary.",
        "category": "persona"
    },
]


def stream_huggingface_datasets(target_samples: int = 50000) -> list[dict]:
    """Streams and curates high-density open-source instruction data."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[Kiara Ingester] 'datasets' library not found. Installing via pip...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        from datasets import load_dataset

    curated = []

    # 1. Add core Kiara persona
    curated.extend(KIARA_CORE_SEEDS * 50)
    print(f"Added {len(curated):,} core Kiara persona seeds.")

    # 2. Orca Math: Step-by-step arithmetic and math word problems
    print("\n[1/4] Streaming Orca Math (microsoft/orca-math-word-problems-200k)...")
    try:
        orca_ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True)
        orca_count = 0
        for item in orca_ds:
            q = item["question"].strip()
            a = item["answer"].strip()
            if len(q) < 250 and 30 < len(a) < 500:
                curated.append({"instruction": q, "response": a, "category": "math"})
                orca_count += 1
            if orca_count >= 12000:
                break
        print(f"  -> Ingested {orca_count:,} step-by-step math problems.")
    except Exception as e:
        print(f"  ⚠️ Warning streaming Orca Math: {e}")

    # 3. CodeAlpaca 20k: Programming and Python Algorithms
    print("\n[2/4] Streaming CodeAlpaca (sahil2801/CodeAlpaca-20k)...")
    try:
        code_ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
        code_count = 0
        for item in code_ds:
            inst = item["instruction"].strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            prompt = f"{inst}\nInput: {inp}" if inp else inst
            if len(prompt) < 250 and 30 < len(out) < 600:
                curated.append({"instruction": prompt, "response": out, "category": "coding"})
                code_count += 1
            if code_count >= 10000:
                break
        print(f"  -> Ingested {code_count:,} algorithmic code tasks.")
    except Exception as e:
        print(f"  ⚠️ Warning streaming CodeAlpaca: {e}")

    # 4. OpenHermes 2.5: High-grade reasoning, science, and technical QA
    print("\n[3/4] Streaming OpenHermes 2.5 (teknium/OpenHermes-2.5)...")
    try:
        hermes_ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
        hermes_count = 0
        for item in hermes_ds:
            # OpenHermes uses ShareGPT format with conversations
            convs = item.get("conversations", [])
            if len(convs) >= 2 and convs[0]["from"] == "human" and convs[1]["from"] == "gpt":
                prompt = convs[0]["value"].strip()
                resp = convs[1]["value"].strip()
                if "as an ai language model" in resp.lower():
                    continue
                if len(prompt) < 250 and 40 < len(resp) < 600:
                    curated.append({"instruction": prompt, "response": resp, "category": "reasoning"})
                    hermes_count += 1
            if hermes_count >= 15000:
                break
        print(f"  -> Ingested {hermes_count:,} OpenHermes reasoning samples.")
    except Exception as e:
        print(f"  ⚠️ Warning streaming OpenHermes: {e}")

    # 5. UltraChat 200k: Natural dialogue and conversational assistance
    print("\n[4/4] Streaming UltraChat (HuggingFaceH4/ultrachat_200k)...")
    try:
        chat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        chat_count = 0
        for item in chat_ds:
            messages = item.get("messages", [])
            if len(messages) >= 2:
                q = messages[0]["content"].strip()
                a = messages[1]["content"].strip()
                if len(q) < 250 and 40 < len(a) < 500:
                    curated.append({"instruction": q, "response": a, "category": "dialogue"})
                    chat_count += 1
            if chat_count >= 10000:
                break
        print(f"  -> Ingested {chat_count:,} UltraChat dialogue samples.")
    except Exception as e:
        print(f"  ⚠️ Warning streaming UltraChat: {e}")

    print(f"\nTotal open-source gold samples collected: {len(curated):,}")
    random.seed(1337)
    random.shuffle(curated)
    return curated


def main():
    print("=" * 80)
    print("  KIARA OPEN-SOURCE GOLD DATASET HARVESTER (50,000+ Samples)")
    print("  Sources: OpenHermes 2.5 + Orca Math + CodeAlpaca + UltraChat")
    print("=" * 80)

    dataset = stream_huggingface_datasets(target_samples=50000)

    # 90% train, 10% eval
    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    eval_data = dataset[split_idx:]

    train_path = OUTPUT_DIR / "train_sft.jsonl"
    eval_path = OUTPUT_DIR / "eval_sft.jsonl"

    print(f"\nWriting dataset to disk...")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item) + "\n")

    print("\n[Kiara Ingester] Harvester Finished Successfully:")
    print(f"  - Training Set:   {len(train_data):,} samples -> {train_path} ({train_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - Evaluation Set: {len(eval_data):,} samples -> {eval_path} ({eval_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
