"""
data/instructions/acquire_sft_v2.py
===================================
Acquires and curates SFT v2 dataset combining:
  1. Alpaca-Cleaned (general reasoning, coding, writing, Q&A)
  2. Targeted Python algorithms and math solutions
  3. Kiara Persona & self-identity definitions
Filters to concise samples that fit comfortably inside the 256-token context.
"""

from __future__ import annotations
import json
import random
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "instructions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPACA_URL = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"

KIARA_PERSONA_SAMPLES = [
    {
        "instruction": "Who are you?",
        "response": "I am Kiara, an intelligent reasoning-first foundation model built from first principles.",
        "category": "persona"
    },
    {
        "instruction": "What is your name?",
        "response": "My name is Kiara. I am an open, transparent foundation AI model designed for reasoning and code.",
        "category": "persona"
    },
    {
        "instruction": "What can you do?",
        "response": "I can help you write and debug Python code, solve mathematical and algorithmic problems, explain technical and scientific concepts, and answer analytical questions.",
        "category": "persona"
    },
    {
        "instruction": "Explain your architecture.",
        "response": "I am a decoder-only transformer with Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU activation, and a frozen 32,768 Byte-Level BPE tokenizer.",
        "category": "persona"
    },
    {
        "instruction": "Write a Python function to check if a number is prime.",
        "response": "def is_prime(n: int) -> bool:\n    if n <= 1:\n        return False\n    if n <= 3:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True",
        "category": "coding"
    },
    {
        "instruction": "Write a Python function to check if a string is a palindrome.",
        "response": "def is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]",
        "category": "coding"
    },
    {
        "instruction": "Write a Python function to reverse a list in-place.",
        "response": "def reverse_list(items: list) -> None:\n    left, right = 0, len(items) - 1\n    while left < right:\n        items[left], items[right] = items[right], items[left]\n        left += 1\n        right -= 1",
        "category": "coding"
    },
    {
        "instruction": "How do you calculate the Fibonacci sequence in Python?",
        "response": "def fibonacci(n: int) -> list[int]:\n    if n <= 0:\n        return []\n    if n == 1:\n        return [0]\n    fibs = [0, 1]\n    while len(fibs) < n:\n        fibs.append(fibs[-1] + fibs[-2])\n    return fibs",
        "category": "coding"
    },
]


def fetch_and_curate_sft_v2(target_samples: int = 5000):
    print("Fetching Alpaca Cleaned dataset from GitHub...")
    req = urllib.request.Request(ALPACA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_data = json.loads(resp.read().decode("utf-8"))

    print(f"Loaded {len(raw_data):,} raw instructions. Filtering for high quality and concise length...")

    curated = []
    # Seed with Kiara persona
    curated.extend(KIARA_PERSONA_SAMPLES * 8)

    random.seed(1337)
    random.shuffle(raw_data)

    for item in raw_data:
        inst = item["instruction"].strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()

        if inp:
            full_prompt = f"{inst}\nInput: {inp}"
        else:
            full_prompt = inst

        # Length filters: prompt < 300 chars, response between 40 and 600 chars (~10 to 150 tokens)
        if len(full_prompt) < 300 and 30 < len(out) < 600:
            curated.append({
                "instruction": full_prompt,
                "response": out,
                "category": "general"
            })

        if len(curated) >= target_samples:
            break

    random.shuffle(curated)
    print(f"Total curated SFT v2 samples: {len(curated):,}")

    # 90/10 split
    split_idx = int(len(curated) * 0.9)
    train_samples = curated[:split_idx]
    eval_samples = curated[split_idx:]

    train_path = OUTPUT_DIR / "train_sft.jsonl"
    eval_path = OUTPUT_DIR / "eval_sft.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for s in eval_samples:
            f.write(json.dumps(s) + "\n")

    print(f"Saved {len(train_samples):,} training samples to {train_path}")
    print(f"Saved {len(eval_samples):,} eval samples to {eval_path}")


if __name__ == "__main__":
    fetch_and_curate_sft_v2(target_samples=5000)
