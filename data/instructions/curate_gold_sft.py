"""
data/instructions/curate_gold_sft.py — Curated, Leak-Free SFT Dataset Builder
=============================================================================
Builds a verified, high-density instruction dataset with:
  - 0% leakage between train and evaluation sets (deterministic hash split)
  - 0% verbatim duplication
  - High-grade Python coding patterns (including LRUCache with OrderedDict)
  - Step-by-step arithmetic and word problem reasoning
  - Strict instruction and format-following constraints (JSON, Yes/No, numbered lists)
  - Broad general knowledge from Alpaca-Cleaned

Output:
  - data/instructions/train_sft.jsonl
  - data/instructions/eval_sft.jsonl
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "instructions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPACA_URL = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"

# ─── 1. Core Persona & Architecture Seeds ────────────────────────────────────

PERSONA_SEEDS = [
    {
        "instruction": "Who are you?",
        "response": "I am Kiara, an intelligent reasoning-first AI foundation model built from first principles.",
        "category": "persona",
    },
    {
        "instruction": "What is your name?",
        "response": "My name is Kiara. I am an open, transparent foundation model built with Grouped Query Attention and RoPE positional encoding.",
        "category": "persona",
    },
    {
        "instruction": "What can you do?",
        "response": "I can assist you with writing and debugging Python code, solving multi-step mathematical problems, explaining scientific concepts, and providing clear, structured analysis.",
        "category": "persona",
    },
    {
        "instruction": "Explain your architecture.",
        "response": "I am a decoder-only transformer featuring Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU non-linearities, and a 49,152-token vocabulary.",
        "category": "persona",
    },
    {
        "instruction": "Hello! How are you today?",
        "response": "Hello! I am Kiara, ready to help you with coding, mathematics, and analysis. What would you like to work on today?",
        "category": "persona",
    },
]

# ─── 2. Golden Python Coding Demonstrations ───────────────────────────────────

CODING_SEEDS = [
    {
        "instruction": "Write a complete Python implementation of a LRU Cache using collections.OrderedDict.",
        "response": (
            "```python\n"
            "from collections import OrderedDict\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = capacity\n"
            "        self.cache = OrderedDict()\n\n"
            "    def get(self, key: int) -> int:\n"
            "        if key not in self.cache:\n"
            "            return -1\n"
            "        self.cache.move_to_end(key)\n"
            "        return self.cache[key]\n\n"
            "    def put(self, key: int, value: int) -> None:\n"
            "        if key in self.cache:\n"
            "            self.cache.move_to_end(key)\n"
            "        self.cache[key] = value\n"
            "        if len(self.cache) > self.capacity:\n"
            "            self.cache.popitem(last=False)\n"
            "```\n\n"
            "### Explanation:\n"
            "- `move_to_end(key)` marks the accessed key as most recently used in $O(1)$ time.\n"
            "- `popitem(last=False)` removes the least recently used element from the front when capacity is exceeded."
        ),
        "category": "coding",
    },
    {
        "instruction": "Write a Python function to check if a number is prime.",
        "response": (
            "```python\n"
            "def is_prime(n: int) -> bool:\n"
            "    if n <= 1:\n"
            "        return False\n"
            "    if n <= 3:\n"
            "        return True\n"
            "    if n % 2 == 0 or n % 3 == 0:\n"
            "        return False\n"
            "    i = 5\n"
            "    while i * i <= n:\n"
            "        if n % i == 0 or n % (i + 2) == 0:\n"
            "            return False\n"
            "        i += 6\n"
            "    return True\n"
            "```"
        ),
        "category": "coding",
    },
    {
        "instruction": "Implement binary search in Python.",
        "response": (
            "```python\n"
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            left = mid + 1\n"
            "        else:\n"
            "            right = mid - 1\n"
            "    return -1\n"
            "```"
        ),
        "category": "coding",
    },
    {
        "instruction": "Write a Python function to calculate the factorial of a number.",
        "response": (
            "```python\n"
            "def factorial(n: int) -> int:\n"
            "    if n < 0:\n"
            "        raise ValueError('Factorial is not defined for negative numbers.')\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
            "```"
        ),
        "category": "coding",
    },
    {
        "instruction": "Write a Python function to check if a string is a palindrome.",
        "response": (
            "```python\n"
            "def is_palindrome(s: str) -> bool:\n"
            "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
            "    return cleaned == cleaned[::-1]\n"
            "```"
        ),
        "category": "coding",
    },
    {
        "instruction": "Write a Python function to reverse a string without using slicing.",
        "response": (
            "```python\n"
            "def reverse_string(s: str) -> str:\n"
            "    chars = []\n"
            "    for i in range(len(s) - 1, -1, -1):\n"
            "        chars.append(s[i])\n"
            "    return ''.join(chars)\n"
            "```"
        ),
        "category": "coding",
    },
    {
        "instruction": "Write a Python function find_max(lst) that returns the maximum number in a list of numbers.",
        "response": (
            "```python\n"
            "def find_max(lst: list[int | float]) -> int | float:\n"
            "    if not lst:\n"
            "        raise ValueError('List is empty.')\n"
            "    max_val = lst[0]\n"
            "    for x in lst[1:]:\n"
            "        if x > max_val:\n"
            "            max_val = x\n"
            "    return max_val\n"
            "```"
        ),
        "category": "coding",
    },
]

# ─── 3. Multi-Step Mathematics & Word Problems ───────────────────────────────

MATH_SEEDS = [
    {
        "instruction": (
            "Solve this step by step: A store sells apples for 2 dollars each and oranges for 3 dollars each. "
            "If Sarah buys 4 apples and 5 oranges, how much does she spend?"
        ),
        "response": (
            "Here is the step-by-step solution:\n\n"
            "1. Cost of apples: 4 apples * $2 each = $8.\n"
            "2. Cost of oranges: 5 oranges * $3 each = $15.\n"
            "3. Total expenditure: $8 + $15 = $23.\n\n"
            "Sarah spends $23 in total."
        ),
        "category": "math",
    },
    {
        "instruction": "A shirt costs $80 and is discounted by 25%. What is the final price?",
        "response": (
            "Step 1: Calculate the discount amount:\n"
            "Discount = 25% of $80 = 0.25 * 80 = $20.\n\n"
            "Step 2: Subtract the discount from the original price:\n"
            "Final Price = $80 - $20 = $60.\n\n"
            "The final price is $60."
        ),
        "category": "math",
    },
    {
        "instruction": "Solve for x: 3x + 15 = 45.",
        "response": (
            "Step 1: Subtract 15 from both sides:\n"
            "3x = 45 - 15\n"
            "3x = 30\n\n"
            "Step 2: Divide both sides by 3:\n"
            "x = 30 / 3\n"
            "x = 10\n\n"
            "The solution is x = 10."
        ),
        "category": "math",
    },
    {
        "instruction": "What is the area of a rectangle with length 8 and width 5?",
        "response": (
            "The area of a rectangle is calculated as Length multiplied by Width.\n"
            "Area = 8 * 5 = 40.\n"
            "The area is 40 square units."
        ),
        "category": "math",
    },
    {
        "instruction": "What is 2 raised to the power of 8 (2^8)?",
        "response": (
            "2^8 = 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 = 256.\n"
            "The answer is 256."
        ),
        "category": "math",
    },
    {
        "instruction": "Evaluate: (15 + 5) * (10 - 2).",
        "response": (
            "Step 1: Calculate inside the parentheses:\n"
            "(15 + 5) = 20\n"
            "(10 - 2) = 8\n\n"
            "Step 2: Multiply the results:\n"
            "20 * 8 = 160.\n\n"
            "The answer is 160."
        ),
        "category": "math",
    },
]

# ─── 4. Instruction Following & Structural Constraints ─────────────────────────

FORMAT_SEEDS = [
    {
        "instruction": "Answer with ONLY a single word: is the earth round? (Yes or No)",
        "response": "Yes",
        "category": "format",
    },
    {
        "instruction": "List 3 colors of the rainbow. Format your answer as a numbered list with exactly 3 items.",
        "response": "1. Red\n2. Yellow\n3. Blue",
        "category": "format",
    },
    {
        "instruction": "Output a valid JSON object with keys 'language' (value 'Python') and 'version' (value '3'). Do not write any other text.",
        "response": '{"language": "Python", "version": "3"}',
        "category": "format",
    },
    {
        "instruction": "Translate the English word 'apple' into Spanish. Provide ONLY the Spanish word.",
        "response": "manzana",
        "category": "format",
    },
]

# ─── 5. Algorithmic Synthetic Arithmetic Generator ───────────────────────────

def generate_diverse_arithmetic(count: int = 1500) -> list[dict]:
    items = []
    random.seed(1337)
    seen = set()

    for _ in range(count):
        op = random.choice(["+", "-", "*", "/"])
        if op == "+":
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            res = a + b
            q = random.choice([
                f"What is {a} + {b}?",
                f"Calculate {a} + {b}.",
                f"Add {a} and {b}.",
                f"{a} + {b}",
            ])
            ans = f"{a} + {b} = {res}."
        elif op == "-":
            a = random.randint(10, 150)
            b = random.randint(1, a)
            res = a - b
            q = random.choice([
                f"What is {a} - {b}?",
                f"Calculate {a} minus {b}.",
                f"Subtract {b} from {a}.",
                f"{a} - {b}",
            ])
            ans = f"{a} - {b} = {res}."
        elif op == "*":
            a = random.randint(2, 25)
            b = random.randint(2, 25)
            res = a * b
            q = random.choice([
                f"What is {a} * {b}?",
                f"Multiply {a} by {b}.",
                f"What is the product of {a} and {b}?",
                f"{a} * {b}",
            ])
            ans = f"{a} * {b} = {res}."
        else:
            b = random.randint(2, 12)
            res = random.randint(2, 15)
            a = b * res
            q = random.choice([
                f"What is {a} / {b}?",
                f"Divide {a} by {b}.",
                f"{a} / {b}",
            ])
            ans = f"{a} / {b} = {res}."

        if q not in seen:
            seen.add(q)
            items.append({"instruction": q, "response": ans, "category": "math"})

    return items


# ─── 6. Build Curated Dataset ────────────────────────────────────────────────

def build_curated_gold_dataset(target_total: int = 10000) -> tuple[list[dict], list[dict]]:
    print("=" * 80)
    print("  KIARA SFT GOLD DATASET CURATOR")
    print(f"  Target Samples: {target_total:,} | Strict Deduplication: Yes | 0% Leakage: Yes")
    print("=" * 80)

    curated = []
    seen_instructions: set[str] = set()

    def add_unique(item: dict) -> bool:
        norm = re.sub(r"\s+", " ", item["instruction"].strip().lower())
        if norm in seen_instructions:
            return False
        seen_instructions.add(norm)
        curated.append(item)
        return True

    # 1. Add seeds
    for s in PERSONA_SEEDS:
        add_unique(s)
    for s in CODING_SEEDS:
        add_unique(s)
    for s in MATH_SEEDS:
        add_unique(s)
    for s in FORMAT_SEEDS:
        add_unique(s)

    # 2. Add diverse arithmetic
    for item in generate_diverse_arithmetic(count=2000):
        add_unique(item)

    print(f"Added {len(curated):,} curated seeds & arithmetic samples.")

    # 3. Stream from Alpaca-Cleaned to reach target
    remaining = target_total - len(curated)
    print(f"Fetching Alpaca-Cleaned for remaining {remaining:,} diverse general knowledge samples...")

    req = urllib.request.Request(ALPACA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        alpaca_raw = json.loads(resp.read().decode("utf-8"))

    random.seed(42)
    random.shuffle(alpaca_raw)

    added_alpaca = 0
    for item in alpaca_raw:
        inst = item["instruction"].strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()

        # Quality filtering
        low_out = out.lower()
        if "as an ai language model" in low_out or "i do not have access" in low_out:
            continue
        if len(out) < 20 or len(out) > 600:
            continue

        full_prompt = f"{inst}\nInput: {inp}" if inp else inst
        if add_unique({"instruction": full_prompt, "response": out, "category": "general"}):
            added_alpaca += 1
            if added_alpaca >= remaining:
                break

    print(f"Total unique curated samples assembled: {len(curated):,}")

    # 4. Deterministic Hash-based Split (guarantees 0% leakage)
    train_set = []
    eval_set = []

    for item in curated:
        # Hash instruction string to deterministically assign to train or eval
        h = int(hashlib.md5(item["instruction"].encode("utf-8")).hexdigest(), 16)
        if (h % 10) == 0:
            eval_set.append(item)
        else:
            train_set.append(item)

    # Verify zero leakage
    train_insts = set(x["instruction"].strip().lower() for x in train_set)
    eval_insts = set(x["instruction"].strip().lower() for x in eval_set)
    intersection = train_insts.intersection(eval_insts)
    assert len(intersection) == 0, f"Critical error: {len(intersection)} leaked samples detected!"

    print(f"  -> Train samples: {len(train_set):,}")
    print(f"  -> Eval samples:  {len(eval_set):,}")
    print(f"  -> Leakage rate:  0.0% (0 overlapping instructions)")

    return train_set, eval_set


def main():
    train_set, eval_set = build_curated_gold_dataset(target_total=10000)

    train_path = OUTPUT_DIR / "train_sft.jsonl"
    eval_path = OUTPUT_DIR / "eval_sft.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_set:
            f.write(json.dumps(s) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for s in eval_set:
            f.write(json.dumps(s) + "\n")

    print(f"\n[Done] Curated dataset written to:")
    print(f"  - {train_path} ({train_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - {eval_path} ({eval_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
