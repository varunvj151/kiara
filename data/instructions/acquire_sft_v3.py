"""
data/instructions/acquire_sft_v3.py — Kiara SFT v3 Super-Dataset Generator
==========================================================================
Constructs a 10,000-sample high-density instruction dataset across 6 pillars:
  1. Python Conceptual Explanations (lists vs tuples, == vs is, decorators, GIL, generators)
  2. Ground-Truth Arithmetic & Word Problems (1+1, 2+3, 4*5, percentages, equations)
  3. Creative Writing & Poetry (Mars robot, space, coding, nature)
  4. Algorithmic Python Code (is_prime, binary search, sorting, data structures)
  5. Kiara Identity & Architecture (persona without refusal bleed-over)
  6. High-Quality General Knowledge (from curated Alpaca-Cleaned)
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

# --- 1. Python Conceptual Explanations ---
PYTHON_CONCEPT_PAIRS = [
    (
        "What is the difference between a list and a tuple in Python?",
        "In Python, the primary differences between a list and a tuple are:\n1. Mutability: Lists are mutable (elements can be added, modified, or removed). Tuples are immutable (cannot be changed after creation).\n2. Syntax: Lists use square brackets `[1, 2, 3]`, while tuples use parentheses `(1, 2, 3)`.\n3. Performance: Tuples are slightly faster and consume less memory.\n4. Hashability: Tuples can be used as dictionary keys if they contain immutable elements, whereas lists cannot."
    ),
    (
        "What is the difference between == and is in Python?",
        "`==` checks for value equality (whether two objects have the same content), whereas `is` checks for identity (whether two references point to the exact same object in memory)."
    ),
    (
        "What is a generator in Python?",
        "A generator is a function that produces a sequence of values on-demand using the `yield` statement instead of `return`. It maintains its execution state across iterations, saving memory because values are computed lazily."
    ),
    (
        "What is a decorator in Python?",
        "A decorator is a design pattern in Python that modifies or extends the behavior of a function or method without altering its source code. It wraps the target function using the `@decorator_name` syntax."
    ),
    (
        "Explain the difference between deepcopy and shallow copy in Python.",
        "A shallow copy constructs a new collection object and inserts references to the original child objects. A deepcopy recursively copies all child objects, creating an independent clone where changes to nested objects do not affect the original."
    ),
    (
        "What is the Python Global Interpreter Lock (GIL)?",
        "The Global Interpreter Lock (GIL) is a mutex in CPython that ensures only one native thread executes Python bytecode at a time. This prevents race conditions in memory management but limits CPU-bound multi-threading to a single core."
    ),
    (
        "What are *args and **kwargs in Python?",
        "`*args` allows a function to accept any number of positional arguments as a tuple, while `**kwargs` allows accepting arbitrary keyword arguments as a dictionary."
    ),
    (
        "What is the difference between a set and a list in Python?",
        "A list is an ordered sequence of elements that allows duplicates and supports index-based lookup. A set is an unordered collection of unique elements with $O(1)$ average time complexity for membership testing."
    ),
    (
        "How does Python manage memory?",
        "Python uses automatic memory management with private heap allocation, reference counting to track object references, and a cyclic garbage collector to detect and reclaim isolated reference cycles."
    ),
    (
        "What is the purpose of __init__ in Python classes?",
        "`__init__` is the constructor method in Python. It is automatically called when a new instance of a class is created to initialize its attributes."
    ),
]

# --- 2. Creative Writing & Poetry ---
CREATIVE_SAMPLES = [
    (
        "Write a short poem about a robot exploring Mars.",
        "Across the crimson desert sand,\nA lonely rover walks the land.\nWith copper wheels and lens of glass,\nIt watches ancient shadows pass.\nUnderneath the starlit dome,\nA quiet scout so far from home.\nSearching dust for signs of life,\nUntouched by human fear or strife."
    ),
    (
        "Write a poem about coding at night.",
        "The screen glows soft with midnight blue,\nAs logic builds a world anew.\nBrackets close and loops unwind,\nA silent architecture in the mind.\nWhen morning breaks and tests all green,\nThe quiet craft remains unseen."
    ),
    (
        "Write a short story about an autonomous AI assisting an astronaut.",
        "Dr. Elena rested her forehead against the observation dome of Station Iris. Outside, Earth hung like a sapphire marble in velvet dark.\n'Heart rate elevated by fifteen percent, Elena,' Kiara's gentle voice resonated through the cabin audio.\n'Just looking home, Kiara,' she smiled faintly.\n'Understood,' Kiara answered, dimming the ambient console lights to amber. 'Atmospheric scrubber cycle is optimal, trajectory lock is steady at 0.002 degrees variance. You may rest. I will keep the watch.'"
    ),
    (
        "Write a poem about nature and mountains.",
        "High above the rolling mist,\nGranite peaks by sunlight kissed.\nSilent guardians of the sky,\nWhere the eagle dares to fly.\nRivers carve the valley floor,\nWhispering of evermore."
    ),
]

# --- 3. Ground-Truth Algorithms ---
CODING_ALGORITHMS = [
    (
        "Write a Python function to check if a number is prime.",
        "def is_prime(n: int) -> bool:\n    if n <= 1:\n        return False\n    if n <= 3:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True"
    ),
    (
        "Write a Python function to check if a string is a palindrome.",
        "def is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]"
    ),
    (
        "Implement binary search in Python.",
        "def binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
    ),
    (
        "Write a Python function to calculate the factorial of a number.",
        "def factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError('Factorial is not defined for negative numbers.')\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"
    ),
    (
        "Write a Python function to reverse a list in-place.",
        "def reverse_list(items: list) -> None:\n    left, right = 0, len(items) - 1\n    while left < right:\n        items[left], items[right] = items[right], items[left]\n        left += 1\n        right -= 1"
    ),
    (
        "Write a function to remove duplicates from a list while maintaining order.",
        "def remove_duplicates(items: list) -> list:\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result"
    ),
    (
        "Write a Python function that adds two numbers.",
        "def add(a: float, b: float) -> float:\n    \"\"\"Return the sum of two numbers.\"\"\"\n    return a + b"
    ),
]

# --- 4. Kiara Identity & Persona ---
PERSONA_SAMPLES = [
    (
        "Who are you?",
        "I am Kiara, an intelligent reasoning-first foundation model built from first principles."
    ),
    (
        "What is your name?",
        "My name is Kiara. I am an open, transparent foundation model built with Grouped Query Attention and RoPE positional encoding."
    ),
    (
        "What can you help me with?",
        "I can help you with Python programming, algorithmic problem-solving, mathematics, technical explanations, and conceptual analysis."
    ),
    (
        "Explain your architecture.",
        "I am an autoregressive transformer model designed with Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU non-linearities, and a 32,768-token Byte-BPE vocabulary."
    ),
    (
        "Hello!",
        "Hello! I am Kiara. How can I help you with your project today?"
    ),
    (
        "Hi",
        "Hi! I am Kiara. What would you like to build or explore today?"
    ),
]


def generate_arithmetic_samples(count: int = 1200) -> list[dict]:
    """Generates ground-truth single and multi-digit arithmetic examples."""
    samples = []
    random.seed(42)

    # Shorthand additions: 1+1, 2+1, 2+2, etc.
    for a in range(1, 10):
        for b in range(1, 10):
            samples.append({
                "instruction": f"{a}+{b}",
                "response": f"{a} + {b} = {a + b}",
                "category": "arithmetic"
            })
            samples.append({
                "instruction": f"{a} + {b}",
                "response": f"{a} + {b} = {a + b}",
                "category": "arithmetic"
            })
            samples.append({
                "instruction": f"What is {a} + {b}?",
                "response": f"{a} + {b} is {a + b}.",
                "category": "arithmetic"
            })
            samples.append({
                "instruction": f"What is {a} plus {b}?",
                "response": f"{a} plus {b} equals {a + b}.",
                "category": "arithmetic"
            })
            samples.append({
                "instruction": f"{a} * {b}",
                "response": f"{a} * {b} = {a * b}",
                "category": "arithmetic"
            })
            samples.append({
                "instruction": f"What is {a} multiplied by {b}?",
                "response": f"{a} multiplied by {b} is {a * b}.",
                "category": "arithmetic"
            })

    # Two-digit math with step-by-step
    for _ in range(count):
        a = random.randint(10, 99)
        b = random.randint(2, 50)
        op = random.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
            samples.append({
                "instruction": f"Calculate {a} + {b}.",
                "response": f"Step 1: Add {a} and {b}.\n{a} + {b} = {ans}.\nThe answer is {ans}.",
                "category": "arithmetic"
            })
        elif op == "-":
            ans = a - b
            samples.append({
                "instruction": f"What is {a} minus {b}?",
                "response": f"Subtracting {b} from {a}:\n{a} - {b} = {ans}.\nThe result is {ans}.",
                "category": "arithmetic"
            })
        elif op == "*":
            ans = a * b
            samples.append({
                "instruction": f"Multiply {a} by {b}.",
                "response": f"Multiplying {a} and {b}:\n{a} * {b} = {ans}.\nThe product is {ans}.",
                "category": "arithmetic"
            })

    return samples


def build_sft_v3_dataset(target_total: int = 10000) -> tuple[list[dict], list[dict]]:
    print("Building Kiara SFT v3 Super-Dataset...")
    curated = []

    # 1. Multiply core persona & conceptual pairs so the model memorizes them reliably
    for inst, resp in PERSONA_SAMPLES:
        for _ in range(40):
            curated.append({"instruction": inst, "response": resp, "category": "persona"})

    for inst, resp in PYTHON_CONCEPT_PAIRS:
        for _ in range(35):
            curated.append({"instruction": inst, "response": resp, "category": "python_concept"})

    for inst, resp in CREATIVE_SAMPLES:
        for _ in range(40):
            curated.append({"instruction": inst, "response": resp, "category": "creative"})

    for inst, resp in CODING_ALGORITHMS:
        for _ in range(35):
            curated.append({"instruction": inst, "response": resp, "category": "coding"})

    # 2. Add ground-truth arithmetic
    arithmetic_data = generate_arithmetic_samples(count=1500)
    curated.extend(arithmetic_data)
    print(f"Added {len(curated):,} curated core seeds (Persona, Python concepts, Poetry, Arithmetic).")

    # 3. Pull Alpaca Cleaned to fill the remainder
    remaining_needed = target_total - len(curated)
    print(f"Fetching Alpaca Cleaned for remaining {remaining_needed:,} samples...")

    req = urllib.request.Request(ALPACA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        alpaca_raw = json.loads(resp.read().decode("utf-8"))

    random.seed(1337)
    random.shuffle(alpaca_raw)

    added_alpaca = 0
    for item in alpaca_raw:
        inst = item["instruction"].strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()

        # Filter out overly long responses or refusal boilerplates
        if "as an ai language model" in out.lower() or "i do not have access" in out.lower():
            continue

        full_prompt = f"{inst}\nInput: {inp}" if inp else inst

        if len(full_prompt) < 250 and 30 < len(out) < 500:
            curated.append({
                "instruction": full_prompt,
                "response": out,
                "category": "general"
            })
            added_alpaca += 1

        if added_alpaca >= remaining_needed:
            break

    print(f"Total curated samples assembled: {len(curated):,}")

    # Shuffle thoroughly
    random.shuffle(curated)

    # 90% train / 10% eval
    split_idx = int(len(curated) * 0.9)
    train_set = curated[:split_idx]
    eval_set = curated[split_idx:]

    return train_set, eval_set


def main():
    train_set, eval_set = build_sft_v3_dataset(target_total=10000)

    train_path = OUTPUT_DIR / "train_sft.jsonl"
    eval_path = OUTPUT_DIR / "eval_sft.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_set:
            f.write(json.dumps(sample) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for sample in eval_set:
            f.write(json.dumps(sample) + "\n")

    print(f"\n[Kiara Dataset] SFT v3 Super-Dataset Successfully Saved:")
    print(f"   - Training:   {len(train_set):,} samples -> {train_path} ({train_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"   - Evaluation: {len(eval_set):,} samples -> {eval_path} ({eval_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
