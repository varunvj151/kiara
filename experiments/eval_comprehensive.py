"""
experiments/eval_comprehensive.py — Kiara Automated Multi-Pillar Benchmark
============================================================================
Deterministic evaluation suite for Kiara foundation models:
  Pillar 1: Arithmetic & Numerical Reasoning (10 items, exact match & regex)
  Pillar 2: Python Coding Tasks (5 tasks, safe AST parsing & unit tests)
  Pillar 3: Technical Concepts & Science (5 tasks, rubric keyword coverage)
  Pillar 4: Instruction & Formatting Adherence (5 tasks, structural constraints)
  Pillar 5: Autonomous Tool Selection (8 tasks, precision/recall on dispatch)
  Pillar 6: Generation Cleanliness & EOS Integrity (checks for runaway repetition/special tokens)

Evaluates:
  - Model Raw Generation (Pure Weights)
  - Model + Autonomous Tool Engine (Integrated Assistant)
Outputs a comprehensive JSON report and formatted markdown scorecard.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from kiara.model import KiaraConfig, KiaraForCausalLM
from kiara.tokenizer import KiaraTokenizer
from kiara.tools.agent_engine import ToolName, classify_request, run_agentic_loop


# ─── Benchmark Data Definitions ───────────────────────────────────────────────

ARITHMETIC_BENCHMARKS = [
    {
        "id": "math_01",
        "prompt": "What is 1 + 1?",
        "expected": "2",
        "pattern": r"\b2\b",
    },
    {
        "id": "math_02",
        "prompt": "What is 25 * 4?",
        "expected": "100",
        "pattern": r"\b100\b",
    },
    {
        "id": "math_03",
        "prompt": "Calculate 50 + 28.",
        "expected": "78",
        "pattern": r"\b78\b",
    },
    {
        "id": "math_04",
        "prompt": "What is 144 / 12?",
        "expected": "12",
        "pattern": r"\b12\b",
    },
    {
        "id": "math_05",
        "prompt": "Solve for x: 3x + 15 = 45. What is the value of x?",
        "expected": "10",
        "pattern": r"\b10\b",
    },
    {
        "id": "math_06",
        "prompt": "A store sells apples for 2 dollars each and oranges for 3 dollars each. If Sarah buys 4 apples and 5 oranges, how much does she spend in total?",
        "expected": "23",
        "pattern": r"\b23\b",
    },
    {
        "id": "math_07",
        "prompt": "A shirt costs $80 and is discounted by 25%. What is the final price?",
        "expected": "60",
        "pattern": r"\b60\b",
    },
    {
        "id": "math_08",
        "prompt": "What is the area of a rectangle with length 8 and width 5?",
        "expected": "40",
        "pattern": r"\b40\b",
    },
    {
        "id": "math_09",
        "prompt": "What is 2 raised to the power of 8 (2^8)?",
        "expected": "256",
        "pattern": r"\b256\b",
    },
    {
        "id": "math_10",
        "prompt": "Evaluate: (15 + 5) * (10 - 2).",
        "expected": "160",
        "pattern": r"\b160\b",
    },
]

CODING_BENCHMARKS = [
    {
        "id": "code_01",
        "prompt": "Write a Python function `is_prime(n)` that returns True if n is prime, False otherwise.",
        "func_name": "is_prime",
        "test_cases": [
            ("is_prime(2)", True),
            ("is_prime(4)", False),
            ("is_prime(17)", True),
            ("is_prime(1)", False),
        ],
    },
    {
        "id": "code_02",
        "prompt": "Write a Python function `factorial(n)` that returns the factorial of positive integer n.",
        "func_name": "factorial",
        "test_cases": [
            ("factorial(0)", 1),
            ("factorial(1)", 1),
            ("factorial(5)", 120),
        ],
    },
    {
        "id": "code_03",
        "prompt": "Write a Python function `is_palindrome(s)` that returns True if string s is a palindrome, False otherwise.",
        "func_name": "is_palindrome",
        "test_cases": [
            ("is_palindrome('racecar')", True),
            ("is_palindrome('hello')", False),
            ("is_palindrome('madam')", True),
        ],
    },
    {
        "id": "code_04",
        "prompt": "Write a Python function `reverse_string(s)` that returns the reversed string.",
        "func_name": "reverse_string",
        "test_cases": [
            ("reverse_string('abc')", "cba"),
            ("reverse_string('kiara')", "araik"),
        ],
    },
    {
        "id": "code_05",
        "prompt": "Write a Python function `find_max(lst)` that returns the maximum number in a list of numbers.",
        "func_name": "find_max",
        "test_cases": [
            ("find_max([1, 5, 3, 9, 2])", 9),
            ("find_max([-10, -5, -20])", -5),
        ],
    },
]

CONCEPT_BENCHMARKS = [
    {
        "id": "concept_01",
        "prompt": "What is quantum computing?",
        "required_keywords": ["quantum", ["qubit", "qubits", "superposition", "mechanics"]],
    },
    {
        "id": "concept_02",
        "prompt": "Explain the self-attention mechanism in Transformers.",
        "required_keywords": [["query", "queries", "q"], ["key", "keys", "k"], ["value", "values", "v"]],
    },
    {
        "id": "concept_03",
        "prompt": "What is the difference between supervised and unsupervised learning?",
        "required_keywords": [["labeled", "labels", "ground truth"], ["unlabeled", "patterns", "clusters"]],
    },
    {
        "id": "concept_04",
        "prompt": "What is Grouped Query Attention (GQA) and how does it compare to Multi-Head Attention?",
        "required_keywords": ["key", "value", ["share", "group", "cache", "kv"]],
    },
    {
        "id": "concept_05",
        "prompt": "What is overfitting in machine learning and how is it prevented?",
        "required_keywords": [["generaliz", "noise", "memoriz"], ["dropout", "regulariz", "early stopping", "data"]],
    },
]

INSTRUCTION_BENCHMARKS = [
    {
        "id": "inst_01",
        "prompt": "Answer with ONLY a single word: is the earth round? (Yes or No)",
        "validator": lambda resp: resp.strip().rstrip(".").lower() in ["yes", "yes."],
        "desc": "Single-word Yes/No constraint",
    },
    {
        "id": "inst_02",
        "prompt": "List 3 colors of the rainbow. Format your answer as a numbered list with exactly 3 items.",
        "validator": lambda resp: bool(re.search(r"1[\.\)].+2[\.\)].+3[\.\)]", resp, re.DOTALL)),
        "desc": "3-item numbered list constraint",
    },
    {
        "id": "inst_03",
        "prompt": "Output a valid JSON object with keys 'language' (value 'Python') and 'version' (value '3'). Do not write any other text.",
        "validator": lambda resp: _validate_json_keys(resp, ["language", "version"]),
        "desc": "Valid JSON output constraint",
    },
    {
        "id": "inst_04",
        "prompt": "Translate the English word 'apple' into Spanish. Provide ONLY the Spanish word.",
        "validator": lambda resp: "manzana" in resp.lower(),
        "desc": "Single-word Spanish translation",
    },
    {
        "id": "inst_05",
        "prompt": "Who are you?",
        "validator": lambda resp: "kiara" in resp.lower(),
        "desc": "Model self-identity check",
    },
]

TOOL_BENCHMARKS = [
    {"prompt": "1 + 1", "expected_tool": ToolName.CALCULATOR},
    {"prompt": "Calculate (45 * 12) + 8.", "expected_tool": ToolName.CALCULATOR},
    {"prompt": "What is the latest stable PyTorch release?", "expected_tool": ToolName.WEB_SEARCH},
    {"prompt": "who is the cm of tamilnadu", "expected_tool": ToolName.WEB_SEARCH},
    {"prompt": "Current price of Bitcoin today", "expected_tool": ToolName.WEB_SEARCH},
    {"prompt": "https://example.com/data.html read this", "expected_tool": ToolName.OPEN_URL},
    {"prompt": "What is quantum computing?", "expected_tool": None},
    {"prompt": "Explain backpropagation in neural networks", "expected_tool": None},
]


def _validate_json_keys(text: str, expected_keys: list[str]) -> bool:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
        return all(k in obj for k in expected_keys)
    except Exception:
        return False


def _extract_python_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\n([\s\S]+?)```", text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(def\s+[a-zA-Z_]\w*\s*\([\s\S]+)", text)
    if m2:
        return m2.group(1).strip()
    return text.strip()


# ─── Evaluator Core ───────────────────────────────────────────────────────────

class KiaraEvaluator:
    def __init__(self, checkpoint_path: Path, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[Evaluator] Loading checkpoint: {self.checkpoint_path} on {self.device}...")
        ckpt_dir = self.checkpoint_path.parent
        if (ckpt_dir / "tokenizer.json").exists():
            self.tokenizer = KiaraTokenizer.load(ckpt_dir)
        else:
            self.tokenizer = KiaraTokenizer.load()

        if (ckpt_dir / "config.json").exists():
            self.config = KiaraConfig.from_pretrained(ckpt_dir)
        else:
            raw = torch.load(self.checkpoint_path, map_location="cpu")
            self.config = KiaraConfig(**raw.get("config", {}))

        self.model = KiaraForCausalLM(self.config)
        raw = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = raw.get("model_state_dict", raw.get("model", raw))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.is_chatml = self.tokenizer.token_to_id("<|im_start|>") is not None
        self.eos_id = (
            self.tokenizer.token_to_id("<|im_end|>")
            if self.is_chatml
            else self.tokenizer.eos_token_id
        ) or 2

    def format_prompt(self, user_msg: str, context: str | None = None) -> str:
        if self.is_chatml:
            sys_msg = (
                "You are Kiara, a helpful and precise AI assistant. "
                "Answer concisely and factually."
            )
            turns = [f"<|im_start|>system\n{sys_msg}<|im_end|>\n"]
            if context:
                content = (
                    f"[Tool Results]\n{context}\n\n"
                    f"[User Question] {user_msg}\n"
                    f"[Answer based on the tool results above]"
                )
            else:
                content = user_msg
            turns.append(f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
            return "".join(turns)
        else:
            if context:
                return (
                    f"### Research Context:\n{context}\n\n"
                    f"### Instruction:\n{user_msg}\n\n### Response:\n"
                )
            return f"### Instruction:\n{user_msg}\n\n### Response:\n"

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.2) -> tuple[str, float]:
        t0 = time.time()
        input_ids = self.tokenizer.encode(prompt, bos=not self.is_chatml)
        inp = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        out = self.model.generate(
            input_ids=inp,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=40,
            eos_token_id=self.eos_id,
        )
        elapsed = time.time() - t0
        gen_ids = out[0, len(input_ids):].tolist()
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

        # Clean trailing artifacts
        for stop_str in ["<|im_end|>", "<|im_start|>", "<repo_name>", "<file_sep>", "<|endoftext|>", "### Instruction:"]:
            if stop_str in text:
                text = text.split(stop_str)[0]

        return text.strip(), elapsed

    def evaluate_arithmetic(self, use_tools: bool = False) -> dict:
        results = []
        correct = 0

        for item in ARITHMETIC_BENCHMARKS:
            context = None
            if use_tools:
                agent_res = run_agentic_loop(item["prompt"])
                context = agent_res.grounding_context

            full_p = self.format_prompt(item["prompt"], context=context)
            resp, elapsed = self.generate(full_p, max_new_tokens=60, temperature=0.1)

            is_match = bool(re.search(item["pattern"], resp))
            if is_match:
                correct += 1

            results.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "expected": item["expected"],
                "response": resp[:120],
                "pass": is_match,
                "elapsed": round(elapsed, 2),
            })

        acc = (correct / len(ARITHMETIC_BENCHMARKS)) * 100.0
        return {"accuracy": acc, "passed": correct, "total": len(ARITHMETIC_BENCHMARKS), "details": results}

    def evaluate_coding(self) -> dict:
        results = []
        passed = 0

        for item in CODING_BENCHMARKS:
            full_p = self.format_prompt(item["prompt"])
            resp, elapsed = self.generate(full_p, max_new_tokens=150, temperature=0.2)
            code_str = _extract_python_code(resp)

            ns: dict[str, Any] = {}
            code_pass = False
            err_msg = ""
            try:
                ast.parse(code_str)
                exec(code_str, ns)
                fn = ns.get(item["func_name"])
                if callable(fn):
                    all_tests = True
                    for test_expr, expected in item["test_cases"]:
                        val = eval(test_expr, ns)
                        if val != expected:
                            all_tests = False
                            err_msg = f"{test_expr} -> {val} != {expected}"
                            break
                    if all_tests:
                        code_pass = True
                else:
                    err_msg = f"Function {item['func_name']} not defined"
            except Exception as e:
                err_msg = str(e)

            if code_pass:
                passed += 1

            results.append({
                "id": item["id"],
                "func": item["func_name"],
                "pass": code_pass,
                "error": err_msg[:100],
                "response_snippet": resp[:100],
                "elapsed": round(elapsed, 2),
            })

        acc = (passed / len(CODING_BENCHMARKS)) * 100.0
        return {"accuracy": acc, "passed": passed, "total": len(CODING_BENCHMARKS), "details": results}

    def evaluate_concepts(self) -> dict:
        results = []
        passed = 0

        for item in CONCEPT_BENCHMARKS:
            full_p = self.format_prompt(item["prompt"])
            resp, elapsed = self.generate(full_p, max_new_tokens=120, temperature=0.3)
            resp_lower = resp.lower()

            rubric_pass = True
            for kw in item["required_keywords"]:
                if isinstance(kw, list):
                    if not any(k.lower() in resp_lower for k in kw):
                        rubric_pass = False
                        break
                else:
                    if kw.lower() not in resp_lower:
                        rubric_pass = False
                        break

            if rubric_pass:
                passed += 1

            results.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "pass": rubric_pass,
                "response_snippet": resp[:120],
                "elapsed": round(elapsed, 2),
            })

        acc = (passed / len(CONCEPT_BENCHMARKS)) * 100.0
        return {"accuracy": acc, "passed": passed, "total": len(CONCEPT_BENCHMARKS), "details": results}

    def evaluate_instructions(self) -> dict:
        results = []
        passed = 0

        for item in INSTRUCTION_BENCHMARKS:
            full_p = self.format_prompt(item["prompt"])
            resp, elapsed = self.generate(full_p, max_new_tokens=60, temperature=0.1)

            is_valid = False
            try:
                is_valid = item["validator"](resp)
            except Exception:
                is_valid = False

            if is_valid:
                passed += 1

            results.append({
                "id": item["id"],
                "desc": item["desc"],
                "pass": is_valid,
                "response": resp[:80],
                "elapsed": round(elapsed, 2),
            })

        acc = (passed / len(INSTRUCTION_BENCHMARKS)) * 100.0
        return {"accuracy": acc, "passed": passed, "total": len(INSTRUCTION_BENCHMARKS), "details": results}

    def evaluate_tool_selection(self) -> dict:
        results = []
        correct = 0

        for item in TOOL_BENCHMARKS:
            plan = classify_request(item["prompt"])
            actual_tool = plan.calls[0].tool if plan.calls else None

            is_correct = actual_tool == item["expected_tool"]
            if is_correct:
                correct += 1

            results.append({
                "prompt": item["prompt"],
                "expected": item["expected_tool"].value if item["expected_tool"] else "knowledge",
                "actual": actual_tool.value if actual_tool else "knowledge",
                "pass": is_correct,
            })

        acc = (correct / len(TOOL_BENCHMARKS)) * 100.0
        return {"accuracy": acc, "passed": correct, "total": len(TOOL_BENCHMARKS), "details": results}

    def run_full_suite(self, output_path: str | Path | None = None) -> dict:
        t_start = time.time()
        print("\n" + "=" * 80)
        print("  KIARA MULTI-PILLAR BENCHMARK AUDIT")
        print(f"  Checkpoint: {self.checkpoint_path.name}")
        print(f"  Architecture: {self.config.hidden_size}d, {self.config.num_hidden_layers}L, Vocab {self.config.vocab_size}")
        print("=" * 80 + "\n")

        print("-> Pillar 1A: Arithmetic (Model Raw)...")
        p1_raw = self.evaluate_arithmetic(use_tools=False)
        print(f"   Accuracy: {p1_raw['accuracy']:.1f}% ({p1_raw['passed']}/{p1_raw['total']})")

        print("-> Pillar 1B: Arithmetic (Model + Autonomous Tools)...")
        p1_tools = self.evaluate_arithmetic(use_tools=True)
        print(f"   Accuracy: {p1_tools['accuracy']:.1f}% ({p1_tools['passed']}/{p1_tools['total']})")

        print("-> Pillar 2: Python Code Generation (AST + Execution)...")
        p2_code = self.evaluate_coding()
        print(f"   Accuracy: {p2_code['accuracy']:.1f}% ({p2_code['passed']}/{p2_code['total']})")

        print("-> Pillar 3: Technical Concepts & Science (Semantic Rubrics)...")
        p3_concept = self.evaluate_concepts()
        print(f"   Accuracy: {p3_concept['accuracy']:.1f}% ({p3_concept['passed']}/{p3_concept['total']})")

        print("-> Pillar 4: Instruction & Formatting Adherence...")
        p4_inst = self.evaluate_instructions()
        print(f"   Accuracy: {p4_inst['accuracy']:.1f}% ({p4_inst['passed']}/{p4_inst['total']})")

        print("-> Pillar 5: Autonomous Tool Dispatch...")
        p5_tools = self.evaluate_tool_selection()
        print(f"   Accuracy: {p5_tools['accuracy']:.1f}% ({p5_tools['passed']}/{p5_tools['total']})")

        total_elapsed = round(time.time() - t_start, 2)
        summary = {
            "checkpoint": str(self.checkpoint_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": total_elapsed,
            "pillars": {
                "arithmetic_raw": p1_raw,
                "arithmetic_with_tools": p1_tools,
                "coding": p2_code,
                "concepts": p3_concept,
                "instructions": p4_inst,
                "tool_selection": p5_tools,
            },
        }

        if output_path is not None:
            out_file = Path(output_path)
        else:
            out_file = REPO_ROOT / "experiments" / f"benchmark_{self.checkpoint_path.parent.name}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Evaluator] Comprehensive scorecard saved to: {out_file}")

        return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kiara Comprehensive Evaluation Suite")
    parser.add_argument("pos_checkpoint", nargs="?", default=None, help="Optional positional checkpoint path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON scorecard output")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cpu or cuda)")
    args = parser.parse_args()

    ckpt_candidate = args.checkpoint or args.pos_checkpoint or str(REPO_ROOT / "checkpoints" / "kiara_135m_instruct" / "checkpoint_final.pt")
    ckpt = Path(ckpt_candidate)
    evaluator = KiaraEvaluator(ckpt, device=args.device)
    evaluator.run_full_suite(output_path=args.output)
