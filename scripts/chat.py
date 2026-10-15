"""
scripts/chat.py — Interactive Text Generation & Chat with Kiara
==============================================================
Runs autoregressive generation on CPU/GPU with streaming-style output.
- Thermal-safe: caps CPU threads to 4.
- Works with Stage 26C (Kiara-Tiny 40.9M) and upcoming Stage 26D checkpoints.
- Usage:
    python scripts/chat.py
    python scripts/chat.py --prompt "Once upon a time"
    python scripts/chat.py --checkpoint checkpoints/stage_26C/checkpoint_final.pt --temperature 0.7
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

# Protect laptop thermals: cap CPU threads
torch.set_num_threads(4)

from kiara.model import KiaraConfig, KiaraForCausalLM
from kiara.tokenizer import KiaraTokenizer
from kiara.trainer import load_checkpoint

DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "stage_26C" / "checkpoint_final.pt"

TINY_CONFIG = dict(
    vocab_size=32768,
    hidden_size=512,
    intermediate_size=1536,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=128,
    tie_word_embeddings=True,
)

SMALL_CONFIG = dict(
    vocab_size=32768,
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=3,
    max_position_embeddings=128,
    tie_word_embeddings=True,
)


def load_model_and_tokenizer(ckpt_path: Path, device: torch.device):
    tokenizer = KiaraTokenizer.load()

    # Detect architecture based on path or checkpoint
    if "stage_26D" in str(ckpt_path):
        config = KiaraConfig(**SMALL_CONFIG)
    else:
        config = KiaraConfig(**TINY_CONFIG)

    model = KiaraForCausalLM(config)
    print(f"Loading checkpoint from: {ckpt_path}...")
    load_checkpoint(ckpt_path, model=model)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_text(
    model: KiaraForCausalLM,
    tokenizer: KiaraTokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    top_k: int = 40,
    device: torch.device = torch.device("cpu"),
) -> str:
    input_ids = tokenizer.encode(prompt, bos=True)
    t_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    t0 = time.time()
    out = model.generate(
        input_ids=t_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    elapsed = time.time() - t0

    generated_ids = out[0].tolist()
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_gen = len(generated_ids) - len(input_ids)
    tok_per_sec = tokens_gen / max(0.001, elapsed)

    return full_text, tokens_gen, tok_per_sec


def main():
    parser = argparse.ArgumentParser(description="Chat / Text Generation with Kiara")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT), help="Path to checkpoint .pt")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to generate and exit")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Active CPU Threads: {torch.get_num_threads()}")
    model, tokenizer = load_model_and_tokenizer(ckpt_path, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Kiara Ready! Parameters: {param_count:,}")
    print("=" * 60)

    # One-shot mode
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        output, n_tok, tps = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(f"Generated: {output}")
        print(f"[{n_tok} tokens, {tps:.1f} tok/s]\n")
        return

    # Interactive REPL mode
    print("Entering interactive mode. Type your prompt, or 'exit' / 'quit' to stop.\n")
    while True:
        try:
            prompt = input("You > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            print("Kiara > ", end="", flush=True)
            output, n_tok, tps = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
            )
            # Print only continuation
            continuation = output[len(prompt):].strip()
            print(f"{continuation}  [{n_tok} tokens @ {tps:.1f} tok/s]\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
