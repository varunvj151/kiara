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

_PREFERRED_135M = REPO_ROOT / "checkpoints" / "kiara_135m_instruct" / "checkpoint_final.pt"
_PREFERRED_INSTRUCT = REPO_ROOT / "checkpoints" / "kiara_instruct" / "checkpoint_final1.pt"
if _PREFERRED_135M.exists():
    DEFAULT_CKPT = _PREFERRED_135M
elif _PREFERRED_INSTRUCT.exists():
    DEFAULT_CKPT = _PREFERRED_INSTRUCT
else:
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
    if (ckpt_path.parent / "tokenizer.json").exists():
        tokenizer = KiaraTokenizer.load(ckpt_path.parent)
    else:
        tokenizer = KiaraTokenizer.load()

    # Detect architecture based on config.json, checkpoint dict, or preset
    config = None
    raw = torch.load(ckpt_path, map_location="cpu")
    if "config" in raw and isinstance(raw["config"], dict):
        config = KiaraConfig(**raw["config"])
    elif (ckpt_path.parent / "config.json").exists():
        config = KiaraConfig.from_pretrained(ckpt_path.parent)
    elif "stage_26D" in str(ckpt_path) or "small" in str(ckpt_path).lower():
        config = KiaraConfig(
            vocab_size=32768, hidden_size=768, intermediate_size=2048,
            num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=3,
            max_position_embeddings=512, tie_word_embeddings=True
        )
    else:
        config = KiaraConfig(
            vocab_size=32768, hidden_size=512, intermediate_size=1536,
            num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=2,
            max_position_embeddings=512, tie_word_embeddings=True
        )

    model = KiaraForCausalLM(config)
    print(f"Loading checkpoint from: {ckpt_path} (d_model={config.hidden_size}, layers={config.num_hidden_layers}, max_pos={config.max_position_embeddings})...")
    if "model_state_dict" in raw:
        model.load_state_dict(raw["model_state_dict"])
    else:
        model.load_state_dict(raw)
    del raw

    model.to(device)
    model.eval()
    return model, tokenizer


def generate_text(
    model: KiaraForCausalLM,
    tokenizer: KiaraTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    instruct: bool = False,
    device: torch.device = torch.device("cpu"),
) -> tuple[str, int, float]:
    is_chatml = tokenizer.token_to_id("<|im_start|>") is not None
    if instruct:
        if is_chatml:
            formatted_prompt = (
                f"<|im_start|>system\nYou are Kiara, a helpful AI assistant.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    else:
        formatted_prompt = prompt

    input_ids = tokenizer.encode(formatted_prompt, bos=not is_chatml)
    t_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    stop_ids = {tokenizer.eos_token_id, 2, 3}
    for tok_name in ["<|im_end|>", "<|endoftext|>", "<repo_name>", "<file_sep>"]:
        tid = tokenizer.token_to_id(tok_name)
        if tid is not None:
            stop_ids.add(tid)

    t0 = time.time()
    out = model.generate(
        input_ids=t_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=list(stop_ids),
        use_cache=True,
    )
    elapsed = time.time() - t0

    generated_ids = out[0].tolist()
    tokens_gen = len(generated_ids) - len(input_ids)
    tok_per_sec = tokens_gen / max(0.001, elapsed)

    continuation_ids = generated_ids[len(input_ids):]
    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=False).strip()

    # Clean trailing stop markers
    for stop_str in ["<|im_end|>", "<|im_start|>", "<repo_name>", "<file_sep>", "<|endoftext|>", "###"]:
        if stop_str in continuation:
            continuation = continuation.split(stop_str)[0].strip()

    return continuation, tokens_gen, tok_per_sec


def main():
    parser = argparse.ArgumentParser(description="Chat / Text Generation with Kiara")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT), help="Path to checkpoint .pt")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to generate and exit")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--instruct", action="store_true", help="Format prompt with instruction template")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    # Auto-enable instruct mode if checkpoint path has 'instruct'
    instruct_mode = args.instruct or ("instruct" in str(ckpt_path).lower())

    # Set smart temperature default: 0.3 for instruct (reasoning/code), 0.7 for creative completion
    temp = args.temperature if args.temperature is not None else (0.3 if instruct_mode else 0.7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Active CPU Threads: {torch.get_num_threads()} | Instruct Mode: {instruct_mode} | Temp: {temp} | RepPenalty: {args.repetition_penalty}")
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
            temperature=temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            instruct=instruct_mode,
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
                temperature=temp,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                instruct=instruct_mode,
                device=device,
            )
            # Print only continuation
            continuation = output.strip()
            print(f"{continuation}  [{n_tok} tokens @ {tps:.1f} tok/s]\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
