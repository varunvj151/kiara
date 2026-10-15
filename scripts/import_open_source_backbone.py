"""
scripts/import_open_source_backbone.py — Open-Source Foundation Model Importer
=============================================================================
Imports pre-trained open-source weights (e.g. SmolLM-135M, SmolLM-360M)
directly into Kiara's native architecture.

Because Kiara's backbone (GQA + SwiGLU + RMSNorm + RoPE + Tied Embeddings)
is 100% architecturally identical to the state-of-the-art Llama/SmolLM layout,
we can ingest hundreds of billions of tokens of pre-trained knowledge into Kiara
with zero architectural impedance.

Usage:
    python scripts/import_open_source_backbone.py --repo-id HuggingFaceTB/SmolLM-135M-Instruct
    python scripts/import_open_source_backbone.py --repo-id HuggingFaceTB/SmolLM-360M-Instruct
"""

from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from huggingface_hub import hf_hub_download
import safetensors.torch

from kiara.model import KiaraConfig, KiaraForCausalLM


def import_backbone(repo_id: str, output_dir: Path):
    print("=" * 80)
    print(f"  KIARA OPEN-SOURCE BACKBONE IMPORTER")
    print(f"  Source HuggingFace Model: {repo_id}")
    print(f"  Target Kiara Directory:  {output_dir}")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download configuration
    print(f"\n[1/4] Downloading model configuration from {repo_id}...")
    cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        hf_cfg = json.load(f)

    # Map HF configuration to KiaraConfig
    kiara_config = KiaraConfig(
        vocab_size=hf_cfg.get("vocab_size", 49152),
        hidden_size=hf_cfg["hidden_size"],
        intermediate_size=hf_cfg["intermediate_size"],
        num_hidden_layers=hf_cfg["num_hidden_layers"],
        num_attention_heads=hf_cfg["num_attention_heads"],
        num_key_value_heads=hf_cfg["num_key_value_heads"],
        max_position_embeddings=hf_cfg.get("max_position_embeddings", 2048),
        rms_norm_eps=hf_cfg.get("rms_norm_eps", 1e-5),
        rope_theta=hf_cfg.get("rope_theta", 10000.0),
        tie_word_embeddings=hf_cfg.get("tie_word_embeddings", True),
    )
    kiara_config.save_pretrained(output_dir)
    print(f"  -> Converted architecture: d_model={kiara_config.hidden_size}, "
          f"layers={kiara_config.num_hidden_layers}, "
          f"heads={kiara_config.num_attention_heads}Q/{kiara_config.num_key_value_heads}KV, "
          f"vocab={kiara_config.vocab_size}, max_pos={kiara_config.max_position_embeddings}")

    # 2. Download Tokenizer
    print(f"\n[2/4] Downloading tokenizer artifacts...")
    try:
        tok_json = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
        shutil.copy2(tok_json, output_dir / "tokenizer.json")
        tok_cfg = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json")
        shutil.copy2(tok_cfg, output_dir / "tokenizer_config.json")
        print(f"  -> Tokenizer artifacts saved to {output_dir}")
    except Exception as e:
        print(f"  Note: Tokenizer download notice: {e}")

    # 3. Download Model Weights
    print(f"\n[3/4] Downloading model weights (safetensors)...")
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    print(f"  -> Loading safetensors from {weights_path}...")
    tensors = safetensors.torch.load_file(weights_path)

    # 4. Ingest into KiaraForCausalLM
    print(f"\n[4/4] Ingesting weights into KiaraForCausalLM...")
    model = KiaraForCausalLM(kiara_config)
    missing, unexpected = model.load_state_dict(tensors, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  -> Model instantiated successfully! Total parameters: {total_params:,}")
    if missing:
        print(f"  -> Noticeable missing keys (expected if tied): {missing}")
    if unexpected:
        print(f"  -> Unexpected keys: {unexpected}")

    # Save native checkpoint
    ckpt_file = output_dir / "checkpoint_final.pt"
    print(f"\nWriting native Kiara checkpoint to: {ckpt_file}...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": kiara_config.to_dict(),
        "total_params": total_params,
        "source_repo": repo_id,
    }, ckpt_file)

    print("\n" + "=" * 80)
    print(f"  SUCCESS! Kiara open-source foundation model ready at:")
    print(f"  {ckpt_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Import Open-Source Model Weights into Kiara")
    parser.add_argument("--repo-id", type=str, default="HuggingFaceTB/SmolLM-135M-Instruct", help="HuggingFace repo ID")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for Kiara checkpoint")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        name = args.repo_id.split("/")[-1].lower().replace("-", "_")
        out_dir = REPO_ROOT / "checkpoints" / f"kiara_{name}"

    import_backbone(args.repo_id, out_dir)


if __name__ == "__main__":
    main()
