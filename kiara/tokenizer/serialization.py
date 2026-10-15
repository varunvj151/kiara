"""
kiara/tokenizer/serialization.py — Tokenizer Serialization and Artifact I/O
=============================================================================
Manages loading and saving of the canonical Kiara tokenizer artifacts:
  - tokenizer.json       (HuggingFace serialized tokenizer)
  - tokenizer_config.json (Kiara TokenizerConfig)
  - metadata.json        (Training metadata, seed, decision provenance)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from kiara.tokenizer.config import TokenizerConfig


def get_default_artifact_dir() -> Path:
    """Return the default canonical artifact directory."""
    # Try locating relative to kiara package root (repo_root/artifacts/tokenizer)
    pkg_dir = Path(__file__).resolve().parent  # kiara/tokenizer
    repo_root = pkg_dir.parent.parent         # repo root
    candidate = repo_root / "artifacts" / "tokenizer"
    if candidate.exists():
        return candidate
    return candidate


def save_tokenizer_artifacts(
    backend: Tokenizer,
    config: TokenizerConfig,
    target_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save the tokenizer backend, config, and metadata to target_dir.

    Args:
        backend: HuggingFace Tokenizer instance.
        config: Kiara TokenizerConfig instance.
        target_dir: Destination directory.
        metadata: Optional dictionary with provenance details.

    Returns:
        Path to the saved directory.
    """
    out_path = Path(target_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save backend tokenizer
    tokenizer_file = out_path / "tokenizer.json"
    backend.save(str(tokenizer_file))

    # 2. Save config
    config_file = out_path / "tokenizer_config.json"
    config.save(config_file)

    # 3. Save metadata
    if metadata is not None:
        meta_file = out_path / "metadata.json"
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return out_path


def load_tokenizer_artifacts(
    source_dir: str | Path | None = None,
) -> tuple[Tokenizer, TokenizerConfig, dict[str, Any]]:
    """Load backend tokenizer, config, and metadata from source_dir.

    Args:
        source_dir: Directory containing tokenizer.json and tokenizer_config.json.
                    If None, defaults to the canonical artifact directory.

    Returns:
        tuple of (Tokenizer, TokenizerConfig, metadata_dict)
    """
    if source_dir is None:
        dir_path = get_default_artifact_dir()
    else:
        dir_path = Path(source_dir)

    tokenizer_path = dir_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in: {dir_path}")

    backend = Tokenizer.from_file(str(tokenizer_path))

    config_path = dir_path / "tokenizer_config.json"
    if config_path.exists():
        config = TokenizerConfig.from_json_file(config_path)
    else:
        config = TokenizerConfig()

    meta_path = dir_path / "metadata.json"
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    return backend, config, metadata
