"""
Quick verification of Stage 26C pre-training components.
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.pretrain_real_v2 import create_model_and_tokenizer, load_datasets

model, tokenizer = create_model_and_tokenizer()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
train_ds, eval_ds, _ = load_datasets(tokenizer)
print(f"Train chunks: {len(train_ds):,}, Eval chunks: {len(eval_ds):,}")
print("Stage 26C verification passed!")
