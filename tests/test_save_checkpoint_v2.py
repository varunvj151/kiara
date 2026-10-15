import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from kiara.model import KiaraConfig, KiaraForCausalLM
from kiara.trainer import build_optimizer, get_cosine_schedule_with_warmup, save_checkpoint, load_checkpoint

config = KiaraConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32)
model = KiaraForCausalLM(config)
optimizer = build_optimizer(model, lr=3e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, 10, 100)

tmp_path = REPO_ROOT / "checkpoints" / "stage_26C" / "test_tmp.pt"
save_checkpoint(tmp_path, model=model, optimizer=optimizer, scheduler=scheduler, global_step=500, epoch=0, loss=1.5, config=config)
assert tmp_path.exists()
meta = load_checkpoint(tmp_path, model=model)
assert meta["global_step"] == 500
tmp_path.unlink()
print("save_checkpoint test PASSED!")
