"""Quick test to verify pretrain_real.py configuration."""
import sys
sys.path.insert(0, ".")
from experiments.pretrain_real import MODEL_CONFIG, TRAIN_CONFIG

p = MODEL_CONFIG.calculate_expected_parameters()
print("Stage 26A pretrain_real.py verification:")
print("  Total params :", f"{p['total']:,}")
print("  Train steps  :", TRAIN_CONFIG["train_steps"])
print("  Warmup steps :", TRAIN_CONFIG["warmup_steps"])
print("  Batch size   :", TRAIN_CONFIG["batch_size"])
print("  Learning rate:", TRAIN_CONFIG["learning_rate"])
print("Import status  : OK")
