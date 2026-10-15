"""Verify pretrain.py imports and Stage C config."""
import sys
sys.path.insert(0, '.')
from experiments.pretrain import STAGE_CONFIGS, MODEL_CONFIG, build_datasets_v3

cfg = STAGE_CONFIGS['C']
p = MODEL_CONFIG.calculate_expected_parameters()

print('Stage C config:')
for k, v in cfg.items():
    print(f'  {k:20s}: {v}')
print()
print('Model params:')
for k, v in p.items():
    print(f'  {k:20s}: {v:,}')
print()
print('Import: OK')
