"""
kiara/trainer — Model Training and Optimization Subsystem
==========================================================
Provides optimization, learning rate scheduling, checkpoint persistence,
and pre-training loops for Kiara.
"""

from kiara.trainer.checkpoint import load_checkpoint, save_checkpoint
from kiara.trainer.optimizer import build_optimizer
from kiara.trainer.scheduler import get_cosine_schedule_with_warmup
from kiara.trainer.trainer import Trainer, TrainerConfig

__all__ = [
    "Trainer",
    "TrainerConfig",
    "build_optimizer",
    "get_cosine_schedule_with_warmup",
    "save_checkpoint",
    "load_checkpoint",
]
