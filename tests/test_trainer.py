"""
tests/test_trainer.py — Test Suite for Step 24 Single-Batch Overfit & Training Subsystem
========================================================================================
Validates:
  - Gate 1: Single-batch overfit test (loss drops near zero, 100% token accuracy)
  - Gate 2: Causal objective & exact sequence memorization
  - Gate 3: Optimizer weight-decay parameter group splitting
  - Gate 4: Learning rate schedule with linear warmup and cosine decay
  - Gate 5: Checkpoint integrity gate (resumed training produces bit-identical path)
  - Gate 6: Full Trainer loop execution
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from kiara.data import DataCollatorForCausalLM, IGNORE_INDEX
from kiara.model import KiaraConfig, KiaraForCausalLM
from kiara.tokenizer import KiaraTokenizer
from kiara.trainer import (
    Trainer,
    TrainerConfig,
    build_optimizer,
    get_cosine_schedule_with_warmup,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def small_config() -> KiaraConfig:
    """Fast, lightweight configuration for overfit and trainer tests."""
    return KiaraConfig(
        vocab_size=500,
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )


class TestSingleBatchOverfitGate:
    """Critical Gate 1: Verifies that Kiara can rapidly overfit and memorize a fixed batch."""

    def test_single_batch_overfit_and_memorization(self, small_config: KiaraConfig):
        torch.manual_seed(1337)
        model = KiaraForCausalLM(small_config)
        model.train()

        # Fixed batch of 2 sequences, length 12
        seq1 = [10, 25, 42, 88, 105, 12, 19, 33, 55, 77, 99, 101]
        seq2 = [5, 15, 20, 30, 45, 60, 75, 90, 110, 120, 130, 140]
        input_ids = torch.tensor([seq1, seq2], dtype=torch.long)
        labels = input_ids.clone()

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0)

        initial_loss = None
        final_loss = None

        # Train for 70 steps on the exact same batch
        for step in range(70):
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        # 1. Verify that initial loss was high and decreased substantially
        assert initial_loss is not None and initial_loss > 5.0
        assert final_loss is not None and final_loss < 0.05, f"Final loss too high: {final_loss}"

        # 2. Verify next-token prediction accuracy reaches 100% on the batch
        model.eval()
        with torch.no_grad():
            eval_out = model(input_ids=input_ids, labels=labels)
            shift_logits = eval_out.logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            predictions = torch.argmax(shift_logits, dim=-1)

            correct = (predictions == shift_labels).sum().item()
            total = shift_labels.numel()
            accuracy = correct / total

            assert accuracy == 1.0, f"Expected 100% next-token accuracy, got {accuracy*100}%"


class TestOptimizerAndScheduler:
    """Verifies optimizer parameter grouping and cosine scheduler."""

    def test_optimizer_weight_decay_splitting(self, small_config: KiaraConfig):
        model = KiaraForCausalLM(small_config)
        optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1)

        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        assert decay_group["weight_decay"] == 0.1
        assert no_decay_group["weight_decay"] == 0.0

        # Verify no 1D parameter is in decay group
        for p in decay_group["params"]:
            assert p.ndim >= 2

        # Verify RMSNorm weights are in no_decay group
        for name, p in model.named_parameters():
            if "norm" in name:
                assert any(p is nd_p for nd_p in no_decay_group["params"])

    def test_cosine_schedule_with_warmup(self, small_config: KiaraConfig):
        model = KiaraForCausalLM(small_config)
        optimizer = build_optimizer(model, lr=1.0, weight_decay=0.0)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
            min_lr_ratio=0.1,
        )

        # Step 0: lr = 0.0
        assert scheduler.get_last_lr()[0] == 0.0

        # Step 5: halfway through warmup (lr = 0.5)
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        assert pytest.approx(scheduler.get_last_lr()[0], abs=1e-3) == 0.5

        # Step 10: end of warmup (lr = 1.0)
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        assert pytest.approx(scheduler.get_last_lr()[0], abs=1e-3) == 1.0

        # Step 100: end of training (lr = min_lr_ratio = 0.1)
        for _ in range(90):
            optimizer.step()
            scheduler.step()
        assert pytest.approx(scheduler.get_last_lr()[0], abs=1e-3) == 0.1


class TestCheckpointIntegrityGate:
    """Critical Gate 4: Resuming from checkpoint produces identical forward path and weights."""

    def test_checkpoint_resume_exact_parity(self, small_config: KiaraConfig):
        torch.manual_seed(42)

        # Prepare synthetic dataset (15 batches of size 4)
        x_data = torch.randint(0, small_config.vocab_size, (60, 16))
        dataset = [{"input_ids": row, "labels": row} for row in x_data]
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        # -------------------------------------------------------------
        # Run A: Continuous training for 10 steps
        # -------------------------------------------------------------
        torch.manual_seed(42)
        model_a = KiaraForCausalLM(small_config)
        opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
        sched_a = get_cosine_schedule_with_warmup(opt_a, num_warmup_steps=2, num_training_steps=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "step_5.pt"

            # Train steps 1 to 5
            batch_iter = iter(loader)
            for step in range(1, 6):
                batch = next(batch_iter)
                opt_a.zero_grad()
                out = model_a(input_ids=batch["input_ids"], labels=batch["labels"])
                out.loss.backward()
                opt_a.step()
                sched_a.step()

            # Save checkpoint at step 5
            save_checkpoint(
                target_path=ckpt_path,
                model=model_a,
                optimizer=opt_a,
                scheduler=sched_a,
                global_step=5,
                epoch=1,
                loss=out.loss.item(),
                config=small_config,
            )

            # Continue Run A steps 6 to 10
            losses_a = []
            for step in range(6, 11):
                batch = next(batch_iter)
                opt_a.zero_grad()
                out = model_a(input_ids=batch["input_ids"], labels=batch["labels"])
                out.loss.backward()
                opt_a.step()
                sched_a.step()
                losses_a.append(out.loss.item())

            # -------------------------------------------------------------
            # Run B: Fresh model resuming from step 5 checkpoint
            # -------------------------------------------------------------
            model_b = KiaraForCausalLM(small_config)
            opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
            sched_b = get_cosine_schedule_with_warmup(opt_b, num_warmup_steps=2, num_training_steps=10)

            load_meta = load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model_b,
                optimizer=opt_b,
                scheduler=sched_b,
            )
            assert load_meta["global_step"] == 5

            # Recreate loader iterator to step 6
            batch_iter_b = iter(loader)
            for _ in range(5):
                next(batch_iter_b)

            losses_b = []
            for step in range(6, 11):
                batch = next(batch_iter_b)
                opt_b.zero_grad()
                out = model_b(input_ids=batch["input_ids"], labels=batch["labels"])
                out.loss.backward()
                opt_b.step()
                sched_b.step()
                losses_b.append(out.loss.item())

            # -------------------------------------------------------------
            # Verification: Losses and weights must match identically
            # -------------------------------------------------------------
            for la, lb in zip(losses_a, losses_b):
                assert pytest.approx(la, abs=1e-6) == lb

            for (na, pa), (nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
                assert torch.equal(pa, pb), f"Mismatch in parameter: {na}"


class TestTrainerEndToEnd:
    """Verifies complete Trainer execution on synthetic dataset."""

    def test_trainer_run_and_evaluation(self, small_config: KiaraConfig):
        # Build small training and eval datasets
        train_data = torch.randint(0, small_config.vocab_size, (20, 8))
        eval_data = torch.randint(0, small_config.vocab_size, (8, 8))

        train_ds = [{"input_ids": row, "labels": row} for row in train_data]
        eval_ds = [{"input_ids": row, "labels": row} for row in eval_data]

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=4, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_cfg = TrainerConfig(
                output_dir=tmpdir,
                learning_rate=1e-3,
                warmup_steps=2,
                max_steps=15,
                eval_interval=5,
                save_interval=5,
                log_interval=5,
            )

            model = KiaraForCausalLM(small_config)
            trainer = Trainer(
                model=model,
                config=trainer_cfg,
                train_dataloader=train_loader,
                eval_dataloader=eval_loader,
            )

            result = trainer.train()

            assert result["global_step"] == 15
            assert len(result["history"]) > 0
            assert (Path(tmpdir) / "checkpoint_final.pt").exists()
            assert (Path(tmpdir) / "checkpoint_step_10.pt").exists()

            # Evaluation check
            eval_metrics = trainer.evaluate()
            assert "eval_loss" in eval_metrics
            assert "eval_perplexity" in eval_metrics
            assert not math.isnan(eval_metrics["eval_loss"])
