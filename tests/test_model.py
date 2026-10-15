"""
tests/test_model.py — Test Suite for Step 23 Kiara Transformer Architecture
=============================================================================
Validates from-first-principles implementation of:
  - Exact parameter count accounting (tied & untied)
  - RMSNorm numerical correctness
  - RoPE relative position invariance and shapes
  - Grouped Query Attention (GQA) & strict causal masking isolation
  - SwiGLU Feed-Forward Network
  - Full model forward pass and loss computation with IGNORE_INDEX
  - Critical KV-cache vs full-forward generation equivalence gate
  - Determinism, gradient flow, and save/load identity
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from kiara.data.constants import IGNORE_INDEX
from kiara.model import (
    CausalLMOutput,
    GroupedQueryAttention,
    KiaraConfig,
    KiaraForCausalLM,
    KiaraModel,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUMLP,
    TransformerBlock,
)


@pytest.fixture
def tiny_config() -> KiaraConfig:
    """Return a fast, lightweight configuration for unit tests."""
    return KiaraConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )


class TestKiaraConfigAndParameterAccounting:
    """Verifies architectural configuration validation and parameter counting."""

    def test_config_divisibility_validation(self):
        # Invalid hidden_size / num_heads
        with pytest.raises(ValueError, match="divisible by num_attention_heads"):
            KiaraConfig(hidden_size=65, num_attention_heads=4)

        # Invalid num_heads / num_kv_heads
        with pytest.raises(ValueError, match="divisible by num_key_value_heads"):
            KiaraConfig(hidden_size=500, num_attention_heads=5, num_key_value_heads=2)

        # Invalid head_dim (must be even for RoPE)
        with pytest.raises(ValueError, match="head_dim .* must be even for Rotary Position Embeddings"):
            KiaraConfig(hidden_size=30, num_attention_heads=6, num_key_value_heads=2)  # head_dim = 5 (odd)

    def test_exact_parameter_accounting_tied_embeddings(self):
        config = KiaraConfig(
            vocab_size=32768,
            hidden_size=512,
            intermediate_size=1536,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
        expected = config.calculate_expected_parameters(tie_word_embeddings=True)
        # Mathematical derivation:
        # Embedding: 32768 * 512 = 16,777,216
        # Attention per block: 512*512 (Q) + 512*128 (K) + 512*128 (V) + 512*512 (O) = 655,360
        # SwiGLU per block: 3 * (512 * 1536) = 2,359,296
        # Norms per block: 2 * 512 = 1024
        # Single block total: 655,360 + 2,359,296 + 1024 = 3,015,680
        # 8 blocks total: 8 * 3,015,680 = 24,125,440
        # Final Norm: 512
        # Tied Total: 16,777,216 + 24,125,440 + 512 = 40,903,168
        assert expected["total"] == 40903168
        assert expected["non_embedding"] == 24125952

        model = KiaraForCausalLM(config)
        assert model.get_num_params(exclude_embeddings=False) == 40903168
        assert model.get_num_params(exclude_embeddings=True) == 24125952

    def test_exact_parameter_accounting_untied_embeddings(self):
        config = KiaraConfig(
            vocab_size=32768,
            hidden_size=512,
            intermediate_size=1536,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
        expected = config.calculate_expected_parameters(tie_word_embeddings=False)
        # Untied adds separate LM Head: + 32768 * 512 = 16,777,216
        # Untied Total: 40,903,168 + 16,777,216 = 57,680,384
        assert expected["total"] == 57680384
        model = KiaraForCausalLM(config)
        assert model.get_num_params(exclude_embeddings=False) == 57680384


class TestRMSNorm:
    """Verifies numerical properties of RMSNorm."""

    def test_rmsnorm_forward_and_shapes(self):
        norm = RMSNorm(dim=64, eps=1e-5)
        x = torch.randn(2, 10, 64)
        out = norm(x)

        assert out.shape == x.shape
        # Check that root mean square of output is approximately 1.0 (since weight=1.0)
        rms = torch.sqrt(torch.mean(out.pow(2), dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


class TestRotaryEmbedding:
    """Verifies Rotary Position Embeddings (RoPE)."""

    def test_rope_shapes_and_cache(self):
        rope = RotaryEmbedding(dim=16, max_position_embeddings=32)
        x = torch.randn(2, 4, 10, 16)
        cos, sin = rope(x, seq_len=10)

        assert cos.shape == (10, 16)
        assert sin.shape == (10, 16)

    def test_rope_relative_position_invariance(self):
        from kiara.model.rope import apply_rotary_pos_emb

        dim = 16
        rope = RotaryEmbedding(dim=dim, max_position_embeddings=64)
        cos, sin = rope(torch.empty(1, 1, 64, dim), seq_len=64)

        # Vector q at position m=5, Vector k at position n=2 (diff = 3)
        q = torch.randn(1, 1, 1, dim)
        k = torch.randn(1, 1, 1, dim)

        q_m, _ = apply_rotary_pos_emb(q, k, cos, sin, position_ids=torch.tensor([[5]]))
        _, k_n = apply_rotary_pos_emb(q, k, cos, sin, position_ids=torch.tensor([[2]]))
        dot_product_1 = torch.sum(q_m * k_n)

        # Shift both by +10: m=15, n=12 (diff = 3)
        q_m_shifted, _ = apply_rotary_pos_emb(q, k, cos, sin, position_ids=torch.tensor([[15]]))
        _, k_n_shifted = apply_rotary_pos_emb(q, k, cos, sin, position_ids=torch.tensor([[12]]))
        dot_product_2 = torch.sum(q_m_shifted * k_n_shifted)

        # Inner product should depend only on relative distance (m - n = 3)
        assert torch.allclose(dot_product_1, dot_product_2, atol=1e-5)


class TestAttentionAndCausalIsolation:
    """Verifies Grouped Query Attention and strict causal isolation."""

    def test_gqa_forward_shapes(self, tiny_config: KiaraConfig):
        attn = GroupedQueryAttention(tiny_config)
        rope = RotaryEmbedding(dim=tiny_config.head_dim, max_position_embeddings=tiny_config.max_position_embeddings)
        x = torch.randn(2, 8, tiny_config.hidden_size)
        cos, sin = rope(x, seq_len=8)

        out, kv_cache = attn(x, cos=cos, sin=sin, use_cache=True)
        assert out.shape == (2, 8, tiny_config.hidden_size)
        assert kv_cache is not None
        assert kv_cache[0].shape == (2, tiny_config.num_key_value_heads, 8, tiny_config.head_dim)

    def test_strict_causal_isolation(self, tiny_config: KiaraConfig):
        """Mutating token at index j > i must NOT affect hidden state at index i."""
        model = KiaraModel(tiny_config)
        model.eval()

        input_ids = torch.tensor([[10, 20, 30, 40, 50]])

        # Base forward pass
        out_base, _ = model(input_ids)

        # Mutate the last token (index 4)
        input_ids_mutated = torch.tensor([[10, 20, 30, 40, 999]])
        out_mutated, _ = model(input_ids_mutated)

        # Hidden states for positions 0, 1, 2, 3 must be strictly identical
        assert torch.equal(out_base[:, :4, :], out_mutated[:, :4, :])
        # Hidden state for mutated position 4 must differ
        assert not torch.equal(out_base[:, 4:, :], out_mutated[:, 4:, :])


class TestSwiGLUAndBlock:
    """Verifies SwiGLU MLP and TransformerBlock."""

    def test_swiglu_shapes(self, tiny_config: KiaraConfig):
        mlp = SwiGLUMLP(tiny_config)
        x = torch.randn(2, 5, tiny_config.hidden_size)
        out = mlp(x)
        assert out.shape == x.shape

    def test_transformer_block_forward(self, tiny_config: KiaraConfig):
        block = TransformerBlock(tiny_config)
        rope = RotaryEmbedding(dim=tiny_config.head_dim, max_position_embeddings=tiny_config.max_position_embeddings)
        x = torch.randn(2, 6, tiny_config.hidden_size)
        cos, sin = rope(x, seq_len=6)

        out, present_kv = block(x, cos=cos, sin=sin)
        assert out.shape == x.shape


class TestFullModelAndLoss:
    """Verifies KiaraForCausalLM forward pass, loss masking, and gradient flow."""

    def test_causal_lm_forward_and_loss(self, tiny_config: KiaraConfig):
        model = KiaraForCausalLM(tiny_config)
        input_ids = torch.tensor([[5, 12, 19, 24], [7, 8, 9, 10]])
        labels = input_ids.clone()

        output = model(input_ids=input_ids, labels=labels)
        assert isinstance(output, CausalLMOutput)
        assert output.logits.shape == (2, 4, tiny_config.vocab_size)
        assert output.loss is not None
        assert output.loss.ndim == 0
        assert not torch.isnan(output.loss)

    def test_loss_ignores_padding_index(self, tiny_config: KiaraConfig):
        model = KiaraForCausalLM(tiny_config)
        input_ids = torch.tensor([[5, 12, 19, tiny_config.pad_token_id]])
        labels = torch.tensor([[5, 12, 19, IGNORE_INDEX]])

        out = model(input_ids=input_ids, labels=labels)
        assert out.loss is not None
        assert not torch.isnan(out.loss)

    def test_gradient_flow_backpropagation(self, tiny_config: KiaraConfig):
        model = KiaraForCausalLM(tiny_config)
        model.train()
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
        labels = input_ids.clone()

        output = model(input_ids=input_ids, labels=labels)
        output.loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient!"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"

    def test_deterministic_forward_pass(self, tiny_config: KiaraConfig):
        model = KiaraForCausalLM(tiny_config)
        model.eval()
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16))

        with torch.no_grad():
            out1 = model(input_ids).logits
            out2 = model(input_ids).logits

        assert torch.equal(out1, out2)


class TestKVCacheEquivalenceGate:
    """Critical Verification Gate: KV-cache generation must be identical to full forward pass."""

    def test_kv_cache_step_by_step_equivalence(self, tiny_config: KiaraConfig):
        model = KiaraForCausalLM(tiny_config)
        model.eval()

        # Prompt sequence of length 6
        prompt = torch.tensor([[10, 25, 42, 88, 105, 210]])

        # 1. Full-sequence forward pass
        with torch.no_grad():
            full_logits = model(prompt).logits

        # 2. Incremental autoregressive forward pass with KV-cache
        cached_logits = []
        past_kv = None

        with torch.no_grad():
            for t in range(prompt.shape[1]):
                token_step = prompt[:, t : t + 1]
                step_out = model(
                    input_ids=token_step,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = step_out.past_key_values
                cached_logits.append(step_out.logits)

        cached_logits_tensor = torch.cat(cached_logits, dim=1)

        # Logits must match to numerical tolerance
        assert torch.allclose(full_logits, cached_logits_tensor, atol=1e-5)


class TestModelPersistence:
    """Verifies save and load round-trip identity."""

    def test_save_and_from_pretrained(self, tiny_config: KiaraConfig):
        model = KiaraForCausalLM(tiny_config)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            model.save_pretrained(save_path)

            assert (save_path / "config.json").exists()
            assert (save_path / "model.pt").exists()

            reloaded = KiaraForCausalLM.from_pretrained(save_path)
            reloaded.eval()

            x = torch.randint(0, tiny_config.vocab_size, (1, 8))
            with torch.no_grad():
                out_orig = model(x).logits
                out_reloaded = reloaded(x).logits

            assert torch.equal(out_orig, out_reloaded)
