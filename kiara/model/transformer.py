"""
kiara/model/transformer.py — Complete Kiara Decoder-Only Transformer
=====================================================================
Implements the full foundation model architecture:
  - KiaraModel: Core Transformer backbone with embeddings, blocks, and final norm
  - KiaraForCausalLM: Model with LM head, weight tying, causal cross-entropy loss, and autoregressive generation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from kiara.data.constants import IGNORE_INDEX
from kiara.model.block import TransformerBlock
from kiara.model.config import KiaraConfig
from kiara.model.rms_norm import RMSNorm
from kiara.model.rope import RotaryEmbedding


@dataclass
class CausalLMOutput:
    """Output container for KiaraForCausalLM forward pass."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    hidden_states: torch.Tensor | None = None


class KiaraModel(nn.Module):
    """Core Transformer decoder backbone."""

    def __init__(self, config: KiaraConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights from normal distribution."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Forward pass through the transformer backbone.

        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            attention_mask: Optional padding mask [batch_size, seq_len]
            position_ids: Optional position indices [batch_size, seq_len]
            past_key_values: Optional list of cached (k, v) tuples per layer
            use_cache: Whether to compute and return updated KV cache

        Returns:
            tuple of (final_hidden_states, next_past_key_values)
        """
        batch_size, seq_len = input_ids.shape

        # 1. Token Embeddings
        hidden_states = self.embed_tokens(input_ids)

        # 2. Position IDs and RoPE Frequencies
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        total_seq_len = seq_len + past_key_values_length

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                total_seq_len,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0).expand(batch_size, -1)

        cos, sin = self.rotary_emb(hidden_states, seq_len=total_seq_len)

        # 3. Convert 2D attention_mask to 4D additive mask if provided
        extended_attention_mask = None
        if attention_mask is not None:
            # Convert 1 (attend) to 0.0 and 0 (mask) to -inf
            # shape: [batch, 1, 1, seq_len]
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * torch.finfo(hidden_states.dtype).min

        # 4. Iterate over Transformer Blocks
        next_decoder_cache = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None

            hidden_states, present_kv = layer(
                hidden_states=hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
            )

            if use_cache and present_kv is not None:
                next_decoder_cache.append(present_kv)

        # 5. Final RMSNorm
        hidden_states = self.norm(hidden_states)

        return hidden_states, next_decoder_cache


class KiaraForCausalLM(nn.Module):
    """Kiara Decoder-Only Language Model with LM Head for Causal Pre-training."""

    def __init__(self, config: KiaraConfig) -> None:
        super().__init__()
        self.config = config
        self.model = KiaraModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Optional Weight Tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_num_params(self, exclude_embeddings: bool = False) -> int:
        """Count total parameters in the model."""
        if exclude_embeddings:
            # Exclude embedding and (if untied) lm_head
            return sum(
                p.numel()
                for name, p in self.named_parameters()
                if "embed_tokens" not in name and "lm_head" not in name
            )
        # Deduplicate shared memory parameters (e.g. tied embedding/head)
        visited: set[int] = set()
        total = 0
        for p in self.parameters():
            p_ptr = p.data_ptr()
            if p_ptr not in visited:
                visited.add(p_ptr)
                total += p.numel()
        return total

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        """Forward pass computing logits and optional causal cross-entropy loss."""
        hidden_states, next_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        eos_token_id: int | list[int] | set[int] | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively generate tokens from prompt input_ids."""
        self.eval()
        if eos_token_id is not None:
            if isinstance(eos_token_id, (list, tuple, set)):
                stop_ids = {int(x) for x in eos_token_id if x is not None}
            else:
                stop_ids = {int(eos_token_id)}
        else:
            stop_ids = {int(self.config.eos_token_id)} if self.config.eos_token_id is not None else set()
        generated = input_ids.clone()
        past_key_values = None
        prompt_len = input_ids.shape[-1]

        for _ in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                # Only pass the newly generated token
                current_input = generated[:, -1:]
            else:
                current_input = generated

            outputs = self.forward(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :].clone()

            # Apply repetition penalty to previously generated tokens
            if repetition_penalty != 1.0 and generated.shape[-1] > prompt_len:
                for b in range(generated.shape[0]):
                    prev_tokens = torch.unique(generated[b, prompt_len:])
                    for token_id in prev_tokens:
                        logit_val = next_token_logits[b, token_id]
                        if logit_val > 0:
                            next_token_logits[b, token_id] = logit_val / repetition_penalty
                        else:
                            next_token_logits[b, token_id] = logit_val * repetition_penalty

            # Temperature scaling
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-K filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right so we keep at least the first token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for b in range(next_token_logits.shape[0]):
                    indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                    next_token_logits[b, indices_to_remove] = float("-inf")

            probs = F.softmax(next_token_logits, dim=-1)

            # Greedy argmax if temperature=0 else multinomial sampling
            if temperature == 0.0:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token), dim=-1)

            if stop_ids and any(tok.item() in stop_ids for tok in next_token.flatten()):
                break

        return generated

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save model weights and configuration."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(out_dir)
        torch.save(self.state_dict(), out_dir / "model.pt")

    @classmethod
    def from_pretrained(cls, load_directory: str | Path) -> KiaraForCausalLM:
        """Load model weights and configuration."""
        in_dir = Path(load_directory)
        config = KiaraConfig.from_pretrained(in_dir)
        model = cls(config)
        model.load_state_dict(torch.load(in_dir / "model.pt", map_location="cpu", weights_only=True))
        return model
