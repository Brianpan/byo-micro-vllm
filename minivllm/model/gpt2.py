"""GPT-2 written for a serving engine (Step 7 dense, Step 10 paged).

Mirrors micro_vllm/models/llama.py in shape: the model takes a FLAT batch of
tokens (no [batch, seq] padding). input_ids and positions are both [T]. With
attention metadata + KV caches it runs paged attention (Step 10); without
them it runs plain causal attention over a single sequence (Step 7).

Note: HF's GPT-2 stores Linear weights as Conv1D with shape [in, out]; we use
nn.Linear ([out, in]) so every one of them must be transposed when loading.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from minivllm.attention.metadata import AttentionMetadata
from minivllm.attention.paged_attention import paged_attention


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_epsilon: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_hf(cls, hf_config) -> "GPT2Config":
        """[Step 7] Copy the six fields from a transformers GPT2Config."""
        raise NotImplementedError("Step 7: implement from_hf")


class GPT2Attention(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        """[Step 7] c_attn: E -> 3E (q,k,v), c_proj: E -> E."""
        raise NotImplementedError("Step 7: implement __init__")

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """[Step 7] x: [T, E] -> [T, E]. Project to q, k, v ([T, H, D] each), then
        dispatch: _dense() when metadata is None, _paged() otherwise (Step 10)."""
        raise NotImplementedError("Step 7: implement forward")

    def _dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """[Step 7] Causal attention over ONE sequence, no cache. [T, H, D] -> [T, H, D]."""
        raise NotImplementedError("Step 7: implement _dense")

    def _paged(self, q, k, v, kv_cache: torch.Tensor, metadata: AttentionMetadata) -> torch.Tensor:
        """[Step 10] Hand the new tokens' q/k/v to paged_attention (Step 9)."""
        raise NotImplementedError("Step 10: implement _paged")


class GPT2MLP(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        """[Step 7] c_fc: E -> 4E, gelu (tanh approx, a.k.a. gelu_new), c_proj: 4E -> E."""
        raise NotImplementedError("Step 7: implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[Step 7]"""
        raise NotImplementedError("Step 7: implement forward")


class GPT2Block(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        """[Step 7] Pre-LN: x + attn(ln_1(x)); x + mlp(ln_2(x))."""
        raise NotImplementedError("Step 7: implement __init__")

    def forward(self, x, kv_cache=None, metadata=None):
        """[Step 7]"""
        raise NotImplementedError("Step 7: implement forward")


class GPT2LMHeadModel(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        """[Step 7] wte, wpe, h[n_layer], ln_f. lm_head is TIED to wte (no own weight)."""
        raise NotImplementedError("Step 7: implement __init__")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor] | None = None,
        metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """[Step 7] input_ids, positions: [T] -> hidden states [T, E] (after ln_f).
        [Step 10] kv_caches[i] is layer i's cache, all layers share one metadata."""
        raise NotImplementedError("Step 7: implement forward")

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """[Step 7] [N, E] -> [N, vocab] using the tied embedding matrix."""
        raise NotImplementedError("Step 7: implement compute_logits")

    # ---- weight loading ---------------------------------------------------
    @torch.no_grad()
    def load_hf_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """[Step 7] Copy a transformers GPT2LMHeadModel state_dict into this model.
        Strip the "transformer." prefix, skip attn.bias / attn.masked_bias / lm_head,
        and transpose Conv1D weights (c_attn, c_proj, c_fc). Every parameter of
        ours must be loaded exactly once."""
        raise NotImplementedError("Step 7: implement load_hf_state_dict")

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "GPT2LMHeadModel":
        """[Step 7] Load with transformers, convert config + weights."""
        raise NotImplementedError("Step 7: implement from_pretrained")


def kv_cache_bytes_per_block(cfg: GPT2Config, block_size: int, dtype: torch.dtype) -> int:
    """[Step 12] K and V, every layer, one block: 2 * L * block_size * E * itemsize."""
    raise NotImplementedError("Step 12: implement kv_cache_bytes_per_block")

