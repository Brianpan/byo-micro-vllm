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
        return cls(
            vocab_size=hf_config.vocab_size,
            n_positions=hf_config.n_positions,
            n_embd=hf_config.n_embd,
            n_layer=hf_config.n_layer,
            n_head=hf_config.n_head,
            layer_norm_epsilon=hf_config.layer_norm_epsilon,
        )


class GPT2Attention(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        """[Step 7] c_attn: E -> 3E (q,k,v), c_proj: E -> E."""
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """[Step 7] x: [T, E] -> [T, E]. Project to q, k, v ([T, H, D] each), then
        dispatch: _dense() when metadata is None, _paged() otherwise (Step 10)."""
        T = x.shape[0]
        q, k, v = self.c_attn(x).split(x.shape[-1], dim=-1)
        q = q.view(T, self.n_head, self.head_dim)
        k = k.view(T, self.n_head, self.head_dim)
        v = v.view(T, self.n_head, self.head_dim)
        if metadata is None:
            out = self._dense(q, k, v)
        else:
            out = self._paged(q, k, v, kv_cache, metadata)
        return self.c_proj(out.reshape(T, -1))

    def _dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """[Step 7] Causal attention over ONE sequence, no cache. [T, H, D] -> [T, H, D]."""
        return F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True
        ).transpose(0, 1)

    def _paged(self, q, k, v, kv_cache: torch.Tensor, metadata: AttentionMetadata) -> torch.Tensor:
        """[Step 10] Hand the new tokens' q/k/v to paged_attention (Step 9)."""
        return paged_attention(q, k, v, kv_cache, metadata)


class GPT2MLP(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        """[Step 7] c_fc: E -> 4E, gelu (tanh approx, a.k.a. gelu_new), c_proj: 4E -> E."""
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[Step 7]"""
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class GPT2Block(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        """[Step 7] Pre-LN: x + attn(ln_1(x)); x + mlp(ln_2(x))."""
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = GPT2Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.mlp = GPT2MLP(cfg)

    def forward(self, x, kv_cache=None, metadata=None):
        """[Step 7]"""
        x = x + self.attn(self.ln_1(x), kv_cache, metadata)
        return x + self.mlp(self.ln_2(x))


class GPT2LMHeadModel(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        """[Step 7] wte, wpe, h[n_layer], ln_f. lm_head is TIED to wte (no own weight)."""
        super().__init__()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.h = nn.ModuleList(GPT2Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor] | None = None,
        metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """[Step 7] input_ids, positions: [T] -> hidden states [T, E] (after ln_f).
        [Step 10] kv_caches[i] is layer i's cache, all layers share one metadata."""
        x = self.wte(input_ids) + self.wpe(positions)
        for i, block in enumerate(self.h):
            x = block(x, None if kv_caches is None else kv_caches[i], metadata)
        return self.ln_f(x)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """[Step 7] [N, E] -> [N, vocab] using the tied embedding matrix."""
        return hidden @ self.wte.weight.t()

    # ---- weight loading ---------------------------------------------------
    @torch.no_grad()
    def load_hf_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """[Step 7] Copy a transformers GPT2LMHeadModel state_dict into this model.
        Strip the "transformer." prefix, skip attn.bias / attn.masked_bias / lm_head,
        and transpose Conv1D weights (c_attn, c_proj, c_fc). Every parameter of
        ours must be loaded exactly once."""
        own = dict(self.named_parameters())
        loaded = set()
        for name, tensor in sd.items():
            name = name.removeprefix("transformer.")
            # attn.bias / attn.masked_bias are causal-mask buffers, not weights
            if name.startswith("lm_head") or name.endswith((".attn.bias", ".attn.masked_bias")):
                continue
            if name not in own:
                raise KeyError(f"unexpected weight {name}")
            if name.endswith(".weight") and any(k in name for k in ("c_attn", "c_proj", "c_fc")):
                tensor = tensor.t()
            if own[name].shape != tensor.shape:
                raise ValueError(f"shape mismatch for {name}: {tuple(own[name].shape)} vs {tuple(tensor.shape)}")
            own[name].copy_(tensor)
            loaded.add(name)
        missing = set(own) - loaded
        if missing:
            raise KeyError(f"missing weights: {sorted(missing)}")

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "GPT2LMHeadModel":
        """[Step 7] Load with transformers, convert config + weights."""
        from transformers import GPT2LMHeadModel as HFGPT2

        hf = HFGPT2.from_pretrained(name_or_path)
        model = cls(GPT2Config.from_hf(hf.config))
        model.load_hf_state_dict(hf.state_dict())
        return model


def kv_cache_bytes_per_block(cfg: GPT2Config, block_size: int, dtype: torch.dtype) -> int:
    """[Step 12] K and V, every layer, one block: 2 * L * block_size * E * itemsize."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    return 2 * cfg.n_layer * block_size * cfg.n_embd * itemsize

