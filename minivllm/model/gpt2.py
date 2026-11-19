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

from httpx2 import query
import torch
import torch.nn.functional as F
from torch import nn

from minivllm.attention.metadata import AttentionMetadata
from minivllm.attention.paged_attention import paged_attention

#             ┌─ request A (fresh prefill) ─┐ ┌ B (decode) ┐ ┌─ C (prefill chunk 2) ─┐
# input_ids = [ 464, 3290, 318,              812,           55, 901, 23 ]
# positions = [   0,    1,   2,               40,           16,  17, 18 ]

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
        self.config = cfg
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
        q, k, v = self.c_attn(x).chunk(3, dim=-1)

        # do [T, H, D] reshape
        head_dim = self.config.n_embd // self.config.n_head
        head_num = self.config.n_head
        t_size = x.size(0)
        q = q.view(t_size, head_num, head_dim)
        k = k.view(t_size, head_num, head_dim)
        v = v.view(t_size, head_num, head_dim)

        if metadata is None:
            out = self._dense(q, k, v)
        else:
            out = self._paged(q, k, v, kv_cache, metadata)

        # reshape back to [T, E]
        out = out.view(t_size, self.config.n_embd)
        # do output projection
        out = self.c_proj(out)

        return out

    def _dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """[Step 7] Causal attention over ONE sequence, no cache. [T, H, D] -> [T, H, D]."""
        # SDPA attends over dim -2, so heads must lead: [T, H, D] -> [H, T, D]
        # original code
        # attn_weight = query @ key.transpose(-2, -1) * scale_factor
        # attn_weight += attn_bias
        # attn_weight = torch.softmax(attn_weight, dim=-1)
        # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        # end original code

        q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return attn.transpose(0, 1).contiguous()  # back to [T, H, D]

    def _paged(self, q, k, v, kv_cache: torch.Tensor, metadata: AttentionMetadata) -> torch.Tensor:
        """[Step 10] Hand the new tokens' q/k/v to paged_attention (Step 9)."""
        raise NotImplementedError("Step 10: implement _paged")

# [64]  ──c_fc──►  [256]  ──gelu──►  [256]  ──c_proj──►  [64]
#        expand 4×        on/off switch       compress back
class GPT2MLP(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        """[Step 7] c_fc: E -> 4E, gelu (tanh approx, a.k.a. gelu_new), c_proj: 4E -> E."""
        super().__init__()
        self.config = cfg
        self.c_fc = nn.Linear(cfg.n_embd, 4*cfg.n_embd)
        self.c_proj = nn.Linear(4*cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[Step 7]"""
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)

        return x

# from layer perspective, each block has its own attention and MLP with pre-LN.
class GPT2Block(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        """[Step 7] Pre-LN: x + attn(ln_1(x)); x + mlp(ln_2(x))."""
        super().__init__()
        self.attn = GPT2Attention(cfg)
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.mlp = GPT2MLP(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)

    def forward(self, x, kv_cache=None, metadata=None):
        """[Step 7]"""
        residual = x
        x = self.ln_1(x)
        x = self.attn(x, kv_cache=kv_cache, metadata=metadata)
        x = x + residual

        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class GPT2LMHeadModel(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        """[Step 7] wte, wpe, h[n_layer], ln_f. lm_head is TIED to wte (no own weight)."""
        super().__init__()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.h = nn.ModuleList([GPT2Block(cfg) for _ in range(cfg.n_layer)])
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
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, kv_cache=kv_cache, metadata=metadata)
        x = self.ln_f(x)
        return x

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """[Step 7] [N, E] -> [N, vocab] using the tied embedding matrix."""
        return hidden @ self.wte.weight.T

    # ---- weight loading ---------------------------------------------------
    @torch.no_grad()
    def load_hf_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """[Step 7] Copy a transformers GPT2LMHeadModel state_dict into this model.
        Strip the "transformer." prefix, skip attn.bias / attn.masked_bias / lm_head,
        and transpose Conv1D weights (c_attn, c_proj, c_fc). Every parameter of
        ours must be loaded exactly once."""
        set_keys = set()
        for k, v in sd.items():
            if k.startswith('transformer.'):
                k = k[len('transformer.'):]
            if k.startswith('attn.bias') or k.startswith('attn.masked_bias') or k.startswith('lm_head'):
                continue
            if k.endswith('weight') and 'c_attn' in k or 'c_proj' in k or 'c_fc' in k:
                v = v.t()
            if k in self.state_dict():
                self.state_dict()[k].copy_(v)
                set_keys.add(k)

        missing_keys = set(self.state_dict().keys()) - set_keys

        if missing_keys:
            raise KeyError(f"Missing keys in state_dict: {missing_keys}")
        

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "GPT2LMHeadModel":
        """[Step 7] Load with transformers, convert config + weights."""
        from transformers import GPT2LMHeadModel as HFModel
        hf_model = HFModel.from_pretrained(name_or_path)
        hf_config =hf_model.config
        config = GPT2Config.from_hf(hf_config)
        model = cls(config)
        model.load_hf_state_dict(hf_model.state_dict())
        return model

def kv_cache_bytes_per_block(cfg: GPT2Config, block_size: int, dtype: torch.dtype) -> int:
    """[Step 12] K and V, every layer, one block: 2 * L * block_size * E * itemsize."""
    raise NotImplementedError("Step 12: implement kv_cache_bytes_per_block")

