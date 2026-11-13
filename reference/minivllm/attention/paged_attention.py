"""Readable paged attention (Step 9). Mirrors micro_vllm/attention/operators/reference.py.

One call handles a mixed batch: request A may be prefilling 100 tokens while
request B decodes 1 token, all in the same flat [T, H, D] tensors. Production
engines replace the Python loop below with one fused kernel (FlashAttention
varlen / PagedAttention / micro-vllm's triton_paged.py); the math is identical.
"""

import math

import torch

from minivllm.attention.metadata import AttentionMetadata, gather_kv_cache, write_kv_cache


def paged_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_kv_cache: torch.Tensor,
    metadata: AttentionMetadata,
) -> torch.Tensor:
    """[Step 9] query/key/value: [T, H, D] for the NEW tokens only. Returns [T, H, D].

    1. write the new K/V into the cache (so a token can attend to itself),
    2. per request, gather its full history [0, context_len) from the cache,
    3. causal mask: the query at absolute position p sees keys 0..p, where the
       request's queries occupy positions [context_len - q_len, context_len),
    4. softmax(QK^T / sqrt(D)) V in float32, cast back to query dtype.
    """
    write_kv_cache(key, value, layer_kv_cache, metadata.slot_mapping)
    out = torch.empty_like(query)
    scale = 1.0 / math.sqrt(query.shape[-1])
    qsl = metadata.query_start_loc.tolist()
    ctx = metadata.context_lens.tolist()
    for b in range(metadata.num_requests):
        qs, qe, L = qsl[b], qsl[b + 1], ctx[b]
        k, v = gather_kv_cache(layer_kv_cache, metadata.block_tables[b], L, metadata.block_size)
        q = query[qs:qe].float()
        scores = torch.einsum("qhd,khd->hqk", q, k.float()) * scale
        q_pos = torch.arange(L - (qe - qs), L, device=query.device)
        k_pos = torch.arange(L, device=query.device)
        scores.masked_fill_((k_pos[None, :] > q_pos[:, None])[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[qs:qe] = torch.einsum("hqk,khd->qhd", probs, v.float()).to(query.dtype)
    return out
