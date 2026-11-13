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
    raise NotImplementedError("Step 9: implement paged_attention")
