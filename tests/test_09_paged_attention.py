"""Step 9 - Paged attention over a mixed prefill/decode batch."""

import pytest
import torch

from minivllm.attention.metadata import AttentionMetadata, map_cache_slots
from minivllm.attention.paged_attention import paged_attention

H, D, BS, NUM_BLOCKS = 4, 8, 4, 32


def dense_causal(q, k, v):
    """Ground truth for ONE full sequence: [L, H, D] each."""
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True).transpose(0, 1)


def make_metadata(chunks, tables, null_block):
    """chunks: list of (num_computed, num_new). tables: list of block id lists."""
    slots, qsl, ctx = [], [0], []
    width = max(len(t) for t in tables)
    bt = torch.full((len(tables), width), null_block)
    for b, ((done, new), table) in enumerate(zip(chunks, tables)):
        bt[b, : len(table)] = torch.tensor(table)
        slots.append(map_cache_slots(torch.arange(done, done + new), bt[b], BS))
        qsl.append(qsl[-1] + new)
        ctx.append(done + new)
    return AttentionMetadata(torch.cat(slots), torch.tensor(qsl), torch.tensor(ctx), bt, BS)


def test_single_prefill_equals_dense():
    torch.manual_seed(0)
    L = 11
    q, k, v = (torch.randn(L, H, D) for _ in range(3))
    cache = torch.zeros(2, NUM_BLOCKS, BS, H, D)
    md = make_metadata([(0, L)], [[9, 3, 17]], NUM_BLOCKS - 1)
    torch.testing.assert_close(paged_attention(q, k, v, cache, md), dense_causal(q, k, v), atol=1e-5, rtol=1e-5)


def test_chunked_prefill_then_decode_equals_dense():
    torch.manual_seed(1)
    L = 13
    q, k, v = (torch.randn(L, H, D) for _ in range(3))
    expected = dense_causal(q, k, v)
    cache = torch.zeros(2, NUM_BLOCKS, BS, H, D)
    table = [4, 0, 22, 8]
    for start, end in [(0, 5), (5, 12), (12, 13)]:  # chunk, chunk, single-token decode
        md = make_metadata([(start, end - start)], [table], NUM_BLOCKS - 1)
        out = paged_attention(q[start:end], k[start:end], v[start:end], cache, md)
        torch.testing.assert_close(out, expected[start:end], atol=1e-5, rtol=1e-5)


def test_mixed_batch_equals_per_request_dense():
    """Request A: fresh prefill of 6. Request B: 9 cached tokens + 1 decode. C: 3 cached + chunk of 4."""
    torch.manual_seed(2)
    lens = {"A": 6, "B": 10, "C": 7}
    seqs = {n: [torch.randn(L, H, D) for _ in range(3)] for n, L in lens.items()}
    tables = {"A": [1, 2], "B": [10, 5, 30], "C": [7, 25]}
    done = {"A": 0, "B": 9, "C": 3}
    cache = torch.zeros(2, NUM_BLOCKS, BS, H, D)
    # warm the cache for the already-computed prefix of B and C
    for n in ("B", "C"):
        q, k, v = (t[: done[n]] for t in seqs[n])
        paged_attention(q, k, v, cache, make_metadata([(0, done[n])], [tables[n]], NUM_BLOCKS - 1))
    order = ["A", "B", "C"]
    md = make_metadata([(done[n], lens[n] - done[n]) for n in order], [tables[n] for n in order], NUM_BLOCKS - 1)
    q, k, v = (torch.cat([seqs[n][i][done[n]:] for n in order]) for i in range(3))
    out = paged_attention(q, k, v, cache, md)
    expected = torch.cat([dense_causal(*seqs[n])[done[n]:] for n in order])
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_null_block_padding_is_ignored():
    """Padding entries point at the null block; filling it with garbage must not matter."""
    torch.manual_seed(3)
    cache = torch.zeros(2, NUM_BLOCKS, BS, H, D)
    cache[:, NUM_BLOCKS - 1] = 1e4
    q, k, v = (torch.randn(3, H, D) for _ in range(3))
    md = make_metadata([(0, 3), (0, 9)], [[0], [1, 2, 3]], NUM_BLOCKS - 1)
    q2, k2, v2 = (torch.randn(9, H, D) for _ in range(3))
    out = paged_attention(torch.cat([q, q2]), torch.cat([k, k2]), torch.cat([v, v2]), cache, md)
    torch.testing.assert_close(out[:3], dense_causal(q, k, v), atol=1e-5, rtol=1e-5)


def test_on_device_half(device):
    if device != "cuda":
        pytest.skip("needs a CUDA GPU")
    torch.manual_seed(4)
    q, k, v = (torch.randn(9, H, D, device=device, dtype=torch.float16) for _ in range(3))
    cache = torch.zeros(2, NUM_BLOCKS, BS, H, D, device=device, dtype=torch.float16)
    md = make_metadata([(0, 9)], [[5, 6, 7]], NUM_BLOCKS - 1)
    md = AttentionMetadata(*(t.to(device) for t in (md.slot_mapping, md.query_start_loc, md.context_lens, md.block_tables)), BS)
    out = paged_attention(q, k, v, cache, md)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), dense_causal(q.float(), k.float(), v.float()), atol=2e-3, rtol=2e-3)
