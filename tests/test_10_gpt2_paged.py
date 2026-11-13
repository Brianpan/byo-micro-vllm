"""Step 10 - GPT-2 running on the paged KV cache (incremental decoding)."""

import pytest
import torch

from minivllm.attention.metadata import AttentionMetadata, map_cache_slots
from minivllm.model.gpt2 import GPT2LMHeadModel

BS, NUM_BLOCKS = 4, 40


@pytest.fixture
def model(tiny_gpt2):
    return GPT2LMHeadModel.from_pretrained(tiny_gpt2[0]).eval()


def new_caches(model):
    c = model.config
    return [torch.zeros(2, NUM_BLOCKS, BS, c.n_head, c.head_dim) for _ in range(c.n_layer)]


def flat_batch(items):
    """items: list of (token_ids_full, num_computed, num_new, block_table)."""
    ids, pos, slots, qsl, ctx = [], [], [], [0], []
    width = max(len(t) for *_, t in items)
    bt = torch.full((len(items), width), NUM_BLOCKS - 1)
    for b, (toks, done, new, table) in enumerate(items):
        bt[b, : len(table)] = torch.tensor(table)
        p = torch.arange(done, done + new)
        ids.append(torch.tensor(toks[done: done + new]))
        pos.append(p)
        slots.append(map_cache_slots(p, bt[b], BS))
        qsl.append(qsl[-1] + new)
        ctx.append(done + new)
    md = AttentionMetadata(torch.cat(slots), torch.tensor(qsl), torch.tensor(ctx), bt, BS)
    return torch.cat(ids), torch.cat(pos), md


@torch.no_grad()
def test_prefill_equals_dense_forward(model):
    toks = list(range(3, 21))
    caches = new_caches(model)
    ids, pos, md = flat_batch([(toks, 0, len(toks), [7, 2, 11, 30, 5])])
    paged = model(ids, pos, caches, md)
    dense = model(ids, pos)
    torch.testing.assert_close(paged, dense, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_incremental_decode_matches_hf(model, tiny_gpt2):
    _, hf = tiny_gpt2
    torch.manual_seed(0)
    toks = torch.randint(0, model.config.vocab_size, (19,)).tolist()
    expected = hf(torch.tensor([toks])).logits[0]
    caches = new_caches(model)
    table = [12, 3, 25, 0, 9]
    # 7-token chunk, 6-token chunk, then 6 single-token decode steps
    boundaries = [(0, 7), (7, 13)] + [(i, i + 1) for i in range(13, 19)]
    for start, end in boundaries:
        ids, pos, md = flat_batch([(toks, start, end - start, table)])
        logits = model.compute_logits(model(ids, pos, caches, md))
        torch.testing.assert_close(logits, expected[start:end], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_batched_requests_match_individual(model, tiny_gpt2):
    """One forward containing: A's prefill, B's decode, C's middle chunk."""
    _, hf = tiny_gpt2
    torch.manual_seed(1)
    seqs = {n: torch.randint(0, model.config.vocab_size, (L,)).tolist() for n, L in (("A", 9), ("B", 14), ("C", 17))}
    tables = {"A": [1, 2, 3], "B": [10, 11, 12, 13], "C": [20, 21, 22, 23, 24]}
    done = {"A": 0, "B": 13, "C": 6}
    upto = {"A": 9, "B": 14, "C": 12}
    caches = new_caches(model)
    for n in ("B", "C"):                                  # warm prefixes, one request at a time
        ids, pos, md = flat_batch([(seqs[n], 0, done[n], tables[n])])
        model(ids, pos, caches, md)
    order = ["A", "B", "C"]
    ids, pos, md = flat_batch([(seqs[n], done[n], upto[n] - done[n], tables[n]) for n in order])
    logits = model.compute_logits(model(ids, pos, caches, md))
    expected = torch.cat([hf(torch.tensor([seqs[n]])).logits[0, done[n]:upto[n]] for n in order])
    torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_cache_is_actually_used(model):
    """Corrupting the cached prefix must change a decode step's output."""
    toks = list(range(1, 10))
    caches = new_caches(model)
    table = [0, 1, 2]
    ids, pos, md = flat_batch([(toks, 0, 8, table)])
    model(ids, pos, caches, md)
    ids, pos, md = flat_batch([(toks, 8, 1, table)])
    good = model(ids, pos, [c.clone() for c in caches], md)
    for c in caches:
        c[:, 0].normal_()
    bad = model(ids, pos, caches, md)
    assert not torch.allclose(good, bad)
