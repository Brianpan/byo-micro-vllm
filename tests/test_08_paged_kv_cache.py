"""Step 8 - Paged KV cache addressing: slot mapping, write, gather."""

import torch

from minivllm.attention.metadata import gather_kv_cache, map_cache_slots, write_kv_cache

NUM_BLOCKS, BS, H, D = 8, 4, 2, 3


def new_cache():
    return torch.zeros(2, NUM_BLOCKS, BS, H, D)


def test_map_cache_slots():
    table = torch.tensor([5, 2, 7])
    pos = torch.arange(10)
    expected = [20, 21, 22, 23, 8, 9, 10, 11, 28, 29]
    assert map_cache_slots(pos, table, BS).tolist() == expected


def test_map_cache_slots_subset_positions():
    table = torch.tensor([5, 2, 7])
    assert map_cache_slots(torch.tensor([9, 4]), table, BS).tolist() == [29, 8]


def test_write_lands_in_the_right_page():
    cache = new_cache()
    k = torch.arange(H * D, dtype=torch.float32).view(1, H, D) + 1
    v = -k
    write_kv_cache(k, v, cache, torch.tensor([3 * BS + 2]))   # block 3, offset 2
    assert torch.equal(cache[0, 3, 2], k[0])
    assert torch.equal(cache[1, 3, 2], v[0])
    cache[0, 3, 2] = 0
    cache[1, 3, 2] = 0
    assert cache.abs().sum() == 0                              # nothing else was touched


def test_write_then_gather_roundtrip_non_contiguous_blocks():
    cache = new_cache()
    table = torch.tensor([6, 1, 4])
    k = torch.randn(10, H, D)
    v = torch.randn(10, H, D)
    # write in two chunks, like a chunked prefill
    write_kv_cache(k[:6], v[:6], cache, map_cache_slots(torch.arange(6), table, BS))
    write_kv_cache(k[6:], v[6:], cache, map_cache_slots(torch.arange(6, 10), table, BS))
    gk, gv = gather_kv_cache(cache, table, 10, BS)
    assert torch.equal(gk, k) and torch.equal(gv, v)
    gk, _ = gather_kv_cache(cache, table, 5, BS)              # partial history
    assert torch.equal(gk, k[:5])


def test_two_requests_do_not_collide():
    cache = new_cache()
    ta, tb = torch.tensor([0, 2]), torch.tensor([1, 3])
    ka, kb = torch.randn(7, H, D), torch.randn(5, H, D)
    write_kv_cache(ka, ka, cache, map_cache_slots(torch.arange(7), ta, BS))
    write_kv_cache(kb, kb, cache, map_cache_slots(torch.arange(5), tb, BS))
    assert torch.equal(gather_kv_cache(cache, ta, 7, BS)[0], ka)
    assert torch.equal(gather_kv_cache(cache, tb, 5, BS)[0], kb)


def test_works_with_cache_dtype(device):
    cache = torch.zeros(2, NUM_BLOCKS, BS, H, D, dtype=torch.float16, device=device)
    k = torch.randn(3, H, D, device=device)
    table = torch.tensor([2], device=device)
    write_kv_cache(k, k, cache, map_cache_slots(torch.arange(3, device=device), table, BS))
    torch.testing.assert_close(gather_kv_cache(cache, table, 3, BS)[0].float(), k, atol=1e-2, rtol=1e-2)
