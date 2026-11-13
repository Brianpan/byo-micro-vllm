"""Step 15 - Automatic prefix caching: share computed KV blocks by content hash."""

import pytest

from minivllm.config import EngineConfig
from minivllm.core.block import BlockPool
from minivllm.core.kv_cache_manager import KVCacheManager, hash_block
from minivllm.llm import LLM
from minivllm.sampling.params import SamplingParams

BS = 4


# ---- unit level -------------------------------------------------------------
def test_hash_chain_depends_on_prefix():
    h1 = hash_block(None, (1, 2, 3, 4))
    assert h1 == hash_block(None, (1, 2, 3, 4))
    assert hash_block(h1, (5, 6, 7, 8)) != hash_block(hash_block(None, (9, 9, 9, 9)), (5, 6, 7, 8))
    assert len(h1) == 32


def test_pool_cache_index_and_reclaim():
    pool = BlockPool(4)
    (a,) = pool.allocate(1)
    assert pool.cache_block(a, b"h" * 32)
    assert pool.get_cached_block(b"h" * 32) == a
    pool.release(a)                                   # freed but still indexed (reusable)
    assert pool.num_free_blocks == 3 and pool.get_cached_block(b"h" * 32) == a
    assert pool.allocate(0, cached_block_ids=[a]) == ()
    assert pool.ref_count(a) == 1 and pool.num_free_blocks == 2   # pulled out of the free queue


def test_pool_evicts_lru_cached_blocks():
    pool = BlockPool(3)                                # 2 usable
    a, b = pool.allocate(2)
    pool.cache_block(a, b"a" * 32)
    pool.cache_block(b, b"b" * 32)
    pool.release(a)
    pool.release(b)
    (c,) = pool.allocate(1)                            # reuses the least recently freed: a
    assert c == a
    assert pool.get_cached_block(b"a" * 32) is None    # evicted
    assert pool.get_cached_block(b"b" * 32) == b


def test_manager_hit_and_shared_refcounts():
    kv = KVCacheManager(num_blocks=16, block_size=BS, enable_prefix_cache=True)
    prompt = tuple(range(1, 11))                       # 10 tokens = 2 full blocks + 2
    kv.register_request("a")
    kv.allocate_slots("a", 0, 10)
    kv.cache_full_blocks("a", prompt, 10)
    hit = kv.find_prefix_hit(prompt)
    assert hit.num_cached_tokens == 8 and hit.block_ids == kv.get_block_table("a")[:2]

    kv.register_request("b")
    table = kv.allocate_slots("b", 8, 2, hit)
    assert table[:2] == hit.block_ids and table[2] != kv.get_block_table("a")[2]
    assert all(kv.block_pool.ref_count(x) == 2 for x in hit.block_ids)
    kv.free("a")
    assert all(kv.block_pool.ref_count(x) == 1 for x in hit.block_ids)


def test_never_cache_the_whole_prompt():
    kv = KVCacheManager(num_blocks=16, block_size=BS, enable_prefix_cache=True)
    prompt = tuple(range(1, 9))                        # exactly 2 full blocks
    kv.register_request("a")
    kv.allocate_slots("a", 0, 8)
    kv.cache_full_blocks("a", prompt, 8)
    assert kv.find_prefix_hit(prompt).num_cached_tokens == 4   # must leave >=1 token to compute


def test_disabled_means_no_hits():
    kv = KVCacheManager(num_blocks=16, block_size=BS, enable_prefix_cache=False)
    kv.register_request("a")
    kv.allocate_slots("a", 0, 12)
    kv.cache_full_blocks("a", tuple(range(12)), 12)
    assert kv.find_prefix_hit(tuple(range(13))).num_cached_tokens == 0


# ---- engine level -----------------------------------------------------------
SYSTEM = list(range(200, 230))                         # a 30-token shared "system prompt"
PROMPTS = [SYSTEM + [1, 2, 3], SYSTEM + [4, 5], SYSTEM + [6, 7, 8, 9]]


def test_outputs_identical_with_prefix_cache(tiny_gpt2, device, hf_greedy):
    cfg = dict(model=tiny_gpt2[0], device=device, block_size=BS, num_blocks=128)
    llm = LLM(EngineConfig(enable_prefix_cache=True, **cfg))
    first = llm.generate(PROMPTS[:1], SamplingParams(max_tokens=6))        # warms the cache
    rest = llm.generate(PROMPTS[1:], SamplingParams(max_tokens=6))
    llm.shutdown()
    got = [list(o.output_token_ids) for o in first + rest]
    assert got == [hf_greedy(p, 6) for p in PROMPTS]


def test_cache_hit_skips_prefill_compute(tiny_gpt2, device):
    llm = LLM(EngineConfig(model=tiny_gpt2[0], device=device, block_size=BS, num_blocks=128,
                           enable_prefix_cache=True))
    llm.generate(PROMPTS[:1], SamplingParams(max_tokens=2))
    llm.engine.add_request("b", PROMPTS[1], SamplingParams(max_tokens=2))
    plan = llm.scheduler.schedule()
    (item,) = plan.scheduled_requests
    llm.scheduler.update(plan, llm.executor.execute(plan))
    llm.shutdown()
    assert item.num_computed_tokens == 28              # 7 full shared blocks reused
    assert item.num_scheduled_tokens == len(PROMPTS[1]) - 28


def test_prefix_cache_under_preemption(tiny_gpt2, device, hf_greedy):
    llm = LLM(EngineConfig(model=tiny_gpt2[0], device=device, block_size=BS, num_blocks=24,
                           max_num_batched_tokens=24, enable_prefix_cache=True))
    outs = llm.generate(PROMPTS * 2, SamplingParams(max_tokens=10))
    llm.shutdown()
    assert [list(o.output_token_ids) for o in outs] == [hf_greedy(p, 10) for p in PROMPTS * 2]
