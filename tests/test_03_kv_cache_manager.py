"""Step 3 - KVCacheManager: per-request block tables on top of the pool."""

import pytest

from minivllm.core.kv_cache_manager import KVCacheManager


@pytest.fixture
def kv():
    return KVCacheManager(num_blocks=9, block_size=4)  # 8 usable blocks = 32 token slots


def test_num_blocks_for(kv):
    assert [kv.num_blocks_for(n) for n in (0, 1, 4, 5, 8, 9)] == [0, 1, 1, 2, 2, 3]


def test_prefill_then_decode_growth(kv):
    kv.register_request("a")
    assert kv.allocate_slots("a", 0, 6) == (0, 1)        # 6 tokens -> 2 blocks
    assert kv.allocate_slots("a", 6, 1) == (0, 1)        # token 7 fits in block 1
    assert kv.allocate_slots("a", 7, 1) == (0, 1)        # token 8 fills block 1
    assert kv.allocate_slots("a", 8, 1) == (0, 1, 2)     # token 9 needs block 2
    assert kv.num_free_blocks == 5


def test_two_requests_interleave(kv):
    kv.register_request("a")
    kv.register_request("b")
    kv.allocate_slots("a", 0, 4)
    kv.allocate_slots("b", 0, 4)
    kv.allocate_slots("a", 4, 1)
    assert kv.get_block_table("a") == (0, 2)  # non-contiguous: that is the point of paging
    assert kv.get_block_table("b") == (1,)


def test_out_of_blocks_is_atomic(kv):
    kv.register_request("a")
    kv.register_request("b")
    kv.allocate_slots("a", 0, 28)                        # 7 blocks
    assert kv.allocate_slots("b", 0, 5) is None          # needs 2, only 1 free
    assert kv.get_block_table("b") == ()
    assert kv.num_free_blocks == 1


def test_free_blocks_keeps_registration(kv):
    kv.register_request("a")
    kv.allocate_slots("a", 0, 9)
    kv.free_blocks("a")                                  # preemption
    assert kv.get_block_table("a") == ()
    assert kv.num_free_blocks == 8
    assert kv.allocate_slots("a", 0, 2) is not None


def test_free_forgets_request(kv):
    kv.register_request("a")
    kv.allocate_slots("a", 0, 3)
    kv.free("a")
    assert kv.num_free_blocks == 8
    with pytest.raises(KeyError):
        kv.get_block_table("a")
    kv.register_request("a")  # id can be reused afterwards


def test_duplicate_registration(kv):
    kv.register_request("a")
    with pytest.raises(ValueError):
        kv.register_request("a")
