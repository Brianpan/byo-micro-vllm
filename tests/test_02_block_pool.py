"""Step 2 - BlockPool: a ref-counted allocator of KV pages."""

import pytest

from minivllm.core.block import BlockPool


def test_null_block_reserved():
    pool = BlockPool(5)
    assert pool.num_blocks == 5
    assert pool.null_block_id == 4
    assert pool.num_free_blocks == 4


def test_fifo_allocation_order():
    pool = BlockPool(6)
    assert pool.allocate(2) == (0, 1)
    assert pool.allocate(1) == (2,)
    assert pool.num_free_blocks == 2
    assert all(pool.ref_count(b) == 1 for b in (0, 1, 2))


def test_allocate_zero():
    pool = BlockPool(3)
    assert pool.allocate(0) == ()
    assert pool.num_free_blocks == 2


def test_allocation_is_all_or_nothing():
    pool = BlockPool(4)  # 3 usable
    pool.allocate(2)
    assert pool.allocate(2) is None
    assert pool.num_free_blocks == 1  # nothing leaked by the failed attempt
    assert pool.allocate(1) == (2,)


def test_release_returns_block_to_tail():
    pool = BlockPool(5)
    a, b, c = pool.allocate(3)
    pool.release(a)
    assert pool.ref_count(a) == 0
    # block 3 was free before block 0 was released -> FIFO hands out 3 first
    assert pool.allocate(2) == (3, a)


def test_double_free_and_null_release_rejected():
    pool = BlockPool(3)
    (a,) = pool.allocate(1)
    pool.release(a)
    with pytest.raises(ValueError):
        pool.release(a)
    with pytest.raises(ValueError):
        pool.release(pool.null_block_id)


def test_never_hands_out_null_block():
    pool = BlockPool(4)
    ids = pool.allocate(3)
    assert pool.null_block_id not in ids
    assert pool.allocate(1) is None
