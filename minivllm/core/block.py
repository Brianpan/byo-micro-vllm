"""Physical KV-block allocator (Step 2; prefix-cache hooks in Step 15).

Mirrors micro_vllm/core/block.py. The pool knows nothing about tensors: a
"block" is just an integer id in [0, num_blocks). The ModelRunner owns a tensor
with exactly num_blocks pages, so block id N == page N of every layer's cache.

The LAST id is permanently reserved as the *null block*. Padded entries in a
batched block table point at it, so they are always a valid index but never
hold real data.
"""

from collections import deque
from dataclasses import dataclass
from typing import Iterable

BlockHash = bytes


@dataclass(slots=True)
class KVCacheBlock:
    block_id: int
    ref_count: int = 0
    block_hash: BlockHash | None = None  # set only when prefix caching (Step 15)


class BlockPool:
    def __init__(self, num_blocks: int) -> None:
        """[Step 2] Create num_blocks blocks; ids 0..num_blocks-2 start free (FIFO order)."""
        raise NotImplementedError("Step 2: implement __init__")

    @property
    def num_blocks(self) -> int:
        """[Step 2]"""
        raise NotImplementedError("Step 2: implement num_blocks")

    @property
    def null_block_id(self) -> int:
        """[Step 2]"""
        raise NotImplementedError("Step 2: implement null_block_id")

    @property
    def num_free_blocks(self) -> int:
        """[Step 2]"""
        raise NotImplementedError("Step 2: implement num_free_blocks")

    def ref_count(self, block_id: int) -> int:
        """[Step 2]"""
        raise NotImplementedError("Step 2: implement ref_count")

    def allocate(self, count: int, cached_block_ids: Iterable[int] = ()) -> tuple[int, ...] | None:
        """[Step 2] Atomically allocate ``count`` fresh blocks (ref_count=1) from the
        head of the free queue, or return None and change nothing.

        [Step 15] Also take one reference on each ``cached_block_ids`` entry. A
        cached block with ref_count 0 sits in the free queue, so it must be pulled
        out of the queue and counted against the free-block budget. A fresh block
        that still carries an old hash loses it (that is cache eviction).
        """
        raise NotImplementedError("Step 2: implement allocate")

    def release(self, block_id: int) -> None:
        """[Step 2] Drop one reference; at zero the block goes to the TAIL of the
        free queue (so recently freed, possibly-cached blocks are reused last = LRU).
        Releasing the null block or a free block is an error."""
        raise NotImplementedError("Step 2: implement release")

    # ---- Step 15: prefix cache index -------------------------------------
    @property
    def num_cached_blocks(self) -> int:
        """[Step 15]"""
        raise NotImplementedError("Step 15: implement num_cached_blocks")

    def block_hash(self, block_id: int) -> BlockHash | None:
        """[Step 15]"""
        raise NotImplementedError("Step 15: implement block_hash")

    def cache_block(self, block_id: int, block_hash: BlockHash) -> bool:
        """[Step 15] Index a full, computed block by content hash. If another
        block already owns this hash keep the first one and return False."""
        raise NotImplementedError("Step 15: implement cache_block")

    def get_cached_block(self, block_hash: BlockHash) -> int | None:
        """[Step 15] Block id holding this hash, or None."""
        raise NotImplementedError("Step 15: implement get_cached_block")

    def _evict(self, block: KVCacheBlock) -> None:
        """[Step 15] Remove a block's hash from the index (only if the index points at it)."""
        raise NotImplementedError("Step 15: implement _evict")
