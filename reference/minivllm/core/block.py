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
        if num_blocks < 2:
            raise ValueError("need at least one usable block plus the null block")
        self._blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self._null_block_id = num_blocks - 1
        self._free: deque[int] = deque(range(self._null_block_id))
        self._cached: dict[BlockHash, KVCacheBlock] = {}

    @property
    def num_blocks(self) -> int:
        """[Step 2]"""
        return len(self._blocks)

    @property
    def null_block_id(self) -> int:
        """[Step 2]"""
        return self._null_block_id

    @property
    def num_free_blocks(self) -> int:
        """[Step 2]"""
        return len(self._free)

    def ref_count(self, block_id: int) -> int:
        """[Step 2]"""
        return self._blocks[block_id].ref_count

    def allocate(self, count: int, cached_block_ids: Iterable[int] = ()) -> tuple[int, ...] | None:
        """[Step 2] Atomically allocate ``count`` fresh blocks (ref_count=1) from the
        head of the free queue, or return None and change nothing.

        [Step 15] Also take one reference on each ``cached_block_ids`` entry. A
        cached block with ref_count 0 sits in the free queue, so it must be pulled
        out of the queue and counted against the free-block budget. A fresh block
        that still carries an old hash loses it (that is cache eviction).
        """
        cached_ids = tuple(cached_block_ids)
        reclaimed = sum(1 for b in cached_ids if self._blocks[b].ref_count == 0)
        if count + reclaimed > len(self._free):
            return None
        for block_id in cached_ids:
            block = self._blocks[block_id]
            if block.ref_count == 0:
                self._free.remove(block_id)
            block.ref_count += 1
        new_ids = []
        for _ in range(count):
            block = self._blocks[self._free.popleft()]
            if block.block_hash is not None:  # only ever set once Step 15 exists
                self._evict(block)
            block.ref_count = 1
            new_ids.append(block.block_id)
        return tuple(new_ids)

    def release(self, block_id: int) -> None:
        """[Step 2] Drop one reference; at zero the block goes to the TAIL of the
        free queue (so recently freed, possibly-cached blocks are reused last = LRU).
        Releasing the null block or a free block is an error."""
        if block_id == self._null_block_id:
            raise ValueError("cannot release the null block")
        block = self._blocks[block_id]
        if block.ref_count <= 0:
            raise ValueError(f"double free of block {block_id}")
        block.ref_count -= 1
        if block.ref_count == 0:
            self._free.append(block_id)

    # ---- Step 15: prefix cache index -------------------------------------
    @property
    def num_cached_blocks(self) -> int:
        """[Step 15]"""
        return len(self._cached)

    def block_hash(self, block_id: int) -> BlockHash | None:
        """[Step 15]"""
        return self._blocks[block_id].block_hash

    def cache_block(self, block_id: int, block_hash: BlockHash) -> bool:
        """[Step 15] Index a full, computed block by content hash. If another
        block already owns this hash keep the first one and return False."""
        if block_hash in self._cached:
            return False
        block = self._blocks[block_id]
        self._evict(block)
        block.block_hash = block_hash
        self._cached[block_hash] = block
        return True

    def get_cached_block(self, block_hash: BlockHash) -> int | None:
        """[Step 15] Block id holding this hash, or None."""
        block = self._cached.get(block_hash)
        return None if block is None else block.block_id

    def _evict(self, block: KVCacheBlock) -> None:
        """[Step 15] Remove a block's hash from the index (only if the index points at it)."""
        if block.block_hash is not None:
            if self._cached.get(block.block_hash) is block:
                del self._cached[block.block_hash]
            block.block_hash = None
