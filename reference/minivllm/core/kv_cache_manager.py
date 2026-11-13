"""Logical KV-cache bookkeeping (Step 3; prefix caching in Step 15).

Mirrors micro_vllm/core/kv_cache_manager.py with one cache group (GPT-2 only
has full attention, so we drop micro-vllm's sliding-window cache groups).

Every request owns a *block table*: list position i = logical block i of the
sequence = tokens [i*block_size, (i+1)*block_size). The value is the physical
block id in the BlockPool.
"""

import hashlib
from dataclasses import dataclass, field

from minivllm.core.block import BlockHash, BlockPool


def hash_block(parent: BlockHash | None, token_ids: tuple[int, ...]) -> BlockHash:
    """[Step 15] Chained content hash: a block's identity includes its whole prefix."""
    h = hashlib.sha256()
    h.update(parent if parent is not None else b"\x00" * 32)
    for t in token_ids:
        h.update(int(t).to_bytes(8, "little"))
    return h.digest()


@dataclass(frozen=True, slots=True)
class PrefixHit:
    """[Step 15] Result of a prefix-cache lookup."""

    num_cached_tokens: int = 0
    block_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class _RequestState:
    block_ids: list[int] = field(default_factory=list)
    num_hashed_blocks: int = 0  # Step 15: how many leading blocks are in the hash index
    last_hash: BlockHash | None = None


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = False) -> None:
        """[Step 3] Own one BlockPool plus a per-request block table."""
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.block_pool = BlockPool(num_blocks)
        self._requests: dict[str, _RequestState] = {}

    @property
    def num_free_blocks(self) -> int:
        """[Step 3]"""
        return self.block_pool.num_free_blocks

    def num_blocks_for(self, num_tokens: int) -> int:
        """[Step 3] ceil(num_tokens / block_size)."""
        return -(-num_tokens // self.block_size)

    def register_request(self, request_id: str) -> None:
        """[Step 3] Start an empty block table. Duplicate ids are an error."""
        if request_id in self._requests:
            raise ValueError(f"duplicate request {request_id}")
        self._requests[request_id] = _RequestState()

    def get_block_table(self, request_id: str) -> tuple[int, ...]:
        """[Step 3]"""
        return tuple(self._requests[request_id].block_ids)

    def allocate_slots(
        self,
        request_id: str,
        num_computed_tokens: int,
        num_new_tokens: int,
        prefix_hit: "PrefixHit | None" = None,
    ) -> tuple[int, ...] | None:
        """[Step 3] Make sure the block table covers num_computed_tokens + num_new_tokens
        tokens. Allocate only the missing blocks. All-or-nothing: on failure return
        None with no state change. On success return the full block table.

        [Step 15] With a prefix_hit the table must be empty; it becomes
        prefix_hit.block_ids (shared, ref_count += 1) followed by fresh blocks.
        """
        state = self._requests[request_id]
        if prefix_hit is not None and prefix_hit.block_ids:
            if state.block_ids:
                raise ValueError("prefix hit only applies to an empty block table")
            cached = prefix_hit.block_ids
        else:
            cached = ()
        have = len(state.block_ids) + len(cached)
        need = self.num_blocks_for(num_computed_tokens + num_new_tokens) - have
        new_ids = self.block_pool.allocate(max(need, 0), cached)
        if new_ids is None:
            return None
        if cached:
            state.block_ids.extend(cached)
            state.num_hashed_blocks = len(cached)
            state.last_hash = self.block_pool.block_hash(cached[-1])
        state.block_ids.extend(new_ids)
        return tuple(state.block_ids)

    def free_blocks(self, request_id: str) -> None:
        """[Step 3] Release every block of the request but keep it registered
        (used by preemption: the request will be recomputed later)."""
        state = self._requests[request_id]
        # Release in reverse so a sequence's tail blocks are evicted before its head.
        for block_id in reversed(state.block_ids):
            self.block_pool.release(block_id)
        self._requests[request_id] = _RequestState()

    def free(self, request_id: str) -> None:
        """[Step 3] free_blocks + forget the request entirely."""
        self.free_blocks(request_id)
        del self._requests[request_id]

    # ---- Step 15 ----------------------------------------------------------
    def find_prefix_hit(self, token_ids: tuple[int, ...]) -> PrefixHit:
        """[Step 15] Longest run of leading FULL blocks whose chained hash is
        indexed. Never cover the last token: the model must compute at least one
        token to produce logits, so look at most at (len-1)//block_size blocks."""
        if not self.enable_prefix_cache:
            return PrefixHit()
        bs = self.block_size
        max_blocks = (len(token_ids) - 1) // bs
        parent: BlockHash | None = None
        block_ids: list[int] = []
        for i in range(max_blocks):
            parent = hash_block(parent, tuple(token_ids[i * bs:(i + 1) * bs]))
            block_id = self.block_pool.get_cached_block(parent)
            if block_id is None:
                break
            block_ids.append(block_id)
        return PrefixHit(len(block_ids) * bs, tuple(block_ids))

    def cache_full_blocks(self, request_id: str, token_ids: tuple[int, ...], num_computed_tokens: int) -> None:
        """[Step 15] After a step, hash and index every block that is now full
        and computed but not yet indexed."""
        if not self.enable_prefix_cache:
            return
        state = self._requests[request_id]
        bs = self.block_size
        num_full = num_computed_tokens // bs
        while state.num_hashed_blocks < num_full:
            i = state.num_hashed_blocks
            state.last_hash = hash_block(state.last_hash, tuple(token_ids[i * bs:(i + 1) * bs]))
            self.block_pool.cache_block(state.block_ids[i], state.last_hash)
            state.num_hashed_blocks += 1
