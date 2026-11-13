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
    raise NotImplementedError("Step 15: implement hash_block")


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
        raise NotImplementedError("Step 3: implement __init__")

    @property
    def num_free_blocks(self) -> int:
        """[Step 3]"""
        raise NotImplementedError("Step 3: implement num_free_blocks")

    def num_blocks_for(self, num_tokens: int) -> int:
        """[Step 3] ceil(num_tokens / block_size)."""
        raise NotImplementedError("Step 3: implement num_blocks_for")

    def register_request(self, request_id: str) -> None:
        """[Step 3] Start an empty block table. Duplicate ids are an error."""
        raise NotImplementedError("Step 3: implement register_request")

    def get_block_table(self, request_id: str) -> tuple[int, ...]:
        """[Step 3]"""
        raise NotImplementedError("Step 3: implement get_block_table")

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
        raise NotImplementedError("Step 3: implement allocate_slots")

    def free_blocks(self, request_id: str) -> None:
        """[Step 3] Release every block of the request but keep it registered
        (used by preemption: the request will be recomputed later)."""
        raise NotImplementedError("Step 3: implement free_blocks")

    def free(self, request_id: str) -> None:
        """[Step 3] free_blocks + forget the request entirely."""
        raise NotImplementedError("Step 3: implement free")

    # ---- Step 15 ----------------------------------------------------------
    def find_prefix_hit(self, token_ids: tuple[int, ...]) -> PrefixHit:
        """[Step 15] Longest run of leading FULL blocks whose chained hash is
        indexed. Never cover the last token: the model must compute at least one
        token to produce logits, so look at most at (len-1)//block_size blocks."""
        raise NotImplementedError("Step 15: implement find_prefix_hit")

    def cache_full_blocks(self, request_id: str, token_ids: tuple[int, ...], num_computed_tokens: int) -> None:
        """[Step 15] After a step, hash and index every block that is now full
        and computed but not yet indexed."""
        raise NotImplementedError("Step 15: implement cache_full_blocks")
