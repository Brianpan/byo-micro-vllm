"""Paged KV cache addressing (Step 8). Mirrors micro_vllm/attention/attention.py.

Physical layout of ONE layer's cache (allocated by the ModelRunner, Step 12):

    layer_kv_cache: [2, num_blocks, block_size, num_heads, head_dim]
                     ^K/V

Flattening dims 1-2 gives num_blocks * block_size "slots". Token at logical
position p of a request lives in slot

    block_table[p // block_size] * block_size + p % block_size
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, eq=False)
class AttentionMetadata:
    """Everything attention needs to find a FLAT batch's K/V in the paged cache.

    T = total scheduled tokens in the batch, B = number of requests.
    slot_mapping:    [T]   where to WRITE each new token's K/V
    query_start_loc: [B+1] request b owns query rows [qsl[b], qsl[b+1])
    context_lens:    [B]   tokens visible to request b AFTER this step
                           (= num_computed_tokens + num_scheduled_tokens)
    block_tables:    [B, max_blocks] padded with the null block id
    """

    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    block_size: int

    @property
    def num_requests(self) -> int:
        return self.context_lens.numel()


def map_cache_slots(positions: torch.Tensor, block_table: torch.Tensor, block_size: int) -> torch.Tensor:
    """[Step 8] Logical positions [N] + one request's block table -> flat slot ids [N]."""
    return block_table[positions // block_size] * block_size + positions % block_size


def write_kv_cache(key: torch.Tensor, value: torch.Tensor, layer_kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    """[Step 8] Scatter key/value [T, H, D] into their slots, in place."""
    flat = layer_kv_cache.view(2, -1, *layer_kv_cache.shape[3:])
    flat[0].index_copy_(0, slot_mapping, key.to(flat.dtype))
    flat[1].index_copy_(0, slot_mapping, value.to(flat.dtype))


def gather_kv_cache(
    layer_kv_cache: torch.Tensor, block_table: torch.Tensor, context_len: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """[Step 8] Read positions [0, context_len) of one request -> (K, V) each [context_len, H, D]."""
    flat = layer_kv_cache.view(2, -1, *layer_kv_cache.shape[3:])
    positions = torch.arange(context_len, device=block_table.device)
    slots = map_cache_slots(positions, block_table, block_size)
    return flat[0].index_select(0, slots), flat[1].index_select(0, slots)
