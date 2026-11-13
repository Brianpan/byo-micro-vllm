"""Owns the model and the physical KV cache on ONE device (Step 12).

Mirrors micro_vllm/worker/model_runner.py minus CUDA graphs / multi-group caches.

execute(plan):
    SchedulerOutput (pure Python)  --build_model_input-->  GPU tensors
    model.forward(flat batch, paged KV)  -> hidden [T, E]
    pick the LAST row of each request that requires sampling -> logits [S, V]
    Sampler -> ModelRunnerOutput
"""

from dataclasses import dataclass

import torch

from minivllm.attention.metadata import AttentionMetadata, map_cache_slots
from minivllm.config import EngineConfig
from minivllm.core.scheduler_output import SchedulerOutput
from minivllm.executor.interface import ModelRunnerOutput
from minivllm.model.gpt2 import GPT2LMHeadModel, kv_cache_bytes_per_block
from minivllm.sampling.sampler import Sampler

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True, slots=True, eq=False)
class ModelInput:
    input_ids: torch.Tensor         # [T]
    positions: torch.Tensor         # [T] absolute positions
    metadata: AttentionMetadata
    sample_rows: torch.Tensor       # [S] row index into hidden states, one per sampled request
    sample_request_ids: tuple[str, ...]


class ModelRunner:
    def __init__(self, config: EngineConfig) -> None:
        """[Step 12] Resolve device/dtype; model and cache are created later."""
        raise NotImplementedError("Step 12: implement __init__")

    # ---- init -------------------------------------------------------------
    def initialize_model(self) -> None:
        """[Step 12] Load GPT-2 from config.model, move to device/dtype, eval()."""
        raise NotImplementedError("Step 12: implement initialize_model")

    def determine_num_blocks(self) -> int:
        """[Step 12] config.num_blocks if given. Else on CUDA, profile:
            budget = total * gpu_memory_utilization - used_now - peak_activation
        where peak_activation comes from a dummy max_num_batched_tokens forward.
        Else (CPU) enough for max_num_seqs full-length sequences, + 1 null block."""
        raise NotImplementedError("Step 12: implement determine_num_blocks")

    def initialize_kv_cache(self) -> None:
        """[Step 12] Allocate ONE zero tensor [L, 2, num_blocks, block_size, H, D]."""
        raise NotImplementedError("Step 12: implement initialize_kv_cache")

    # ---- per step ---------------------------------------------------------
    def build_model_input(self, plan: SchedulerOutput) -> ModelInput:
        """[Step 12] Flatten the plan: concat input ids, absolute positions
        (num_computed .. num_computed+num_scheduled), slot mapping via each
        request's block table, query_start_loc, context_lens, padded block tables
        (pad = null block = num_blocks - 1), and the hidden-state row of the LAST
        token of every request with requires_sampling."""
        raise NotImplementedError("Step 12: implement build_model_input")

    @torch.inference_mode()
    def execute(self, plan: SchedulerOutput) -> ModelRunnerOutput:
        """[Step 12] build_model_input -> forward with the per-layer caches
        (kv_cache[i]) -> logits only for sample_rows -> Sampler."""
        raise NotImplementedError("Step 12: implement execute")

    def shutdown(self) -> None:
        """[Step 12] Drop model and cache references (lets the allocator reclaim them)."""
        raise NotImplementedError("Step 12: implement shutdown")
