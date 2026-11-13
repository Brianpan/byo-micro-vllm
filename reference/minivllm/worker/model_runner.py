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
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = DTYPES[config.dtype]
        self.block_size = config.block_size
        self.model: GPT2LMHeadModel | None = None
        self.kv_cache: torch.Tensor | None = None  # [L, 2, num_blocks, block_size, H, D]
        self.num_blocks: int | None = None
        self.sampler = Sampler()

    # ---- init -------------------------------------------------------------
    def initialize_model(self) -> None:
        """[Step 12] Load GPT-2 from config.model, move to device/dtype, eval()."""
        model = GPT2LMHeadModel.from_pretrained(self.config.model)
        self.model = model.to(device=self.device, dtype=self.dtype).eval()

    def determine_num_blocks(self) -> int:
        """[Step 12] config.num_blocks if given. Else on CUDA, profile:
            budget = total * gpu_memory_utilization - used_now - peak_activation
        where peak_activation comes from a dummy max_num_batched_tokens forward.
        Else (CPU) enough for max_num_seqs full-length sequences, + 1 null block."""
        cfg = self.config
        if cfg.num_blocks is not None:
            return cfg.num_blocks
        per_block = kv_cache_bytes_per_block(self.model.config, self.block_size, self.dtype)
        if self.device.type != "cuda":
            return cfg.max_num_seqs * -(-cfg.max_model_len // self.block_size) + 1
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        base = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        n = min(cfg.max_num_batched_tokens, self.model.config.n_positions)
        with torch.inference_mode():
            ids = torch.zeros(n, dtype=torch.long, device=self.device)
            self.model.compute_logits(self.model(ids, torch.arange(n, device=self.device)))
        torch.cuda.synchronize(self.device)
        peak = torch.cuda.max_memory_allocated(self.device) - base
        torch.cuda.empty_cache()
        budget = total * cfg.gpu_memory_utilization - (total - free) - peak
        num_blocks = int(budget // per_block)
        if num_blocks < 2:
            raise RuntimeError("not enough GPU memory for the KV cache")
        return num_blocks

    def initialize_kv_cache(self) -> None:
        """[Step 12] Allocate ONE zero tensor [L, 2, num_blocks, block_size, H, D]."""
        c = self.model.config
        self.num_blocks = self.determine_num_blocks()
        self.kv_cache = torch.zeros(
            (c.n_layer, 2, self.num_blocks, self.block_size, c.n_head, c.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

    # ---- per step ---------------------------------------------------------
    def build_model_input(self, plan: SchedulerOutput) -> ModelInput:
        """[Step 12] Flatten the plan: concat input ids, absolute positions
        (num_computed .. num_computed+num_scheduled), slot mapping via each
        request's block table, query_start_loc, context_lens, padded block tables
        (pad = null block = num_blocks - 1), and the hidden-state row of the LAST
        token of every request with requires_sampling."""
        input_ids, positions, slots, qsl, ctx, rows, rids = [], [], [], [0], [], [], []
        dev = self.device
        reqs = plan.scheduled_requests
        max_blocks = max(len(r.block_ids) for r in reqs)
        block_tables = torch.full((len(reqs), max_blocks), self.num_blocks - 1, dtype=torch.long)
        for b, r in enumerate(reqs):
            pos = torch.arange(r.num_computed_tokens, r.num_computed_tokens + r.num_scheduled_tokens)
            block_tables[b, : len(r.block_ids)] = torch.tensor(r.block_ids, dtype=torch.long)
            input_ids.extend(r.input_token_ids)
            positions.append(pos)
            slots.append(map_cache_slots(pos, block_tables[b], self.block_size))
            qsl.append(qsl[-1] + r.num_scheduled_tokens)
            ctx.append(r.num_computed_tokens + r.num_scheduled_tokens)
            if r.requires_sampling:
                rows.append(qsl[-1] - 1)
                rids.append(r.request_id)
        metadata = AttentionMetadata(
            slot_mapping=torch.cat(slots).to(dev),
            query_start_loc=torch.tensor(qsl, dtype=torch.long, device=dev),
            context_lens=torch.tensor(ctx, dtype=torch.long, device=dev),
            block_tables=block_tables.to(dev),
            block_size=self.block_size,
        )
        return ModelInput(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=dev),
            positions=torch.cat(positions).to(dev),
            metadata=metadata,
            sample_rows=torch.tensor(rows, dtype=torch.long, device=dev),
            sample_request_ids=tuple(rids),
        )

    @torch.inference_mode()
    def execute(self, plan: SchedulerOutput) -> ModelRunnerOutput:
        """[Step 12] build_model_input -> forward with the per-layer caches
        (kv_cache[i]) -> logits only for sample_rows -> Sampler."""
        if self.model is None or self.kv_cache is None:
            raise RuntimeError("model runner not initialized")
        if not plan.scheduled_requests:
            return ModelRunnerOutput()
        mi = self.build_model_input(plan)
        hidden = self.model(mi.input_ids, mi.positions, list(self.kv_cache.unbind(0)), mi.metadata)
        if mi.sample_rows.numel() == 0:
            return ModelRunnerOutput()
        logits = self.model.compute_logits(hidden[mi.sample_rows])
        by_id = {r.request_id: r for r in plan.scheduled_requests}
        params = tuple(by_id[rid].sampling_params for rid in mi.sample_request_ids)
        indices = tuple(by_id[rid].sample_index for rid in mi.sample_request_ids)
        return self.sampler(logits, mi.sample_request_ids, params, indices)

    def shutdown(self) -> None:
        """[Step 12] Drop model and cache references (lets the allocator reclaim them)."""
        self.model = None
        self.kv_cache = None
