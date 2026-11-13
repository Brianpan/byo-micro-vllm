"""Continuous-batching scheduler (Steps 4-5; prefix cache hook Step 15).

Mirrors micro_vllm/core/scheduler.py. One call to schedule() builds a plan
under a token budget (max_num_batched_tokens) and a sequence budget
(max_num_seqs):

  1. RUNNING requests first (decode = 1 token, or the next prefill chunk).
     If the KV cache is full, preempt the newest running request (Step 5).
  2. Then admit WAITING requests FCFS while budget and cache allow
     (a long prompt is split across steps = chunked prefill).

There is no separate "prefill phase" and "decode phase": every request just
has num_pending_tokens, and the scheduler hands out token budget.
"""

from collections import deque

from minivllm.config import EngineConfig
from minivllm.core.kv_cache_manager import KVCacheManager, PrefixHit
from minivllm.core.request import FinishReason, Request, RequestStatus
from minivllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from minivllm.executor.interface import ModelRunnerOutput


class Scheduler:
    def __init__(self, config: EngineConfig, kv_cache_manager: KVCacheManager) -> None:
        """[Step 4]"""
        raise NotImplementedError("Step 4: implement __init__")

    # ---- request lifecycle ------------------------------------------------
    def add_request(self, request: Request) -> None:
        """[Step 4] Validate (WAITING, prompt < max_model_len, unique id), register
        with the KV manager and append to the waiting queue."""
        raise NotImplementedError("Step 4: implement add_request")

    def get_request(self, request_id: str) -> Request | None:
        """[Step 4]"""
        raise NotImplementedError("Step 4: implement get_request")

    def has_unfinished_requests(self) -> bool:
        """[Step 4]"""
        raise NotImplementedError("Step 4: implement has_unfinished_requests")

    def abort_request(self, request_id: str) -> bool:
        """[Step 5] Finish with ABORTED, drop from queues, free all blocks.
        Return False if unknown or already finished."""
        raise NotImplementedError("Step 5: implement abort_request")

    # ---- planning ---------------------------------------------------------
    def schedule(self) -> SchedulerOutput:
        """[Step 4] Build one plan (see module docstring). Only one plan may be
        in flight: calling schedule() again before update() raises RuntimeError.
        Raise RuntimeError if the very first request of a plan cannot fit in the
        whole KV cache (it never could)."""
        raise NotImplementedError("Step 4: implement schedule")

    def _try_allocate(
        self, request: Request, num_computed: int, num_new: int, hit: PrefixHit | None = None
    ) -> ScheduledRequest | None:
        """[Step 4] Ask the KV manager for slots; on success build the immutable
        ScheduledRequest (input slice, block table, sampling info)."""
        raise NotImplementedError("Step 4: implement _try_allocate")

    def _preempt_last(self) -> str:
        """[Step 5] Recompute-style preemption of the NEWEST running request:
        pop it, reset its progress, free its blocks, put it at the FRONT of the
        waiting queue. Return its id."""
        raise NotImplementedError("Step 5: implement _preempt_last")

    # ---- commit -----------------------------------------------------------
    def update(self, plan: SchedulerOutput, result: ModelRunnerOutput) -> None:
        """[Step 4] Commit one executed plan: advance computed tokens, append
        sampled tokens, decide finish reasons (EOS > LENGTH > MODEL_LENGTH),
        free finished requests. Skip requests aborted mid-flight."""
        raise NotImplementedError("Step 4: implement update")
