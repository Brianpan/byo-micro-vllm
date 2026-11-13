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
        self.config = config
        self.kv = kv_cache_manager
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.requests: dict[str, Request] = {}
        self._pending_plan: SchedulerOutput | None = None

    # ---- request lifecycle ------------------------------------------------
    def add_request(self, request: Request) -> None:
        """[Step 4] Validate (WAITING, prompt < max_model_len, unique id), register
        with the KV manager and append to the waiting queue."""
        if request.status is not RequestStatus.WAITING:
            raise ValueError("request must be WAITING")
        if request.num_prompt_tokens >= self.config.max_model_len:
            raise ValueError("prompt must be shorter than max_model_len")
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request_id {request.request_id}")
        self.kv.register_request(request.request_id)
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def get_request(self, request_id: str) -> Request | None:
        """[Step 4]"""
        return self.requests.get(request_id)

    def has_unfinished_requests(self) -> bool:
        """[Step 4]"""
        return bool(self.waiting or self.running)

    def abort_request(self, request_id: str) -> bool:
        """[Step 5] Finish with ABORTED, drop from queues, free all blocks.
        Return False if unknown or already finished."""
        request = self.requests.get(request_id)
        if request is None or request.is_finished:
            return False
        if request.status is RequestStatus.WAITING:
            self.waiting.remove(request)
        else:
            self.running.remove(request)
        request.finish(FinishReason.ABORTED)
        self.kv.free(request_id)
        return True

    # ---- planning ---------------------------------------------------------
    def schedule(self) -> SchedulerOutput:
        """[Step 4] Build one plan (see module docstring). Only one plan may be
        in flight: calling schedule() again before update() raises RuntimeError.
        Raise RuntimeError if the very first request of a plan cannot fit in the
        whole KV cache (it never could)."""
        if self._pending_plan is not None:
            raise RuntimeError("previous plan has not been updated")
        cfg = self.config
        budget = cfg.max_num_batched_tokens
        scheduled: list[ScheduledRequest] = []
        preempted: list[str] = []
        cache_full = False

        # 1) running requests
        i = 0
        while i < len(self.running):
            if budget == 0 or len(scheduled) == cfg.max_num_seqs:
                break
            request = self.running[i]
            n = min(request.num_pending_tokens, budget, cfg.max_model_len - request.num_computed_tokens)
            item = self._try_allocate(request, request.num_computed_tokens, n)
            while item is None and self.running[-1] is not request:
                preempted.append(self._preempt_last())             # Step 5
                item = self._try_allocate(request, request.num_computed_tokens, n)
            if item is None:
                if not scheduled:
                    raise RuntimeError(f"{request.request_id} cannot fit in the KV cache")
                cache_full = True
                break
            scheduled.append(item)
            budget -= n
            i += 1

        # 2) waiting requests (FCFS; never skip the head of the queue)
        while self.waiting and not cache_full and budget > 0 and len(self.running) < cfg.max_num_seqs:
            request = self.waiting[0]
            hit = None
            if cfg.enable_prefix_cache:  # Step 15
                hit = self.kv.find_prefix_hit(request.all_token_ids)
            start = hit.num_cached_tokens if hit is not None else 0
            n = min(request.num_known_tokens - start, budget, cfg.max_model_len - start)
            item = self._try_allocate(request, start, n, hit)
            if item is None:
                if not scheduled:
                    raise RuntimeError(f"{request.request_id} cannot fit in the KV cache")
                break
            self.waiting.popleft()
            request.resume(start)
            self.running.append(request)
            scheduled.append(item)
            budget -= n

        plan = SchedulerOutput(tuple(scheduled), tuple(preempted))
        if scheduled:
            self._pending_plan = plan
        return plan

    def _try_allocate(
        self, request: Request, num_computed: int, num_new: int, hit: PrefixHit | None = None
    ) -> ScheduledRequest | None:
        """[Step 4] Ask the KV manager for slots; on success build the immutable
        ScheduledRequest (input slice, block table, sampling info)."""
        block_ids = self.kv.allocate_slots(request.request_id, num_computed, num_new, hit)
        if block_ids is None:
            return None
        end = num_computed + num_new
        sample = end == request.num_known_tokens
        return ScheduledRequest(
            request_id=request.request_id,
            input_token_ids=request.all_token_ids[num_computed:end],
            num_computed_tokens=num_computed,
            num_scheduled_tokens=num_new,
            block_ids=block_ids,
            requires_sampling=sample,
            sampling_params=request.sampling_params if sample else None,
            sample_index=request.num_output_tokens if sample else None,
        )

    def _preempt_last(self) -> str:
        """[Step 5] Recompute-style preemption of the NEWEST running request:
        pop it, reset its progress, free its blocks, put it at the FRONT of the
        waiting queue. Return its id."""
        victim = self.running.pop()
        victim.preempt()
        self.kv.free_blocks(victim.request_id)
        self.waiting.appendleft(victim)
        return victim.request_id

    # ---- commit -----------------------------------------------------------
    def update(self, plan: SchedulerOutput, result: ModelRunnerOutput) -> None:
        """[Step 4] Commit one executed plan: advance computed tokens, append
        sampled tokens, decide finish reasons (EOS > LENGTH > MODEL_LENGTH),
        free finished requests. Skip requests aborted mid-flight."""
        if plan is not self._pending_plan:
            raise ValueError("plan is not the pending plan")
        self._pending_plan = None
        sampled = {t.request_id: t.token_id for t in result.sampled_tokens}
        finished: set[str] = set()
        for item in plan.scheduled_requests:
            request = self.requests[item.request_id]
            if request.is_finished:
                continue
            request.advance_computed_tokens(item.num_scheduled_tokens)
            token = sampled.get(item.request_id)
            if item.requires_sampling:
                if token is None:
                    raise ValueError(f"executor did not sample {item.request_id}")
                request.append_sampled_token(token)
                p = request.sampling_params
                reason = None
                if p.eos_token_id is not None and token == p.eos_token_id and not p.ignore_eos:
                    reason = FinishReason.EOS
                elif request.num_output_tokens >= p.max_tokens:
                    reason = FinishReason.LENGTH
                elif request.num_known_tokens >= self.config.max_model_len:
                    reason = FinishReason.MODEL_LENGTH
                if reason is not None:
                    request.finish(reason)
                    finished.add(request.request_id)
            if self.config.enable_prefix_cache:  # Step 15
                self.kv.cache_full_blocks(request.request_id, request.all_token_ids, request.num_computed_tokens)
        if finished:
            self.running = [r for r in self.running if r.request_id not in finished]
            for rid in finished:
                self.kv.free(rid)
