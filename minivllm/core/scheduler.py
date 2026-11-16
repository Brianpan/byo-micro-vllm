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
    engine_config: EngineConfig
    kv: KVCacheManager
    waiting_queue: deque[Request] = deque()
    request_map: dict[str, Request] = {}
    in_flight: bool = False
    
    def __init__(self, config: EngineConfig, kv: KVCacheManager) -> None:
        """[Step 4]"""
        self.engine_config = config
        self.kv = kv
        self.waiting_queue = deque()
        self.running_queue = deque()
        self.request_map = {}
        self.in_flight = False

    # ---- request lifecycle ------------------------------------------------
    def add_request(self, request: Request) -> None:
        """[Step 4] Validate (WAITING, prompt < max_model_len, unique id), register
        with the KV manager and append to the waiting queue."""
        if request.status != RequestStatus.WAITING:
            raise ValueError(f"Request must be in WAITING status, got {request.status}")
        if any(r.request_id == request.request_id for r in self.waiting_queue):
            raise ValueError(f"Request with id {request.request_id} already exists")
        # check prompt length
        if len(request.prompt_token_ids) >= self.engine_config.max_model_len:
            raise ValueError(f"Request prompt exceeds max model length {self.engine_config.max_model_len}")

        self.kv.register_request(request.request_id)
        self.waiting_queue.append(request)
        self.request_map[request.request_id] = request

    def get_request(self, request_id: str) -> Request | None:
        """[Step 4]"""
        return self.request_map.get(request_id)

    def has_unfinished_requests(self) -> bool:
        """[Step 4]"""
        return len(self.waiting_queue) > 0 or len(self.running_queue) > 0

    def abort_request(self, request_id: str) -> bool:
        """[Step 5] Finish with ABORTED, drop from queues, free all blocks.
        Return False if unknown or already finished."""
        request = self.request_map.get(request_id)
        if request == None or request.status != RequestStatus.WAITING:
            return False
        del self.request_map[request_id]
        self.waiting_queue = deque(r for r in self.waiting_queue if r.request_id != request_id)
        self.kv.free_blocks(request_id)

        return True

    # ---- planning ---------------------------------------------------------
    def schedule(self) -> SchedulerOutput:
        """[Step 4] Build one plan (see module docstring). Only one plan may be
        in flight: calling schedule() again before update() raises RuntimeError.
        Raise RuntimeError if the very first request of a plan cannot fit in the
        whole KV cache (it never could)."""
        if not self.waiting_queue and not self.running_queue:
            return SchedulerOutput(scheduled_requests=tuple([]))
        if self.in_flight:
            raise RuntimeError("A plan is already in flight")
        self.in_flight = True
        # check availble from config 
        max_num_batched_tokens = self.engine_config.max_num_batched_tokens
        max_seq_len = self.engine_config.max_num_seqs

        curr_model_len = 0
        curr_seq_len = 0
        scheduled_requests: list[ScheduledRequest] = []
        # running queue is considered for scheduling as well
    
        rq_idx = 0
        while rq_idx < len(self.running_queue) and curr_model_len < max_num_batched_tokens and curr_seq_len < max_seq_len:
            request = self.running_queue[rq_idx]

            num_new = min(request.num_pending_tokens, max_num_batched_tokens - curr_model_len)
            curr_model_len += num_new
            curr_seq_len += 1
            scheduled_request = self._try_allocate(request, num_computed=request.num_computed_tokens, num_new=num_new)
            if scheduled_request is not None:
                scheduled_requests.append(scheduled_request)
            else:
                break
            rq_idx += 1
    
        # no queue jumping: always take the first request in the waiting queue
        while len(self.waiting_queue) > 0 and curr_model_len < max_num_batched_tokens and curr_seq_len < max_seq_len:
            request = self.waiting_queue[0]

            num_new = min(request.num_pending_tokens, max_num_batched_tokens - curr_model_len)
            curr_model_len += num_new
            curr_seq_len += 1

            # if we can fit into kv cache
            if request.num_pending_tokens >= (self.kv.block_pool.num_blocks - 1) * self.kv.block_size:
                raise RuntimeError("Request too large to fit in KV cache")

            scheduled_request = self._try_allocate(request, num_computed=request.num_computed_tokens, num_new=num_new)
            if scheduled_request is not None:
                scheduled_requests.append(scheduled_request)
            else:
                break
            
            self.waiting_queue.popleft()
            self.running_queue.append(request)
            request.status = RequestStatus.RUNNING

        return SchedulerOutput(scheduled_requests=tuple(scheduled_requests))
        

    def _try_allocate(
        self, request: Request, num_computed: int, num_new: int, hit: PrefixHit | None = None
    ) -> ScheduledRequest | None:
        """[Step 4] Ask the KV manager for slots; on success build the immutable
        ScheduledRequest (input slice, block table, sampling info)."""
        block_ids = self.kv.allocate_slots(request.request_id, num_computed_tokens=num_computed, num_new_tokens=num_new, prefix_hit=hit)
        if block_ids is None:
            return None

        end = num_computed + num_new
        sampling = end == request.num_known_tokens
        # prefill phase: input_token_ids is the prompt_token_ids
        input_token_ids = tuple(request.prompt_token_ids)
        if sampling and len(request.output_token_ids) > 0:
            input_token_ids = tuple([request.output_token_ids[-1]])
        # ask for physical blocks from the KV cache manager
        return ScheduledRequest(
            request_id=request.request_id,
            input_token_ids=input_token_ids,
            num_computed_tokens=num_computed,
            num_scheduled_tokens=num_new,
            block_ids=block_ids,
            requires_sampling=sampling,
            sampling_params=request.sampling_params,
            sample_index=len(request.output_token_ids)
        )

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

        if not self.in_flight:
            raise ValueError("No plan in flight to update")

        sample_token_map = {st.request_id: st for st in result.sampled_tokens}

        for sc in plan.scheduled_requests:
            request_id = sc.request_id
            request = self.request_map[request_id]
            if request is None:
                raise ValueError(f"Request {request_id} not found in request_map")
            # skip aborted requests
            if request.finish_reason == FinishReason.ABORTED:
                continue
            
            # we need to consider chunks of the prefill
            # update the number of computed tokens
            request.num_computed_tokens += sc.num_scheduled_tokens
            # if is decode step, put the token into the output token list
            if sc.requires_sampling:
                sample_token = sample_token_map.get(request_id, None)
                if sample_token is not None:
                    token_id = sample_token.token_id
                    request.output_token_ids.append(token_id)
            
                    # check ending conditions
                    if not request.sampling_params.ignore_eos and token_id == request.sampling_params.eos_token_id:
                        request.finish(FinishReason.EOS)
                    elif request.num_output_tokens >= request.sampling_params.max_tokens:
                        request.finish(FinishReason.LENGTH)
                    elif request.num_known_tokens >= self.engine_config.max_model_len:
                        request.finish(FinishReason.MODEL_LENGTH)

            # free the blocks associated with this request if it has finished
            if request.finish_reason is not None:
                self.kv.free_blocks(request_id)
                request.status = RequestStatus.FINISHED
                # remove from the running queue and waiting queue if present
                if request in self.running_queue:
                    self.running_queue.remove(request)
                if request in self.waiting_queue:
                    self.waiting_queue.remove(request)

        self.in_flight = False