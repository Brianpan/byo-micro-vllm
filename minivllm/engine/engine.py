"""The synchronous control loop (Step 6). Mirrors micro_vllm/engine/engine.py.

    while has_unfinished_requests():
        plan   = scheduler.schedule()     # CPU: decide WHAT to compute
        result = executor.execute(plan)   # GPU: compute it
        scheduler.update(plan, result)    # CPU: commit, finish, free
"""

from dataclasses import dataclass

from minivllm.core.request import FinishReason, Request, RequestStatus
from minivllm.core.scheduler import Scheduler
from minivllm.executor.interface import Executor
from minivllm.sampling.params import SamplingParams


@dataclass(frozen=True, slots=True)
class RequestOutput:
    request_id: str
    output_token_ids: tuple[int, ...]  # cumulative, not just the new token
    finished: bool
    finish_reason: FinishReason | None = None


class Engine:
    scheduler: Scheduler
    executor: Executor

    def __init__(self, scheduler: Scheduler, executor: Executor) -> None:
        """[Step 6]"""
        self.scheduler = scheduler
        self.executor = executor

    def add_request(self, request_id: str, prompt_token_ids: list[int], params: SamplingParams) -> None:
        """[Step 6] Wrap in a Request and hand it to the scheduler."""
        request = Request(request_id=request_id, prompt_token_ids=prompt_token_ids, sampling_params=params)

        self.scheduler.add_request(request)

    def step(self) -> tuple[RequestOutput, ...]:
        """[Step 6] schedule -> execute -> update. Return one RequestOutput for
        every request that sampled a token this step (empty tuple if idle)."""
        if not self.has_unfinished_requests():
            return tuple()

        plan = self.scheduler.schedule()
        result = self.executor.execute(plan)
        self.scheduler.update(plan, result)

        return tuple(
            RequestOutput(
                request_id=st.request_id,
                # Make a copy to avoid accidental mutation of the original list.
                output_token_ids=req.output_token_ids.copy(),     
                finished=req.status == RequestStatus.FINISHED,
                finish_reason=req.finish_reason,
            ) 
            for st in result.sampled_tokens
            for req in (self.scheduler.get_request(st.request_id),)
        )

    def has_unfinished_requests(self) -> bool:
        """[Step 6]"""
        return self.scheduler.has_unfinished_requests()

    def abort_request(self, request_id: str) -> bool:
        """[Step 6]"""
        return self.scheduler.abort_request(request_id)

    def run_until_done(self) -> dict[str, RequestOutput]:
        """[Step 6] Step until idle; return the final output of each request."""
        result: dict[str, RequestOutput] = {}
        while self.has_unfinished_requests():
            for output in self.step():
                result[output.request_id] = output
        return result
