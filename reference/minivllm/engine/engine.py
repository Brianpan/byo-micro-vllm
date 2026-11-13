"""The synchronous control loop (Step 6). Mirrors micro_vllm/engine/engine.py.

    while has_unfinished_requests():
        plan   = scheduler.schedule()     # CPU: decide WHAT to compute
        result = executor.execute(plan)   # GPU: compute it
        scheduler.update(plan, result)    # CPU: commit, finish, free
"""

from dataclasses import dataclass

from minivllm.core.request import FinishReason, Request
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
    def __init__(self, scheduler: Scheduler, executor: Executor) -> None:
        """[Step 6]"""
        self.scheduler = scheduler
        self.executor = executor

    def add_request(self, request_id: str, prompt_token_ids: list[int], params: SamplingParams) -> None:
        """[Step 6] Wrap in a Request and hand it to the scheduler."""
        self.scheduler.add_request(Request(request_id, list(prompt_token_ids), params))

    def step(self) -> tuple[RequestOutput, ...]:
        """[Step 6] schedule -> execute -> update. Return one RequestOutput for
        every request that sampled a token this step (empty tuple if idle)."""
        plan = self.scheduler.schedule()
        if not plan.scheduled_requests:
            return ()
        result = self.executor.execute(plan)
        self.scheduler.update(plan, result)
        outputs = []
        for item in plan.scheduled_requests:
            if not item.requires_sampling:
                continue
            r = self.scheduler.get_request(item.request_id)
            if r.finish_reason is FinishReason.ABORTED:
                continue
            outputs.append(RequestOutput(r.request_id, tuple(r.output_token_ids), r.is_finished, r.finish_reason))
        return tuple(outputs)

    def has_unfinished_requests(self) -> bool:
        """[Step 6]"""
        return self.scheduler.has_unfinished_requests()

    def abort_request(self, request_id: str) -> bool:
        """[Step 6]"""
        return self.scheduler.abort_request(request_id)

    def run_until_done(self) -> dict[str, RequestOutput]:
        """[Step 6] Step until idle; return the final output of each request."""
        final: dict[str, RequestOutput] = {}
        while self.has_unfinished_requests():
            for out in self.step():
                if out.finished:
                    final[out.request_id] = out
        return final
