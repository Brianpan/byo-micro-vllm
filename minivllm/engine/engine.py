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
        raise NotImplementedError("Step 6: implement __init__")

    def add_request(self, request_id: str, prompt_token_ids: list[int], params: SamplingParams) -> None:
        """[Step 6] Wrap in a Request and hand it to the scheduler."""
        raise NotImplementedError("Step 6: implement add_request")

    def step(self) -> tuple[RequestOutput, ...]:
        """[Step 6] schedule -> execute -> update. Return one RequestOutput for
        every request that sampled a token this step (empty tuple if idle)."""
        raise NotImplementedError("Step 6: implement step")

    def has_unfinished_requests(self) -> bool:
        """[Step 6]"""
        raise NotImplementedError("Step 6: implement has_unfinished_requests")

    def abort_request(self, request_id: str) -> bool:
        """[Step 6]"""
        raise NotImplementedError("Step 6: implement abort_request")

    def run_until_done(self) -> dict[str, RequestOutput]:
        """[Step 6] Step until idle; return the final output of each request."""
        raise NotImplementedError("Step 6: implement run_until_done")
