"""Executor contract consumed by the Engine (Step 6). Mirrors micro_vllm/executor/interface.py."""

from dataclasses import dataclass
from typing import Protocol

from minivllm.core.scheduler_output import SchedulerOutput


@dataclass(frozen=True, slots=True)
class SampledToken:
    request_id: str
    token_id: int


@dataclass(frozen=True, slots=True)
class ModelRunnerOutput:
    sampled_tokens: tuple[SampledToken, ...] = ()


class Executor(Protocol):
    """Anything that turns a plan into sampled tokens: a fake, one GPU in-process,
    one GPU in another process, N GPUs with tensor parallelism..."""

    num_blocks: int  # physical KV pages the scheduler may hand out

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput: ...

    def shutdown(self) -> None: ...
