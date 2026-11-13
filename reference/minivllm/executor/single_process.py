"""SingleProcessExecutor: the Worker lives in this process (Step 13).
Mirrors micro_vllm/executor/single_process.py."""

from minivllm.core.scheduler_output import SchedulerOutput
from minivllm.executor.interface import ModelRunnerOutput
from minivllm.worker.worker import Worker


class SingleProcessExecutor:
    def __init__(self, worker: Worker) -> None:
        """[Step 13]"""
        self.worker = worker

    @property
    def num_blocks(self) -> int:
        """[Step 13]"""
        return self.worker.num_blocks

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        """[Step 13] Forward unchanged."""
        return self.worker.execute(plan)

    def shutdown(self) -> None:
        """[Step 13]"""
        self.worker.shutdown()
