"""MultiProcessExecutor: the Worker lives in a child process (Step 13).

Plans and results are pickled over a Pipe. Same Executor contract as the
single-process version, so the Engine cannot tell the difference. This is the
seam vLLM uses for one process per GPU (and, with N workers + collectives,
tensor parallelism).
"""

import multiprocessing as mp
import traceback

from minivllm.config import EngineConfig
from minivllm.core.scheduler_output import SchedulerOutput
from minivllm.executor.interface import ModelRunnerOutput
from minivllm.worker.worker import Worker


def _worker_main(config: EngineConfig, conn) -> None:
    """[Step 13] Child-process loop. Build a Worker, send ("ready", num_blocks),
    then answer ("execute", plan) with ("ok", result) until ("shutdown", None).
    Any exception is sent back as ("error", traceback_string)."""
    raise NotImplementedError("Step 13: implement _worker_main")


class MultiProcessExecutor:
    def __init__(self, config: EngineConfig) -> None:
        """[Step 13] Start a 'spawn' child (CUDA cannot be forked) running
        _worker_main and wait for its ready message."""
        raise NotImplementedError("Step 13: implement __init__")

    def _recv(self):
        """[Step 13] Receive one message; raise RuntimeError on ("error", tb)."""
        raise NotImplementedError("Step 13: implement _recv")

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        """[Step 13]"""
        raise NotImplementedError("Step 13: implement execute")

    def shutdown(self) -> None:
        """[Step 13] Idempotent: tell the child to stop and join it."""
        raise NotImplementedError("Step 13: implement shutdown")
