"""Device-process orchestration around one ModelRunner (Step 13).

Mirrors micro_vllm/worker/worker.py. The Worker is the thing that lives on a
GPU (in vLLM: one Worker process per GPU). It builds the runner in the right
order: model first, then measure leftover memory, then allocate KV cache.
"""

from minivllm.config import EngineConfig
from minivllm.core.scheduler_output import SchedulerOutput
from minivllm.executor.interface import ModelRunnerOutput
from minivllm.worker.model_runner import ModelRunner


class Worker:
    def __init__(self, config: EngineConfig) -> None:
        """[Step 13] Create the runner, initialize_model(), then initialize_kv_cache()."""
        raise NotImplementedError("Step 13: implement __init__")

    @property
    def num_blocks(self) -> int:
        """[Step 13] Physical KV pages created by the runner."""
        raise NotImplementedError("Step 13: implement num_blocks")

    def execute(self, plan: SchedulerOutput) -> ModelRunnerOutput:
        """[Step 13]"""
        raise NotImplementedError("Step 13: implement execute")

    def shutdown(self) -> None:
        """[Step 13] Idempotent."""
        raise NotImplementedError("Step 13: implement shutdown")
