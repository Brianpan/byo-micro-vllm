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
        self.config = config
        self.runner: ModelRunner | None = ModelRunner(config)
        self.runner.initialize_model()
        self.runner.initialize_kv_cache()

    @property
    def num_blocks(self) -> int:
        """[Step 13] Physical KV pages created by the runner."""
        if self.runner is None:
            raise RuntimeError("worker is shut down")
        return self.runner.num_blocks

    def execute(self, plan: SchedulerOutput) -> ModelRunnerOutput:
        """[Step 13]"""
        if self.runner is None:
            raise RuntimeError("worker is shut down")
        return self.runner.execute(plan)

    def shutdown(self) -> None:
        """[Step 13] Idempotent."""
        if self.runner is not None:
            self.runner.shutdown()
            self.runner = None
