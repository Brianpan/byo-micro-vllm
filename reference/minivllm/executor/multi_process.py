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
    try:
        worker = Worker(config)
        conn.send(("ready", worker.num_blocks))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return
    while True:
        cmd, payload = conn.recv()
        if cmd == "shutdown":
            worker.shutdown()
            conn.send(("ok", None))
            return
        try:
            conn.send(("ok", worker.execute(payload)))
        except Exception:
            conn.send(("error", traceback.format_exc()))


class MultiProcessExecutor:
    def __init__(self, config: EngineConfig) -> None:
        """[Step 13] Start a 'spawn' child (CUDA cannot be forked) running
        _worker_main and wait for its ready message."""
        ctx = mp.get_context("spawn")
        self._conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(target=_worker_main, args=(config, child_conn), daemon=True)
        self._proc.start()
        self.num_blocks = self._recv()
        self._closed = False

    def _recv(self):
        """[Step 13] Receive one message; raise RuntimeError on ("error", tb)."""
        status, payload = self._conn.recv()
        if status == "error":
            raise RuntimeError(f"worker process failed:\n{payload}")
        return payload

    def execute(self, plan: SchedulerOutput, /) -> ModelRunnerOutput:
        """[Step 13]"""
        self._conn.send(("execute", plan))
        return self._recv()

    def shutdown(self) -> None:
        """[Step 13] Idempotent: tell the child to stop and join it."""
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.send(("shutdown", None))
            self._recv()
        finally:
            self._proc.join(timeout=30)
