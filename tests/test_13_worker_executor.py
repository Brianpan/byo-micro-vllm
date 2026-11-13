"""Step 13 - Worker and executors (in-process and out-of-process)."""

import pytest
import torch

from minivllm.config import EngineConfig
from minivllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from minivllm.executor.interface import Executor
from minivllm.executor.multi_process import MultiProcessExecutor
from minivllm.executor.single_process import SingleProcessExecutor
from minivllm.sampling.params import SamplingParams
from minivllm.worker.worker import Worker


@pytest.fixture
def config(tiny_gpt2, device):
    return EngineConfig(model=tiny_gpt2[0], device=device, block_size=4, num_blocks=16)


def decode_loop(executor: Executor, prompt, n):
    """Drive an executor by hand (we are the scheduler here)."""
    toks = list(prompt)
    blocks = tuple(range(8))
    done = 0
    for i in range(n):
        new = len(toks) - done
        plan = SchedulerOutput((ScheduledRequest(
            "r", tuple(toks[done:]), done, new, blocks, True, SamplingParams(max_tokens=n), i),))
        (tok,) = executor.execute(plan).sampled_tokens
        done = len(toks)
        toks.append(tok.token_id)
    return toks[len(prompt):]


def test_worker_initializes_in_order(config):
    w = Worker(config)
    assert w.num_blocks == 16
    assert w.runner.model is not None and w.runner.kv_cache is not None
    w.shutdown()
    w.shutdown()                       # idempotent
    with pytest.raises(RuntimeError):
        w.execute(SchedulerOutput(()))


def test_single_process_matches_hf(config, hf_greedy):
    ex = SingleProcessExecutor(Worker(config))
    assert ex.num_blocks == 16
    assert decode_loop(ex, [5, 6, 7, 8, 9], 10) == hf_greedy([5, 6, 7, 8, 9], 10)
    ex.shutdown()


def test_multi_process_matches_single(config):
    single = SingleProcessExecutor(Worker(config))
    multi = MultiProcessExecutor(config)
    try:
        assert multi.num_blocks == single.num_blocks
        assert decode_loop(multi, [1, 2, 3], 8) == decode_loop(single, [1, 2, 3], 8)
    finally:
        multi.shutdown()
        multi.shutdown()               # idempotent
        single.shutdown()
    assert not multi._proc.is_alive()


def test_multi_process_reports_worker_errors(tiny_gpt2):
    bad = EngineConfig(model=tiny_gpt2[0] + "-does-not-exist", device="cpu", num_blocks=8)
    with pytest.raises(RuntimeError, match="worker process failed"):
        MultiProcessExecutor(bad)


def test_plans_are_picklable():
    import pickle

    plan = SchedulerOutput((ScheduledRequest("r", (1, 2), 0, 2, (0,), True, SamplingParams(), 0),))
    assert pickle.loads(pickle.dumps(plan)) == plan


def test_executor_owns_no_scheduler_state(config):
    """Running the same plan twice on a fresh worker is deterministic: the worker
    holds tensors, never request bookkeeping."""
    plan = SchedulerOutput((ScheduledRequest("r", (4, 5, 6), 0, 3, (2,), True, SamplingParams(), 0),))
    w1, w2 = Worker(config), Worker(config)
    assert w1.execute(plan) == w2.execute(plan)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
