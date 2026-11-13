"""Step 6 - Engine control loop against a fake executor (no GPU, no model yet).

FakeExecutor behaves like a real model with a KV cache: it remembers the tokens
it has "computed" per request (its fake KV cache), and the sampled token is a
deterministic function of the ENTIRE visible sequence. If the scheduler ever
feeds wrong positions/chunks, or forgets a preemption, outputs change.
"""

import pytest

from minivllm.config import EngineConfig
from minivllm.core.kv_cache_manager import KVCacheManager
from minivllm.core.request import FinishReason
from minivllm.core.scheduler import Scheduler
from minivllm.engine.engine import Engine
from minivllm.executor.interface import ModelRunnerOutput, SampledToken
from minivllm.sampling.params import SamplingParams

VOCAB = 97


def next_token(seq):
    return (sum((i + 1) * t for i, t in enumerate(seq)) + len(seq)) % VOCAB


def reference(prompt, n):
    seq = list(prompt)
    for _ in range(n):
        seq.append(next_token(seq))
    return seq[len(prompt):]


class FakeExecutor:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.cache: dict[str, list[int]] = {}
        self.calls = 0

    def execute(self, plan):
        self.calls += 1
        out = []
        for r in plan.scheduled_requests:
            seq = self.cache.setdefault(r.request_id, [])
            del seq[r.num_computed_tokens:]          # preempted -> num_computed restarts at 0
            assert len(seq) == r.num_computed_tokens, "gap in computed tokens"
            seq.extend(r.input_token_ids)
            if r.requires_sampling:
                out.append(SampledToken(r.request_id, next_token(seq)))
        return ModelRunnerOutput(tuple(out))

    def shutdown(self):
        pass


def build(num_blocks=64, block_size=4, **cfg):
    config = EngineConfig(block_size=block_size, num_blocks=num_blocks, **cfg)
    executor = FakeExecutor(num_blocks)
    return Engine(Scheduler(config, KVCacheManager(num_blocks, block_size)), executor), executor


PROMPTS = [[3, 1, 4, 1, 5], [9, 2, 6], list(range(10, 31)), [8], [2, 7, 1, 8, 2, 8, 1, 8, 2, 8]]


def test_step_outputs_are_cumulative():
    engine, _ = build()
    engine.add_request("a", [1, 2, 3], SamplingParams(max_tokens=3))
    o1 = engine.step()
    o2 = engine.step()
    assert len(o1[0].output_token_ids) == 1 and len(o2[0].output_token_ids) == 2
    assert o2[0].output_token_ids[:1] == o1[0].output_token_ids
    assert not o2[0].finished
    (o3,) = engine.step()
    assert o3.finished and o3.finish_reason is FinishReason.LENGTH
    assert engine.step() == ()


def test_chunked_steps_emit_nothing_until_sampling():
    engine, _ = build(max_num_batched_tokens=4)
    engine.add_request("a", list(range(1, 11)), SamplingParams(max_tokens=1))
    assert engine.step() == ()                 # tokens 0..3: nothing to sample yet
    assert engine.step() == ()                 # tokens 4..7
    (out,) = engine.step()                     # tokens 8..9 reach the end -> sample
    assert out.finished and len(out.output_token_ids) == 1


@pytest.mark.parametrize(
    "cfg",
    [
        dict(),                                                # roomy
        dict(max_num_batched_tokens=7),                        # chunked prefill
        dict(max_num_seqs=2),                                  # queueing
        dict(num_blocks=12, max_num_batched_tokens=16),        # heavy preemption
    ],
)
def test_outputs_independent_of_scheduling(cfg):
    engine, ex = build(**cfg)
    for i, p in enumerate(PROMPTS):
        engine.add_request(f"r{i}", p, SamplingParams(max_tokens=12))
    final = engine.run_until_done()
    for i, p in enumerate(PROMPTS):
        assert list(final[f"r{i}"].output_token_ids) == reference(p, 12), cfg
    assert not engine.has_unfinished_requests()


def test_continuous_batching_admits_mid_flight():
    engine, ex = build()
    engine.add_request("a", [1, 2], SamplingParams(max_tokens=5))
    engine.step()
    engine.add_request("b", [3, 4], SamplingParams(max_tokens=2))  # joins a running batch
    final = engine.run_until_done()
    assert list(final["a"].output_token_ids) == reference([1, 2], 5)
    assert list(final["b"].output_token_ids) == reference([3, 4], 2)
    assert ex.calls == 5                                         # b never waited for a to finish


def test_abort_through_engine():
    engine, _ = build()
    engine.add_request("a", [1, 2], SamplingParams(max_tokens=50))
    engine.step()
    assert engine.abort_request("a")
    assert not engine.has_unfinished_requests()
