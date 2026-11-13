"""Step 4 - Scheduler: continuous batching, chunked prefill, finish reasons."""

import pytest

from minivllm.config import EngineConfig
from minivllm.core.kv_cache_manager import KVCacheManager
from minivllm.core.request import FinishReason, Request, RequestStatus
from minivllm.core.scheduler import Scheduler
from minivllm.executor.interface import ModelRunnerOutput, SampledToken
from minivllm.sampling.params import SamplingParams


def build(num_blocks=64, block_size=4, **cfg):
    config = EngineConfig(block_size=block_size, num_blocks=num_blocks, **cfg)
    return Scheduler(config, KVCacheManager(num_blocks, block_size))


def req(rid, n_prompt, **kw):
    kw.setdefault("max_tokens", 4)
    return Request(rid, list(range(1, n_prompt + 1)), SamplingParams(**kw))


def run_step(s: Scheduler, token=7):
    """Pretend to be an executor that always samples `token`."""
    plan = s.schedule()
    out = ModelRunnerOutput(
        tuple(SampledToken(r.request_id, token) for r in plan.scheduled_requests if r.requires_sampling)
    )
    s.update(plan, out)
    return plan


def test_single_request_prefill_then_decode():
    s = build()
    s.add_request(req("a", 5, max_tokens=3))
    p = run_step(s)
    (a,) = p.scheduled_requests
    assert a.input_token_ids == (1, 2, 3, 4, 5)
    assert (a.num_computed_tokens, a.num_scheduled_tokens, a.requires_sampling) == (0, 5, True)
    assert a.block_ids == (0, 1)
    assert a.sample_index == 0

    p = run_step(s)
    (a,) = p.scheduled_requests
    assert a.input_token_ids == (7,)                 # decode feeds the token sampled last step
    assert (a.num_computed_tokens, a.num_scheduled_tokens) == (5, 1)
    assert a.sample_index == 1

    run_step(s)
    r = s.get_request("a")
    assert r.is_finished and r.finish_reason is FinishReason.LENGTH
    assert r.output_token_ids == [7, 7, 7]
    assert not s.has_unfinished_requests()
    assert s.kv.num_free_blocks == 63                # finished requests return their blocks


def test_chunked_prefill_under_token_budget():
    s = build(max_num_batched_tokens=4)
    s.add_request(req("a", 10))
    chunks = []
    for _ in range(3):
        (a,) = run_step(s).scheduled_requests
        chunks.append((a.num_computed_tokens, a.num_scheduled_tokens, a.requires_sampling))
    assert chunks == [(0, 4, False), (4, 4, False), (8, 2, True)]
    assert s.get_request("a").output_token_ids == [7]


def test_running_first_then_waiting_share_budget():
    s = build(max_num_batched_tokens=6)
    s.add_request(req("a", 4))
    run_step(s)                                       # a prefilled
    s.add_request(req("b", 10))
    p = run_step(s)
    got = [(r.request_id, r.num_scheduled_tokens) for r in p.scheduled_requests]
    assert got == [("a", 1), ("b", 5)]                # a decodes, b gets the rest of the budget


def test_max_num_seqs():
    s = build(max_num_seqs=2)
    for i in range(3):
        s.add_request(req(f"r{i}", 3))
    p = run_step(s)
    assert [r.request_id for r in p.scheduled_requests] == ["r0", "r1"]
    assert s.get_request("r2").status is RequestStatus.WAITING


def test_fcfs_head_of_line_blocking():
    s = build(num_blocks=5, block_size=4)             # 4 usable blocks
    s.add_request(req("a", 8))                        # 2 blocks
    s.add_request(req("big", 12))                     # 3 blocks -> doesn't fit after a
    s.add_request(req("small", 2))                    # would fit, but must not jump the queue
    p = run_step(s)
    assert [r.request_id for r in p.scheduled_requests] == ["a"]


def test_eos_and_ignore_eos():
    s = build()
    s.add_request(req("stop", 2, eos_token_id=7, max_tokens=10))
    s.add_request(req("go", 2, eos_token_id=7, ignore_eos=True, max_tokens=2))
    run_step(s, token=7)
    assert s.get_request("stop").finish_reason is FinishReason.EOS
    assert not s.get_request("go").is_finished
    run_step(s, token=7)
    assert s.get_request("go").finish_reason is FinishReason.LENGTH


def test_max_model_len():
    s = build(max_model_len=6)
    s.add_request(req("a", 4, max_tokens=100))
    run_step(s)
    run_step(s)
    r = s.get_request("a")
    assert r.finish_reason is FinishReason.MODEL_LENGTH and r.num_known_tokens == 6


def test_add_request_validation():
    s = build(max_model_len=8)
    with pytest.raises(ValueError):
        s.add_request(req("long", 8))
    s.add_request(req("a", 2))
    with pytest.raises(ValueError):
        s.add_request(req("a", 2))


def test_one_plan_in_flight():
    s = build()
    s.add_request(req("a", 2))
    plan = s.schedule()
    with pytest.raises(RuntimeError):
        s.schedule()
    s.update(plan, ModelRunnerOutput((SampledToken("a", 1),)))
    with pytest.raises(ValueError):
        s.update(plan, ModelRunnerOutput((SampledToken("a", 1),)))


def test_empty_plan_when_idle():
    s = build()
    assert s.schedule().scheduled_requests == ()
    assert s.schedule().scheduled_requests == ()  # idle plans are not "in flight"


def test_request_larger_than_cache_raises():
    s = build(num_blocks=3, block_size=4)             # 2 usable blocks = 8 tokens
    s.add_request(req("huge", 9))
    with pytest.raises(RuntimeError):
        s.schedule()
