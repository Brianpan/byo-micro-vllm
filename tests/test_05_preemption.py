"""Step 5 - Preemption (recompute) when the KV cache runs out, and abort."""

from minivllm.config import EngineConfig
from minivllm.core.kv_cache_manager import KVCacheManager
from minivllm.core.request import FinishReason, Request, RequestStatus
from minivllm.core.scheduler import Scheduler
from minivllm.executor.interface import ModelRunnerOutput, SampledToken
from minivllm.sampling.params import SamplingParams


def build(num_blocks, block_size=4, **cfg):
    config = EngineConfig(block_size=block_size, num_blocks=num_blocks, **cfg)
    return Scheduler(config, KVCacheManager(num_blocks, block_size))


def req(rid, n_prompt, max_tokens=8):
    return Request(rid, list(range(1, n_prompt + 1)), SamplingParams(max_tokens=max_tokens))


def run_step(s, token=7):
    plan = s.schedule()
    s.update(plan, ModelRunnerOutput(tuple(
        SampledToken(r.request_id, token) for r in plan.scheduled_requests if r.requires_sampling)))
    return plan


def test_newest_request_is_preempted():
    s = build(num_blocks=3)                 # 2 usable blocks, 4 tokens each
    s.add_request(req("old", 4))
    s.add_request(req("new", 4))
    p = run_step(s)                         # both prefill: cache now full
    assert len(p.scheduled_requests) == 2 and s.kv.num_free_blocks == 0

    p = run_step(s)                         # "old" needs a 2nd block for token 5
    assert p.preempted_request_ids == ("new",)
    assert [r.request_id for r in p.scheduled_requests] == ["old"]
    new = s.get_request("new")
    assert new.status is RequestStatus.WAITING
    assert new.num_computed_tokens == 0 and new.output_token_ids == [7]
    assert list(s.waiting)[0] is new        # preempted requests go to the FRONT


def test_preempted_request_recomputes_prompt_and_outputs():
    s = build(num_blocks=3)
    s.add_request(req("old", 4, max_tokens=2))
    s.add_request(req("new", 4))
    run_step(s)
    run_step(s)                              # preempts "new"; "old" finishes (2 tokens)
    assert s.get_request("old").is_finished
    p = run_step(s)                          # "new" resumes: recompute 4 prompt + 1 output
    (n,) = p.scheduled_requests
    assert n.request_id == "new"
    assert n.input_token_ids == (1, 2, 3, 4, 7)
    assert (n.num_computed_tokens, n.num_scheduled_tokens, n.requires_sampling) == (0, 5, True)
    assert n.sample_index == 1               # it already owns one output token


def test_preemption_keeps_everyone_progressing():
    s = build(num_blocks=5, max_num_batched_tokens=64)     # 4 usable blocks
    for i in range(3):
        s.add_request(req(f"r{i}", 3, max_tokens=6))
    steps = 0
    while s.has_unfinished_requests():
        run_step(s)
        steps += 1
        assert steps < 200, "livelock"
    for i in range(3):
        r = s.get_request(f"r{i}")
        assert r.finish_reason is FinishReason.LENGTH and len(r.output_token_ids) == 6
    assert s.kv.num_free_blocks == 4


def test_abort_waiting_and_running():
    s = build(num_blocks=16, max_num_seqs=1)
    s.add_request(req("run", 4))
    s.add_request(req("wait", 4))
    run_step(s)
    free_before = s.kv.num_free_blocks
    assert s.abort_request("run") is True
    assert s.kv.num_free_blocks == free_before + 1
    assert s.abort_request("wait") is True
    assert s.abort_request("wait") is False       # already finished
    assert s.abort_request("missing") is False
    assert s.get_request("run").finish_reason is FinishReason.ABORTED
    assert not s.has_unfinished_requests()
