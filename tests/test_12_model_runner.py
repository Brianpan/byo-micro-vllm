"""Step 12 - ModelRunner: KV-cache sizing, SchedulerOutput -> tensors -> tokens."""

import pytest
import torch

from minivllm.config import EngineConfig
from minivllm.core.scheduler_output import ScheduledRequest, SchedulerOutput
from minivllm.model.gpt2 import GPT2Config, kv_cache_bytes_per_block
from minivllm.sampling.params import SamplingParams
from minivllm.worker.model_runner import ModelRunner



def sched(rid, toks, done, new, blocks, sample=True, index=0):
    return ScheduledRequest(
        request_id=rid, input_token_ids=tuple(toks[done: done + new]), num_computed_tokens=done,
        num_scheduled_tokens=new, block_ids=tuple(blocks), requires_sampling=sample,
        sampling_params=SamplingParams(max_tokens=8) if sample else None, sample_index=index if sample else None)


@pytest.fixture
def runner(tiny_gpt2, device):
    cfg = EngineConfig(model=tiny_gpt2[0], device=device, block_size=4, num_blocks=32)
    r = ModelRunner(cfg)
    r.initialize_model()
    r.initialize_kv_cache()
    yield r
    r.shutdown()


def test_bytes_per_block():
    cfg = GPT2Config()  # real gpt2-small: 12 layers, 768 hidden
    assert kv_cache_bytes_per_block(cfg, 16, torch.float16) == 2 * 12 * 16 * 768 * 2  # 589,824 B


def test_kv_cache_allocation(runner):
    c = runner.model.config
    assert runner.num_blocks == 32
    assert tuple(runner.kv_cache.shape) == (c.n_layer, 2, 32, 4, c.n_head, c.head_dim)


def test_default_num_blocks_on_cpu(tiny_gpt2):
    cfg = EngineConfig(model=tiny_gpt2[0], device="cpu", block_size=16, max_num_seqs=3, max_model_len=100)
    r = ModelRunner(cfg)
    r.initialize_model()
    assert r.determine_num_blocks() == 3 * 7 + 1


def test_build_model_input(runner):
    a = list(range(1, 7))     # 6 tokens, fresh prefill, blocks [5, 9]
    b = list(range(1, 12))    # 11 tokens, decoding the 11th, blocks [3, 0, 7]
    plan = SchedulerOutput((sched("a", a, 0, 6, [5, 9]), sched("b", b, 10, 1, [3, 0, 7], index=4)))
    mi = runner.build_model_input(plan)
    assert mi.input_ids.tolist() == a + [11]
    assert mi.positions.tolist() == [0, 1, 2, 3, 4, 5, 10]
    assert mi.metadata.slot_mapping.tolist() == [20, 21, 22, 23, 36, 37, 7 * 4 + 2]
    assert mi.metadata.query_start_loc.tolist() == [0, 6, 7]
    assert mi.metadata.context_lens.tolist() == [6, 11]
    assert mi.metadata.block_tables.tolist() == [[5, 9, 31], [3, 0, 7]]   # padded with null block 31
    assert mi.sample_rows.tolist() == [5, 6]
    assert mi.sample_request_ids == ("a", "b")


def test_only_finished_chunks_are_sampled(runner):
    toks = list(range(1, 11))
    plan = SchedulerOutput((sched("a", toks, 0, 4, [0, 1, 2], sample=False), sched("b", toks, 0, 3, [3])))
    mi = runner.build_model_input(plan)
    assert mi.sample_rows.tolist() == [6] and mi.sample_request_ids == ("b",)
    out = runner.execute(plan)
    assert [t.request_id for t in out.sampled_tokens] == ["b"]


def test_execute_matches_hf_greedy(runner, tiny_gpt2):
    _, hf = tiny_gpt2
    torch.manual_seed(0)
    a = torch.randint(0, 512, (9,)).tolist()
    b = torch.randint(0, 512, (5,)).tolist()
    with torch.no_grad():
        exp_a = int(hf(torch.tensor([a])).logits[0, -1].argmax())
        exp_b = int(hf(torch.tensor([b])).logits[0, -1].argmax())
    # a: two chunks across two calls; b: one shot inside the second call
    runner.execute(SchedulerOutput((sched("a", a, 0, 5, [10, 11, 12], sample=False),)))
    out = runner.execute(SchedulerOutput((sched("a", a, 5, 4, [10, 11, 12]), sched("b", b, 0, 5, [20, 21]))))
    got = {t.request_id: t.token_id for t in out.sampled_tokens}
    assert got == {"a": exp_a, "b": exp_b}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_profiled_num_blocks_on_gpu(tiny_gpt2):
    cfg = EngineConfig(model=tiny_gpt2[0], device="cuda", block_size=16, gpu_memory_utilization=0.5)
    r = ModelRunner(cfg)
    r.initialize_model()
    r.initialize_kv_cache()
    per_block = kv_cache_bytes_per_block(r.model.config, 16, torch.float32)
    _, total = torch.cuda.mem_get_info()
    assert r.num_blocks > 100
    assert r.num_blocks * per_block <= 0.5 * total
    r.shutdown()
