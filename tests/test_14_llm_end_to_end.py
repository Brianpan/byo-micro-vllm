"""Step 14 - LLM facade: everything wired together, checked against HF generate()."""

import pytest
import torch

from minivllm.config import EngineConfig
from minivllm.core.request import FinishReason
from minivllm.llm import LLM
from minivllm.sampling.params import SamplingParams

PROMPTS = [
    [11, 22, 33],
    list(range(40, 77)),            # long: spans many blocks, gets chunked
    [5],
    [400, 300, 200, 100, 50, 25],
    list(range(100, 119)),
]
N = 12


@pytest.fixture
def expected(hf_greedy):
    return [hf_greedy(p, N) for p in PROMPTS]


def make(tiny_gpt2, device, **kw):
    kw.setdefault("block_size", 4)
    return LLM(EngineConfig(model=tiny_gpt2[0], device=device, **kw))


@pytest.mark.parametrize(
    "kw",
    [
        dict(num_blocks=128),                                     # plain continuous batching
        dict(num_blocks=128, max_num_batched_tokens=8),           # chunked prefill everywhere
        dict(num_blocks=128, max_num_seqs=2),                     # queueing
        dict(num_blocks=16, max_num_batched_tokens=32),           # cache pressure -> preemption
    ],
    ids=["batching", "chunked", "queueing", "preemption"],
)
def test_greedy_matches_hf(tiny_gpt2, device, expected, kw):
    llm = make(tiny_gpt2, device, **kw)
    outs = llm.generate(PROMPTS, SamplingParams(max_tokens=N))
    llm.shutdown()
    assert [list(o.output_token_ids) for o in outs] == expected
    assert all(o.finish_reason is FinishReason.LENGTH for o in outs)
    assert [list(o.prompt_token_ids) for o in outs] == PROMPTS          # prompt order preserved


def test_preemption_really_happened(tiny_gpt2, device):
    llm = make(tiny_gpt2, device, num_blocks=16, max_num_batched_tokens=32)
    for i, p in enumerate(PROMPTS):
        llm.engine.add_request(f"x{i}", p, SamplingParams(max_tokens=N))
    preempted = 0
    while llm.engine.has_unfinished_requests():
        plan = llm.scheduler.schedule()
        preempted += len(plan.preempted_request_ids)
        llm.scheduler.update(plan, llm.executor.execute(plan))
    llm.shutdown()
    assert preempted > 0


def test_eos_stops_early(tiny_gpt2, device, expected):
    eos = expected[0][3]
    llm = make(tiny_gpt2, device, num_blocks=64)
    (out,) = llm.generate([PROMPTS[0]], SamplingParams(max_tokens=N, eos_token_id=eos))
    llm.shutdown()
    first = expected[0].index(eos)
    assert list(out.output_token_ids) == expected[0][: first + 1]
    assert out.finish_reason is FinishReason.EOS


def test_seeded_sampling_is_batch_invariant(tiny_gpt2, device):
    p = SamplingParams(max_tokens=8, temperature=0.8, top_p=0.9, seed=42)
    llm = make(tiny_gpt2, device, num_blocks=128)
    alone = llm.generate([PROMPTS[3]], p)[0].output_token_ids
    batched = llm.generate(PROMPTS, p)[3].output_token_ids
    llm.shutdown()
    assert alone == batched


def test_multi_process_executor(tiny_gpt2, device, expected):
    llm = LLM(EngineConfig(model=tiny_gpt2[0], device=device, block_size=4, num_blocks=64), executor="multi")
    try:
        outs = llm.generate(PROMPTS[:2], SamplingParams(max_tokens=N))
    finally:
        llm.shutdown()
    assert [list(o.output_token_ids) for o in outs] == expected[:2]


def _real_gpt2_available():
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("gpt2")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _real_gpt2_available(), reason="needs network access to the 'gpt2' checkpoint")
def test_real_gpt2_text(device):
    from transformers import AutoTokenizer, GPT2LMHeadModel

    prompts = ["The capital of France is", "def fibonacci(n):", "Once upon a time"]
    llm = LLM(EngineConfig(model="gpt2", device=device, num_blocks=None if device == "cuda" else 256))
    outs = llm.generate(prompts, SamplingParams(max_tokens=20))
    llm.shutdown()
    tok = AutoTokenizer.from_pretrained("gpt2")
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    for prompt, out in zip(prompts, outs):
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            ref = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=20,
                              do_sample=False, pad_token_id=tok.eos_token_id)
        assert out.output_text == tok.decode(ref[0, ids.shape[1]:]), prompt
