"""Step 11 - Sampler: greedy, temperature, top-k, top-p, per-request seeds."""

import pytest
import torch

from minivllm.sampling.params import SamplingParams
from minivllm.sampling.sampler import Sampler, apply_top_k, apply_top_p

NEG = float("-inf")


def test_top_k():
    x = torch.tensor([1.0, 5.0, 3.0, 4.0])
    assert apply_top_k(x, 2).tolist() == [NEG, 5.0, NEG, 4.0]
    assert torch.equal(apply_top_k(x, None), x)
    assert torch.equal(apply_top_k(x, 10), x)


def test_top_p():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    x = probs.log()
    kept = lambda p: (apply_top_p(x, p) > NEG).tolist()
    assert kept(0.45) == [True, False, False, False]
    assert kept(0.7) == [True, True, False, False]
    assert kept(0.85) == [True, True, True, False]  # 0.5+0.3 < 0.85 -> need the third
    assert kept(1.0) == [True, True, True, True]
    assert kept(0.01) == [True, False, False, False]  # top-1 always survives


def test_greedy_batch():
    logits = torch.tensor([[0.1, 2.0, 0.3], [5.0, 1.0, 1.0]])
    out = Sampler()(logits, ("a", "b"), (SamplingParams(), SamplingParams()), (0, 0))
    assert [(t.request_id, t.token_id) for t in out.sampled_tokens] == [("a", 1), ("b", 0)]


def test_top_k_1_is_greedy():
    torch.manual_seed(0)
    logits = torch.randn(50)
    p = SamplingParams(temperature=2.0, top_k=1)
    assert all(Sampler().sample_row(logits, p, i) == int(logits.argmax()) for i in range(20))


def test_seed_reproducible_and_index_dependent():
    logits = torch.zeros(1000)
    p = SamplingParams(temperature=1.0, seed=1234)
    s = Sampler()
    first = [s.sample_row(logits, p, i) for i in range(10)]
    again = [s.sample_row(logits, p, i) for i in range(10)]
    assert first == again                       # same (seed, index) -> same token
    assert len(set(first)) > 1                  # different index -> different draws


def test_seeded_sampling_independent_of_batch_position():
    torch.manual_seed(0)
    logits = torch.randn(3, 100)
    params = (SamplingParams(temperature=1.0, seed=7),) * 3
    s = Sampler()
    batched = s(logits, ("a", "b", "c"), params, (4, 4, 4))
    alone = s(logits[1:2], ("b",), params[:1], (4,))
    assert batched.sampled_tokens[1].token_id == alone.sampled_tokens[0].token_id


def test_temperature_sampling_matches_softmax():
    torch.manual_seed(0)
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    p = SamplingParams(temperature=0.7, seed=3)
    s = Sampler()
    n = 4000
    counts = torch.bincount(torch.tensor([s.sample_row(logits, p, i) for i in range(n)]), minlength=4)
    expected = torch.softmax(logits / 0.7, -1)
    assert torch.allclose(counts / n, expected, atol=0.03)


@pytest.mark.parametrize("kw,allowed", [(dict(top_k=2), {0, 1}), (dict(top_p=0.6), {0, 1})])
def test_filters_restrict_support(kw, allowed):
    logits = torch.tensor([3.0, 2.5, 0.0, -1.0, -2.0])
    p = SamplingParams(temperature=1.0, seed=0, **kw)
    s = Sampler()
    assert {s.sample_row(logits, p, i) for i in range(300)} <= allowed
