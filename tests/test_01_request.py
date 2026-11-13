"""Step 1 - Request state machine and SamplingParams."""

import dataclasses

import pytest

from minivllm.core.request import FinishReason, Request, RequestStatus
from minivllm.sampling.params import SamplingParams


def make(prompt=(1, 2, 3), **kw):
    return Request("r0", list(prompt), SamplingParams(**kw))


class TestSamplingParams:
    def test_defaults_are_greedy(self):
        p = SamplingParams()
        assert p.is_greedy and p.max_tokens == 16

    @pytest.mark.parametrize(
        "kw",
        [
            dict(max_tokens=0),
            dict(temperature=-0.1),
            dict(temperature=float("nan")),
            dict(temperature=1.0, top_k=0),
            dict(temperature=1.0, top_p=0.0),
            dict(temperature=1.0, top_p=1.5),
            dict(seed=-1),
            dict(top_k=5),        # top_k with greedy
            dict(top_p=0.9),      # top_p with greedy
        ],
    )
    def test_invalid(self, kw):
        with pytest.raises(ValueError):
            SamplingParams(**kw)

    def test_frozen(self):
        p = SamplingParams()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.max_tokens = 3


class TestRequest:
    def test_validation(self):
        with pytest.raises(ValueError):
            Request("", [1], SamplingParams())
        with pytest.raises(ValueError):
            Request("r", [], SamplingParams())
        with pytest.raises(ValueError):
            Request("r", [1, -2], SamplingParams())

    def test_prompt_is_copied(self):
        prompt = [5, 6]
        r = Request("r", prompt, SamplingParams())
        prompt.append(7)
        assert r.prompt_token_ids == [5, 6]

    def test_counters(self):
        r = make(prompt=[1, 2, 3, 4])
        assert (r.num_prompt_tokens, r.num_output_tokens, r.num_known_tokens, r.num_pending_tokens) == (4, 0, 4, 4)
        r.resume()
        r.advance_computed_tokens(3)       # chunk 1 of the prefill
        assert r.num_pending_tokens == 1
        r.advance_computed_tokens(1)       # chunk 2 completes the prefill
        r.append_sampled_token(9)
        assert r.all_token_ids == (1, 2, 3, 4, 9)
        assert r.num_known_tokens == 5 and r.num_computed_tokens == 4
        assert r.num_pending_tokens == 1   # the new token's K/V is computed by the NEXT step

    def test_cannot_sample_with_pending_tokens(self):
        r = make()
        r.resume()
        r.advance_computed_tokens(2)
        with pytest.raises(ValueError):
            r.append_sampled_token(1)

    def test_cannot_overcompute(self):
        r = make()
        r.resume()
        with pytest.raises(ValueError):
            r.advance_computed_tokens(4)
        with pytest.raises(ValueError):
            r.advance_computed_tokens(0)

    def test_transitions(self):
        r = make()
        assert r.status is RequestStatus.WAITING
        with pytest.raises(ValueError):
            r.transition_to(RequestStatus.WAITING)
        r.resume(num_computed_tokens=2)
        assert r.status is RequestStatus.RUNNING and r.num_computed_tokens == 2

    def test_preempt_resets_progress_but_keeps_outputs(self):
        r = make()
        r.resume()
        r.advance_computed_tokens(3)
        r.append_sampled_token(7)
        r.preempt()
        assert r.status is RequestStatus.WAITING
        assert r.num_computed_tokens == 0
        assert r.output_token_ids == [7]
        assert r.num_pending_tokens == 4  # recompute prompt + the generated token

    def test_finish_is_terminal(self):
        r = make()
        r.finish(FinishReason.ABORTED)
        assert r.is_finished and r.finish_reason is FinishReason.ABORTED
        with pytest.raises(ValueError):
            r.transition_to(RequestStatus.RUNNING)
        with pytest.raises(ValueError):
            r.finish(FinishReason.EOS)
