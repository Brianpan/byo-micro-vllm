"""Logits -> token ids (Step 11). Mirrors micro_vllm/sampling/sampler.py."""

import torch

from minivllm.executor.interface import ModelRunnerOutput, SampledToken
from minivllm.sampling.params import SamplingParams


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """[Step 11] Keep the k largest logits of a 1-D row, set the rest to -inf."""
    if top_k is None or top_k >= logits.shape[-1]:
        return logits
    values, indices = torch.topk(logits, top_k)
    return torch.full_like(logits, float("-inf")).scatter(0, indices, values)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """[Step 11] Nucleus filter on a 1-D row: keep the smallest set of highest
    probability tokens whose cumulative probability reaches top_p (the top-1
    token is always kept)."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    remove = probs.cumsum(-1) - probs > top_p  # mass BEFORE this token already exceeds top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)


class Sampler:
    def __call__(
        self,
        logits: torch.Tensor,
        request_ids: tuple[str, ...],
        params: tuple[SamplingParams, ...],
        sample_indices: tuple[int, ...],
    ) -> ModelRunnerOutput:
        """[Step 11] One row of logits per request -> ModelRunnerOutput."""
        tokens = [self.sample_row(row, p, i) for row, p, i in zip(logits, params, sample_indices)]
        return ModelRunnerOutput(tuple(SampledToken(r, t) for r, t in zip(request_ids, tokens)))

    def sample_row(self, logits: torch.Tensor, params: SamplingParams, sample_index: int) -> int:
        """[Step 11] Greedy = argmax. Otherwise: /temperature -> top-k -> top-p ->
        softmax -> multinomial. If params.seed is set use a fresh torch.Generator
        seeded with (seed + sample_index) so the k-th token of a request is
        reproducible no matter how requests were batched or preempted."""
        if params.is_greedy:
            return int(torch.argmax(logits).item())
        x = logits.float() / params.temperature
        x = apply_top_p(apply_top_k(x, params.top_k), params.top_p)
        probs = torch.softmax(x, dim=-1)
        gen = None
        if params.seed is not None:
            gen = torch.Generator(device=probs.device)
            gen.manual_seed(params.seed + sample_index)
        return int(torch.multinomial(probs, 1, generator=gen).item())
