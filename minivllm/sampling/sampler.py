"""Logits -> token ids (Step 11). Mirrors micro_vllm/sampling/sampler.py."""

import torch

from minivllm.executor.interface import ModelRunnerOutput, SampledToken
from minivllm.sampling.params import SamplingParams


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """[Step 11] Keep the k largest logits of a 1-D row, set the rest to -inf."""
    raise NotImplementedError("Step 11: implement apply_top_k")


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """[Step 11] Nucleus filter on a 1-D row: keep the smallest set of highest
    probability tokens whose cumulative probability reaches top_p (the top-1
    token is always kept)."""
    raise NotImplementedError("Step 11: implement apply_top_p")


class Sampler:
    def __call__(
        self,
        logits: torch.Tensor,
        request_ids: tuple[str, ...],
        params: tuple[SamplingParams, ...],
        sample_indices: tuple[int, ...],
    ) -> ModelRunnerOutput:
        """[Step 11] One row of logits per request -> ModelRunnerOutput."""
        raise NotImplementedError("Step 11: implement __call__")

    def sample_row(self, logits: torch.Tensor, params: SamplingParams, sample_index: int) -> int:
        """[Step 11] Greedy = argmax. Otherwise: /temperature -> top-k -> top-p ->
        softmax -> multinomial. If params.seed is set use a fresh torch.Generator
        seeded with (seed + sample_index) so the k-th token of a request is
        reproducible no matter how requests were batched or preempted."""
        raise NotImplementedError("Step 11: implement sample_row")
