"""Immutable execution plan: the ONLY thing that crosses scheduler -> executor.

Mirrors micro_vllm/core/scheduler_output.py. Everything is frozen and made of
plain Python values so it can be pickled to a worker process (Step 13).
"""

from dataclasses import dataclass

from minivllm.sampling.params import SamplingParams


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    request_id: str
    input_token_ids: tuple[int, ...]   # tokens [num_computed, num_computed + num_scheduled)
    num_computed_tokens: int           # K/V already in cache = absolute position of input[0]
    num_scheduled_tokens: int
    block_ids: tuple[int, ...]         # block table after allocation
    requires_sampling: bool            # True when this chunk reaches the end of known tokens
    sampling_params: SamplingParams | None = None
    sample_index: int | None = None    # == num_output_tokens; seeds per-token RNG (Step 11)


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    scheduled_requests: tuple[ScheduledRequest, ...]
    preempted_request_ids: tuple[str, ...] = ()

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(r.num_scheduled_tokens for r in self.scheduled_requests)
