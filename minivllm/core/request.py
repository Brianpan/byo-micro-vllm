"""Request state machine (Step 1). Mirrors micro_vllm/core/request.py."""

from dataclasses import dataclass, field
from enum import Enum, auto

from minivllm.sampling.params import SamplingParams


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class FinishReason(Enum):
    EOS = auto()
    LENGTH = auto()        # hit sampling_params.max_tokens
    MODEL_LENGTH = auto()  # hit config.max_model_len
    ABORTED = auto()


# FINISHED is terminal and is entered only through finish().
ALLOWED_TRANSITIONS = {
    RequestStatus.WAITING: {RequestStatus.RUNNING},
    RequestStatus.RUNNING: {RequestStatus.WAITING},
    RequestStatus.FINISHED: set(),
}


@dataclass(slots=True)
class Request:
    """Mutable per-request state. Only the Scheduler mutates it.

    The single most important counter is ``num_computed_tokens``: how many of
    the request's known tokens already have K/V written in the paged cache.
    Everything the scheduler does is "advance num_computed_tokens toward
    num_known_tokens, then sample one more token".
    """

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    output_token_ids: list[int] = field(default_factory=list, init=False)
    num_computed_tokens: int = field(default=0, init=False)
    status: RequestStatus = field(default=RequestStatus.WAITING, init=False)
    finish_reason: FinishReason | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """[Step 1] Reject empty ids / empty prompts / negative token ids; copy the prompt list."""
        raise NotImplementedError("Step 1: implement __post_init__")

    @property
    def num_prompt_tokens(self) -> int:
        """[Step 1]"""
        raise NotImplementedError("Step 1: implement num_prompt_tokens")

    @property
    def num_output_tokens(self) -> int:
        """[Step 1]"""
        raise NotImplementedError("Step 1: implement num_output_tokens")

    @property
    def num_known_tokens(self) -> int:
        """[Step 1] prompt + generated tokens."""
        raise NotImplementedError("Step 1: implement num_known_tokens")

    @property
    def num_pending_tokens(self) -> int:
        """[Step 1] Known tokens whose K/V is not in the cache yet."""
        raise NotImplementedError("Step 1: implement num_pending_tokens")

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        """[Step 1] Immutable snapshot prompt + outputs."""
        raise NotImplementedError("Step 1: implement all_token_ids")

    @property
    def is_finished(self) -> bool:
        """[Step 1]"""
        raise NotImplementedError("Step 1: implement is_finished")

    def transition_to(self, new_status: RequestStatus) -> None:
        """[Step 1] Apply a legal transition from ALLOWED_TRANSITIONS or raise ValueError."""
        raise NotImplementedError("Step 1: implement transition_to")

    def append_sampled_token(self, token_id: int) -> None:
        """[Step 1] A token may only be appended once every known token is computed."""
        raise NotImplementedError("Step 1: implement append_sampled_token")

    def advance_computed_tokens(self, num_tokens: int) -> None:
        """[Step 1] Commit num_tokens (> 0, <= pending) newly computed tokens."""
        raise NotImplementedError("Step 1: implement advance_computed_tokens")

    def preempt(self) -> None:
        """[Step 1] RUNNING -> WAITING and forget all computed K/V (recompute later)."""
        raise NotImplementedError("Step 1: implement preempt")

    def resume(self, num_computed_tokens: int = 0) -> None:
        """[Step 1] WAITING -> RUNNING, starting at a (possibly prefix-cached) boundary."""
        raise NotImplementedError("Step 1: implement resume")

    def finish(self, reason: FinishReason) -> None:
        """[Step 1] Enter the terminal state. Finishing twice is an error."""
        raise NotImplementedError("Step 1: implement finish")
