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

    # check will run after __init__
    def __post_init__(self) -> None:
        """[Step 1] Reject empty ids / empty prompts / negative token ids; copy the prompt list."""
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if any(tid < 0 for tid in self.prompt_token_ids):
            raise ValueError("prompt_token_ids must not contain negative values")
        # Copy the prompt list to prevent external modifications.
        self.prompt_token_ids = self.prompt_token_ids.copy()
        
    @property
    def num_prompt_tokens(self) -> int:
        """[Step 1]"""
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        """[Step 1]"""
        return len(self.output_token_ids)

    @property
    def num_known_tokens(self) -> int:
        """[Step 1] prompt + generated tokens."""
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def num_pending_tokens(self) -> int:
        """[Step 1] Known tokens whose K/V is not in the cache yet."""
        return self.num_known_tokens - self.num_computed_tokens

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        """[Step 1] Immutable snapshot prompt + outputs."""
        return tuple(self.prompt_token_ids + self.output_token_ids)

    @property
    def is_finished(self) -> bool:
        """[Step 1]"""
        return self.status == RequestStatus.FINISHED

    def transition_to(self, new_status: RequestStatus) -> None:
        """[Step 1] Apply a legal transition from ALLOWED_TRANSITIONS or raise ValueError."""
        if new_status not in ALLOWED_TRANSITIONS.get(self.status, []):
            raise ValueError(f"Illegal transition from {self.status} to {new_status}")
        self.status = new_status

    def append_sampled_token(self, token_id: int) -> None:
        """[Step 1] A token may only be appended once every known token is computed."""
        if self.num_pending_tokens != 0:
            raise ValueError("Cannot append token before all pending tokens are computed")
        self.output_token_ids.append(token_id)

    def advance_computed_tokens(self, num_tokens: int) -> None:
        """[Step 1] Commit num_tokens (> 0, <= pending) newly computed tokens."""
        if num_tokens <= 0 or num_tokens > self.num_pending_tokens:
            raise ValueError("num_tokens must be > 0 and <= pending tokens")
        self.num_computed_tokens += num_tokens

    def preempt(self) -> None:
        """[Step 1] RUNNING -> WAITING and forget all computed K/V (recompute later)."""
        self.transition_to(RequestStatus.WAITING)
        self.num_computed_tokens = 0

    def resume(self, num_computed_tokens: int = 0) -> None:
        """[Step 1] WAITING -> RUNNING, starting at a (possibly prefix-cached) boundary."""
        self.num_computed_tokens = num_computed_tokens
        self.transition_to(RequestStatus.RUNNING)

    def finish(self, reason: FinishReason) -> None:
        """[Step 1] Enter the terminal state. Finishing twice is an error."""
        if self.is_finished:
            raise ValueError("Request is already finished")
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason
