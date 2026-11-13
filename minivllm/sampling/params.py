"""Per-request sampling parameters (Step 1)."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Immutable generation limits and token-selection policy."""

    max_tokens: int = 16
    eos_token_id: int | None = None
    ignore_eos: bool = False
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        """[Step 1] Validate every field; raise ValueError on bad input.

        Rules: max_tokens > 0; temperature finite and >= 0; top_k None or > 0;
        0 < top_p <= 1; seed None or >= 0; greedy (temperature == 0) must not
        set top_k / top_p.
        """
        raise NotImplementedError("Step 1: implement __post_init__")

    @property
    def is_greedy(self) -> bool:
        """[Step 1] Greedy decoding means temperature == 0."""
        raise NotImplementedError("Step 1: implement is_greedy")
