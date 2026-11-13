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
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive int")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and >= 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be >= 0 or None")
        if self.is_greedy and (self.top_k is not None or self.top_p != 1.0):
            raise ValueError("top_k/top_p require temperature > 0")

    @property
    def is_greedy(self) -> bool:
        """[Step 1] Greedy decoding means temperature == 0."""
        return self.temperature == 0.0
