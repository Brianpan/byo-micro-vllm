"""Offline generation facade (Step 14). Mirrors micro_vllm/llm.py.

Wiring order matters: the Worker must exist first because only it knows how
many KV blocks fit on the device; the KVCacheManager/Scheduler are sized from
that number.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from minivllm.config import EngineConfig
from minivllm.core.kv_cache_manager import KVCacheManager
from minivllm.core.request import FinishReason
from minivllm.core.scheduler import Scheduler
from minivllm.engine.engine import Engine
from minivllm.executor.multi_process import MultiProcessExecutor
from minivllm.executor.single_process import SingleProcessExecutor
from minivllm.sampling.params import SamplingParams
from minivllm.worker.worker import Worker


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    finish_reason: FinishReason
    output_text: str | None = None


class LLM:
    def __init__(self, config: EngineConfig, executor: str = "single", tokenizer=None) -> None:
        """[Step 14] Build executor ("single" | "multi"), then KVCacheManager
        (executor.num_blocks), Scheduler, Engine. Load a HF tokenizer from
        config.model if none is given and one is available (else None)."""
        raise NotImplementedError("Step 14: implement __init__")

    def generate(
        self,
        prompts: Sequence[str] | Sequence[Sequence[int]],
        params: SamplingParams | Sequence[SamplingParams],
    ) -> list[GenerationOutput]:
        """[Step 14] Accept text prompts (needs tokenizer) or token-id lists.
        Default eos_token_id to tokenizer.eos_token_id when the caller left it
        None. Submit everything, step until all are finished, return outputs
        in PROMPT order (not finish order), decoded when a tokenizer exists."""
        raise NotImplementedError("Step 14: implement generate")

    def shutdown(self) -> None:
        """[Step 14]"""
        raise NotImplementedError("Step 14: implement shutdown")
