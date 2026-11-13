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
        self.config = config
        if executor == "single":
            self.executor = SingleProcessExecutor(Worker(config))
        elif executor == "multi":
            self.executor = MultiProcessExecutor(config)
        else:
            raise ValueError(f"unknown executor {executor}")
        kv = KVCacheManager(self.executor.num_blocks, config.block_size, config.enable_prefix_cache)
        self.scheduler = Scheduler(config, kv)
        self.engine = Engine(self.scheduler, self.executor)
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(config.model)
            except Exception:
                tokenizer = None
        self.tokenizer = tokenizer
        self._counter = 0

    def generate(
        self,
        prompts: Sequence[str] | Sequence[Sequence[int]],
        params: SamplingParams | Sequence[SamplingParams],
    ) -> list[GenerationOutput]:
        """[Step 14] Accept text prompts (needs tokenizer) or token-id lists.
        Default eos_token_id to tokenizer.eos_token_id when the caller left it
        None. Submit everything, step until all are finished, return outputs
        in PROMPT order (not finish order), decoded when a tokenizer exists."""
        if isinstance(params, SamplingParams):
            params = [params] * len(prompts)
        if len(params) != len(prompts):
            raise ValueError("need one SamplingParams per prompt")
        ids_list = []
        for p in prompts:
            if isinstance(p, str):
                if self.tokenizer is None:
                    raise ValueError("text prompts need a tokenizer")
                ids_list.append(list(self.tokenizer.encode(p)))
            else:
                ids_list.append(list(p))
        request_ids = []
        for ids, sp in zip(ids_list, params):
            if sp.eos_token_id is None and self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
                sp = replace(sp, eos_token_id=self.tokenizer.eos_token_id)
            rid = f"req-{self._counter}"
            self._counter += 1
            self.engine.add_request(rid, ids, sp)
            request_ids.append(rid)
        final = {}
        pending = set(request_ids)
        while pending:
            for out in self.engine.step():
                if out.finished and out.request_id in pending:
                    final[out.request_id] = out
                    pending.discard(out.request_id)
        results = []
        for rid, ids in zip(request_ids, ids_list):
            out = final[rid]
            text = self.tokenizer.decode(out.output_token_ids) if self.tokenizer is not None else None
            results.append(GenerationOutput(tuple(ids), out.output_token_ids, out.finish_reason, text))
        return results

    def shutdown(self) -> None:
        """[Step 14]"""
        self.executor.shutdown()
