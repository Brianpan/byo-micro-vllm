"""Run your engine on the real GPT-2 checkpoint.

    python examples/generate.py                     # your minivllm/
    MINIVLLM_IMPL=reference python examples/generate.py
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference" if os.environ.get("MINIVLLM_IMPL") == "reference" else ROOT))

import torch  # noqa: E402

from minivllm.config import EngineConfig  # noqa: E402
from minivllm.llm import LLM  # noqa: E402
from minivllm.sampling.params import SamplingParams  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
llm = LLM(EngineConfig(
    model="gpt2",
    device=device,
    dtype="float16" if device == "cuda" else "float32",
    num_blocks=None if device == "cuda" else 512,   # None -> profile free GPU memory
    enable_prefix_cache=True,
))
print(f"device={device} kv_blocks={llm.executor.num_blocks}")

prompts = [
    "The key idea behind paged attention is",
    "In a distributed system, the hardest problem is",
    "Once upon a time, in a small village,",
    "def quicksort(arr):",
] * 4

start = time.perf_counter()
outs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.8, top_p=0.95, seed=0))
elapsed = time.perf_counter() - start
for o in outs[:4]:
    print("-" * 60)
    print(o.output_text)
n = sum(len(o.output_token_ids) for o in outs)
print("-" * 60)
print(f"{len(prompts)} requests, {n} tokens in {elapsed:.2f}s -> {n / elapsed:.1f} tok/s")
llm.shutdown()
