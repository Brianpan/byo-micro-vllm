"""Shared fixtures.

By default the tests import YOUR package at <repo>/minivllm. Run them against
the reference solution with:

    MINIVLLM_IMPL=reference pytest tests/
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_impl = os.environ.get("MINIVLLM_IMPL", "")
_impl_root = ROOT / "reference" if _impl == "reference" else (Path(_impl) if _impl else ROOT)
sys.path.insert(0, str(_impl_root))


def pytest_report_header(config):
    return f"minivllm implementation under test: {_impl_root / 'minivllm'}"


@pytest.fixture(scope="session")
def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


TINY_GPT2 = dict(vocab_size=512, n_positions=256, n_embd=64, n_layer=2, n_head=4)


@pytest.fixture(scope="session")
def tiny_gpt2(tmp_path_factory):
    """A small random GPT-2 saved like a real checkpoint: (path, hf_model).

    No network needed. A larger init std makes logits peaked so greedy decoding
    has no near-ties (otherwise float noise could flip an argmax)."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(**TINY_GPT2, initializer_range=0.1, attn_implementation="eager")
    hf = GPT2LMHeadModel(cfg).eval()
    path = tmp_path_factory.mktemp("tiny-gpt2")
    hf.save_pretrained(path)
    return str(path), hf


@pytest.fixture
def hf_greedy(tiny_gpt2):
    """Reference greedy continuation computed with plain HF transformers (CPU, fp32)."""
    import torch

    _, hf = tiny_gpt2

    def run(prompt: list[int], max_tokens: int) -> list[int]:
        ids = torch.tensor([prompt])
        with torch.no_grad():
            out = hf.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=max_tokens,
                min_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=0,
            )
        return out[0, len(prompt):].tolist()

    return run
