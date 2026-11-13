"""Step 7 - GPT-2 as a flat-token model, checked against HF transformers."""

import pytest
import torch

from minivllm.model.gpt2 import GPT2Config, GPT2LMHeadModel
from conftest import TINY_GPT2


@pytest.fixture
def model(tiny_gpt2):
    path, _ = tiny_gpt2
    return GPT2LMHeadModel.from_pretrained(path).eval()


def test_config_from_hf(tiny_gpt2):
    _, hf = tiny_gpt2
    cfg = GPT2Config.from_hf(hf.config)
    assert (cfg.vocab_size, cfg.n_embd, cfg.n_layer, cfg.n_head) == (
        TINY_GPT2["vocab_size"], TINY_GPT2["n_embd"], TINY_GPT2["n_layer"], TINY_GPT2["n_head"])
    assert cfg.head_dim == 16


def test_parameter_count_matches_hf(model, tiny_gpt2):
    _, hf = tiny_gpt2
    ours = sum(p.numel() for p in model.parameters())
    theirs = sum(p.numel() for p in hf.parameters())  # HF also ties lm_head, so counts once
    assert ours == theirs


def test_lm_head_is_tied(model):
    hidden = torch.randn(3, model.config.n_embd)
    assert torch.allclose(model.compute_logits(hidden), hidden @ model.wte.weight.T)


def test_conv1d_weights_are_transposed(model, tiny_gpt2):
    _, hf = tiny_gpt2
    theirs = hf.transformer.h[0].attn.c_attn.weight      # Conv1D: [in, out]
    ours = model.h[0].attn.c_attn.weight                 # Linear: [out, in]
    assert ours.shape == (3 * 64, 64)
    assert torch.equal(ours, theirs.T)


def test_bad_state_dict_rejected(model, tiny_gpt2):
    _, hf = tiny_gpt2
    sd = dict(hf.state_dict())
    sd.pop("transformer.ln_f.weight")
    with pytest.raises(KeyError):
        GPT2LMHeadModel(model.config).load_hf_state_dict(sd)


@torch.no_grad()
@pytest.mark.parametrize("length", [1, 7, 33])
def test_logits_match_hf(model, tiny_gpt2, length):
    _, hf = tiny_gpt2
    torch.manual_seed(length)
    ids = torch.randint(0, model.config.vocab_size, (length,))
    expected = hf(ids[None]).logits[0]
    hidden = model(ids, torch.arange(length))               # flat [T] inputs, no batch dim
    assert hidden.shape == (length, model.config.n_embd)
    got = model.compute_logits(hidden)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_runs_on_device(model, device):
    model = model.to(device)
    ids = torch.arange(5, device=device)
    assert model.compute_logits(model(ids, ids)).device.type == device
