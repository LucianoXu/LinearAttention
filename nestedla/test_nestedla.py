"""Correctness tests for the nested linear-attention model.

The critical one is `test_chunkwise_matches_reference`: the efficient
chunkwise-parallel training path must equal the explicit token-recurrent +
per-chunk-slow reference, for BOTH the flat baseline (slow_memory=False) and the
nested model. Run in the container:

    python -m pytest nestedla/test_nestedla.py -q
"""

import torch

from .model import ModelArgs, NestedLA, NestedMemory, RoPE


def _args(**kw):
    base = dict(vocab_size=64, dim=32, v_dim_mult=1, head=4, ffn_mult=2,
                n_layers=2, context_len=24, chunk_size=6)
    base.update(kw)
    return ModelArgs(**base)


@torch.no_grad()
def test_memory_chunkwise_matches_reference():
    """Single NestedMemory layer: chunkwise == reference (float64, tight)."""
    for slow in (False, True):
        torch.manual_seed(0)
        args = _args(slow_memory=slow)
        rope = RoPE(args).double()
        mem = NestedMemory(args, rope).double()
        x = torch.randn(3, args.context_len, args.dim, dtype=torch.float64)
        yc = mem(x, reference=False)
        yr = mem(x, reference=True)
        err = (yc - yr).abs().max().item()
        assert err < 1e-9, f"slow={slow}: chunkwise vs reference max err {err}"


@torch.no_grad()
def test_full_model_chunkwise_matches_reference():
    for slow in (False, True):
        torch.manual_seed(1)
        args = _args(slow_memory=slow)
        model = NestedLA(args).double()
        x = torch.randint(0, args.vocab_size, (2, args.context_len))
        lc = model(x, reference=False)
        lr = model(x, reference=True)
        err = (lc - lr).abs().max().item()
        assert err < 1e-9, f"slow={slow}: full-model max err {err}"


@torch.no_grad()
def test_causality_chunkwise():
    """Output at position t must not depend on inputs > t (linear-attn causal)."""
    torch.manual_seed(2)
    args = _args(slow_memory=True)
    model = NestedLA(args).double()
    x = torch.randint(0, args.vocab_size, (1, args.context_len))
    y1 = model(x)
    t = 10
    x2 = x.clone()
    x2[0, t + 1:] = torch.randint(0, args.vocab_size, (args.context_len - t - 1,))
    y2 = model(x2)
    assert (y1[0, :t + 1] - y2[0, :t + 1]).abs().max().item() < 1e-9


def test_baseline_has_fewer_params_than_nested():
    flat = NestedLA(_args(slow_memory=False))
    nested = NestedLA(_args(slow_memory=True))
    n_flat = sum(p.numel() for p in flat.parameters())
    n_nested = sum(p.numel() for p in nested.parameters())
    # slow memory adds only the per-head read gates (a few params/layer)
    assert n_nested > n_flat
    overhead = (n_nested - n_flat) / n_flat
    assert overhead < 0.05, f"slow-memory overhead {overhead:.3%} unexpectedly large"


def test_gradients_flow_both_toggles():
    for slow in (False, True):
        args = _args(slow_memory=slow)
        model = NestedLA(args)
        x = torch.randint(0, args.vocab_size, (2, args.context_len))
        loss = model(x).square().mean()
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
