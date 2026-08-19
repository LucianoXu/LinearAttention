"""Nested multi-timescale linear-attention LM.

The memory primitive is RetNet-style per-head decay linear attention, but the
*training* path is **chunkwise-parallel** (O(L*C) instead of the O(L^2) form in
`retnet/model.py`): the sequence is split into chunks of size C, intra-chunk
interactions are computed in parallel, and a recurrent KV state is carried across
chunk boundaries.

On top of the per-token **fast** memory we add an optional **slow** memory: a
second associative state updated once *per chunk* (a lower-frequency timescale)
with a slower decay, read by every token through a learned per-head gate. This is
the Nested-Learning multi-frequency-memory idea -- fast (per-token) memory nested
inside slow (per-chunk) memory. `slow_memory=False` recovers a flat single-
timescale baseline, so the ablation is one config toggle.

Two forward implementations share the same projections:
  * `_chunkwise` -- the efficient training path.
  * `_reference` -- explicit token-recurrent fast + per-chunk slow, used to unit-
    test `_chunkwise` for exactness (see test_nestedla.py). It also documents the
    exact recurrent semantics.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class ModelArgs:
    vocab_size: int = 760
    dim: int = 768
    v_dim_mult: int = 1
    head: int = 8
    ffn_mult: int = 4
    n_layers: int = 12
    context_len: int = 2048

    chunk_size: int = 256          # C: chunk length for the chunkwise scan
    slow_memory: bool = True       # toggle the nested slow-memory level (A/B knob)
    slow_gamma_power: float = 0.25  # gamma_slow = gamma_fast ** p  (p<1 -> slower)
    fast_decay_min_exp: int = 5    # fast head gammas = 1 - 2^-(e..e+H-1); head
                                   # horizons ~= 2^e .. 2^(e+H-1) tokens. The
                                   # default (e=5) makes the slowest fast head
                                   # span ~2^12 > context, so a separate slow
                                   # memory is redundant; lowering e localizes
                                   # fast memory and gives the slow level a job.


class RoPE(nn.Module):
    """Rotary position embedding applied to q/k before the dot product.

    (Same construction as retnet/model.py; absolute-position rotation makes the
    q.k dot product depend on relative position, which is preserved by chunking.)
    """
    m: torch.Tensor

    def __init__(self, args: ModelArgs):
        super().__init__()
        assert (args.dim // args.head) % 2 == 0
        self.args = args
        self.prepared_L = 0
        self.register_buffer("m", torch.zeros(()), persistent=False)
        self.prepare_m(args.context_len)

    def prepare_m(self, L: int):
        if L > self.prepared_L:
            device, dtype = self.m.device, torch.float32
            hd = self.args.dim // self.args.head
            idxk = torch.arange(0, hd // 2, device=device, dtype=dtype) / (hd // 2)
            phase = torch.outer(torch.arange(0, L, device=device, dtype=dtype),
                                torch.pow(10000, -idxk))
            m_sin, m_cos = torch.sin(phase), torch.cos(phase)
            self.register_buffer(
                "m",
                torch.stack([m_cos, m_sin, -m_sin, m_cos], dim=-1).reshape(L, hd // 2, 2, 2),
                persistent=False,
            )
            self.prepared_L = L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, L, Dh)
        self.prepare_m(x.shape[-2])
        m = self.m[: x.shape[-2]].to(x.dtype)        # (L, Dh/2, 2, 2)
        x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x = torch.einsum("ldmn,bhldn->bhldm", m, x)
        return x.reshape(*x.shape[:-2], -1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.eps = 1e-8
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.gamma


class NestedMemory(nn.Module):
    """Multi-head decay linear attention with a fast (per-token) and an optional
    slow (per-chunk) memory level."""

    gamma: torch.Tensor
    gamma_slow: torch.Tensor

    def __init__(self, args: ModelArgs, rope: RoPE):
        super().__init__()
        assert args.dim % args.head == 0
        self.args = args
        self.rope = rope
        H = args.head
        self.Dh = args.dim // H
        self.Dv = args.dim * args.v_dim_mult // H

        # per-head fast decay (RetNet schedule); slow decay is closer to 1
        gamma = 1 - torch.pow(2.0, -float(args.fast_decay_min_exp) - torch.arange(0, H))
        self.register_buffer("gamma", gamma, persistent=False)
        self.register_buffer("gamma_slow", torch.pow(gamma, args.slow_gamma_power),
                             persistent=False)

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim * args.v_dim_mult, bias=False)
        self.wg = nn.Linear(args.dim, args.dim * args.v_dim_mult, bias=False)
        self.wo = nn.Linear(args.dim * args.v_dim_mult, args.dim, bias=False)
        self.rms_gamma = nn.Parameter(torch.ones(H, self.Dv))

        if args.slow_memory:
            # per-head read gate for the slow memory; init bias negative so the
            # slow path starts ~closed (model behaves like the flat baseline at
            # init, then learns to open the gate).
            self.slow_gate = nn.Linear(args.dim, H, bias=True)
            nn.init.zeros_(self.slow_gate.weight)
            nn.init.constant_(self.slow_gate.bias, -2.0)

    # -- shared projection + post-processing -------------------------------
    def _project(self, x):
        B, L, _ = x.shape
        H, Dh = self.args.head, self.Dh
        q = self.rope(self.wq(x).reshape(B, L, H, Dh).transpose(1, 2))   # (B,H,L,Dh)
        k = self.rope(self.wk(x).reshape(B, L, H, Dh).transpose(1, 2))
        v = self.wv(x).reshape(B, L, H, self.Dv).transpose(1, 2)         # (B,H,L,Dv)
        return q, k, v

    def _post(self, x, y):
        # y: (B,H,L,Dv) -> per-head RMSNorm -> swish gate -> output projection
        B, H, L, Dv = y.shape
        rms = torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        y = (y * rms * self.rms_gamma.view(1, H, 1, Dv))
        y = y.transpose(1, 2).reshape(B, L, H * Dv)
        return self.wo(nn.functional.silu(self.wg(x)) * y)

    # -- efficient training path -------------------------------------------
    def _chunkwise(self, x):
        B, L, _ = x.shape
        H, Dh, Dv, C = self.args.head, self.Dh, self.Dv, self.args.chunk_size
        assert L % C == 0, f"context_len {L} must be divisible by chunk_size {C}"
        NC = L // C
        scale = 1.0 / math.sqrt(Dh)

        q, k, v = self._project(x)
        # -> chunks: (B,H,NC,C,*)
        qc = q.reshape(B, H, NC, C, Dh)
        kc = k.reshape(B, H, NC, C, Dh)
        vc = v.reshape(B, H, NC, C, Dv)

        g = self.gamma.to(x.dtype).view(H, 1, 1)              # (H,1,1)
        i = torch.arange(C, device=x.device)
        ij = i.view(C, 1) - i.view(1, C)                      # (C,C)  = local (i - j)
        D_intra = torch.where(ij >= 0, g.view(H, 1, 1) ** ij.clamp(min=0).float(),
                              torch.zeros((), device=x.device, dtype=x.dtype))  # (H,C,C)
        decay_cross = self.gamma.to(x.dtype).view(H, 1) ** (i + 1).float()      # (H,C)
        kv_local_decay = self.gamma.to(x.dtype).view(H, 1) ** (C - 1 - i).float()  # (H,C)
        chunk_decay = self.gamma.to(x.dtype) ** C                               # (H,)

        # intra-chunk (parallel within each chunk)
        attn = torch.einsum("bhncd,bhnmd->bhncm", qc, kc) * D_intra.view(1, H, 1, C, C)
        inner = torch.einsum("bhncm,bhnmv->bhncv", attn, vc)                    # (B,H,NC,C,Dv)

        # per-chunk local KV contributions for the FAST state carry
        kv_fast = torch.einsum("bhncd,bhncv->bhndv",
                               kc * kv_local_decay.view(1, H, 1, C, 1), vc)     # (B,H,NC,Dh,Dv)

        # slow memory: one coarse KV summary per chunk (averaged, no intra decay)
        use_slow = self.args.slow_memory
        if use_slow:
            kv_slow = torch.einsum("bhncd,bhncv->bhndv", kc, vc) / C            # (B,H,NC,Dh,Dv)
            gs = self.gamma_slow.to(x.dtype)

        # sequential scan over the (few) chunks to build start-of-chunk states
        out = torch.empty(B, H, NC, C, Dv, device=x.device, dtype=v.dtype)
        S = torch.zeros(B, H, Dh, Dv, device=x.device, dtype=v.dtype)           # fast state
        if use_slow:
            Z = torch.zeros(B, H, Dh, Dv, device=x.device, dtype=v.dtype)       # slow state
            slow_gate = torch.sigmoid(self.slow_gate(x)).reshape(B, NC, C, H)   # (B,NC,C,H)

        for c in range(NC):
            cross = torch.einsum("bhcd,bhdv->bhcv",
                                 qc[:, :, c] * decay_cross.view(1, H, C, 1), S)  # (B,H,C,Dv)
            y = (inner[:, :, c] + cross) * scale
            if use_slow:
                slow_read = torch.einsum("bhcd,bhdv->bhcv", qc[:, :, c], Z) * scale
                gate = slow_gate[:, c].permute(0, 2, 1).unsqueeze(-1)           # (B,H,C,1)
                y = y + gate * slow_read
                Z = gs.view(1, H, 1, 1) * Z + kv_slow[:, :, c]
            out[:, :, c] = y
            S = chunk_decay.view(1, H, 1, 1) * S + kv_fast[:, :, c]

        return self._post(x, out.reshape(B, H, L, Dv))

    # -- reference path (for equivalence tests; explicit recurrence) -------
    def _reference(self, x):
        B, L, _ = x.shape
        H, Dh, Dv, C = self.args.head, self.Dh, self.Dv, self.args.chunk_size
        scale = 1.0 / math.sqrt(Dh)
        q, k, v = self._project(x)
        g = self.gamma.to(x.dtype)
        use_slow = self.args.slow_memory
        if use_slow:
            gs = self.gamma_slow.to(x.dtype)
            gate = torch.sigmoid(self.slow_gate(x))                            # (B,L,H)

        out = torch.zeros(B, H, L, Dv, device=x.device, dtype=v.dtype)
        S = torch.zeros(B, H, Dh, Dv, device=x.device, dtype=v.dtype)
        Z = torch.zeros(B, H, Dh, Dv, device=x.device, dtype=v.dtype)
        chunk_kv = torch.zeros(B, H, Dh, Dv, device=x.device, dtype=v.dtype)   # accumulates current chunk
        for t in range(L):
            if use_slow and t % C == 0:
                # slow memory updates ONCE per chunk, with the *previous* chunk's
                # averaged summary; reads use the state from before this chunk.
                Z = gs.view(1, H, 1, 1) * Z + (chunk_kv / C if t > 0 else 0.0 * chunk_kv)
                chunk_kv = torch.zeros_like(chunk_kv)
            qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]                    # (B,H,Dh/Dv)
            S = g.view(1, H, 1, 1) * S + torch.einsum("bhi,bhj->bhij", kt, vt)
            y = torch.einsum("bhi,bhij->bhj", qt, S) * scale
            if use_slow:
                slow_read = torch.einsum("bhi,bhij->bhj", qt, Z) * scale
                y = y + gate[:, t].unsqueeze(-1) * slow_read
                chunk_kv = chunk_kv + torch.einsum("bhi,bhj->bhij", kt, vt)
            out[:, :, t] = y
        return self._post(x, out)

    def forward(self, x, reference: bool = False):
        return self._reference(x) if reference else self._chunkwise(x)


class FNN(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.l1 = nn.Linear(args.dim, args.dim * args.ffn_mult, bias=False)
        self.activation = nn.GELU()
        self.l2 = nn.Linear(args.dim * args.ffn_mult, args.dim, bias=False)

    def forward(self, x):
        return self.l2(self.activation(self.l1(x)))


class Block(nn.Module):
    def __init__(self, args: ModelArgs, rope: RoPE):
        super().__init__()
        self.rmsnorm1 = RMSNorm(args.dim)
        self.att = NestedMemory(args, rope)
        self.rmsnorm2 = RMSNorm(args.dim)
        self.fnn = FNN(args)

    def forward(self, x, reference: bool = False):
        x = x + self.att(self.rmsnorm1(x), reference=reference)
        x = x + self.fnn(self.rmsnorm2(x))
        return x


class NestedLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.rope = RoPE(args)
        self.blocks = nn.ModuleList([Block(args, self.rope) for _ in range(args.n_layers)])
        self.head_norm = RMSNorm(args.dim)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, x, mask=None, reference: bool = False):
        # `mask` is accepted (and ignored) so the training loop can call the model
        # the same way it calls the transformer; linear attention is causal by
        # construction (decay + chunkwise carry), no explicit mask needed.
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, reference=reference)
        return self.head(self.head_norm(x))
