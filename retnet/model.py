from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
import math

@dataclass
class ModelArgs:
    vocab_size: int = 256
    dim: int = 512
    v_dim_mult: int = 2
    head: int = 16
    ffn_mult: int = 3
    n_layers: int = 8
    context_len : int = 128


class RoPE(nn.Module):
    m: torch.Tensor   # declares the registered buffer's type so Pylance treats self.m as a Tensor

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.dim % 2 == 0
        self.args = args

        self.prepared_L = 0

        self.register_buffer('m', torch.zeros(()), persistent=False)
        self.prepare_m(args.context_len)


    def prepare_m(self, L: int):

        if L > self.prepared_L:
            device : torch.device = self.m.device
            dtype : torch.dtype = self.m.dtype
            idxk = torch.arange(0, self.args.dim // self.args.head // 2, device=device, dtype=dtype) / (self.args.dim // self.args.head // 2)
            phase = torch.outer(torch.arange(0, L, device=device, dtype=dtype), torch.pow(10000, -idxk,))
            m_sin = torch.sin(phase)    # (L, head_dim/2)
            m_cos = torch.cos(phase)    # (L, head_dim/2)

            # m : (L, head_dim/2, 2, 2)  -- RoPE now operates per head
            self.register_buffer(
                'm',
                torch.stack([m_cos, m_sin, -m_sin, m_cos], dim=-1).reshape(L, self.args.dim // self.args.head // 2, 2, 2),
                persistent=False
            )

            self.prepared_L = L


    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None):

        # x : (..., l, head_dim)   e.g. parallel (B, H, L, head_dim)
        # pos: (b, l) absolute positions for the recurrent path; None -> 0..L-1

        if pos is None:
            self.prepare_m(x.shape[-2])

            # slice the matrix
            m = self.m[:x.shape[-2], ...]                       # (L, head_dim/2, 2, 2)

            x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)   # (..., L, head_dim/2, 2)
            x = torch.einsum('ldmn,...ldn->...ldm', m, x)
            x = x.reshape(*x.shape[:-2], -1)

            return x

        else:

            self.prepare_m(int(pos.max().item()) + 1)

            # slice the matrix
            m = self.m[pos]                                     # (B, S, head_dim/2, 2, 2)

            x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)   # (B, H, S, head_dim/2, 2)
            x = torch.einsum('bsdmn,bhsdn->bhsdm', m, x)        # broadcast rotation over heads
            x = x.reshape(*x.shape[:-2], -1)

            return x
    
class RMSNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.eps = 1e-8
        self.args = args
        self.gamma = nn.Parameter(data = torch.ones(args.dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms * self.gamma
        return x


class MSR(nn.Module):

    gamma: torch.Tensor
    grid: torch.Tensor

    def __init__(self, args: ModelArgs, rope: RoPE):
        super().__init__()

        assert args.dim % args.head == 0

        self.register_buffer(
            "gamma", 
            1 - torch.pow(2, (-5 -torch.arange(0, args.head))), # (h,)
            persistent=False
        )

        self.args = args

        self.rope = rope

        self.wq = nn.Linear(
            in_features=self.args.dim,
            out_features=self.args.dim,
            bias = False
        )

        self.wk = nn.Linear(
            in_features=self.args.dim,
            out_features=self.args.dim,
            bias = False
        )

        self.wv = nn.Linear(
            in_features=self.args.dim,
            out_features=self.args.dim * self.args.v_dim_mult,
            bias = False
        )

        self.wg = nn.Linear(
            in_features=self.args.dim,
            out_features=self.args.dim * self.args.v_dim_mult,
            bias = False
        )

        self.wo = nn.Linear(
            in_features=self.args.dim * self.args.v_dim_mult,
            out_features=self.args.dim,
            bias = False
        )

        self.rms_gamma = nn.Parameter(data = torch.ones(args.head, args.dim * self.args.v_dim_mult // args.head))

        self.register_buffer(
            "grid",
            torch.arange(args.context_len).unsqueeze(1) - torch.arange(args.context_len).unsqueeze(0),   # (L, L)
            persistent=False,
        )


    def recurrent_forward(self, x, step_count, kv_state_t: torch.Tensor):
        # RetNet recurrence (Eq 6): S_n = gamma * S_{n-1} + k_n^T v_n ;  out_n = q_n S_n
        # x: (B, 1, dim);  kv_state_t: (B, H, Dh, Dvh)   [Dvh = Dh * v_dim_mult]

        B = x.shape[0]
        H, Dh = self.args.head, self.args.dim // self.args.head

        pos = torch.tensor([step_count], device=x.device).broadcast_to(B, 1)

        q = self.rope(self.wq(x).reshape(B, 1, H, Dh).transpose(1, 2), pos).squeeze(2)   # (B, H, Dh)
        k = self.rope(self.wk(x).reshape(B, 1, H, Dh).transpose(1, 2), pos).squeeze(2)   # (B, H, Dh)
        v = self.wv(x).reshape(B, 1, H, -1).transpose(1, 2).squeeze(2)                   # (B, H, Dvh)

        # decayed matrix-memory update (per-head gamma), then query it
        gamma = self.gamma.reshape(1, H, 1, 1)
        kv_state_tp1 = gamma * kv_state_t + torch.einsum('bhi,bhj->bhij', k, v)          # (B, H, Dh, Dvh)
        out = torch.einsum('bhi,bhij->bhj', q, kv_state_tp1) / math.sqrt(Dh)             # (B, H, Dvh)

        # per-head RMSNorm + swish gate + output proj (same post-processing as parallel)
        rms = torch.rsqrt(out.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        y = (out * rms * self.rms_gamma).reshape(B, 1, -1)                               # (B, 1, dim*v_dim_mult)
        x = self.wo(nn.functional.silu(self.wg(x)) * y)                                  # (B, 1, dim)

        kv_state_t.copy_(kv_state_tp1)

        return x





    def forward(self, x, mask = None):
        # mask: lower-triangular 0/1 causal mask (L, L), or None
        # x : (b, l, dim)

        B, L = x.shape[0], x.shape[1]

        H, Dh = self.args.head, self.args.dim // self.args.head

        Dnm = torch.tril(torch.pow(self.gamma.reshape(-1, 1, 1), self.grid[:L, :L])) # (h, L, L)

        q = self.rope(self.wq(x).reshape(B, L, H, Dh).transpose(1, 2))   # (B,H,L,Dh)
        k = self.rope(self.wk(x).reshape(B, L, H, Dh).transpose(1, 2))
        v = self.wv(x).reshape(B, L, H, -1).transpose(1, 2)
        a = torch.einsum('bhid,bhjd->bhij', q, k) * Dnm            # (B,H,L,L)
        a = a / math.sqrt(Dh)
        y = torch.einsum('bhij,bhjd->bihd', a, v)                  # (B, L, H, Dh)

        
        rms = torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        y = y * rms * self.rms_gamma
        y = y.reshape(B, L, -1)

        x = self.wo(nn.functional.silu(self.wg(x)) * y)

        return x


class FNN(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args

        self.l1 = nn.Linear(
            in_features=self.args.dim,
            out_features=self.args.dim * self.args.ffn_mult,
            bias = False
        )

        self.activation = nn.GELU()

        self.l2 = nn.Linear(
            in_features=self.args.dim * self.args.ffn_mult,
            out_features=self.args.dim,
            bias = False
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        return x

class Blocks(nn.Module):
    def __init__(self, args: ModelArgs, rope: RoPE):
        super().__init__()

        self.args = args
        self.rmsnorm1 = RMSNorm(self.args)
        self.att = MSR(self.args, rope)
        self.rmsnorm2 = RMSNorm(self.args)
        self.fnn = FNN(self.args)

    def recurrent_forward(self, x, step_count, kv_state_t: torch.Tensor):
        dx = self.rmsnorm1(x)
        dx = self.att.recurrent_forward(dx, step_count, kv_state_t)
        x = x + dx

        dx = self.rmsnorm2(x)
        dx = self.fnn(dx)
        x = x + dx
        
        return x
    
    def forward(self, x, mask=None):
        dx = self.rmsnorm1(x)
        dx = self.att(dx, mask)
        x = x + dx

        dx = self.rmsnorm2(x)
        dx = self.fnn(dx)
        x = x + dx
        
        return x


class RetNetHead(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args

        self.rms = RMSNorm(self.args)

        self.l = nn.Linear(
            in_features=self.args.dim,
            out_features=self.args.vocab_size,
            bias = False
        )

    def forward(self, x):
        # (B, L, D)
        x = self.rms(x)
        x = self.l(x)
        return x


class RetNet(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args
        
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.dim)
        self.rope = RoPE(self.args)
        self.blocks = nn.ModuleList([Blocks(self.args, self.rope) for _ in range(self.args.n_layers)])
        self.head = RetNetHead(self.args)

    def recurrent_forward(self, x, step_count, kv_state_t: torch.Tensor | None):
        B, _ = x.shape
        H, Dh = self.args.head, self.args.dim // self.args.head
        Dvh = self.args.dim * self.args.v_dim_mult // self.args.head

        if kv_state_t is None:
            kv_state_t = torch.zeros(B, self.args.n_layers, H, Dh, Dvh, device=x.device)

        x = self.embedding(x)
        for i, block in enumerate(self.blocks): # type: ignore
            block: Blocks
            x = block.recurrent_forward(x, step_count, kv_state_t[:, i, ...])

        x = self.head(x)

        return x, kv_state_t

    def forward(self, x, mask = None):

        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.head(x)

        return x