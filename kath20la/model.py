from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

import math


@dataclass
class ModelArgs:
    vocab_size: int = 256
    dim: int = 512
    ffn_mult: int = 4
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
            idxk = torch.arange(0, self.args.dim // 2, device=device, dtype=dtype) / (self.args.dim // 2)
            phase = torch.outer(torch.arange(0, L, device=device, dtype=dtype), torch.pow(10000, -idxk,))
            m_sin = torch.sin(phase)    # (L, dim/2)
            m_cos = torch.cos(phase)    # (L, dim/2)

            # m : (L, dim/2, 2, 2)
            self.register_buffer(
                'm', 
                torch.stack([m_cos, m_sin, -m_sin, m_cos], dim=-1).reshape(L, self.args.dim // 2, 2, 2),
                persistent=False
            )

            self.prepared_L = L


    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None):
        
        # x : (b, l, d)
        # pos: (b, l), assume it to be ascending

        if pos is None:
            self.prepare_m(x.shape[1])

            # slice the matrix
            m = self.m[:x.shape[1], ...]

            x = x.reshape(x.shape[0], x.shape[1], -1, 2)
            x = torch.einsum('ldmn,bldn->bldm', m, x)
            x = x.reshape(x.shape[0], x.shape[1], -1)

            return x

        else:

            self.prepare_m(int(pos.max().item()) + 1)

            # slice the matrix
            m = self.m[pos]

            x = x.reshape(x.shape[0], x.shape[1], -1, 2)
            x = torch.einsum('bldmn,bldn->bldm', m, x)
            x = x.reshape(x.shape[0], x.shape[1], -1)

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


class LinearAttention(nn.Module):
    def __init__(self, args: ModelArgs, rope: RoPE):
        super().__init__()

        self.args = args

        self.rope = rope

        self.att_coef = 1 / math.sqrt(self.args.dim)

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
            out_features=self.args.dim,
            bias = False
        )

        self.wo = nn.Linear(
            in_features=self.args.dim,
            out_features=self.args.dim,
            bias = False
        )

    def recurrent_forward(self, x, step_count, kv_state_t: torch.Tensor, phik_state_t: torch.Tensor):
        # new_embedding: (B, 1, D), modified in place
        
        pos = torch.tensor([step_count], device=x.device).broadcast_to(x.shape[0], 1)

        q = self.rope(self.wq(x), pos).squeeze(1)
        phi_q_tp1 = F.elu(q) + 1

        k = self.rope(self.wk(x), pos).squeeze(1)
        phi_k_tp1 = F.elu(k) + 1 # (B, D)

        v_tp1 = self.wv(x).squeeze(1) # (B, D)

        phik_state_tp1 = phik_state_t + phi_k_tp1
        kv_state_tp1 = kv_state_t + torch.einsum('bi,bj->bij', phi_k_tp1, v_tp1)

        den = torch.einsum('bi,bi->b', phi_q_tp1, phik_state_tp1)
        
        out = torch.einsum('bi,bij,b->bj', phi_q_tp1, kv_state_tp1, 1/(den + 1e-7))

        x = self.wo(out).unsqueeze(1)

        phik_state_t.copy_(phik_state_tp1)
        kv_state_t.copy_(kv_state_tp1)
        
        return x





    def forward(self, x, mask = None):
        # mask: 0/1 matrix or None
        # x : (b, l, d)

        q = self.rope(self.wq(x))
        phi_q = F.elu(q) + 1

        k = self.rope(self.wk(x))
        phi_k = F.elu(k) + 1

        v = self.wv(x)

        A = torch.einsum('bix,bjx->bij', phi_q, phi_k) # (B, L, L)

        if mask is not None:
            A = A * mask

        out = torch.einsum('bij,bjd-> bid', A, v)
        den = A.sum(dim=-1, keepdim=True)
        x = out / (den + 1e-7)

        x = self.wo(x)

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

        self.activation = nn.ReLU()

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
        self.att = LinearAttention(self.args, rope)
        self.rmsnorm2 = RMSNorm(self.args)
        self.fnn = FNN(self.args)

    def recurrent_forward(self, x, step_count, kv_state_t: torch.Tensor, phik_state_t: torch.Tensor):
        dx = self.rmsnorm1(x)
        dx = self.att.recurrent_forward(dx, step_count, kv_state_t, phik_state_t)
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


class Kath20LAHead(nn.Module):
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


class Kath20LA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args
        
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.dim)
        self.rope = RoPE(self.args)
        self.blocks = nn.ModuleList([Blocks(self.args, self.rope) for _ in range(self.args.n_layers)])
        self.head = Kath20LAHead(self.args)

    def recurrent_forward(self, x, step_count, kv_state_t: torch.Tensor | None, phik_state_t: torch.Tensor | None):
        B, _ = x.shape
        D = self.args.dim

        if kv_state_t is None:
            kv_state_t = torch.zeros(B, self.args.n_layers, D, D, device=x.device)
        if phik_state_t is None:
            phik_state_t = torch.zeros(B, self.args.n_layers, D, device=x.device)

        x = self.embedding(x)
        for i, block in enumerate(self.blocks): # type: ignore
            block: LinearAttention
            x = block.recurrent_forward(x, step_count, kv_state_t[:, i, ...], phik_state_t[:, i, ...])

        x = self.head(x)

        return x, kv_state_t, phik_state_t

    def forward(self, x, mask = None):

        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.head(x)

        return x