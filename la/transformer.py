from dataclasses import dataclass

import torch
from torch import nn

import math


@dataclass
class TransformerArgs:
    vocab_size: int = 256
    dim: int = 512
    ffn_mult: int = 4
    n_layers: int = 8
    context_len : int = 128


class RoPE(nn.Module):
    m: torch.Tensor   # declares the registered buffer's type so Pylance treats self.m as a Tensor

    def __init__(self, args: TransformerArgs):
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


    def forward(self, x: torch.Tensor):
        
        # x : (b, l, d)

        self.prepare_m(x.shape[1])

        # slice the matrix
        m = self.m[:x.shape[1], ...]

        x = x.reshape(x.shape[0], x.shape[1], -1, 2)
        x = torch.einsum('ldmn,bldn->bldm', m, x)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        return x
    
class RMSNorm(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()

        self.eps = 1e-8
        self.args = args
        self.gamma = nn.Parameter(data = torch.ones(args.dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms * self.gamma
        return x


class SoftMaxAttention(nn.Module):
    def __init__(self, args: TransformerArgs, rope: RoPE):
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


    def forward(self, x, mask = None):
        # mask: 0/1 matrix or None
        # x : (b, l, d)

        q = self.rope(self.wq(x))
        k = self.rope(self.wk(x))
        v = self.wv(x)

        # q,k,v all (b, l, d)

        a = torch.einsum('bid,bjd->bij', q, k) # (b, l, l)

        if mask is not None:
            mask_delta = (torch.ones_like(mask, device=x.device, dtype = x.dtype) - mask) * (-1e+8)
            a = a + mask_delta

        coefs = torch.softmax(a * self.att_coef, dim=-1)

        x = torch.einsum('bij,bjd->bid', coefs, v)

        x = self.wo(x)

        return x


class FNN(nn.Module):
    def __init__(self, args: TransformerArgs):
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
    def __init__(self, args: TransformerArgs, rope: RoPE):
        super().__init__()

        self.args = args
        self.rmsnorm1 = RMSNorm(self.args)
        self.att = SoftMaxAttention(self.args, rope)
        self.rmsnorm2 = RMSNorm(self.args)
        self.fnn = FNN(self.args)
    
    def forward(self, x, mask=None):
        dx = self.rmsnorm1(x)
        dx = self.att(dx, mask)
        x = x + dx

        dx = self.rmsnorm2(x)
        dx = self.fnn(dx)
        x = x + dx
        
        return x


class TransformerHead(nn.Module):
    def __init__(self, args: TransformerArgs):
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


class Transformer(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()

        self.args = args
        
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.dim)
        self.rope = RoPE(self.args)
        self.blocks = nn.ModuleList([Blocks(self.args, self.rope) for _ in range(self.args.n_layers)])
        self.head = TransformerHead(self.args)

    def forward(self, x, mask = None):

        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.head(x)

        return x