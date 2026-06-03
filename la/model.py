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
    def __init__(self, args: TransformerArgs):
        super().__init__()

        assert args.dim % 2 == 0
        self.args = args

        self.prepared_L = 0

        self.prepare_m(args.context_len)


    def prepare_m(self, L: int):

        if L > self.prepared_L:

            idxk = torch.arange(0, self.args.dim // 2) / (self.args.dim // 2)
            phase = torch.outer(torch.arange(0, L), torch.pow(10000, -idxk))
            m_sin = torch.sin(phase)    # (L, dim/2)
            m_cos = torch.cos(phase)    # (L, dim/2)

            # m : (L, dim/2, 2, 2)
            self.m = torch.stack([m_cos, m_sin, -m_sin, m_cos], dim=-1).reshape(L, self.args.dim // 2, 2, 2)
            self.m.requires_grad = False

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

        self.args = args
        self.gamma = nn.Parameter(data = torch.ones(args.dim))

    def forward(self, x):
        stddev = torch.std(x, dim=-1, keepdim=True)
        x = x / stddev * self.gamma
        return x


class SoftMaxAttention(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()

        self.args = args

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

    def forward(self, x, mask = None):
        # mask: 0/1 matrix or None
        # x : (b, l, d)

        q = self.wq(x) 
        k = self.wk(x)
        v = self.wv(x)

        # q,k,v all (b, l, d)

        a = torch.einsum('bid,bjd->bij', q, k) # (b, l, l)

        if mask is not None:
            mask_delta = (torch.ones_like(mask) - mask) * (-1e+8)
            a = a + mask_delta

        coefs = torch.softmax(a * self.att_coef, dim=-1)

        x = torch.einsum('bij,bjd->bid', coefs, v)

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
    def __init__(self, args: TransformerArgs):
        super().__init__()

        self.args = args
        self.rmsnorm1 = RMSNorm(self.args)
        self.att = SoftMaxAttention(self.args)
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

        self.l1 = nn.Linear(
            in_features=self.args.dim,
            out_features=self.args.dim * self.args.ffn_mult,
            bias = False
        )

        self.activation = nn.ReLU()

        self.l2 = nn.Linear(
            in_features=self.args.dim * self.args.ffn_mult,
            out_features=self.args.vocab_size,
            bias = False
        )

    def forward(self, x):
        # (B, L, D)
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()

        self.args = args
        
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.dim)
        self.rope = RoPE(self.args)
        self.blocks = [Blocks(self.args) for i in range(self.args.n_layers)]
        self.head = TransformerHead(self.args)

    def forward(self, x, mask = None):

        x = self.embedding(x)
        x = self.rope(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.head(x)

        return x