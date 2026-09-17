from typing import Optional, Callable, List, Any

import numpy as np
import torch
import torchvision.ops
from einops import rearrange
from torch import nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch import nn, Tensor
import torch.fx
from torch.nn.init import trunc_normal_
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
'''可学习邻接矩阵'''

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout, lam, adj, device):
        super().__init__()
        adj = F.pad(adj, (1, 0, 1, 0), "constant", 0.0)
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.lam = lam
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.aptadj = nn.Parameter(adj, requires_grad=True).to(device)
        self.attend1 = nn.Softmax(dim=-1)
        self.attend2 = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        adj = torch.softmax(torch.relu(self.aptadj), dim=-1)
        attn = (1.0 - self.lam) * self.attend1(dots) + self.lam * adj[None, None, :, :]
        attn = self.attend2(attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class DVGTransformer(nn.Module):
    def __init__(self, node, time, depth, heads, dim_head, mlp_dim, dropout, tadj, sadj, device):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(node, Attention(dim=node, heads=heads, dim_head=node*dim_head, dropout=dropout, lam=0.5, adj=tadj, device=device)),
                PreNorm(node, FeedForward(node, node*mlp_dim, dropout=dropout)),
                PreNorm(time, Attention(dim=time, heads=heads, dim_head=time * dim_head, dropout=dropout, lam=0.5, adj=sadj, device=device)),
                PreNorm(time, FeedForward(time, time * mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn1, ff1, attn2, ff2 in self.layers:
            x = attn1(x) + x
            x = ff1(x) + x
            x = x.permute(0, 2, 1)
            x = attn2(x) + x
            x = ff2(x) + x
            x = x.permute(0, 2, 1)
        return x


class DVGTransformerNet(nn.Module):

    def __init__(self,
                 tadj,
                 sadj,
                 device,
                 node=14,
                 time=30,
                 depth=6,
                 heads=4,
                 dim_head=8,
                 mlp_dim=4,
                 pool='cat',
                 dropout=0.1,
                 emb_dropout=0.1):
        super().__init__()
        assert pool in {'cls', 'cat'}, 'pool type must be either cls (cls token) or cat'
        self.to_time_embedding = nn.Sequential(
            nn.Linear(time, time)
        )
        self.to_node_embedding = nn.Sequential(
            nn.Linear(node, node)
        )
        self.tpre_token = nn.Parameter(torch.randn(1, 1, node))
        self.spre_token = nn.Parameter(torch.randn(1, time+1, 1))

        self.pos_embedding = nn.Parameter(torch.randn(1, time+1, node+1))
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.DVGtransformer = DVGTransformer(node+1, time+1, depth, heads, dim_head, mlp_dim, dropout, tadj, sadj, device)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.fc = nn.Sequential(nn.Linear((time+1)*(node+1), 100), nn.GELU())
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Sequential(nn.Linear(100, 1))

    def forward(self, input):
        """Forward function.input: B, N, T"""
        x = input.permute(0, 2, 1)
        b, t, n = x.shape
        x = self.to_node_embedding(x)
        x = x.permute(0, 2, 1)
        x = self.to_time_embedding(x)
        x = x.permute(0, 2, 1)

        tpre_tokens = repeat(self.tpre_token, '1 1 n -> b 1 n', b=b)
        spre_tokens = repeat(self.spre_token, '1 t 1 -> b t 1', b=b)
        x = torch.cat((tpre_tokens, x), dim=1)
        x = torch.cat((spre_tokens, x), dim=2)

        x += self.pos_embedding
        x = self.emb_dropout(x)

        x = self.DVGtransformer(x)
        x = x.contiguous().view(b, -1) if self.pool == 'cat' else x[:, 0, :].contiguous().view(b, -1)

        x = self.to_latent(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

