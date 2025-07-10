import math

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint


class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        query_channels: int,
        inner_product_channels: int,
        num_heads: int,
        qkv_bias: bool = False,
        use_checkpoint: bool = True,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.c_q = nn.Linear(query_channels, num_heads * inner_product_channels, bias=qkv_bias)
        self.c_kv = nn.Linear(in_channels, num_heads * inner_product_channels * 2, bias=qkv_bias)
        # Output projection yields query_channels so that the residual connection works.
        self.c_proj = nn.Linear(num_heads * inner_product_channels, query_channels)
        self.attention = QKVMultiheadCrossAttention(num_heads=num_heads, use_sdpa=use_sdpa)

    def forward(self, queries, x):
        q = self.c_q(queries)
        kv = self.c_kv(x)
        if self.use_checkpoint:
            x = checkpoint(self.attention, q, kv, use_reentrant=False)
        else:
            x = self.attention(q, kv)
        x = self.c_proj(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, num_heads: int, use_sdpa: bool):
        super().__init__()
        self.num_heads = num_heads
        self.use_sdpa = use_sdpa

    def forward(self, q, kv):
        _, m, width = q.shape
        bs, n, _ = kv.shape
        q = q.view(bs, m, self.num_heads, -1)
        kv = kv.view(bs, n, self.num_heads, -1)
        k, v = kv.chunk(2, dim=-1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.use_sdpa:
            out = scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, m, -1)
        else:
            inner_product_channels = width // self.num_heads
            scale = 1 / math.sqrt(inner_product_channels)
            weight = torch.einsum("bhmc,bhnc->bhmn", q, k * scale)
            weight = weight.softmax(dim=-1)
            out = torch.einsum("bhmn,bhnc->bhmc", weight, v).permute(0, 2, 1, 3).reshape(bs, m, -1)

        return out
