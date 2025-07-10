import math

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint


class SelfAttention(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        inner_product_channels: int,
        num_heads: int,
        qkv_bias: bool = False,
        use_checkpoint: bool = True,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.c_qkv = nn.Linear(in_channels, num_heads * inner_product_channels * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(num_heads * inner_product_channels, in_channels)
        self.attention = QKVMultiheadAttention(num_heads=num_heads, use_sdpa=use_sdpa)

    def forward(self, x):
        qkv = self.c_qkv(x)
        if self.use_checkpoint:
            x = checkpoint(self.attention, qkv, use_reentrant=False)
        else:
            x = self.attention(qkv)
        x = self.c_proj(x)
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, num_heads: int, use_sdpa: bool):
        super().__init__()
        self.num_heads = num_heads
        self.use_sdpa = use_sdpa

    def forward(self, qkv):
        bs, seq_length, width = qkv.shape
        qkv = qkv.view(bs, seq_length, self.num_heads, -1)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.use_sdpa:
            out = (
                scaled_dot_product_attention(q, k, v)
                .permute(0, 2, 1, 3)
                .reshape(bs, seq_length, -1)
            )
        else:
            inner_product_channels = width // self.num_heads
            scale = 1 / math.sqrt(inner_product_channels)
            weight = torch.einsum("bhmc,bhnc->bhmn", q, k * scale)
            weight = weight.softmax(dim=-1)
            out = (
                torch.einsum("bhmn,bhnc->bhmc", weight, v)
                .permute(0, 2, 1, 3)
                .reshape(bs, seq_length, -1)
            )

        return out
