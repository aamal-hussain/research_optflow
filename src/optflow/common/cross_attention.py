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
    ):
        super().__init__()
        self.c_q = nn.Linear(query_channels, num_heads * inner_product_channels, bias=qkv_bias)
        self.c_kv = nn.Linear(in_channels, num_heads * inner_product_channels * 2, bias=qkv_bias)
        # Output projection yields query_channels so that the residual connection works.
        self.c_proj = nn.Linear(num_heads * inner_product_channels, query_channels)
        self.attention = QKVMultiheadCrossAttention(num_heads=num_heads)

    def forward(self, queries, x):
        q = self.c_q(queries)
        kv = self.c_kv(x)
        x = checkpoint(self.attention, q, kv, use_reentrant=False)
        x = self.c_proj(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, kv):
        _, m, _ = q.shape
        bs, n, _ = kv.shape
        q = q.view(bs, m, self.num_heads, -1)
        kv = kv.view(bs, n, self.num_heads, -1)
        k, v = kv.chunk(2, dim=-1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        out = scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, m, -1)

        return out
