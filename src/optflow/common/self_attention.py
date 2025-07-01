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
    ):
        super().__init__()
        self.inner_product_channels = inner_product_channels
        self.c_qkv = nn.Linear(in_channels, num_heads * inner_product_channels * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(num_heads * inner_product_channels, in_channels)
        self.attention = QKVMultiheadAttention(num_heads=num_heads)

    def forward(self, x):
        qkv = self.c_qkv(x)
        x = checkpoint(self.attention, qkv, use_reentrant=False)
        x = self.c_proj(x)
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv):
        bs, seq_length, _ = qkv.shape
        qkv = qkv.view(bs, seq_length, self.num_heads, -1)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        out = scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, seq_length, -1)

        return out
