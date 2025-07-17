"""Attention modules for the Dora architecture.
Cosmetic modifications from the original Dora implementation.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from optflow.common import MLP
from optflow.common.cross_attention import CrossAttention
from optflow.common.self_attention import SelfAttention


class ResidualSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        inner_product_channels: int,
        num_heads: int,
        qkv_bias: bool,
        use_checkpoint: bool,
        use_sdpa: bool,
    ):
        super().__init__()

        width = inner_product_channels * num_heads
        self.use_checkpoint = use_checkpoint

        self.attn = SelfAttention(
            in_channels=in_channels,
            inner_product_channels=inner_product_channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint,
            use_sdpa=use_sdpa,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(in_channels=width, out_channels=width)
        self.ln_2 = nn.LayerNorm(width)

    def _forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        query_channels: int,
        inner_product_channels: int,
        num_heads: int,
        qkv_bias: bool,
        use_checkpoint: bool,
        use_sdpa: bool,
    ):
        super().__init__()

        width = inner_product_channels * num_heads
        self.use_checkpoint = use_checkpoint
        self.attn = CrossAttention(
            in_channels=in_channels,
            query_channels=query_channels,
            inner_product_channels=inner_product_channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint,
            use_sdpa=use_sdpa,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP(in_channels=width, out_channels=width)
        self.ln_3 = nn.LayerNorm(width)

    def _forward(self, queries: torch.Tensor, x: torch.Tensor):
        x = queries + self.attn(self.ln_1(queries), self.ln_2(x))
        x = x + self.mlp(self.ln_3(x))
        return x

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, data, use_reentrant=False)
        else:
            return self._forward(x, data)


class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        inner_product_channels: int,
        num_heads: int,
        depth: int = 12,
        qkv_bias: bool = True,
        use_checkpoint: bool = False,
        use_sdpa: bool,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [
                ResidualSelfAttentionBlock(
                    in_channels=in_channels,
                    inner_product_channels=inner_product_channels,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    use_checkpoint=use_checkpoint,
                    use_sdpa=use_sdpa,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x
