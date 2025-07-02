import torch
from torch import nn

from optflow.common import MLP, PositionalEncoding, SelfAttention


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channels=17):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(in_channels, 2 * in_channels)
        self.ln = nn.LayerNorm(in_channels, elementwise_affine=False)

    def forward(self, x, t):
        embed = self.linear(t)
        scale, shift = torch.chunk(embed, 2, dim=-1)
        x = self.ln(x) * (1 + self.silu(scale)) + shift
        return x


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads, inner_product_channels):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.inner_product_channels = inner_product_channels

        self.attn1 = SelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            inner_product_channels=inner_product_channels,
        )
        self.attn2 = SelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            inner_product_channels=inner_product_channels,
        )
        self.norm1 = AdaptiveLayerNorm(in_channels)
        self.norm2 = AdaptiveLayerNorm(in_channels)
        self.norm3 = nn.LayerNorm(in_channels)
        self.mlp = MLP(in_channels, in_channels)

    def forward(self, x, t):
        x = self.attn1(self.norm1(x, t)) + x
        x = self.attn2(self.norm2(x, t)) + x
        x = self.mlp(self.norm3(x)) + x
        return x


class LatentTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        inner_product_channels,
        out_channels,
        num_heads,
        depth=12,
        num_freqs=8,
        include_pi=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inner_product_channels = inner_product_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        inner_dim = num_heads * inner_product_channels
        self.lifting = nn.Linear(in_channels, inner_dim, bias=False)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(inner_dim, num_heads, inner_product_channels) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(inner_dim)
        self.projection = nn.Linear(inner_dim, out_channels, bias=False)

        self.map_noise = PositionalEncoding(
            num_freqs=num_freqs, in_channels=1, include_pi=include_pi
        )
        self.time_lifting = nn.Sequential(
            nn.Linear(self.map_noise.out_dims, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
            nn.SiLU(),
        )

    def forward(self, x, t):
        t = self.map_noise(t)
        t = self.time_lifting(t).unsqueeze(1)  # Add sequence dimension
        x = self.lifting(x)
        for block in self.transformer_blocks:
            x = block(x, t)
        x = self.norm(x)
        x = self.projection(x)
        return x
