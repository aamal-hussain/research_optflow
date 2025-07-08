import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from optflow.common.positional_encoding import PositionalEncoding
from optflow.dora.attention import ResidualCrossAttentionBlock


class DoraDecoder(nn.Module):
    def __init__(
        self,
        out_dims: int,
        width: int,
        num_heads: int,
        num_freqs: int,
        include_pi: bool,
        qkv_bias: bool,
        use_checkpoint: bool,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.embedder = PositionalEncoding(
            num_freqs=num_freqs, in_channels=3, include_pi=include_pi
        )

        self.query_proj = nn.Linear(self.embedder.out_dims, width)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            in_channels=width,
            query_channels=width,
            inner_product_channels=width // num_heads,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint,
        )

        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_dims)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = (
            checkpoint(self._forward, queries, latents, use_reentrant=False)
            if self.use_checkpoint
            else self._forward(queries, latents)
        )
        return logits
