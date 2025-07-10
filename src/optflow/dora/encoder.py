"""Dora encoder, adapted from the original Dora implementation."""

import torch
from torch import nn
from torch_cluster import fps

from optflow.common import PositionalEncoding
from optflow.dora.attention import Perceiver, ResidualCrossAttentionBlock


class DoraEncoder(nn.Module):
    def __init__(
        self,
        point_feature_channels: int,
        latent_sequence_length: int,
        width: int,
        num_heads: int,
        depth: int,
        num_freqs: int,
        include_pi: bool,
        qkv_bias: bool = True,
        use_checkpoint: bool = False,
        use_sdpa: bool = True,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.latent_sequence_length = latent_sequence_length

        self.embedder = PositionalEncoding(
            num_freqs=num_freqs, in_channels=3, include_pi=include_pi
        )
        self.input_proj = nn.Linear(self.embedder.out_dims + point_feature_channels, width)
        self.input_proj1 = nn.Linear(self.embedder.out_dims + point_feature_channels, width)

        self.cross_attn = ResidualCrossAttentionBlock(
            in_channels=width,
            query_channels=width,
            inner_product_channels=width // num_heads,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint,
            use_sdpa=use_sdpa,
        )
        self.cross_attn1 = ResidualCrossAttentionBlock(
            in_channels=width,
            query_channels=width,
            inner_product_channels=width // num_heads,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint,
            use_sdpa=use_sdpa,
        )

        self.self_attn = Perceiver(
            in_channels=width,
            inner_product_channels=width // num_heads,
            num_heads=num_heads,
            depth=depth,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint,
            use_sdpa=use_sdpa,
        )

        self.ln_post = nn.LayerNorm(width)

    def forward(self, coarse_pc, sharp_pc, coarse_feats, sharp_feats):
        bs, N_coarse, D_coarse = coarse_pc.shape
        bs, N_sharp, D_sharp = sharp_pc.shape

        ### Coarse Point Cloud ###
        coarse_data = self.embedder(coarse_pc)
        coarse_data = torch.cat([coarse_data, coarse_feats], dim=-1)

        coarse_data = self.input_proj(coarse_data)
        coarse_ratio = (self.latent_sequence_length // 2) / N_coarse
        flattened = coarse_pc.view(bs * N_coarse, D_coarse)
        batch = torch.arange(bs).to(coarse_pc.device)
        batch = torch.repeat_interleave(batch, N_coarse)
        pos = flattened
        idx = fps(pos, batch, ratio=coarse_ratio)
        query_coarse = coarse_data.view(bs * N_coarse, -1)[idx].view(bs, -1, coarse_data.shape[-1])

        ### Sharp Point Cloud ###
        sharp_data = self.embedder(sharp_pc)
        sharp_data = torch.cat([sharp_data, sharp_feats], dim=-1)
        sharp_data = self.input_proj1(sharp_data)
        sharp_ratio = (self.latent_sequence_length // 2) / N_sharp
        flattened = sharp_pc.view(bs * N_sharp, D_sharp)
        batch = torch.arange(bs).to(sharp_pc.device)
        batch = torch.repeat_interleave(batch, N_sharp)
        pos = flattened
        idx = fps(pos, batch, ratio=sharp_ratio)
        query_sharp = sharp_data.view(bs * N_sharp, -1)[idx].view(bs, -1, sharp_data.shape[-1])

        query = torch.cat([query_coarse, query_sharp], dim=1)

        latents_coarse = self.cross_attn(query, coarse_data)
        latents_sharp = self.cross_attn1(query, sharp_data)
        latents = latents_coarse + latents_sharp

        latents = self.self_attn(latents)
        latents = self.ln_post(latents)

        return latents
