import logging
from enum import Enum

import torch
from torch import nn
from torch.distributions import Normal

from optflow.dora.attention import Perceiver
from optflow.dora.decoder import DoraDecoder
from optflow.dora.encoder import DoraEncoder

LOGGER = logging.getLogger(__name__)


class VAEMode(Enum):
    """Enumeration for different inference modes of the DoraVAE model."""

    DEFAULT = "default"
    ENCODER = "encoder"
    DECODER = "decoder"

    def __str__(self):
        """Return the string representation of the enum value."""
        return self.value

    def __repr__(self):
        """Return the string representation of the enum value."""
        return self.value


class DoraVAE(nn.Module):
    """DoraVAE model, adapted from the original Dora implementation."""

    def __init__(
        self,
        point_feature_channels: int,
        latent_sequence_length: int,
        embed_dim: int,
        out_dims: int,
        width: int,
        num_heads: int,
        num_freqs: int,
        include_pi: bool,
        encoder_depth: int,
        decoder_depth: int,
        qkv_bias: bool,
        use_checkpoint: bool,
        learn_var: bool,
        use_sdpa: bool,
    ):
        super().__init__()

        self.mode = VAEMode.DEFAULT
        self.encoder = DoraEncoder(
            point_feature_channels=point_feature_channels,
            latent_sequence_length=latent_sequence_length,
            width=width,
            num_heads=num_heads,
            depth=encoder_depth,
            num_freqs=num_freqs,
            include_pi=include_pi,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint,
            use_sdpa=use_sdpa,
        )

        self.pre_kl = nn.Linear(width, embed_dim * 2)

        self.post_kl = nn.Linear(embed_dim, width)

        self.transformer = Perceiver(
            in_channels=width,
            inner_product_channels=width // num_heads,
            num_heads=num_heads,
            depth=decoder_depth,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint,
            use_sdpa=use_sdpa,
        )

        self.decoder = DoraDecoder(
            out_dims=out_dims,
            width=width,
            num_heads=num_heads,
            num_freqs=num_freqs,
            include_pi=include_pi,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint,
            use_sdpa=use_sdpa,
        )

        if learn_var:
            self.var = nn.Parameter(torch.empty(1))
            nn.init.constant(self.var, -3.0)

    def encoder_mode(self):
        """Set the model to encoder mode."""
        self.mode = VAEMode.ENCODER

    def decoder_mode(self):
        """Set the model to decoder mode."""
        self.mode = VAEMode.DECODER

    def default_mode(self):
        """Set the model to default mode."""
        self.mode = VAEMode.DEFAULT

    @classmethod
    def from_pretrained_checkpoint(cls, checkpoint_path: str, use_sdpa: bool = True) -> "DoraVAE":
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model = cls(
            point_feature_channels=3,
            latent_sequence_length=2048,
            embed_dim=64,
            out_dims=1,
            width=768,
            num_heads=12,
            num_freqs=8,
            include_pi=False,
            encoder_depth=8,
            decoder_depth=16,
            qkv_bias=False,
            use_checkpoint=True,
            learn_var=False,
            use_sdpa=use_sdpa,
        )
        model.load_state_dict(state_dict)
        return model

    def forward(
        self,
        coarse_pc: torch.Tensor | None = None,
        coarse_feats: torch.Tensor | None = None,
        sharp_pc: torch.Tensor | None = None,
        sharp_feats: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        query_points: torch.Tensor | None = None,
        sample_posterior: bool = False,
    ) -> torch.Tensor:
        match self.mode:
            case VAEMode.DEFAULT:
                if coarse_pc is None or coarse_feats is None:
                    raise ValueError("coarse_pc and coarse_feats must be provided in DEFAULT mode.")

                if sharp_pc is None or sharp_feats is None:
                    raise ValueError("sharp_pc and sharp_feats must be provided in DEFAULT mode.")

                if query_points is None:
                    raise ValueError("query_points must be provided in DEFAULT mode.")

                shape_latents = self.encoder(
                    coarse_pc=coarse_pc,
                    sharp_pc=sharp_pc,
                    coarse_feats=coarse_feats,
                    sharp_feats=sharp_feats,
                )
                moments = self.pre_kl(shape_latents)
                mu, logvar = torch.chunk(moments, 2, dim=-1)
                logvar = logvar.clamp(min=-30.0, max=20.0)

                if sample_posterior:
                    std = torch.exp(0.5 * logvar)
                    sample = torch.randn_like(std)
                    z = mu + sample * std

                    post_dist = Normal(loc=mu, scale=torch.exp(0.5 * logvar))
                    prior_dist = Normal(loc=torch.zeros_like(mu), scale=torch.ones_like(logvar))
                    entropy = post_dist.log_prob(z).sum() - prior_dist.log_prob(z).sum()
                else:
                    z = mu
                    entropy = None

                z = self.post_kl(z)
                z = self.transformer(z)
                return self.decoder(query_points, z), entropy
            case VAEMode.ENCODER:
                if coarse_pc is None or coarse_feats is None:
                    raise ValueError("coarse_pc and coarse_feats must be provided in ENCODER mode.")
                if sharp_pc is None or sharp_feats is None:
                    raise ValueError("sharp_pc and sharp_feats must be provided in ENCODER mode.")
                if query_points is not None or latents is not None:
                    LOGGER.warning(
                        "query_points and latents are ignored in ENCODER mode. "
                        "Only coarse_pc, coarse_feats, sharp_pc, and sharp_feats are used."
                    )

                shape_latents = self.encoder(
                    coarse_pc=coarse_pc,
                    sharp_pc=sharp_pc,
                    coarse_feats=coarse_feats,
                    sharp_feats=sharp_feats,
                )
                moments = self.pre_kl(shape_latents)
                return moments

            case VAEMode.DECODER:
                if latents is None:
                    raise ValueError("latents must be provided in DECODER mode.")
                if query_points is None:
                    raise ValueError("query_points must be provided in DECODER mode.")

                if (
                    coarse_pc is not None
                    or coarse_feats is not None
                    or sharp_pc is not None
                    or sharp_feats is not None
                ):
                    LOGGER.warning(
                        "coarse_pc, coarse_feats, sharp_pc, and sharp_feats are ignored in "
                        "DECODER mode. Only latents and query_points are used."
                    )
                x = self.post_kl(latents)
                x = self.transformer(x)
                return self.decoder(query_points, x)

            case _:
                raise ValueError(
                    f"Unsupported inference mode: {self._mode}. Supported modes are: "
                    " {InferenceMode.DEFAULT}, {InferenceMode.ENCODER}, {InferenceMode.DECODER}."
                )
