import logging
from enum import Enum

import torch
from torch import nn

from optflow.dora.attention import Perceiver
from optflow.dora.decoder import DoraDecoder
from optflow.dora.encoder import DoraEncoder

LOGGER = logging.getLogger(__name__)


class InferenceMode(Enum):
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
        out_dim: int,
        width: int,
        num_heads: int,
        num_freqs: int,
        include_pi: bool,
        encoder_depth: int,
        decoder_depth: int,
        qkv_bias: bool,
        use_checkpoint: bool,
    ):
        super().__init__()

        self.mode = InferenceMode.DEFAULT
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
        )

        self.decoder = DoraDecoder(
            out_dim=out_dim,
            width=width,
            num_heads=num_heads,
            num_freqs=num_freqs,
            include_pi=include_pi,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint,
        )

    def encoder_mode(self):
        """Set the model to encoder mode."""
        self.mode = InferenceMode.ENCODER

    def decoder_mode(self):
        """Set the model to decoder mode."""
        self.mode = InferenceMode.DECODER

    def default_mode(self):
        """Set the model to default mode."""
        self.mode = InferenceMode.DEFAULT

    @classmethod
    def from_pretrained_checkpoint(
        cls,
        checkpoint_path: str,
    ) -> "DoraVAE":
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model = cls(
            point_feature_channels=3,
            latent_sequence_length=2048,
            embed_dim=64,
            out_dim=1,
            width=768,
            num_heads=12,
            num_freqs=8,
            include_pi=False,
            encoder_depth=8,
            decoder_depth=16,
            qkv_bias=False,
            use_checkpoint=True,
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
    ) -> torch.Tensor:
        match self.mode:
            case InferenceMode.DEFAULT:
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
                std = torch.exp(0.5 * logvar)
                sample = torch.randn_like(std)
                z = mu + sample * std
                z = self.post_kl(z)
                x = self.transformer(z)
                return self.decoder(query_points, x)
            case InferenceMode.ENCODER:
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

            case InferenceMode.DECODER:
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
                        "coarse_pc, coarse_feats, sharp_pc, and sharp_feats are ignored in DECODER mode. "
                        "Only latents and query_points are used."
                    )
                x = self.post_kl(latents)
                x = self.transformer(x)
                return self.decoder(query_points, x)

            case _:
                raise ValueError(
                    f"Unsupported inference mode: {self._mode}. Supported modes are: "
                    " {InferenceMode.DEFAULT}, {InferenceMode.ENCODER}, {InferenceMode.DECODER}."
                )
