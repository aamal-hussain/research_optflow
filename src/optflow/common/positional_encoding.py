import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(
        self, num_freqs: int, in_channels: int, include_input: bool = True, include_pi: bool = True
    ):
        super().__init__()
        frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dims = in_channels * (2 * num_freqs + int(include_input))

    def forward(self, x):
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)
