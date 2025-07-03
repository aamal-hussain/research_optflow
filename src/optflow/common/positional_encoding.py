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

        frequencies = frequencies.view(1, 1, -1)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dims = in_channels * (2 * num_freqs + int(include_input))

    def forward(self, x):
        embed = x.unsqueeze(-1) * self.frequencies
        embed = torch.cat([embed.sin(), embed.cos()], dim=-2)
        embed = embed.flatten(-2)

        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)
        return embed
