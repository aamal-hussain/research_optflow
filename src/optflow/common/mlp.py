from torch import nn


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, multiplier=4):
        super().__init__()

        hidden_channels = in_channels * multiplier
        self.c_fc = nn.Linear(in_channels, hidden_channels)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_channels, out_channels)


    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
