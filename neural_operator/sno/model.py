"""SNO mapping input features to complex output POD coefficients."""
from torch import nn

from .._layers import _ModeBlock


class SVDSNO(nn.Module):
    """Map arbitrary independent input features to complex output POD coefficients."""
    def __init__(self, input_dim, output_modes, width=64, depth=4):
        super().__init__()
        self.output_modes, self.width = output_modes, width
        self.lift = nn.Linear(input_dim, output_modes * width)
        self.blocks = nn.ModuleList(_ModeBlock((output_modes,), width) for _ in range(depth))
        self.project = nn.Linear(width, 2)

    def forward(self, features):
        x = self.lift(features).reshape(-1, self.output_modes, self.width)
        for block in self.blocks:
            x = block(x)
        return self.project(x)

