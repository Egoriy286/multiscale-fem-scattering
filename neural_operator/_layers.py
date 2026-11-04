"""Shared coefficient mixing for the SVD and Chebyshev operators."""
from torch import nn
from torch.nn import functional as F


def _pair(value):
    return (int(value), int(value)) if isinstance(value, int) else tuple(map(int, value))



class _ModeBlock(nn.Module):
    def __init__(self, modes, width):
        super().__init__()
        self.channels = nn.Linear(width, width)
        self.axes = nn.ModuleList(nn.Linear(n, n, bias=False) for n in modes)

    def forward(self, x):
        mixed = self.channels(x)
        for axis, layer in enumerate(self.axes, start=1):
            mixed = layer(mixed.movedim(axis, -1)).movedim(-1, axis)
        return x + F.gelu(mixed)

