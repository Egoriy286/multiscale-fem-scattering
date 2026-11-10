"""Coefficient mixing and shape validation for the spectral operators."""
from numbers import Integral

from torch import nn
from torch.nn import functional as F


def _pair(value):
    values = (value, value) if isinstance(value, Integral) else tuple(value)
    if len(values) != 2:
        raise ValueError("modes must be a positive integer or a pair of positive integers")
    return tuple(_positive_int(n, "modes") for n in values)


def _positive_int(value, name, *, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)



class _ModeBlock(nn.Module):
    """Legacy separable block, retained for saved v1 SNO/Chebyshev weights."""
    def __init__(self, modes, width):
        super().__init__()
        self.channels = nn.Linear(width, width)
        self.axes = nn.ModuleList(nn.Linear(n, n, bias=False) for n in modes)

    def forward(self, x):
        mixed = self.channels(x)
        for axis, layer in enumerate(self.axes, start=1):
            mixed = layer(mixed.movedim(axis, -1)).movedim(-1, axis)
        return x + F.gelu(mixed)


class CoefficientBlock(nn.Module):
    """Parallel per-mode and global mode/channel branches in coefficient space.

    For a table Z this computes Z + GELU(Z W_local + b + B Z W_channel).
    B acts on mode positions; it is unrelated to the spatial POD decoder.
    """

    def __init__(self, modes, width):
        super().__init__()
        self.local = nn.Linear(width, width)
        self.channels = nn.Linear(width, width, bias=False)
        self.modes = nn.Linear(modes, modes, bias=False)

    def forward(self, x):
        mixed = self.modes(self.channels(x).transpose(-2, -1)).transpose(-2, -1)
        return x + F.gelu(self.local(x) + mixed)

