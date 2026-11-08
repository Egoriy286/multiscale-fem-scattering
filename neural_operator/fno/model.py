"""Fourier neural operator with local spatial activations."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .._layers import _pair, _positive_int


class SpectralConv2d(nn.Module):
    """Signed first-axis and nonnegative second-axis rFFT low frequencies."""
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        in_channels = _positive_int(in_channels, "in_channels")
        out_channels = _positive_int(out_channels, "out_channels")
        self.modes = _pair(modes)
        shape = (in_channels, out_channels, *self.modes)
        scale = 1 / math.sqrt(in_channels * out_channels)
        self.weight_positive = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_negative = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.out_channels = out_channels

    def forward(self, x):
        h, w = x.shape[-2:]
        spectrum = torch.fft.rfft2(x, norm="ortho")
        out = spectrum.new_zeros(x.shape[0], self.out_channels, h, w // 2 + 1)
        positive = min(self.modes[0], (h + 1) // 2)
        negative = min(self.modes[0], h // 2)
        second = min(self.modes[1], w // 2 + 1)
        out[:, :, :positive, :second] = torch.einsum("bixy,ioxy->boxy", spectrum[:, :, :positive, :second], self.weight_positive[:, :, :positive, :second])
        if negative:
            # The stored negative axis is [-m, ..., -1]. If a smaller grid
            # clips it, keep its tail so the weight for frequency -1 stays -1.
            out[:, :, -negative:, :second] = torch.einsum("bixy,ioxy->boxy", spectrum[:, :, -negative:, :second], self.weight_negative[:, :, -negative:, :second])
        return torch.fft.irfft2(out, s=(h, w), norm="ortho")



class FNO2d(nn.Module):
    def __init__(self, input_channels=7, modes=16, width=64, depth=4, padding=8):
        super().__init__()
        input_channels = _positive_int(input_channels, "input_channels")
        width = _positive_int(width, "width")
        depth = _positive_int(depth, "depth")
        self.padding = _positive_int(padding, "padding", allow_zero=True)
        modes = _pair(modes)
        self.lift = nn.Linear(input_channels, width)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes) for _ in range(depth))
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(depth))
        self.project = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))

    def forward(self, values):
        if values.ndim != 4 or min(values.shape[1:3]) < 1:
            raise ValueError("FNO values must have shape [batch, height, width, channels]")
        h, w = values.shape[1:3]
        x = self.lift(values).permute(0, 3, 1, 2)
        if self.padding:
            x = F.pad(x, (0, self.padding, 0, self.padding))
        for spectral, local in zip(self.spectral, self.local):
            x = x + F.gelu(spectral(x) + local(x))
        return self.project(x[:, :, :h, :w].permute(0, 2, 3, 1))

