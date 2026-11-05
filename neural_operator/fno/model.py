"""Fourier neural operator with local spatial activations."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .._layers import _pair


class SpectralConv2d(nn.Module):
    """Signed first-axis and nonnegative second-axis rFFT low frequencies."""
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
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
            out[:, :, -negative:, :second] = torch.einsum("bixy,ioxy->boxy", spectrum[:, :, -negative:, :second], self.weight_negative[:, :, :negative, :second])
        return torch.fft.irfft2(out, s=(h, w), norm="ortho")



class FNO2d(nn.Module):
    def __init__(self, input_channels=7, modes=16, width=64, depth=4, padding=8):
        super().__init__()
        self.padding = int(padding)
        if self.padding < 0:
            raise ValueError("padding must be nonnegative")
        self.lift = nn.Linear(input_channels, width)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes) for _ in range(depth))
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(depth))
        self.project = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))

    def forward(self, values):
        h, w = values.shape[1:3]
        x = self.lift(values).permute(0, 3, 1, 2)
        if self.padding:
            x = F.pad(x, (0, self.padding, 0, self.padding))
        for spectral, local in zip(self.spectral, self.local):
            x = x + F.gelu(spectral(x) + local(x))
        return self.project(x[:, :, :h, :w].permute(0, 2, 3, 1))

