"""PyTorch coefficient-space SNOs and a spatial-activation Fourier operator."""
from __future__ import annotations
import math
import torch
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


def chebyshev_matrices(size, *, dtype=torch.float32, device=None):
    """Evaluation and exact DCT-I analysis on ascending Lobatto nodes.

    Coefficients use f(x)=sum_k a_k T_k(x), with no implicit half factors.
    This is a Chebyshev-weighted discrete transform, not Euclidean orthogonal.
    """
    j = torch.arange(size, dtype=dtype, device=device)
    theta = math.pi * (1 - j / (size - 1))
    synthesis = torch.cos(theta[:, None] * j[None, :])
    endpoint = torch.ones(size, dtype=dtype, device=device)
    endpoint[0] = endpoint[-1] = 0.5
    analysis = (2 / (size - 1)) * endpoint[:, None] * synthesis.T * endpoint[None, :]
    return analysis, synthesis


class ChebyshevSNO(nn.Module):
    """Nonlinear mixing of retained tensor Chebyshev coefficients."""
    def __init__(self, input_channels=7, modes=16, width=64, depth=4):
        super().__init__()
        self.modes = _pair(modes)
        self.lift = nn.Linear(input_channels, width)
        self.blocks = nn.ModuleList(_ModeBlock(self.modes, width) for _ in range(depth))
        self.project = nn.Linear(width, 2)
        self.register_buffer("transform_version", torch.tensor(1, dtype=torch.int64))

    def forward(self, values):
        h, w = values.shape[1:3]
        mh, mw = self.modes
        if mh > h or mw > w:
            raise ValueError("Chebyshev modes must not exceed grid dimensions")
        ay, sy = chebyshev_matrices(h, dtype=values.dtype, device=values.device)
        ax, sx = chebyshev_matrices(w, dtype=values.dtype, device=values.device)
        coefficients = torch.einsum("ih,bhwc,jw->bijc", ay[:mh], values, ax[:mw])
        x = self.lift(coefficients)
        for block in self.blocks:
            x = block(x)
        coefficients = self.project(x)
        return torch.einsum("hi,bijc,wj->bhwc", sy[:, :mh], coefficients, sx[:, :mw])


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
