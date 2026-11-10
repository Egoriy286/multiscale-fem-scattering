"""Tensor Chebyshev transforms and coefficient-space neural operator."""
import math

import torch
from torch import nn

from .._layers import _ModeBlock, _pair, _positive_int


def chebyshev_matrices(size, *, dtype=torch.float32, device=None):
    """Evaluation and exact DCT-I analysis on ascending Lobatto nodes.

    Coefficients use f(x)=sum_k a_k T_k(x), with no implicit half factors.
    This is a Chebyshev-weighted discrete transform, not Euclidean orthogonal.
    """
    size = _positive_int(size, "Chebyshev size")
    if size < 2:
        raise ValueError("Chebyshev transforms require at least two Lobatto nodes")
    if not dtype.is_floating_point:
        raise ValueError("Chebyshev transforms require a floating-point dtype")
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
        input_channels = _positive_int(input_channels, "input_channels")
        width = _positive_int(width, "width")
        depth = _positive_int(depth, "depth")
        self.modes = _pair(modes)
        self.lift = nn.Linear(input_channels, width)
        self.blocks = nn.ModuleList(_ModeBlock(self.modes, width) for _ in range(depth))
        self.project = nn.Linear(width, 2)
        self.register_buffer("transform_version", torch.tensor(1, dtype=torch.int64))

    def forward(self, values):
        if values.ndim != 4 or min(values.shape[1:3]) < 2:
            raise ValueError("Chebyshev values need [batch, height>=2, width>=2, channels]")
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

