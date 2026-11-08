"""Versioned coefficient-space SNO with separate input/output POD bases."""
import torch
from torch import nn
from torch.nn import functional as F

from .._layers import CoefficientBlock, _ModeBlock, _positive_int


class SVDSNO(nn.Module):
    """Map material POD coefficients and wave parameters to output coefficients.

    v2 uses tokens [A_j, n_j, cos(theta), sin(theta), scaled_k]. The two
    material coefficient sequences are padded independently, never resampled
    or identified with the output POD basis. ``input_ranks`` specifies their
    lengths in the feature vector. v1 preserves the old dense-lift network for
    loading existing checkpoints; it is not silently upgraded.
    """
    def __init__(self, input_dim, output_modes, width=64, depth=4, *,
                 input_ranks=None, architecture_version=2):
        super().__init__()
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.output_modes = _positive_int(output_modes, "output_modes")
        self.width = _positive_int(width, "width")
        depth = _positive_int(depth, "depth")
        if architecture_version not in (1, 2):
            raise ValueError("SNO architecture_version must be 1 or 2")
        self.architecture_version = architecture_version
        if architecture_version == 1:
            self.token_modes = self.output_modes
            self.input_ranks = None
            self.lift = nn.Linear(self.input_dim, self.output_modes * self.width)
            self.blocks = nn.ModuleList(_ModeBlock((self.output_modes,), self.width)
                                        for _ in range(depth))
        else:
            if input_ranks is None and self.input_dim == 3:
                input_ranks = (0, 0)
            if input_ranks is None or len(input_ranks) != 2:
                raise ValueError("SNO v2 requires input_ranks=(rank_A, rank_n)")
            self.input_ranks = tuple(_positive_int(r, "input rank", allow_zero=True)
                                     for r in input_ranks)
            if sum(self.input_ranks) + 3 != self.input_dim:
                raise ValueError("input_dim must equal rank_A + rank_n + 3")
            self.token_modes = max(self.output_modes, *self.input_ranks)
            self.lift = nn.Linear(5, self.width)
            self.blocks = nn.ModuleList(CoefficientBlock(self.token_modes, self.width)
                                        for _ in range(depth))
        self.project = nn.Linear(width, 2)

    def coefficient_tokens(self, features):
        """Build v2 tokens without losing input modes when output rank is small."""
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [batch, {self.input_dim}]")
        if self.architecture_version != 2:
            raise ValueError("coefficient tokens are defined for SNO v2 only")
        rank_a, rank_n = self.input_ranks
        a = F.pad(features[:, :rank_a], (0, self.token_modes - rank_a))
        n = F.pad(features[:, rank_a:rank_a + rank_n], (0, self.token_modes - rank_n))
        globals_ = features[:, -3:, None].transpose(1, 2).expand(-1, self.token_modes, -1)
        return torch.cat((a[..., None], n[..., None], globals_), dim=-1)

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [batch, {self.input_dim}]")
        if self.architecture_version == 1:
            x = self.lift(features).reshape(-1, self.output_modes, self.width)
        else:
            x = F.gelu(self.lift(self.coefficient_tokens(features)))
        for block in self.blocks:
            x = block(x)
        return self.project(x[:, :self.output_modes])

