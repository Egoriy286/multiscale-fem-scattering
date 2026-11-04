"""Compatibility imports; implementations live in the method packages."""
from ._layers import _ModeBlock, _pair
from .chebyshev.model import ChebyshevSNO, chebyshev_matrices
from .fno.model import FNO2d, SpectralConv2d
from .sno.model import SVDSNO

__all__ = ["SVDSNO", "ChebyshevSNO", "FNO2d", "SpectralConv2d", "chebyshev_matrices"]
