"""Local neural operators for the Helmholtz scattering experiments.

Numerical solvers and training are imported explicitly so loading a dataset
never requires PyTorch, Gmsh, or DOLFINx.
"""

from ._runtime import prepare_native_runtime

prepare_native_runtime()
