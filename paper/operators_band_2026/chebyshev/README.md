# Chebyshev SNO

Run from the repository root after the experiment data are ready:

```sh
VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python paper/operators_band_2026/spectral.py chebyshev
```

The decoder uses the tensor basis `T_j(x/2) T_l(y/2)`, with
`0 <= j,l < 16` and no periodic extension. Its coefficients come from
the FEM mass-weighted normal equations evaluated on the unstructured mesh.
This adapts the polynomial representation in [Fanaskov and Oseledets](https://arxiv.org/html/2205.10573v2)
to the native FEM field and metric, with no Chebyshev-grid resampling.

The network, parameter lift, training loss and split sizes match the
[Fourier-cosine run](../fourier/README.md). Only the output spatial basis
and the corresponding projection change.
