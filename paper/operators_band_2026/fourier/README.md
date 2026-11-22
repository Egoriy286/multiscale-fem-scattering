# Fourier-cosine SNO

Run from the repository root after the experiment data are ready:

```sh
VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python paper/operators_band_2026/spectral.py fourier
```

The decoder uses the real tensor basis
`cos(pi*j*(x+2)/4) cos(pi*l*(y+2)/4)`, with `0 <= j,l < 16`.
This is the cosine part of a Fourier expansion after even reflection about
the sides of the square, of period 8 in each coordinate. The reflection
allows differing values on opposite sides of the square. Finite cosine
sums have vanishing normal derivative on each side, so the basis does not
impose the physical impedance condition. The reference fields solve the
impedance problem before any projection.

The architecture follows the coefficient and channel mixing of Eq. (22),
Appendix C.1.3 of [Fanaskov and Oseledets](https://arxiv.org/html/2205.10573v2).
The learned mode map is factored along the two spectral axes. The four
blocks use residual connections, coefficient-specific bias and GELU.
Seven standardized scalars are lifted directly to a spectral feature
tensor. The two outputs at each mode are coefficients of the real and
imaginary parts. There are no nonlinear layers evaluated on physical grids.

The projection basis is evaluated at the native FEM nodes. For a basis matrix
`V` the normal equations use `G = V.T @ M @ V`, with the consistent FEM
mass matrix `M`. Targets are coefficients of `u_s/k`; the nodal decoder
multiplies by the original `k`. The mean relative loss includes the squared
projection remainder and the coefficient error quadratic form with `G`.
Every reported test error is checked against direct nodal reconstruction.
Training uses the best checkpoint by validation loss for each sample count.

After both full training runs, the separate near-zero diagnostic uses the
frozen (N=1000) checkpoints:

```sh
VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python paper/operators_band_2026/fourier/evaluate_nearzero.py
```

This reads the six queries from `data/nearzero.npz` and writes an independent
`nearzero.csv` in each model directory.
