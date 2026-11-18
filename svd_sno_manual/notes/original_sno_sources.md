# Sources for original SNO chapter

Read on 2026-11-17:

- https://arxiv.org/html/2205.10573v2 — section 4 Eq (3), Appendix C.1.3 Eq (19), (20), (21), (22); context §2, §3.1/3.2. The page declares CC Zero.
- https://arxiv.org/pdf/2205.10573v2 — verified 16-page v2 PDF, equation numbering cross-checked against HTML.
- https://github.com/VLSF/SNO — author repository, CC0-1.0 license.
- https://raw.githubusercontent.com/VLSF/SNO/main/architectures/SNO_1D.py — NN_c right channel multiplication + broadcast row bias; NN_i left mode matrix/right channel matrix/full bias; NN composes c/i/c; vmap introduces batch. Last layer of EACH subnet skips activation (`if i < n-1`).
- https://raw.githubusercontent.com/VLSF/SNO/main/architectures/fSNO_1D.py — complex weights stored as real/imag arrays; same linear endpoint convention, complex_split_softplus.
- https://raw.githubusercontent.com/VLSF/SNO/main/architectures/SNO_2D.py — separate matrices along two modal axes, channel axis independent.
- https://raw.githubusercontent.com/VLSF/SNO/main/functions/utils.py — real softplus and complex_split_softplus definitions.

The chapter is original pedagogical prose, with own component derivations, transform conventions, two original TikZ graphs, parameter-count derivation, and manually chosen numeric example. No benchmark results reproduced. No claim of exact paper implementation by local SVD-SNO v2.

Pitfalls explicitly handled:
1. Row-broadcast bias in coefficient storage is uniform coefficient shift, not generally spatial constant; the paper's prose statement is not repeated as a mathematical identity.
2. Text after Eq21 says N2 decreases channels, but Eq20 + actual composition use final N3; chapter follows equations and code.
3. Eq22 has no residual or parallel local branch. Local v2 is distinct.
4. Schematic Eq21/22 activation differs from affine terminal layers of EACH author subnet.
5. Fourier domain [-1,1], period2, pi frequency factor; nodes starting at -1 need phase convention with FFT. Chapter convention is declared independently, not asserted identical to every author transform routine.
6. Count 51 is reported only as author's modal size; not interpreted as a fully enumerated symmetric frequency band without checking storage.
7. Complex split activation does not automatically preserve conjugate symmetry.
8. Two-dimensional formula is labeled conceptual separable representation; axis ordering of actual JAX dot implementation is not asserted without further adapter checks.

Integration: references primer:chapter, macros R C ii, environments idea/pitfall/worked; only standard TikZ shapes/paths, no extra graph packages. Parent compiles and visually checks layout. Only assigned files written.

## Repository revision and paper/code distinction

GitHub main resolved through https://api.github.com/repos/VLSF/SNO/commits/main to commit `4362226284f22f6f456d429ec4ddec5b8596e894` on this reading. File URLs above were read from main at that time; pinned base: https://github.com/VLSF/SNO/tree/4362226284f22f6f456d429ec4ddec5b8596e894 .

The observation that NN_c and NN_i skip activation on each subnet terminal layer is an observed AUTHOR CODE FACT in SNO_1D.py/fSNO_1D.py/SNO_2D.py, not a property claimed by schematic paper Eq21/Eq22. The text separates them explicitly.

Additional derivations: transparent H/C/K/beta passport avoids material A collision; exact Chebyshev dual normalization h0=pi, hj=pi/2 and weighted kernel integral are spelled out.
