# Article and implementation comparison

This source audit supports the frozen sample-size study. The selected resource-limited setup is recorded in [protocol.json](protocol.json): 100 epochs maximum, batch 50, CPU SNO and MPS FNO. Completed numerical results require all 48 runs and replay verification; the source audit itself does not establish convergence.

## Primary sources

| Source | Exact relevant protocol | Meaning of a sample and limitations |
| --- | --- | --- |
| You, Xu, Cai, final MscaleFNO journal article, JCP 547 (2026), 114530, [DOI](https://doi.org/10.1016/j.jcp.2025.114530), [author-hosted reprint](https://bpb-us-w2.wpmucdn.com/people.smu.edu/dist/e/573/files/2026/04/MultiscaleFNO_reprint.pdf), pp.7,15–16 | Example 4.5: 1,000 samples, 800 training /100 validation /100 test; batch 25; 500 epochs; Adam initial lr 0.001, weight decay 1e-4; cosine annealing per minibatch to zero. Finite differences on 2001×2001, downsampled to 401×401; RTX 4090. | Each sample is a newly drawn medium perturbation and its solution. Domain [0,1]²; homogeneous Dirichlet boundary; fixed forcing. This is not the repository's fixed-medium frequency sweep. |
| [Official MscaleFNO 2D script, pinned commit](https://github.com/yyc-rgb/MscaleFNO/blob/75f12b6f55e53280e6a1baf2f6d96e84a910a5b3/fourier_2d_earlystop_mscale_helm.py) | `ntrain=800`, `nval=100`, `ntest=100`, batch 25, 500 epochs, Adam lr 0.001, weight decay 1e-4, cosine schedule stepped each minibatch. Reads `a_matrix` and `u_matrix`; first 800/next 100/last 100 split. | `r=2` subsamples a nominal 401×401 dataset to 201×201. Uses 100 modes rather than the article's 200. Script and article are distinct reproducibility targets. |
| Fanaskov, Oseledets, [Spectral Neural Operators, arXiv:2205.10573v2](https://arxiv.org/html/2205.10573), Appendix C | Each benchmark has 1,000 train and 1,000 test examples; no separate validation count specified there. 100 grid points per dimension. Adam lr 0.001; multiply lr by 0.5 each 10,000 epochs in 1D /5,000 in 2D. Batch 1,000 in 1D /100 in 2D. 1D: 50,000 epochs. 2D: approximately equal 6.7 GPU hours per method, including approximately 21,600 FNO epochs and 31,500 Fourier-SNO epochs. V100 16GB. | Samples are function/solution pairs for the paper's benchmark operators, not this scattering dataset. Errors for every method are evaluated on a common uniform grid. |
| Kalachikova et al., [GMsFEM scattering article, JCAM 424 (2023), 114977](https://doi.org/10.1016/j.cam.2022.114977), local source [PDF](../../references/10%20GMsFEM%20for%20scattering%20problem.pdf), p.9 | Deterministic fine/coarse FEM comparison; 10×10 and 20×20 coarse grids; different wavenumbers and local basis counts. No neural training, validation, epochs, or sample-count protocol. | Fine FEM is the reference. Article reports separate relative errors for real, imaginary, and absolute solution. This cannot supply an ML data split or a neural convergence theorem. |

The [available MscaleFNO arXiv v1](https://arxiv.org/html/2412.20183v1) is an earlier, materially different version: its Example 4.5 is a 1D out-of-distribution test. Use the final journal reprint for the 2D Example 4.5. The DOI contains 2025, while the final journal volume/year is 547 (2026). The official repository README's year is not authoritative for this distinction.

## Final MscaleFNO article: physics and model

The final article's Example 4.5 learns

\[
 \Delta u+(\lambda^2+c\omega(x,y))u=f(x,y),\qquad u|_{\partial\Omega}=0,
 \qquad G:(\omega,x,y)\mapsto u.
\]

Its random perturbation is `omega = xi * eta / max(abs(eta))`, with `xi` uniform on (0,1), and `eta` a sum over m,n=100,...,110 of sine/sine, sine/cosine, cosine/cosine, and cosine/sine terms with independently drawn coefficients in (-1,1). The forcing is fixed and oscillatory. One medium realization and its numerical solution constitute one sample, regardless of the number of grid nodes.

The printed Example 4.5 specifies `lambda=10` and `c=0.9*lambda^2=3.6`, an arithmetic inconsistency: 0.9×10² is 90. A faithful reproduction must resolve this against generation code or author clarification and disclose the resolution; this study does not resolve it by changing its own physical problem.

For 2D Helmholtz, the final article describes four Fourier blocks, 200 modes per spatial dimension, a width-32 ordinary FNO and eight independent width-10 multiscale branches. Reported parameter counts are 327,697,153 and 256,014,744 respectively. Its multiscale operator is

\[
 \hat u=\sum_{j=1}^{8}\gamma_j\operatorname{FNO}_{\theta_j}(c_j\omega,c_jx,c_jy),
 \quad c_j\in\{1,40,80,100,120,140,180,200\}\text{ initially},\quad\gamma_j=1\text{ initially}.
\]

The scales and mixture weights are trainable. The outer Fourier-block activation is sine; the spectral-path MLP contains GELU. Scaling input values and coordinates does not resample the computational grid. The optimizer's cosine horizon is `epochs * floor(Ntrain / batch_size)` in the article; an implementation retaining partial minibatches must explicitly account for its actual number of updates.

## Official code versus paper and repository adaptation

| Feature | Final article / official source | This study's required declaration |
| --- | --- | --- |
| Grid / bandwidth | Article: 401² and 200 modes per dimension. Official script: 201² after `r=2`, 100 modes. | Frozen 64² grid, 8 modes per axis, four active blocks; width-2 eight-branch FNO and nearest-parameter width-6 ordinary FNO. Compact settings are a resource-limited adaptation. |
| Fourth Fourier block | Official script computes spectral/MLP/local outputs for block four but the sum assignment is commented out, so they do not affect projection. Paper describes four active blocks. | Use and disclose the equation-aligned four active blocks; do not claim bitwise reproduction of the defective script. |
| Padding | Script sets an attribute to 9, but pad/unpad operations are commented out; effective padding is zero. | Record effective behavior, not the unused attribute. |
| Physical mapping | Article: random medium to real solution with Dirichlet boundary and fixed source. | Existing fixed heterogeneous medium, varying k, fixed incident direction, absorbing-boundary scattering; complex scattered-field target. |
| Input / output channels | Official script: medium plus scaled x,y; one real output. | Common physical information; representation-specific channels documented, including k/incident-wave channels when used; real and imaginary outputs reconstruct a complex field. Scaling extra channels is a further adaptation. |
| Checkpoint selection | Official script records validation and test each epoch; saves validation-best checkpoint. Despite filename, no patience termination is implemented. | Common FEM validation-only selection; untouched test evaluated after selection. |
| SNO representation | Original SNO uses Chebyshev/Fourier coefficients; Appendix C specifies softplus. | SVD/POD coefficients and GELU, and the multiscale POD variant, are repository adaptations. |

## Metrics are related, but not identical

MscaleFNO defines per-sample relative L2 error; its official `LpLoss` applies Euclidean norms on flattened regular-grid fields and averages per sample. SNO explicitly recomputes all methods' errors on the same uniform grid. Neither convention justifies comparing one method's grid validation loss with another's FEM validation loss as if they were identical.

The GMsFEM article p.9 prints

\[
 e(u)=100\frac{\|u-u_f\|_{L^2}}{\|u\|_{L^2}},
 \qquad e_{\mathrm{Re}}=e(\mathrm{Re}\,u),\quad
 e_{\mathrm{Im}}=e(\mathrm{Im}\,u),\quad e_{\mathrm{Abs}}=e(|u|),
\]

where the prose identifies `u` as the multiscale solution and `u_f` as the fine-grid reference. Thus its printed denominator is the approximation's norm. This study deliberately uses the **reference-normalized complex scattered-field consistent-mass FEM norm**, specified in [PROTOCOL.md](PROTOCOL.md), for every method and every split. It is a transparent adaptation, not a literal reuse of the paper's component metrics.

## What this experiment can establish

The 400/600/800/1,000 training counts refer to nested fixed-medium frequency queries, with common 100 validation and 100 test queries. The 800-training case matches the article's split counts only. It does not reproduce the article's random-medium dataset, grid, model capacity, source, boundary conditions, hardware, or reported accuracy.

The results can measure empirical learning curves, initialization variability, paired held-out errors, representation errors, and computational cost under the frozen configuration. Three seeds are initialization repeats, not independent draws of media or data splits. No cited result supplies a convergence rate or superiority guarantee for multiscale SVD/POD-SNO. Any advantage must be demonstrated by the actual completed runs, and any nonmonotonicity or failure must remain visible.
