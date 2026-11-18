# Independent scientific and editorial review

Read first: `paper/new_paper/README.md`; writing guidance: `/Users/egoriy/.codex/skills/humanizer/SKILL.md`. This review inspected saved records and source code. It did not retrain networks, rerun FEM solves, or independently reproduce checkpoint predictions. Historical verification records therefore remain archived evidence, not checks executed by this reviewer.

## Recommended argument

Present this as a controlled study of a fixed-medium parametric surrogate with a train-fitted spatial decoder and an explicit error decomposition. The main result concerns one shared model on k in [0.1,4], not two routed frequency specialists. The useful scientific distinction is between representation accuracy, coefficient prediction, and extrapolation. A separate low-frequency representation experiment supports that distinction; it is not an additional sample of the same benchmark. The archived frozen-low-model failure must remain visible.

The original article's restrained progression is appropriate: scattering problem, variational form, approximation spaces, learned coefficient map, then measured errors and limits. Its historical numerical settings are not appropriate to copy. In particular, its random-medium dataset and its preliminary validation-only training statements differ from the completed fixed-medium studies now under discussion.

## Numbers to use and their scope

All percentages below are samplewise relative complex L2 norms, averaged over samples, not the square root of an aggregate energy ratio. Total-field and scattered-field errors have different reference denominators. Shared comparison values come from `experiments/shared_sno_gmsfem/results/summary.csv`; training metadata and decoder floor come from `sno_metrics.json` in that directory.

| Shared study, 424 test fields | Total mean (%) | Total p95 (%) | Total max (%) | Scattered mean (%) |
|---|---:|---:|---:|---:|
| Shared SVD-SNO | 2.365553 | 7.371791 | 27.585601 | 3.327085 |
| FEM h=0.04 | 0.532214 | 1.770053 | 4.051607 | 0.708033 |
| GMsFEM 20x20 Type 2, 16 local modes | 3.243923 | 12.662407 | 30.801100 | 4.292523 |

Keep all four GMsFEM settings in a table or appendix, not only the best one. The main model uses 61 modes per real/imaginary component, 50,648 trainable real parameters, width 48, depth 6, seed 42; epoch 493 was selected from 500 epochs. Split counts are 1712/432/424 fields and 214/54/52 distinct frequencies, respectively. There are 2568 fields and 320 distinct physical frequencies. The shared frequency grid is a union of earlier grids. k=2 has 16 distinct directions and is one test group, so field-weighted averages need not equal frequency-weighted averages.

The shared consistent-mass affine-decoder floor on test is 0.1434719% mean scattered error. This is much smaller than the 3.327085% predictor error; the fixed decoder admits substantially better fields than this trained map predicts. Do not subtract these mean percentages to estimate a coefficient-error component. Orthogonal squared errors decompose for each sample; arithmetic averages of relative norms do not.

Mean superiority has limits. GMsFEM's total median is 1.184844%, below SNO's 1.331596%. GMsFEM also has smaller mean errors in the two intervals up to k=2; the shared network has smaller means in the two higher intervals. This supports a frequency-dependent comparison, not dominance at every query. SNO's worst total error is 27.5856%, so avoid a claim of uniform accuracy.

The separate static-low study has 8000 fields from one medium and 1000 frequencies, with 5360/1360/1280 fields and 670/170/160 frequency groups. It uses h=0.1, k in [1,2], and different material values from the 24-inclusion main reference geometry. Source: `paper/svd_sno_revision_2026-11-06/evidence/current_static_low/{dataset_config,sno_metrics,fno_metrics,chebyshev_metrics}.json`.

| Static-low method | Test scattered mean (%) | Test p95 (%) | Parameters |
|---|---:|---:|---:|
| SVD-SNO, 10 modes/component | 0.947455 | 1.653433 | 4022 |
| FNO, 32x32 grid | 1.659892 | 1.950549 | 445010 |
| Chebyshev-SNO, 8x8 coefficients | 7.936406 | 9.881075 | 2426 |

The SVD test consistent-decoder floor is 0.918758%. Its actual POD-encoded reconstruction is 0.918859%, close but mathematically distinct. Chebyshev decoder lower bounds are validation-only: 7.614263%, 5.890452%, and 4.701619% at 8,12,16 modes per axis. Source: `chebyshev_decoder_validation.json`. These show that the fixed coarse Chebyshev span itself limits accuracy. They do not establish which architecture would win at matched approximation capacity or after tuning. Validation lower bounds cannot be subtracted from test prediction errors. The static timing records mix CPU and MPS and cannot support a hardware-normalized speed claim.

The frozen low-band model on test k>2 has mean total error 131.8231% and scattered error 177.7174%, as recorded in `experiments/reference_sno_gmsfem/results/REPORT.md` and the underlying comparison records. This is a separate model trained on [0.1,2]. The separately trained high-band model is not extrapolation, and neither is the shared [0.1,4] model.

## Mathematics and implementation audit

Inspected `neural_operator/sno/{model,basis}.py`, `_layers.py`, `losses.py`, the relevant preprocessing in `training.py`, reference training, shared baseline import/verification code, and `neural_operator/chebyshev/model.py`.

1. POD uses diagonal lumped weights W: Phi^T W Phi = I, separately for real and imaginary parts. This is not a consistent-mass POD optimization. Rank selection measures validation reconstruction error using the dataset's FEM norm, while coefficients used for that reconstruction are the lumped-weight encoder coefficients.
2. Training uses the actual consistent FEM mass M and decoder buffers. For component c, define G_c = Phi_c^T M Phi_c and a_c^* = G_c^{-1} Phi_c^T M(u_c-mu_c). Then the squared error is the sum over c of (a_hat_c-a_c^*)^T G_c(a_hat_c-a_c^*) plus the two residual energies. Normalized outputs must first be denormalized. `PODFEMLoss` implements this with float64 Gram/projection calculations and rounded decoder buffers. A Euclidean coefficient norm, identity Gram, or lumped encoder target in this identity would be a mathematical error. The implementation avoids these errors.
3. Distinguish the POD encoder's reconstruction error from the best M-projection floor in the same decoder span. Neither is a continuum PDE error bound. The training floor uses labels rounded to float32; mention finite precision only where exactness is discussed.
4. SNO v2 uses independent material coefficient sequences, padding, a shared five-channel lift, dense mode mixing, and a shared two-channel head. Material ranks are zero here, so inputs reduce to (k,cos(theta),sin(theta)). There is no tested material-function generalization. A learned dense matrix on POD positions is not Fourier convolution or an eigenbasis of the PDE operator.
5. Identical input tokens at fixed medium do not force identical final coefficients: unconstrained dense mode mixing can give different row sums. This is not an established symmetry bug. The architecture does omit explicit mode-position embeddings; an ablation would be needed to determine whether that choice limits learning. Do not diagnose the shared prediction gap as a proven implementation defect.
6. Chebyshev-SNO is a separate DCT-I/Lobatto implementation with fixed polynomial coefficients and `_ModeBlock`; it is not the same SNO-v2 block simply fitted with a new POD basis. Its analysis/synthesis endpoint factors are explicit. Avoid calling it CNO, a name used elsewhere for convolutional neural operators.
7. No confirmed model-code bug was found in this inspection. A meaningful remaining numerical caveat is that source and reference triangulations differ: a source P1 field is piecewise affine on its own mesh, not necessarily on each reference triangle. Common reference-point quadrature makes comparisons consistent with the declared protocol but is not automatically exact integration of the cross-mesh error. Do not describe that comparison as an exact common-mesh mass identity. No cross-mesh quadrature-convergence study was performed here.

The source code fits bases and normalizers on train; validation selects rank and checkpoint. Precomputing projections for held-out labels in the loss module does not by itself create leakage if optimizer batches remain train-only, as the inspected loops do. Saved hashes and query identities protect the baseline import. The archived shared verification reloads all 424 predictions and performs four fresh FEM/GMsFEM spot checks; this is stronger than checking a report table, but it is not full independent replication by this reviewer.

## Originality and publication limits

POD plus a learned coefficient map is established reduced modeling. Coefficient-space spectral neural operators and GMsFEM are also established. The present dense mode mixer and affine decoder, without a new theorem or ablation, should not be sold as a new general operator-learning architecture. The defensible contribution is an audited Helmholtz case study that separates decoder capacity, learned-map error, common-reference multiscale comparisons, and failed frequency extrapolation. This is a useful scientific record; evidence for a broad methodological contribution at ICML is presently limited. Formatting as ICML does not establish suitability or acceptance.

For a stronger future methodological claim, a direct POD-MLP or other coefficient regressor with the same decoder would be particularly informative. Multiple seeds, more media, a global POD-Galerkin baseline, rank and budget sweeps, and a controlled cost benchmark would answer distinct missing questions. These are proposed studies, not results to imply in the paper. Do not start them as part of this editorial revision.

Verified primary literature for conservative positioning: [Fanaskov and Oseledets, Spectral Neural Operators](https://arxiv.org/abs/2205.10573), [Li et al., Fourier Neural Operator](https://arxiv.org/abs/2010.08895), [Efendiev, Galvis and Hou, GMsFEM](https://arxiv.org/abs/1301.2866). Their general constructions motivate the categories above; they do not certify this implementation's performance. The SNO source describes Fourier and Chebyshev series, so call the learned POD variant an adaptation rather than an exact reproduction.

## Prose audit

The companion draft uses one argument per paragraph, defined scopes, and plain verbs. No em dashes, promotional novelty claims, invented authors, affiliations, experiments, or uncertainty estimates were added. Its related-work citation keys are suggested integration keys, to be matched to the primary manuscript bibliography by the integrating author.

## Final manuscript and asset-code review

Reviewed the completed `main.tex`, `prepare_assets.py`, bibliography, generated statistics, and `evidence/operator_checks.json` on 2026-11-13. No edits were made to the manuscript or figures. Independent arithmetic checks used saved per-query JSON records and Python's standard-library `statistics` module; no model loading, training, or PDE solving was needed.

The main comparison is correctly qualified throughout: the model and its basis are shared across the declared band; the static representation study has its own medium, mesh, fields, and split; low-band extrapolation is a separate frozen model. The mass-weighted error identity, lumped-POD distinction, float32 decoder/float64 quadratic description, non-equivalent method-size counts, and finite-reference/continuum/cross-mesh limitations are mathematically appropriate. GMsFEM's elliptic stiffness and alpha-weighted mass in the local eigenproblem match `gmsfem/offline.py`, including retention of the smallest eigenvalues. Local snapshots are real here, so the mathematical conjugate transpose agrees with the implementation's ordinary transpose.

Independent numerical checks confirmed:

- The paired total-error comparison gives exactly 139 SNO wins and 285 GMsFEM wins among 424 query identities, with no ties. The figure uses SNO on the horizontal axis and GMsFEM on the vertical axis, so its caption correctly states that points above the diagonal favor SNO.
- There are exactly 200 unique frozen-low-model queries with k>2, ranging from 2.0125786163522013 to 4. Every query has an exact shared-test counterpart. Mean total errors are 131.82310649382603% for the frozen model and 4.560418600295925% for the shared model on this same subset. The frozen mean scattered error is 177.71737165666732%. Thus the 200-query statement and 4.5604% derived comparison are sound; using the 424-query main mean here would be wrong, and the manuscript avoids it.
- Interval field counts are 128,96,112,88. SNO/GMsFEM total means are respectively (0.11248293165501234,0.05446069020202556), (0.7970082655842825,0.5997780145779101), (3.647032375826094,4.460976513117781), and (5.722910158712073,9.218686668874785) percent. All manuscript interval values round correctly.
- Two float32 53246-by-61 decoder matrices occupy exactly 25,984,048 bytes, or 24.780319213867188 MiB. The stated 24.8 MiB is correct and explicitly excludes means, mesh, other bundle metadata, and float64 preprocessing copies. The text does not claim this is complete serialized-bundle size.
- The principal shared/static test means, percentile values, floors, ranks, counts, epoch selections, and parameter counts agree with the records identified in the first review. Main total/scattered denominators and phase-sensitive scoring are consistently distinguished.

Three numerical-precision statements need either conventional rounding or explicit upper-bound wording. `operator_checks.json` gives a maximum scattered-metric discrepancy of 7.561363849717395e-5 percentage points, which rounds to 7.56e-5, rather than the manuscript's 7.57e-5. Its largest Chebyshev inverse entrywise discrepancy is 2.2343238370581275e-15, which rounds to 2.23e-15, rather than 2.24e-15. The archived `results/verification.json` gives a maximum baseline discrepancy of 2.6412205755832474e-12 percentage points, which rounds to 2.64e-12, rather than 2.65e-12. All three manuscript values are valid conservative upper bounds if introduced as `below`, but the wording currently reports them as measured maxima. These are small presentation issues, not changes to the conclusions; they were reported promptly to the integrating author.

The schematic labels the physical matrix A_h(k), while the manuscript defines H_h(k). Aligning the labels would avoid an unnecessary notation switch. The asset script's canonical-data assertions, query matching, shared limits within displayed quantities, and omission of smoothing support the captions. Its field examples are display transfers rather than the scored source P1 quadrature fields, and the manuscript explicitly says so. No favorable-query selection is apparent.

The bibliography's FNO, SNO, GMsFEM, PODNO, and POD-DeepONet references fit the limited related-work claims. The primary [PODNO abstract](https://arxiv.org/abs/2504.18513) directly supports the stated replacement of Fourier transforms with POD transforms. The [Lu et al. primary record](https://arxiv.org/abs/2111.05512) confirms the listed authors and corresponding published DOI. The Kalachikova article DOI did not open through the web tool in this review; its journal metadata agree with the repository's source citation, but an inaccessible publisher page is not claimed as a fresh full-text verification. The related-work paragraph appropriately avoids importing the cited papers' accuracy or speed claims into this experiment.

No confirmed mathematical defect or numerical mismatch affecting the reported comparison was found. This final pass inspected fresh-check records; it did not independently rerun the 91-test suite or the checkpoint reconstruction. The primary agent remains responsible for final compilation, page inspection, and ensuring the evidence ledger accurately records those executed checks.
