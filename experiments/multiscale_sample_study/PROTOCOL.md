# Fixed-medium sample scaling protocol

Status: execution protocol, selected from training-only hardware timings before any new-model training. The machine-readable protocol is frozen on first training. No completed results are claimed until replay verification passes.

## Scientific question and samples

The experiment measures how prediction error changes with the number of training queries for the repository's existing fixed-medium Helmholtz scattering problem. Geometry, mesh, coefficient fields, boundary conditions, frequency interval, and incident direction are inherited from that problem and recorded explicitly in the dataset manifest. No alternative medium family is inferred or introduced.

A sample is one distinct physical frequency query and its complex scattered FEM solution: `(fixed medium, k, fixed theta) -> u_scat`. Spatial nodes, real/imaginary channels, neural initialization seeds, and interpolation points are not additional samples. This is frequency interpolation for a fixed medium, not generalization to unseen random media.

Generate 1,200 unique queries once. Reserve the same 100 validation and 100 test queries for every run. The remaining 1,000 queries have a fixed, frequency-balanced ordering whose first 400, 600, 800, and 1,000 entries define nested training sets. Prefix balancing must be implemented and verified without using solution errors. Train, validation, and test query sets must be disjoint; duplicated frequencies are forbidden. Record exact physical parameters, ordered indices, generation seed, mesh/coefficient provenance, and file hashes. Each FEM label is genuinely solved; interpolation of a smaller set of labels is not dataset enlargement.

The horizontal axis is **training samples**, not total samples. Total counts are consequently 600, 800, 1,000, and 1,200. The 800-training case has the article's 800/100/100 counts, but its physical sample family differs from the article.

## Methods and fitting

The four main methods are `sno_gelu` (ordinary SVD/POD-SNO), `mscale_sno` (multiscale SVD/POD-SNO), `fno_gelu` (ordinary FFT-FNO), and `mscale_fno` (independent scaled FFT-FNO branches). Use initialization seeds 11, 29, and 47 for each size and method: 48 independently initialized training runs. Each run starts from scratch; larger training sets do not warm-start from smaller ones.

Fit normalizers, POD bases, and other learned representation statistics using only that run's training prefix. Validation and test snapshots must not enter SVD/POD fitting, rank selection, or normalization. Freeze the rank policy and architecture settings before training the study; record realized ranks and parameter counts. All methods receive the same physical information, with their representation-specific encoders documented. Train-only preprocessing can differ between prefixes, but the policy cannot be adjusted in response to held-out results.

Parameter matching is approximate and must be reported numerically rather than assumed. SVD/POD-SNO is a repository adaptation of coefficient-space operator learning. Its multiscale variant is an experimental architecture; neither convergence nor superiority follows from the cited SNO or MscaleFNO papers.

## Common objective and evaluation

Let `u_i` be the reference complex scattered field on the original FEM mesh, `uhat_i` the reconstructed predicted field on that same mesh, and `M` the consistent FEM mass matrix. Use the same per-case relative error for training, validation, and test, for all four methods:

\[
 e_i=\sqrt{\frac{\operatorname{Re}[(\hat u_i-u_i)^*M(\hat u_i-u_i)]}
 {\operatorname{Re}[u_i^*Mu_i]}} ,\qquad
 L(S)=\frac1{|S|}\sum_{i\in S}e_i.
\]

Report percentages as `100 * e_i`; training minimizes the unscaled mean. Complex conjugation and the consistent mass matrix are essential. This is a mean of individual relative norms, not a relative norm of the concatenated dataset and not mean squared error. Check and record any numerical denominator floor and zero-reference handling. Clamp only roundoff-negative quadratic forms; significant negatives indicate an error.

For SVD/POD models, a mass-orthonormal coefficient formula is acceptable only if it includes the reference projection residual and equals reconstruction-space FEM error. For FFT models, differentiably reconstruct grid predictions on FEM nodes before evaluating this objective. Grid loss must not replace FEM loss for checkpoint selection or appear as the same convergence quantity. Use training-only fitted output normalization, then evaluate physical errors after undoing it.

Select the checkpoint with the smallest common validation mean `L(validation)`. Predeclare tie handling. No test information enters early stopping, architecture selection, learning-rate choices, or checkpoint selection. Test evaluation occurs after the protocol and checkpoints are fixed. Any later scientific change requires an explicitly new experiment, with prior test exposure disclosed.

## Budget and runtime selection before freezing

The hardware pilot selected native macOS PyTorch: both SNO variants on CPU with two threads, and both FFT variants on MPS. Timing comparisons between ordinary and multiscale models therefore share a backend within each family; cross-family wall-time differences also include backend differences and cannot establish an architecture-only speedup. Complex FFT, backward, and the common differentiable FEM objective were actually exercised.

This initial sample-scaling series uses **100 epochs maximum**, batch size 50, Adam learning rate 0.001 and weight decay 0.0001, with cosine annealing per update. Validation is evaluated at epoch 1, every 5 epochs, and the budget endpoint; 15 checks without improvement terminate a run early. These are evaluated raw points, not measurements at every omitted epoch. A run finishing its 100-epoch budget is not automatically converged. The journal used 500 epochs; this is an explicit resource-limited comparison, not reproduction of its training budget. Extending the schedule requires a separately frozen experiment.

All models retain four blocks. Both multiscale models have eight independent branches and initial scales [1,40,80,100,120,140,180,200]. The FFT grid is 64 by 64 with eight modes per axis; multiscale width 2 has 33,824 real parameters and the nearest ordinary width 6 has 37,634. The original journal models had approximately 256 million and 328 million parameters. SNO uses fixed requested complex POD rank 20 (capped by numerical rank), multiscale width 8 and a nearest-parameter ordinary width selected without validation/test scores. Actual counts and projection errors are recorded.

Online MPS grid loss evaluates the exact projected FEM quadratic form in float32 with a tested custom symmetric-matrix gradient; projection, residual preparation, validation, and final field scoring use float64. Actual-data comparisons against the double-precision implementation bounded relative value/gradient differences below 2e-7 in the pilot. This changes arithmetic precision, not the mass matrix or interpolation space. CPU SNO uses float32 model and online coefficient-loss arithmetic and float64 validation. The frozen machine-readable protocol’s generic `online_dtype` description is imprecise for CPU SNO; per-run configurations and the independent physical audit record the actual precision. All physical dataset fingerprints are file-content based so platform-dependent last bits in assembled matrices are not mistaken for changed PDE cases.

Choose and document one feasible fixed configuration for the complete study before freezing. Do not change resolution, width, Fourier modes, branch count, epoch limit, or optimizer schedule between sample sizes in response to observed errors. Specify optimizer, initial learning rate, weight decay, scheduler units and horizon, batch size, epoch budget, and any stopping rule in the frozen configuration. Budget reduction is permissible as a disclosed resource-limited adaptation, not as article reproduction. Existing one-epoch timing runs cannot establish convergence or accuracy.

For the main sample-size comparison, give all methods the declared epoch budget and report the corresponding update counts and wall time. Equal epochs imply more updates at larger sample sizes; this is part of the declared experiment. Do not describe this as equal compute. Additional equal-time comparisons, if performed, need their own predeclared selection rule.

## Curves, statistics, and verification

Persist actual training and validation errors at each declared evaluation (epoch 1, every fifth epoch, and the final budget epoch), together with epoch, cumulative optimizer steps, cumulative training time, learning rate, and checkpoint decisions. Training-loss rows contain the minibatch losses from that evaluated epoch; omitted epochs have no persisted loss row. Training minibatch means reflect changing parameters; if an end-of-epoch training evaluation is recorded, label it separately. Distinguish raw validation values from running minima. Never manufacture epochs, smooth away divergence, extrapolate unfinished curves, or represent an incomplete run as completed.

Plot the common validation metric against epoch, cumulative wall time, and optimizer steps, with all four methods and three seeds identifiable. Aggregate only supported observations; do not extend a stopped seed's curve with fabricated values. Report test mean error and sample standard deviation across the three initialization seeds, plus casewise p95 and paired per-query method differences on the common test set. State that three initialization seeds measure initialization variability, not independent resampling of the dataset. Pair comparisons by size, seed, and test-query identity. Avoid significance or generalization claims unsupported by this design.

Separately report POD projection residuals and exact-label FEM-to-grid-to-FEM round-trip error, as representation diagnostics. The latter is not a universal mathematical lower bound on learned model error. Test diagnostics must not be used for tuning.

Verify dataset counts, disjointness, nested prefixes, uniqueness, train-only preprocessing, finite histories, checkpoint replay, and agreement of reported errors with direct common-mass recomputation. Save dataset/source/configuration fingerprints, prediction fields or reproducible checkpoint paths, and a machine-readable completion status. Results are reportable as completed only after actual execution and these checks.

## Relation to the articles

See [ARTICLE_COMPARISON.md](ARTICLE_COMPARISON.md) for source-specific counts, physics, architecture, and metric differences. The study follows the principles of common evaluation and validation-only selection, while retaining the repository's physical problem. Matching the article's 800-training count does not reproduce its dataset or numerical result.
