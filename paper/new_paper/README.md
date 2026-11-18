# New paper: scientific and editorial contract

This directory contains a new English research manuscript in the official ICML
2026 LaTeX layout. It is a local research draft, not an accepted ICML paper.
The earlier manuscript and all existing experimental records remain separate.

## Writing style

Use the mathematical organization common to the project's GMsFEM sources:
state the problem, define the approximation spaces, describe the algorithm,
then test the claims. Use the existing manuscript as a sample of the author's
technical vocabulary. The voice should be direct and restrained, with varied
sentence lengths, explicit assumptions, and precise references to equations,
figures, and tables. Do not imitate a named researcher's distinctive wording or
copy passages. The author retains responsibility for the final text.

- Write one argument per paragraph. Use `we` for an action actually performed.
- Introduce every symbol before using it; distinguish fine FEM, reference FEM,
  GMsFEM reduction, POD projection, and learned prediction.
- Use sentence-case headings. Avoid promotional adjectives, chat language,
  mechanical roadmaps, em dashes, `etc.`, and unfinished lists.
- Every quantitative statement must have a traceable local source. Explain
  the denominator, field (total/scattered), split, and reference discretization.
- Figures need readable labels, units, consistent scales, and captions that
  state the experiment and the interpretation. Generate plots from numerical
  records; use original vector drawings for mathematical schematics.

## Evidence and scope

The main candidate is the completed shared SVD-SNO experiment on one fixed
medium over k in [0.1,4], with 424 held-out fields and common-reference GMsFEM
comparisons. The static 8000-field three-architecture experiment is a separate
representation study. Its numbers must not be pooled with the shared study.
The frozen low-band extrapolation result is retained even though it is poor.

<!-- claim-status: reproduced | archived-verified | derived | untested -->
Use these statuses in the evidence ledger, not as prose in the manuscript.
`reproduced` means recomputed during this revision; `archived-verified` means
supported by an existing verification record; `derived` means calculated from
saved data; `untested` must not become a numerical result.

## Working constraints

1. Read the implementation and provenance before writing results. Audit SNO
   and Chebyshev-SNO separately; CNO is not an established model name here.
2. Preserve pre-existing working-tree changes. Do not overwrite checkpoints,
   datasets, historical reports, or the previous paper.
3. Fit bases and normalizers on train, select on validation, and use test only
   for the declared evaluation. Do not tune on test or select flattering fields.
4. No invented experiments, error bars, affiliations, coauthors, citations,
   speedups, convergence claims, or new-medium generalization claims.
5. Do not label a lumped-weight POD as consistent-mass POD. Derive the actual
   decoder loss and representation floor from the implementation.
6. Keep scientific plots deterministic and export PDF plus PNG. Generative
   illustration must never stand in for measured physical data.
7. Use one Medium Effort subagent to review evidence and draft prose, with
   separate files. The primary agent checks the claims and integrates them.
8. Run relevant numerical tests, validate generated tables against source
   records, compile LaTeX, and inspect rendered PDF pages before delivery.
9. Do not start expensive retraining or generate thousands of FEM solves for
   an editorial revision. Record any environment-dependent checks not rerun.
10. Keep a short audit, evidence ledger, and reproducible build command beside
    the manuscript. Clearly separate executed checks from proposed work.

## Deliverables

- `main.tex`, `main.pdf`, `references.bib`: manuscript and bibliography.
- `figures/`: data-derived plots and an original method schematic.
- `prepare_assets.py`, `build.sh`, `validate.py`: reproducible production.
- `AUDIT.md`, `evidence/`: implementation findings and source records.
- `reviews/`: independent Medium Effort editorial and scientific review.

This contract was written before the new manuscript. The final article has
seven pages of main content, followed by references and appendices. The eight
exported PDF/PNG asset pairs include the combined diagnostic figure and its
individual panels. The primary results are historical; fresh inference and
unit checks are recorded separately in [AUDIT.md](AUDIT.md).

## Build and check

From the repository root, using its NumPy/SciPy/PyTorch/Matplotlib environment:

```sh
# Rebuild figures and generated tables from existing numerical records.
.venv/bin/python paper/new_paper/prepare_assets.py

# Compile the article with the included official style and BibTeX files.
sh paper/new_paper/build.sh

# Check hashes, paired statistics, tables, citations, page size, and overflow.
.venv/bin/python paper/new_paper/validate.py
```

LaTeX needs `latexmk`, `pdflatex`, and `bibtex`; PDF checking/rendering needs
Poppler. The validator also finds Poppler in the bundled Codex runtime on this
machine. The figure build uses existing tracked result records and the compact
paper-local field snapshot, without reopening the full training dataset.

Optional independent checkpoint check, requiring the existing local shared
dataset and model (about 2.2 GB of labels, not distributed as paper sources):

```sh
.venv/bin/python paper/new_paper/audit_operators.py
.venv/bin/python paper/new_paper/prepare_assets.py
sh paper/new_paper/build.sh
.venv/bin/python paper/new_paper/validate.py
```

This rechecks hashes, all 424 neural test queries, the coefficient-space loss
identity, and Chebyshev transforms. It does not retrain the physical models or
run new PDE solves. Small differences from archived predictions are expected
across numerical-library versions; explicit tolerances are in the script.

To render pages for visual inspection:

```sh
mkdir -p tmp/pdfs/new_paper
pdftoppm -scale-to 1600 -png paper/new_paper/main.pdf tmp/pdfs/new_paper/page
```

The exact targeted test command and observed result are recorded in
[test_results.json](evidence/test_results.json). Pytest's `--import-mode=importlib`
avoids duplicate test module names in this repository. See
[validation.json](evidence/validation.json) for automatic checks and
[visual_review.json](evidence/visual_review.json) for the final page review.

## Author block and template

The author name follows `CITATION.cff`. Institutional affiliation and contact
details were not supplied, so none was invented. The local manuscript uses a
notice-only preprint footer in place of the template's affiliation/contact
footer. Official files in `vendor/` remain unchanged. Page size, text block,
and fonts follow the official style; references use ordinary bottom alignment
and balanced columns to avoid stretched blank space. This is a research
preprint, not a claim of submission, acceptance, or complete ICML suitability.

## Scientific prose pass

The Medium Effort reviewer inspected the evidence and then the integrated
manuscript. The final prose pass removed repeated roadmap language and
promotional claims, kept one argument per paragraph, and preserved mathematical
terminology. No em dashes, unfinished lists, or `etc.` remain in the manuscript.
The existing article provided terminology and mathematical progression; no
named author's distinctive prose was copied. A personal voice cannot be
matched more precisely without additional confirmed writing samples.
