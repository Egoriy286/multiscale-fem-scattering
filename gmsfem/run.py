"""CLI: python3 -m gmsfem.run --profile demo --case notebook."""

from __future__ import annotations

import argparse
from pathlib import Path


GENERATED_FIGURES = (
    "meshes.png", "basis.png", "convergence.png", "fields_total.png",
    "fields_scattered.png", "fields_dg_style.png", "reference_comparison.png",
)


def clear_generated_figures(output):
    """Remove only this CLI's figures so a reused folder cannot mix studies."""
    for filename in GENERATED_FIGURES:
        (Path(output) / filename).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="GMsFEM: .geo → CG1 → snapshots → спектр → сравнение с МКЭ")
    parser.add_argument("--profile", choices=("quick", "demo", "paper"), default="demo",
                        help="quick: n4,h.1,M≤4; demo: n10/20,h.04,M≤16; paper: h.015 (~100k DOF)")
    parser.add_argument("--case", choices=("notebook", "paper_example1", "paper_example2", "homogeneous"), default="notebook")
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--coarse", type=int, nargs="+", default=None)
    parser.add_argument("--modes", type=int, nargs="+", default=None)
    parser.add_argument("--types", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--direction", type=float, nargs=2, default=(1., 0.))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-reference", action="store_true", help="Пропустить отдельный МКЭ h/2")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    from .experiment import ROOT, compare_refined, run_study, save_study, write_report

    quick = args.profile == "quick"
    h = args.h if args.h is not None else (.1 if quick else .015 if args.profile == "paper" else .04)
    coarse = tuple(args.coarse if args.coarse is not None else (4,) if quick else (10, 20))
    modes = tuple(args.modes if args.modes is not None else (1, 2, 4) if quick else (1, 2, 4, 6, 8, 12, 16))
    output = args.output or ROOT / "results" / f"{args.case}_k{args.k:g}"
    study = run_study(case=args.case, k=args.k, h=h, coarse_grids=coarse,
                      modes=modes, basis_types=tuple(args.types), direction=tuple(args.direction))
    reference = None if args.no_reference else compare_refined(study)
    # Persist the numerical result before rendering: a failed plot must not
    # discard a completed solve. Refresh only report.md after the new PNGs.
    clear_generated_figures(output)
    output = save_study(study, output, reference)
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plotting import (plot_basis, plot_convergence, plot_fields, plot_meshes,
                               plot_refined_comparison, save_figure)

        nc, t, M = max(coarse), max(args.types), max(modes)
        figures = [("meshes.png", plot_meshes(study.problem, nc, reference.problem if reference else None)),
                   ("basis.png", plot_basis(study.problem, study.spaces[nc, t])),
                   ("convergence.png", plot_convergence(study.rows)),
                   ("fields_total.png", plot_fields(study.problem, study.fine,
                       [(f"GMsFEM {n}×{n}, Type {t}, M={M}", study.solutions[n, t, M]) for n in coarse])),
                   ("fields_scattered.png", plot_fields(study.problem, study.fine,
                       [(f"GMsFEM {n}×{n}, Type {t}, M={M}", study.solutions[n, t, M]) for n in coarse], field="scattered")),
                   ("fields_dg_style.png", plot_fields(study.problem, study.fine,
                       [(f"GMsFEM {n}×{n}, Type {t}, M={M}", study.solutions[n, t, M]) for n in coarse], field="paper_mixed"))]
        if reference:
            figures.append(("reference_comparison.png", plot_refined_comparison(study.problem, study.fine,
                           study.solutions[nc, t, M], reference.problem, reference.fine,
                           f"GMsFEM {nc}×{nc}, Type {t}, M={M}")))
        for filename, figure in figures:
            save_figure(figure, output / filename)
            plt.close(figure)
        write_report(study, reference, output)
    print(f"Готово: {output / 'report.md'}")
    return output


if __name__ == "__main__":
    main()
