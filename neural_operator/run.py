"""Command-line equivalents of the local notebooks."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("data", "svd", "train", "compare", "high-frequency"))
    parser.add_argument("--dataset", type=Path, default=base/"data"/"low_frequency")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=("svd", "chebyshev", "fno", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-media", type=int, default=48)
    parser.add_argument("--n-directions", type=int, default=4)
    parser.add_argument("--h", type=float, default=.1)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    from .data import load_dataset
    if args.action == "data":
        from .fem import DatasetConfig, generate_dataset
        ds = generate_dataset(args.dataset, DatasetConfig(n_media=args.n_media,
                              n_directions=args.n_directions, h=args.h))
        print(f"Saved {ds.u_scat.shape[0]} FEM samples on {ds.u_scat.shape[1]} nodes: {ds.directory}")
        return
    dataset = load_dataset(args.dataset)
    if args.action == "svd":
        from .diagnostics import svd_diagnostics
        rows, _, _ = svd_diagnostics(dataset, args.output or base/"runs"/"svd_diagnostics")
        for row in rows:
            print(row)
        return
    from .training import TrainConfig, run_experiment
    methods = ("svd", "chebyshev", "fno") if args.method == "all" else (args.method,)
    low_root = base/"runs"/"low_frequency"
    if args.action == "train":
        root = args.output or low_root
        for method in methods:
            _, metrics, _ = run_experiment(dataset, root/method,
                                           TrainConfig(method=method, epochs=args.epochs), retrain=args.retrain)
            print(method, metrics["test_fem_l2"], flush=True)
    elif args.action == "compare":
        from .experiments import compare_methods
        _, summary, _ = compare_methods(dataset, {m: low_root/m for m in methods},
                                        args.output or base/"runs"/"comparison")
        for row in summary:
            print(row)
    else:
        from .fem import DatasetConfig, generate_dataset
        from .experiments import evaluate_frequency_shift, refinement_study
        import json
        config = DatasetConfig(**json.loads((dataset.directory/"config.json").read_text()))
        high = generate_dataset(base/"data"/"high_frequency", replace(config, k_range=(4., 8.)))
        root = args.output or base/"runs"/"high_frequency"
        for method in methods:
            run_experiment(high, root/method, TrainConfig(method=method, epochs=args.epochs), retrain=args.retrain)
        _, summary, _ = evaluate_frequency_shift({m: low_root/m for m in methods}, high,
                                                 root/"comparison", {m: root/m for m in methods})
        refinement_study(high, root/"comparison")
        for row in summary:
            print(row)


if __name__ == "__main__":
    main()
