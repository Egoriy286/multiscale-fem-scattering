"""Local static-medium datasets and full/randomized neural-operator runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from . import settings as default_settings


def main(settings=default_settings):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("data", "svd", "train", "compare", "high-frequency"))
    parser.add_argument("--dataset", type=Path, default=settings.LOW_DATA_DIR)
    parser.add_argument("--high-dataset", type=Path, default=settings.HIGH_DATA_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=("svd", "chebyshev", "fno", "all"), default="all")
    parser.add_argument("--pod-method", choices=("full", "randomized"), default=settings.POD_METHOD)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-frequencies", type=int, default=settings.N_FREQUENCIES)
    parser.add_argument("--n-directions", type=int, default=settings.N_DIRECTIONS)
    parser.add_argument("--fixed-medium-id", type=int, default=0)
    parser.add_argument("--k-range", type=float, nargs=2, default=(1., 2.))
    parser.add_argument("--h", type=float, default=.1)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    from .data import load_dataset, write_dataset_info
    if args.action == "data":
        from .fem import StaticDatasetConfig, generate_static_dataset
        config = StaticDatasetConfig(n_frequencies=args.n_frequencies, n_directions=args.n_directions,
                                     h=args.h, k_range=tuple(args.k_range), fixed_medium_id=args.fixed_medium_id)
        dataset = generate_static_dataset(args.dataset, config)
        info = write_dataset_info(dataset)
        print(f"Saved {info['samples']} fields of {info['media']} fixed medium: {dataset.directory}")
        print(info["splits"], flush=True)
        return
    dataset = load_dataset(args.dataset, mmap=True)
    if args.action == "svd":
        from .diagnostics import svd_diagnostics
        rows, _, _ = svd_diagnostics(dataset, args.output or settings.SVD_RUNS,
                                    methods=(args.pod_method,))
        for row in rows:
            print(row)
        return
    from .training import TrainConfig, run_experiment
    methods = ("svd", "chebyshev", "fno") if args.method == "all" else (args.method,)

    def train_config(method):
        return TrainConfig(method=method, epochs=args.epochs, pod_method=args.pod_method)

    if args.action == "train":
        root = args.output or settings.LOW_RUNS
        for method in methods:
            _, metrics, _ = run_experiment(dataset, root/method, train_config(method), retrain=args.retrain)
            print(method, metrics["test_fem_l2"], flush=True)
    elif args.action == "compare":
        from .experiments import compare_methods
        _, summary, _ = compare_methods(dataset, {m: settings.LOW_RUNS/m for m in methods},
                                        args.output or settings.COMPARISON_RUNS)
        for row in summary:
            print(row)
    else:
        from .fem import DatasetConfig, StaticDatasetConfig, generate_dataset, generate_static_dataset
        from .experiments import evaluate_frequency_shift, refinement_study
        config = json.loads((dataset.directory/"config.json").read_text())
        static = config.pop("medium_mode", None) == "static"
        config = (StaticDatasetConfig if static else DatasetConfig)(**config)
        del dataset
        generator = generate_static_dataset if static else generate_dataset
        high = generator(args.high_dataset, replace(config, k_range=(4., 8.)))
        write_dataset_info(high)
        root = args.output or settings.HIGH_RUNS
        for method in methods:
            run_experiment(high, root/method, train_config(method), retrain=args.retrain)
        _, summary, _ = evaluate_frequency_shift({m: settings.LOW_RUNS/m for m in methods}, high,
                                                 root/"comparison", {m: root/m for m in methods})
        refinement_study(high, root/"comparison")
        for row in summary:
            print(row)


if __name__ == "__main__":
    main()
