"""Generate, inspect, train and compare operators on a shared FEM dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from . import settings as default_settings

BANDS = {"low": (1., 2.), "high": (4., 8.)}
METHODS = ("svd", "chebyshev", "fno")


def build_parser(settings=default_settings):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action, description in (
        ("data", "Generate FEM fields, without training"),
        ("inspect", "Show saved dataset metadata without loading fields"),
        ("svd", "Inspect train-only POD compression"),
        ("train", "Train operators or reuse matching completed runs"),
        ("compare", "Compare saved models with FEM/GMsFEM"),
        ("high-frequency", "Train high-band models and compare frequency transfer"),
    ):
        command = actions.add_parser(action, help=description, description=description)
        command.add_argument("--dataset", type=Path, help="Dataset directory (or NPZ for inspect)")
        command.add_argument("--band", choices=BANDS, default="low")
        if action in ("data", "inspect"):
            continue
        command.add_argument("--pod-method", choices=("full", "randomized"), default=settings.POD_METHOD)
        command.add_argument("--output", type=Path, help="Results directory")
        if action != "svd":
            command.add_argument("--method", choices=(*METHODS, "all"), default="all")
        if action in ("compare", "high-frequency"):
            command.add_argument("--runs", type=Path, help="Source model root containing method folders")
        if action in ("train", "high-frequency"):
            command.add_argument("--epochs", type=int, default=200)
            command.add_argument("--retrain", action="store_true")
        if action == "high-frequency":
            command.add_argument("--high-dataset", type=Path, default=settings.HIGH_DATA_DIR)
    data = actions.choices["data"]
    data.add_argument("--n-frequencies", type=int, default=settings.N_FREQUENCIES)
    data.add_argument("--n-directions", type=int, default=settings.N_DIRECTIONS)
    data.add_argument("--fixed-medium-id", type=int, default=0)
    data.add_argument("--seed", type=int, default=2026)
    data.add_argument("--k-range", type=float, nargs=2, help="Override the selected band's frequency range")
    data.add_argument("--h", type=float, default=.1, help="FEM mesh size")
    data.add_argument("--n-coarse", type=int, default=4)
    data.add_argument("--dry-run", action="store_true", help="Print the plan without files or FEM solves")
    return parser


def _profile(pod_method):
    if pod_method == "randomized":
        from .randomized import settings
        return settings
    return default_settings


def _run_root(directory, profile):
    """Keep existing checkpoints discoverable; give custom datasets their own runs."""
    for band in ("LOW", "HIGH"):
        if directory == getattr(profile, f"{band}_DATA_DIR").resolve():
            return getattr(profile, f"{band}_RUNS")
    return profile.BASE/"runs"/directory.name


def _data(args):
    from .datasets import StaticDatasetConfig, generate_dataset, plan_dataset, write_dataset_info
    config = StaticDatasetConfig(n_frequencies=args.n_frequencies, n_directions=args.n_directions,
                                 h=args.h, n_coarse=args.n_coarse, seed=args.seed,
                                 k_range=tuple(args.k_range or BANDS[args.band]),
                                 fixed_medium_id=args.fixed_medium_id)
    if args.k_range and not args.dataset:
        raise ValueError("--k-range requires --dataset PATH so custom frequencies have an explicit home")
    directory = args.dataset or default_settings.BASE/"data"/f"static_{args.band}_frequency_{config.n_frequencies}x{config.n_directions}"
    plan = {"directory": str(directory.expanduser().resolve()), **plan_dataset(config)}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.dry_run:
        dataset = generate_dataset(directory, config)
        info = write_dataset_info(dataset)
        print(f"Ready: {info['samples']} fields, {info['media']} medium, {dataset.directory}", flush=True)


def _execute(args):
    if args.action == "data":
        return _data(args)
    from .data import inspect_dataset, load_dataset
    directory = (args.dataset or getattr(default_settings, f"{args.band.upper()}_DATA_DIR")).expanduser().resolve()
    if args.action == "inspect":
        print(json.dumps(inspect_dataset(directory), indent=2))
        return
    profile = _profile(args.pod_method)
    root = _run_root(directory, profile)
    dataset = load_dataset(directory, mmap=True)
    if args.action == "svd":
        from .diagnostics import svd_diagnostics
        output = args.output or root.parent/f"svd_diagnostics_{root.name}"
        rows, _, _ = svd_diagnostics(dataset, output, methods=(args.pod_method,))
    else:
        from .training import TrainConfig, run_experiment
        methods = METHODS if args.method == "all" else (args.method,)

        def train(target, destination):
            for method in methods:
                config = TrainConfig(method=method, epochs=args.epochs, pod_method=args.pod_method)
                _, metrics, _ = run_experiment(target, destination/method, config, retrain=args.retrain)
                print(method, metrics["test_fem_l2"], flush=True)

        if args.action == "train":
            train(dataset, args.output or root)
            return
        runs = {method: (args.runs or root)/method for method in methods}
        if args.action == "compare":
            from .experiments import compare_methods
            _, rows, _ = compare_methods(dataset, runs, args.output or root.parent/f"comparison_{root.name}")
        else:
            from .datasets import generate_dataset, load_config, write_dataset_info
            from .experiments import evaluate_frequency_shift, refinement_study
            config = load_config(dataset.directory)
            if tuple(config.k_range) != BANDS['low']:
                raise ValueError("high-frequency requires a low-band (1, 2) source dataset")
            del dataset
            high = generate_dataset(args.high_dataset, replace(config, k_range=BANDS["high"]))
            write_dataset_info(high)
            destination = args.output or _run_root(high.directory, profile)
            train(high, destination)
            _, rows, _ = evaluate_frequency_shift(runs, high, destination/"comparison",
                                                   {m: destination/m for m in methods})
            refinement_study(high, destination/"comparison")
    for row in rows:
        print(row)


def main(settings=default_settings, argv=None):
    parser = build_parser(settings)
    args = parser.parse_args(argv)
    try:
        _execute(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
