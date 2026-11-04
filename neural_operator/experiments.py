"""Paired comparisons and frequency experiments using shared FEM samples."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .benchmark import compare_classical, compare_refined_fem, mesh_resolution
from .diagnostics import write_csv
from .training import load_predictor, summarize


METHOD_NAMES = {"svd": "SVD-SNO", "chebyshev": "Chebyshev-SNO", "fno": "FNO"}


def per_sample_errors(dataset, indices, prediction, method):
    indices = np.asarray(indices)
    reference = dataset.u_scat[indices]
    relative = dataset.relative_l2(reference, prediction)
    delta = np.asarray(prediction)-reference
    absolute = np.sqrt(np.maximum(np.real(np.sum(delta.conj() *
                       (dataset.mass_matrix @ delta.T).T, axis=1)), 0.))
    return [{"method": method, "sample_index": int(index),
             "split_group_id": int(dataset.split_group_ids[index]), "split_strategy": dataset.split_strategy,
             "medium_id": int(dataset.medium_id[index]), "k": float(dataset.k0[index]),
             "error_fem_l2": float(relative[j]), "absolute_fem_l2": float(absolute[j])}
            for j, index in enumerate(indices)]


def evaluate_run(dataset, run_dir, indices=None):
    """Warm batch timing includes encoding, model execution, and reconstruction."""
    run_dir = Path(run_dir)
    indices = dataset.split_indices()[2] if indices is None else np.asarray(indices)
    predictor = load_predictor(run_dir/"model.pt")
    predictor.predict_dataset(dataset, indices[:1])
    started = perf_counter()
    predictions = predictor.predict_dataset(dataset, indices)
    batch_seconds = perf_counter()-started
    single_times = []
    for _ in range(5):
        started = perf_counter()
        predictor.predict_dataset(dataset, indices[:1])
        single_times.append(perf_counter()-started)
    rows = per_sample_errors(dataset, indices, predictions, METHOD_NAMES[predictor.method])
    metrics = json.loads((run_dir/"metrics.json").read_text())
    summary = {"method": METHOD_NAMES[predictor.method],
               **summarize([row["error_fem_l2"] for row in rows]),
               "samples": len(indices), "media": len(np.unique(dataset.medium_id[indices])),
               "groups": len(np.unique(dataset.split_group_ids[indices])), "split_strategy": dataset.split_strategy,
               "batch_seconds_per_sample": batch_seconds/len(indices),
               "single_seconds_median": float(np.median(single_times)),
               "preprocessing_seconds": metrics["preprocessing_seconds"],
               "training_seconds": metrics["training_seconds"],
               "parameters": metrics["parameters"]}
    return predictor, rows, summary


def compare_methods(dataset, run_dirs, output_dir, *, n_groups=3, n_coarse=4,
                    modes=(1, 2, 4), basis_types=(1, 2)):
    """Classical and learned methods see the same held-out group samples.

    Full-test neural results are saved separately from the smaller, paired
    classical comparison. GMsFEM bases are rebuilt for every changed medium/k.
    """
    test = dataset.split_indices()[2]
    _, first = np.unique(dataset.split_group_ids[test], return_index=True)
    selected = test[first[:n_groups]]
    if not len(selected):
        raise ValueError("No held-out groups for comparison.")
    paired, all_test, full_summaries = [], [], []
    for index in selected:
        paired.extend(compare_classical(dataset, int(index), n_coarse=n_coarse,
                                         modes=modes, basis_types=basis_types))
    for method, directory in run_dirs.items():
        predictor, rows, summary = evaluate_run(dataset, directory)
        all_test.extend(rows)
        full_summaries.append(summary)
        for index in selected:
            started = perf_counter()
            prediction = predictor.predict_dataset(dataset, [index])
            seconds = perf_counter()-started
            row = per_sample_errors(dataset, [index], prediction, METHOD_NAMES[method])[0]
            paired.append({**row, "assembly_seconds": 0.,
                           "offline_seconds": summary["preprocessing_seconds"],
                           "training_seconds": summary["training_seconds"],
                           "online_seconds": seconds, "dofs": "", "parameters": summary["parameters"],
                           **mesh_resolution(dataset, int(index))})
    aggregates = []
    for method in dict.fromkeys(row["method"] for row in paired):
        rows = [row for row in paired if row["method"] == method]
        aggregates.append({"method": method, **summarize([r["error_fem_l2"] for r in rows]),
                           "paired_samples": len(rows),
                           "online_seconds_mean": float(np.mean([r["online_seconds"] for r in rows])),
                           "offline_seconds_mean": float(np.mean([r["offline_seconds"] for r in rows])),
                           "assembly_seconds_mean": float(np.mean([r["assembly_seconds"] for r in rows])),
                           "training_seconds": float(rows[0].get("training_seconds", 0.))})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir/"paired_samples.csv", paired)
    write_csv(output_dir/"paired_summary.csv", aggregates)
    write_csv(output_dir/"neural_full_test.csv", all_test)
    write_csv(output_dir/"neural_full_test_summary.csv", full_summaries)
    return paired, aggregates, full_summaries


def evaluate_frequency_shift(low_run_dirs, high_dataset, output_dir, high_run_dirs=None):
    """Frozen low-band networks vs separately retrained high-band networks."""
    test = high_dataset.split_indices()[2]
    all_rows, summaries = [], []
    regimes = [("frozen_low_band", low_run_dirs)]
    if high_run_dirs:
        regimes.append(("retrained_high_band", high_run_dirs))
    for regime, directories in regimes:
        for method, directory in directories.items():
            predictor, rows, summary = evaluate_run(high_dataset, directory, test)
            all_rows.extend([{**row, "regime": regime} for row in rows])
            summary["regime"] = regime
            if method == "svd":
                br, bi = predictor.bases["real"], predictor.bases["imag"]
                fields = high_dataset.u_scat[test]
                projection = br.decode(br.encode(fields.real)) + 1j*bi.decode(bi.encode(fields.imag))
                summary["pod_projection_mean"] = float(high_dataset.relative_l2(fields, projection).mean())
            summaries.append(summary)
    _, first = np.unique(high_dataset.split_group_ids[test], return_index=True)
    resolution = [{"sample_index": int(index), "k": float(high_dataset.k0[index]),
                   **mesh_resolution(high_dataset, int(index))} for index in test[first]]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir/"frequency_samples.csv", all_rows)
    write_csv(output_dir/"frequency_summary.csv", summaries)
    write_csv(output_dir/"mesh_resolution.csv", resolution)
    return all_rows, summaries, resolution


def refinement_study(dataset, output_dir, n_groups=3):
    test = dataset.split_indices()[2]
    _, first = np.unique(dataset.split_group_ids[test], return_index=True)
    # Largest k among the held-out frequency/material groups is checked first.
    choices = test[first]
    choices = choices[np.argsort(dataset.k0[choices])[::-1]][:n_groups]
    rows = []
    for index in choices:
        rows.append({"sample_index": int(index), "k": float(dataset.k0[index]),
                     **compare_refined_fem(dataset, int(index))})
    write_csv(Path(output_dir)/"refinement.csv", rows)
    return rows
