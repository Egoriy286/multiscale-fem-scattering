"""Reject incomplete or unpaired results before publishing the final report.

Only small query/JSON artifacts are read; validation does not load FEM fields,
checkpoints, or the FEM/PyTorch runtime.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


BASELINE_METHODS = {
    "FEM h=0.04",
    *(f"GMsFEM {grid}x{grid} Type {kind} M=16"
      for grid in (10, 20) for kind in (1, 2)),
}
REQUIRED_METRICS = {
    *(f"{field}_l2_error_{component}"
      for field in ("total", "scattered")
      for component in ("real", "imag", "abs", "complex")),
    *(f"{field}_e_{component}_{denominator}"
      for field in ("total", "scattered")
      for component in ("real", "imag", "abs", "complex")
      for denominator in ("paper", "ref")),
}


def _fail(context, message):
    raise ValueError(f"Report validation failed ({context}): {message}")


def _finite(value, context):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{context}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail(context, "nonfinite numerical value")


def _read(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        _fail(path, f"required JSON artifact is missing or invalid: {error}")
    if not isinstance(value, dict):
        _fail(path, "expected a JSON object")
    _finite(value, str(path))
    return value


def _number(value, context, *, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum):
        _fail(context, f"expected a finite number >= {minimum}")
    return value


def _queries(directory):
    info = _read(directory / "dataset_info.json")
    if info.get("complete") is not True:
        _fail(directory, "reference dataset is incomplete")
    path = directory / "samples.npz"
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != info.get("sha256", {}).get("samples.npz"):
            _fail(path, "reference query checksum differs")
        with np.load(path, allow_pickle=False) as archive:
            split, k, theta = (archive[key] for key in ("split", "k0", "theta"))
    except (OSError, KeyError, ValueError) as error:
        _fail(path, f"cannot read reference queries: {error}")
    if (split.ndim != 1 or k.shape != split.shape or theta.shape != split.shape
            or len(split) != info.get("samples") or split.dtype.kind not in "iu"
            or not np.isin(split, (0, 1, 2)).all()
            or not np.isfinite(k).all() or not np.isfinite(theta).all()):
        _fail(path, "invalid reference sample arrays")
    indices = np.flatnonzero(split == 2)
    if not len(indices):
        _fail(path, "reference test split is empty")
    return indices, k, theta


def _rows(value, path, methods, indices, k, theta):
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        _fail(path, "rows must be a nonempty list")
    expected = set(map(int, indices))
    seen = {method: set() for method in methods}
    for position, row in enumerate(rows):
        context = f"{path}: row {position}"
        if not isinstance(row, dict) or row.get("method") not in methods:
            _fail(context, "unexpected or missing method")
        method, index = row["method"], row.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index not in expected:
            _fail(context, "index is not an expected held-out test sample")
        if index in seen[method]:
            _fail(context, f"duplicate sample index {index} for {method}")
        seen[method].add(index)
        if "sample_index" in row and row["sample_index"] != index:
            _fail(context, "sample_index disagrees with index")
        for name, reference in (("k", k[index]), ("theta", theta[index])):
            actual = _number(row.get(name), f"{context}.{name}", minimum=-math.inf)
            if not math.isclose(actual, float(reference), rel_tol=0, abs_tol=1e-12):
                _fail(context, f"{name} differs from reference sample {index}")
        for metric in REQUIRED_METRICS:
            _number(row.get(metric), f"{context}.{metric}")
    for method, actual in seen.items():
        if actual != expected:
            _fail(path, f"{method} has {len(actual)} samples; expected all {len(expected)} test indices")
    return {method: len(actual) for method, actual in sorted(seen.items())}


def _training(metrics, history, expected_count, context):
    if metrics.get("test_samples") != expected_count:
        _fail(context, "SNO metrics test_samples differs from the reference split")
    epochs = metrics.get("completed_epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        _fail(context, "SNO completed_epochs must be a positive integer")
    for name in ("train", "val", "lr"):
        values = history.get(name)
        if not isinstance(values, list) or len(values) != epochs:
            _fail(context, f"SNO history {name} does not contain {epochs} epochs")
        for position, value in enumerate(values):
            _number(value, f"{context}.history.{name}[{position}]")
    for name in ("mean", "median", "p95", "max"):
        _number(metrics.get("test_fem_l2", {}).get(name), f"{context}.test_fem_l2.{name}")


def _reproduction(summary, context):
    if summary.get("passed") is not True:
        _fail(context, "saved-table reproduction has not passed")
    tolerance = _number(summary.get("tolerance_percentage_points"),
                        f"{context}.tolerance_percentage_points")
    difference = _number(summary.get("max_error_percentage_point_difference"),
                         f"{context}.max_error_percentage_point_difference")
    if difference > tolerance:
        _fail(context, "saved-table reproduction exceeds its declared tolerance")
    count = summary.get("compared_values")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        _fail(context, "saved-table reproduction compared no values")


def validate_results(results_dir, dataset_root):
    """Require both bands, every paired method, and frozen low-to-high evaluation.

    ``dataset_root`` contains ``low/`` and ``high/`` native datasets. A concise
    validation manifest is returned; any failed requirement raises ValueError.
    """
    results_dir, dataset_root = Path(results_dir), Path(dataset_root)
    manifest = {"passed": True, "bands": {}}
    for band in ("low", "high"):
        directory = results_dir / band
        indices, k, theta = _queries(dataset_root / band)
        metrics = _read(directory / "sno_metrics.json")
        history = _read(directory / "sno_history.json")
        _training(metrics, history, len(indices), directory)
        sno_path, baseline_path = directory / "sno.json", directory / "baselines.json"
        counts = _rows(_read(sno_path), sno_path, {f"SNO {band}"}, indices, k, theta)
        counts.update(_rows(_read(baseline_path), baseline_path,
                            BASELINE_METHODS, indices, k, theta))
        reproduction_path = directory / "table_reproduction" / "summary.json"
        reproduction = _read(reproduction_path)
        _reproduction(reproduction, reproduction_path)
        if band == "high":
            extrapolation_path = directory / "extrapolation.json"
            extrapolation_indices = indices[k[indices] > 2]
            if not len(extrapolation_indices):
                _fail(directory, "high-band test split contains no k>2 samples")
            counts.update(_rows(_read(extrapolation_path), extrapolation_path,
                                {"SNO low extrapolation"}, extrapolation_indices, k, theta))
        manifest["bands"][band] = {
            "test_samples": len(indices), "method_samples": counts,
            "reproduction_max_difference_percentage_points":
                reproduction["max_error_percentage_point_difference"],
        }
    return manifest
