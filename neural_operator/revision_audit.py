"""Re-evaluate saved runs on the frozen test split and write compact provenance.

This command never trains or selects a model. It checks dataset fingerprints,
test-row ordering, portable checkpoint inference and archived predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import load_dataset
from .training import dataset_fingerprint, load_predictor, summarize


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_run(dataset, directory, fingerprint):
    directory = Path(directory)
    metrics = json.loads((directory / "metrics.json").read_text())
    if metrics["dataset_fingerprint"] != fingerprint:
        raise ValueError(f"Dataset fingerprint differs for {directory}")
    test = dataset.split_indices()[2]
    predictor = load_predictor(directory / "model.pt", device="cpu")
    prediction = predictor.predict_dataset(dataset, test)
    errors = dataset.relative_l2(dataset.u_scat[test], prediction)
    recomputed = summarize(errors)
    if not np.isfinite(errors).all():
        raise ValueError(f"Nonfinite test errors for {directory}")
    for name, value in recomputed.items():
        np.testing.assert_allclose(value, metrics["test_fem_l2"][name], atol=1e-8, rtol=1e-5)
    with np.load(directory / "test_predictions.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["indices"], test)
        np.testing.assert_allclose(prediction, archive["prediction"], atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(errors, archive["fem_l2"], atol=1e-8, rtol=1e-5)
        max_difference = float(np.max(np.abs(prediction - archive["prediction"])))
    files = ("model.pt", "config.json", "metrics.json", "history.json", "test_predictions.npz")
    return {"directory": str(directory), "model_kwargs": predictor.model_kwargs,
            "saved_metrics": metrics, "recomputed_test_fem_l2": recomputed,
            "verified_test_samples": len(test),
            "max_abs_prediction_difference_from_archive": max_difference,
            "sha256": {name: sha256(directory / name) for name in files}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--current-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--comparison-run", type=Path, action="append", default=[],
                        help="Optional additional completed run; verify on the same frozen test split")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset, mmap=True)
    fingerprint = dataset_fingerprint(dataset)
    current = audit_run(dataset, args.current_run, fingerprint)
    baseline = audit_run(dataset, args.baseline_run, fingerprint)
    source = Path(__file__).resolve().parent
    root = source.parent
    source_paths = ("_layers.py", "sno/model.py", "sno/basis.py", "chebyshev/model.py",
                    "fno/model.py", "training.py", "losses.py", "revision_audit.py")
    report = {
        "schema_version": 1,
        "scope": "One fixed medium; held-out frequency groups; one seed; FEM-label error only",
        "dataset": str(args.dataset), "dataset_fingerprint": fingerprint,
        "split_strategy": dataset.split_strategy,
        "optimizer": {"name": "AdamW", "learning_rate": current["saved_metrics"]["config"]["learning_rate"],
                      "weight_decay": 1e-5, "gradient_clip_norm": 1.},
        "current": current, "baseline": baseline,
        "additional_runs": [audit_run(dataset, path, fingerprint) for path in args.comparison_run],
        "mean_error_difference": (current["recomputed_test_fem_l2"]["mean"] -
                                  baseline["recomputed_test_fem_l2"]["mean"]),
        "source_sha256": {str((source / name).relative_to(root)): sha256(source / name)
                          for name in source_paths},
        "limitations": ["No claim of statistical superiority from one seed",
                        "Training timings are separate runs, not a controlled speed benchmark",
                        "Zero input material POD ranks prevent generalization to new materials",
                        "Surrogate errors do not certify continuum Helmholtz accuracy"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "test_fem_l2": current["recomputed_test_fem_l2"],
                      "baseline_test_fem_l2": baseline["recomputed_test_fem_l2"]}, indent=2))


if __name__ == "__main__":
    main()
