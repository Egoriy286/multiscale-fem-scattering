"""Publication gate checks with tiny metadata; no FEM runtime or field arrays."""
import hashlib
import json

import numpy as np
import pytest

from experiments.reference_sno_gmsfem.report_validation import (
    BASELINE_METHODS, REQUIRED_METRICS, validate_results,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _change(path, update):
    value = json.loads(path.read_text())
    update(value)
    _write(path, value)


@pytest.fixture
def artifacts(tmp_path):
    results, data = tmp_path / "results", tmp_path / "data"
    for band in ("low", "high"):
        folder = data / band
        folder.mkdir(parents=True)
        k = [0.4, 0.8, 1., 2.] if band == "low" else [2.5, 3., 2., 4.]
        theta = [0., .1, .2, .3]
        np.savez(folder / "samples.npz", split=[0, 1, 2, 2], k0=k, theta=theta)
        _write(folder / "dataset_info.json", {
            "complete": True, "samples": 4,
            "sha256": {"samples.npz": hashlib.sha256((folder / "samples.npz").read_bytes()).hexdigest()},
        })

        def rows(method, indices=(2, 3)):
            return [dict(method=method, index=i, k=k[i], theta=theta[i],
                         **{metric: .1 for metric in REQUIRED_METRICS}) for i in indices]

        destination = results / band
        _write(destination / "sno_metrics.json", {
            "test_samples": 2, "completed_epochs": 2,
            "test_fem_l2": {name: .001 for name in ("mean", "median", "p95", "max")},
        })
        _write(destination / "sno_history.json", {name: [.1, .01] for name in ("train", "val", "lr")})
        _write(destination / "sno.json", {"rows": rows(f"SNO {band}")})
        _write(destination / "baselines.json", {
            "rows": [row for method in sorted(BASELINE_METHODS) for row in rows(method)],
        })
        _write(destination / "table_reproduction" / "summary.json", {
            "passed": True, "tolerance_percentage_points": 1e-5,
            "max_error_percentage_point_difference": 1e-7, "compared_values": 100,
        })
        if band == "high":
            _write(destination / "extrapolation.json", {"rows": rows("SNO low extrapolation", (3,))})
    return results, data


def test_complete_paired_report_passes_without_loading_fields(artifacts):
    manifest = validate_results(*artifacts)
    assert manifest["passed"] is True
    assert manifest["bands"]["low"]["test_samples"] == 2
    assert manifest["bands"]["high"]["method_samples"]["SNO low extrapolation"] == 1
    assert len(manifest["bands"]["low"]["method_samples"]) == 6


@pytest.mark.parametrize("relative", [
    "low/sno_metrics.json", "low/sno_history.json", "low/sno.json",
    "high/baselines.json", "high/table_reproduction/summary.json", "high/extrapolation.json",
])
def test_missing_required_artifact_fails(artifacts, relative):
    results, data = artifacts
    (results / relative).unlink()
    with pytest.raises(ValueError, match="required JSON artifact"):
        validate_results(results, data)


def test_missing_entire_baseline_method_fails(artifacts):
    results, data = artifacts
    _change(results / "low/baselines.json", lambda v: v.update(
        rows=[row for row in v["rows"] if row["method"] != "GMsFEM 20x20 Type 2 M=16"]))
    with pytest.raises(ValueError, match="expected all 2 test indices"):
        validate_results(results, data)


def test_duplicate_indices_fail_even_when_row_count_is_correct(artifacts):
    results, data = artifacts
    _change(results / "low/sno.json", lambda v: v["rows"].__setitem__(1, v["rows"][0].copy()))
    with pytest.raises(ValueError, match="duplicate sample index"):
        validate_results(results, data)


@pytest.mark.parametrize("key,value,match", [
    ("k", 1.123, "k differs"), ("theta", 1.123, "theta differs"),
    ("index", 0, "expected held-out"),
    ("scattered_e_complex_ref", None, "finite number"),
    ("total_e_complex_ref", float("nan"), "nonfinite"),
])
def test_mismatched_queries_or_invalid_metrics_fail(artifacts, key, value, match):
    results, data = artifacts
    _change(results / "high/baselines.json", lambda v: v["rows"][0].update({key: value}))
    with pytest.raises(ValueError, match=match):
        validate_results(results, data)


def test_extrapolation_must_exclude_the_shared_k2_endpoint(artifacts):
    results, data = artifacts
    _change(results / "high/extrapolation.json", lambda v: v["rows"][0].update(index=2, k=2., theta=.2))
    with pytest.raises(ValueError, match="expected held-out"):
        validate_results(results, data)


@pytest.mark.parametrize("update,match", [
    ({"passed": False}, "has not passed"),
    ({"max_error_percentage_point_difference": .1}, "exceeds its declared tolerance"),
    ({"tolerance_percentage_points": None}, "finite number"),
    ({"compared_values": 0}, "compared no values"),
])
def test_table_reproduction_requires_measured_pass(artifacts, update, match):
    results, data = artifacts
    _change(results / "low/table_reproduction/summary.json", lambda v: v.update(update))
    with pytest.raises(ValueError, match=match):
        validate_results(results, data)


def test_incomplete_training_history_fails(artifacts):
    results, data = artifacts
    _change(results / "high/sno_history.json", lambda v: v.update(val=[.1]))
    with pytest.raises(ValueError, match="does not contain 2 epochs"):
        validate_results(results, data)


def test_changed_reference_queries_fail(artifacts):
    results, data = artifacts
    with (data / "low/samples.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="reference query checksum differs"):
        validate_results(results, data)
