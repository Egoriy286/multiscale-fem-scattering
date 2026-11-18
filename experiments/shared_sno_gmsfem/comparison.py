"""Strictly paired cached GMsFEM scores and independent numerical spot checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.reference_sno_gmsfem.dataset import _sha256, load_dataset
from experiments.reference_sno_gmsfem.report_validation import (
    BASELINE_METHODS, REQUIRED_METRICS, _queries, _read, _reproduction, _rows,
    _training,
)
from experiments.reference_sno_gmsfem.run import write_csv, write_json
from .dataset import SOURCES


def remap_baseline_rows(source_values, source_queries, shared_queries):
    """Preserve numerical scores and their source identity; change indices only."""
    mapping = {(str(band), int(index)): i for i, (band, index) in enumerate(zip(
        shared_queries["source_band"], shared_queries["source_index"]))}
    if len(mapping) != len(shared_queries["k0"]):
        raise ValueError("Shared source mapping contains duplicates")
    result = []
    for band in SOURCES:
        queries = source_queries[band]
        test = np.flatnonzero(queries["split"] == 2)
        value = source_values[band]
        _rows(value, band, BASELINE_METHODS, test, queries["k0"], queries["theta"])
        for old in value["rows"]:
            index = old["index"]
            if (old.get("sample_index") != index
                    or old.get("frequency_id") != int(queries["frequency_id"][index])):
                raise ValueError("Source baseline sample/frequency identity differs")
            new_index = mapping.get((band, index))
            if new_index is None or shared_queries["split"][new_index] != 2:
                raise ValueError("Source test query missing from shared test")
            if (shared_queries["k0"][new_index] != queries["k0"][index]
                    or shared_queries["theta"][new_index] != queries["theta"][index]):
                raise ValueError("Shared physical query differs from source baseline")
            result.append(dict(old, index=new_index, sample_index=new_index,
                               frequency_id=int(shared_queries["frequency_id"][new_index]),
                               source_band=band, source_index=index,
                               source_frequency_id=old["frequency_id"]))
    _rows({"rows": result}, "shared", BASELINE_METHODS,
          np.flatnonzero(shared_queries["split"] == 2),
          shared_queries["k0"], shared_queries["theta"])
    return result


def import_baselines(data_dir, source_root, results_dir):
    """Reuse exact query scores, with checked mesh/material and artifact hashes."""
    from experiments.reference_sno_gmsfem.baselines import FINE_SHA256, REFERENCE_SHA256
    from gmsfem.fem import MATERIALS

    data_dir, source_root, results_dir = map(Path, (data_dir, source_root, results_dir))
    config = _read(data_dir / "config.json")
    with np.load(data_dir / "samples.npz", allow_pickle=False) as archive:
        shared = {name: archive[name] for name in archive.files}
    values, queries, sources = {}, {}, {}
    materials = {str(tag): list(pair) for tag, pair in MATERIALS["notebook"].items()}
    for band in SOURCES:
        directory = source_root / "data" / band
        info = _read(directory / "dataset_info.json")
        if config["sources"][band]["sha256"] != info["sha256"]:
            raise ValueError("Shared labels do not match baseline source dataset")
        for name, expected in info["sha256"].items():
            if _sha256(directory / name) != expected:
                raise ValueError(f"Baseline source checksum differs: {band}/{name}")
        _queries(directory)
        with np.load(directory / "samples.npz", allow_pickle=False) as archive:
            queries[band] = {name: archive[name] for name in archive.files}
        path = source_root / "results" / band / "baselines.json"
        value = _read(path)
        metadata = value["metadata"]
        for name, expected in (
            ("fine_mesh_sha256", FINE_SHA256), ("reference_mesh_sha256", REFERENCE_SHA256),
            ("case", "notebook"), ("materials", materials),
            ("coarse_grids", [10, 20]), ("basis_types", [1, 2]), ("modes", 16),
        ):
            if metadata.get(name) != expected:
                raise ValueError(f"Baseline {band} physics/settings mismatch: {name}")
        if sorted(metadata["sample_indices"]) != np.flatnonzero(queries[band]["split"] == 2).tolist():
            raise ValueError("Baseline metadata sample_indices differs")
        for row in value["rows"]:
            if row.get("dofs_reference") != info["nodes"] or row.get("dofs_fine") != 17549:
                raise ValueError("Baseline mesh dimensions differ")
            expected_method = ("FEM h=0.04" if row.get("basis_type") == 0 else
                               f"GMsFEM {row['n_coarse']}x{row['n_coarse']} "
                               f"Type {row['basis_type']} M={row['modes']}")
            if row["method"] != expected_method:
                raise ValueError("Baseline method label/settings differ")
        summary_path = path.parent / "table_reproduction" / "summary.json"
        _reproduction(_read(summary_path), summary_path)
        values[band] = value
        sources[band] = dict(path=str(path), sha256=_sha256(path),
                             reproduction_sha256=_sha256(summary_path),
                             metadata=metadata)
    rows = remap_baseline_rows(values, queries, shared)
    write_json(results_dir / "baselines.json", dict(rows=rows, metadata={
        "reuse": "exact source queries and verified reference labels; no interpolation of scores",
        "source_runs": sources, "sample_indices": np.flatnonzero(shared["split"] == 2).tolist(),
        "timing": "Historical independent source runs; k=2 setup and Type1 builds occurred twice",
    }))
    write_csv(results_dir / "baselines.csv", rows)
    print(f"Imported {len(rows)} paired baseline rows for {len(rows)//5} shared test queries", flush=True)


def verify_predictions(data_dir, run_dir, results_dir):
    """Reload every prediction and freshly solve four GMsFEM test problems."""
    import torch
    from neural_operator.training import dataset_fingerprint, load_predictor, summarize
    from experiments.reference_sno_gmsfem.baselines import evaluate_baselines
    from experiments.reference_sno_gmsfem.run import evaluation_rows

    data_dir, run_dir, results_dir = map(Path, (data_dir, run_dir, results_dir))
    data = load_dataset(data_dir)
    torch.set_num_threads(2)
    predictor = load_predictor(run_dir / "model.pt")
    test = data.split_indices()[2]
    with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["indices"], test):
            raise ValueError("Saved predictions have a different test split")
        saved = archive["prediction"]
        fresh = predictor.predict_dataset(data, test, batch_size=32)
        parity = data.relative_l2(saved, fresh)
        if not np.isfinite(parity).all() or parity.max() > 1e-6:
            raise ValueError("Reloaded checkpoint does not reproduce saved predictions")
        errors = data.relative_l2(data.u_scat[test], fresh)
        if not np.allclose(errors, archive["fem_l2"], rtol=1e-6, atol=1e-8):
            raise ValueError("Reloaded test errors differ")
        scored = {r["index"]: r for r in _read(results_dir / "sno.json")["rows"]}
        observed = np.array([scored[int(index)]["scattered_e_complex_ref"] for index in test])
        if not np.allclose(observed, 100 * errors, rtol=1e-6, atol=1e-6):
            raise ValueError("Published SNO scores differ from reloaded field errors")
    del saved, fresh
    metrics = _read(run_dir / "metrics.json")
    if metrics["dataset_fingerprint"] != dataset_fingerprint(data):
        raise ValueError("Checkpoint dataset fingerprint differs")
    summary = summarize(errors)
    if any(not np.isclose(value, metrics["test_fem_l2"][name], rtol=1e-6, atol=1e-8)
           for name, value in summary.items()):
        raise ValueError("Reported SNO summary differs from reloaded test errors")
    recomputed = evaluation_rows(data, predictor, test, "SVD-SNO shared")
    score_difference = max(abs(row[name] - scored[row["index"]][name])
                           for row in recomputed for name in REQUIRED_METRICS)
    if score_difference > 1e-6:
        raise ValueError("Published SNO quadrature metrics differ from reloaded predictions")
    selected = np.array([np.flatnonzero((data.k0 == k) & (data.split == 2))[0]
                         for k in (.1, 1., 2., 4.)])
    baseline = evaluate_baselines(data, results_dir / "verification", selected,
                                  coarse_grids=(20,), basis_types=(2,), modes=16,
                                  reproduce=False, save_fields=True)
    cached = {(row["method"], row["index"]): row
              for row in _read(results_dir / "baselines.json")["rows"]}
    differences = []
    for row in baseline["rows"]:
        previous = cached[row["method"], row["index"]]
        differences.extend(abs(row[name] - previous[name]) for name in REQUIRED_METRICS)
    maximum = max(differences)
    if maximum > 1e-6:
        raise ValueError(f"Fresh baseline spot checks differ from cached scores: {maximum}")
    checks = dict(passed=True, checkpoint_sha256=_sha256(run_dir / "model.pt"),
                  dataset_fingerprint=metrics["dataset_fingerprint"],
                  source_label_sha256=_sha256(data_dir / "u_scat.npy"),
                  all_test_predictions_reloaded=len(test),
                  max_reload_relative_l2=float(parity.max()),
                  max_sno_metric_difference=score_difference,
                  saved_predictions_sha256=_sha256(run_dir / "test_predictions.npz"),
                  history_sha256=_sha256(results_dir / "sno_history.json"),
                  baseline_spot_indices=selected.tolist(), baseline_spot_k=data.k0[selected].tolist(),
                  baseline_spot_methods=sorted(set(row["method"] for row in baseline["rows"])),
                  baseline_max_metric_difference=maximum,
                  baseline_metric_difference_tolerance=1e-6,
                  score_sha256={name: _sha256(results_dir / name)
                                for name in ("sno.json", "baselines.json", "sno_metrics.json")})
    write_json(results_dir / "verification.json", checks)
    print(f"Verified all {len(test)} reloaded predictions and fresh baselines at k=.1,1,2,4", flush=True)


def validate_shared_results(data_dir, run_dir, results_dir):
    """Refuse unpaired, incomplete, stale or physically mismatched reports."""
    from neural_operator.training import dataset_fingerprint

    data_dir, run_dir, results_dir = map(Path, (data_dir, run_dir, results_dir))
    test, k, theta = _queries(data_dir)
    with np.load(data_dir / "samples.npz", allow_pickle=False) as archive:
        groups, split = archive["frequency_id"], archive["split"]
        for frequency in np.unique(k):
            rows = k == frequency
            if len(np.unique(groups[rows])) != 1 or len(np.unique(split[rows])) != 1:
                raise ValueError("Physical frequency is split across groups or partitions")
        band, source_index = archive["source_band"], archive["source_index"]
    metrics = _read(results_dir / "sno_metrics.json")
    history = _read(results_dir / "sno_history.json")
    _training(metrics, history, len(test), results_dir)
    data = load_dataset(data_dir)
    if metrics["dataset_fingerprint"] != dataset_fingerprint(data):
        raise ValueError("Reported model uses a different dataset")
    if (metrics != _read(run_dir / "metrics.json")
            or history != _read(run_dir / "history.json")
            or metrics["config"] != _read(results_dir / "protocol.json")["config"]
            or metrics["best_epoch"] != int(np.argmin(history["val"]) + 1)
            or not np.isclose(metrics["best_val_fem_l2"], min(history["val"]), rtol=1e-12)
            or metrics["train_samples"] != int(np.count_nonzero(split == 0))
            or metrics["val_samples"] != int(np.count_nonzero(split == 1))):
        raise ValueError("Training protocol, selected checkpoint or split counts differ")
    counts = _rows(_read(results_dir / "sno.json"), "shared SNO", {"SVD-SNO shared"}, test, k, theta)
    baselines = _read(results_dir / "baselines.json")
    counts.update(_rows(baselines, "shared baselines", BASELINE_METHODS, test, k, theta))
    original = {}
    for source, metadata in baselines["metadata"]["source_runs"].items():
        path = Path(metadata["path"])
        if _sha256(path) != metadata["sha256"]:
            raise ValueError("Source baseline artifact changed")
        original.update({(source, r["method"], r["index"]): r for r in _read(path)["rows"]})
    for row in baselines["rows"]:
        index = row["index"]
        if (row.get("sample_index") != index or row.get("frequency_id") != int(groups[index])
                or row.get("source_band") != str(band[index])
                or row.get("source_index") != int(source_index[index])):
            raise ValueError("Baseline remapping provenance differs")
        source = original[row["source_band"], row["method"], row["source_index"]]
        if any(row[name] != source[name] for name in REQUIRED_METRICS):
            raise ValueError("Remapped baseline scores differ from source artifact")
    checks = _read(results_dir / "verification.json")
    if (checks.get("passed") is not True
            or checks["checkpoint_sha256"] != _sha256(run_dir / "model.pt")
            or checks["source_label_sha256"] != _sha256(data_dir / "u_scat.npy")
            or checks["all_test_predictions_reloaded"] != len(test)
            or checks["dataset_fingerprint"] != metrics["dataset_fingerprint"]
            or checks["saved_predictions_sha256"] != _sha256(run_dir / "test_predictions.npz")
            or checks["history_sha256"] != _sha256(results_dir / "sno_history.json")
            or any(_sha256(results_dir / name) != digest
                   for name, digest in checks["score_sha256"].items())):
        raise ValueError("Independent verification is missing or stale")
    return dict(passed=True, model_count=1, k_range=[float(k.min()), float(k.max())],
                test_samples=len(test), test_frequency_groups=len(np.unique(groups[test])),
                method_samples=counts, independent_verification=checks)
