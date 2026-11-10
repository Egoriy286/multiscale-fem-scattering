"""Prepare frozen predictions and native P1 fields for solution visualizations.

Run ``python -m experiments.reference_sno_gmsfem.visual_fields`` inside the
complex container. This writes only results/visuals, never trains, and checks
every selected solution against the previously published paired-case metrics.
The 401-square grid is a display grid: integral errors retain the original
reference-mesh quadrature and are not recomputed on display pixels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "results/visuals"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path):
    rows = json.loads(path.read_text())["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing saved paired rows: {path}")
    return rows


def _saved_row(rows, index, method=None):
    matched = [row for row in rows if row["index"] == index
               and (method is None or row["method"] == method)]
    if len(matched) != 1:
        raise ValueError(f"Expected one saved row for sample {index}, method {method}")
    return matched[0]


def _select(dataset, target):
    """Nearest held-out frequency, then smallest saved incidence angle."""
    test = np.flatnonzero(dataset.split == 2)
    frequencies = np.unique(dataset.k0[test])
    frequency = frequencies[np.argmin(np.abs(frequencies-target))]
    candidates = test[dataset.k0[test] == frequency]
    index = int(candidates[np.argmin(dataset.theta[candidates])])
    if dataset.split[index] != 2:
        raise ValueError("Visual case must be held out")
    return index


def _check_metrics(recomputed, saved, context, percentage_tolerance=1e-6,
                   absolute_l2_tolerance=1e-8):
    errors = []
    for name, value in saved.items():
        if not name.startswith(("total_e_", "scattered_e_", "total_l2_error_", "scattered_l2_error_")):
            continue
        if name not in recomputed or not np.isfinite(value) or not np.isfinite(recomputed[name]):
            raise ValueError(f"{context}: nonfinite or missing metric {name}")
        difference = abs(float(recomputed[name])-float(value))
        is_percentage = name.startswith(("total_e_", "scattered_e_"))
        tolerance = percentage_tolerance if is_percentage else absolute_l2_tolerance
        if difference > tolerance:
            raise ValueError(f"{context}: {name} differs by {difference:g}, tolerance {tolerance:g}")
        errors.append((is_percentage, difference))
    if len(errors) != 24:
        raise ValueError(f"{context}: expected all 24 saved field metrics")
    return dict(passed=True, compared_values=len(errors),
                max_percentage_point_difference=max(diff for percent, diff in errors if percent),
                max_absolute_l2_difference=max(diff for percent, diff in errors if not percent),
                tolerance_percentage_points=percentage_tolerance,
                absolute_l2_tolerance=absolute_l2_tolerance)


def _model_features(predictor, theta, k):
    """Exact scalar features presented to this frozen static-medium model."""
    stats = predictor.preprocessing["k_stats"]
    mean, std = float(np.asarray(stats["mean"])), float(np.asarray(stats["std"]))
    ranks = [int(predictor.bases[name].rank) for name in ("A", "n")]
    if ranks != [0, 0]:
        raise ValueError("Expected zero material POD ranks for the fixed-medium model")
    return dict(cos_theta=float(np.cos(theta)), sin_theta=float(np.sin(theta)),
                normalized_k=(k-mean)/std, k_mean=mean, k_std=std,
                material_ranks=ranks)


def prepare_visual_fields(output_dir=DEFAULT_OUTPUT):
    from gmsfem.fem import MATERIALS
    from gmsfem.metrics import BARYCENTRIC
    from gmsfem.offline import build_offline_space, solve_reduced
    from neural_operator.fem import HelmholtzSolver
    from neural_operator.mesh_transfer import P1Transfer
    from neural_operator.training import load_predictor
    from .baselines import (FINE_MESH, FINE_SHA256, REFERENCE_MESH,
                            REFERENCE_SHA256, metrics_on_reference)
    from .dataset import load_dataset

    started = perf_counter()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = {band: HERE/"runs"/band/"model.pt" for band in ("low", "high")}
    source_paths = [
        Path(__file__), HERE/"dataset.py", HERE/"baselines.py",
        ROOT/"gmsfem/fem.py", ROOT/"gmsfem/offline.py", ROOT/"gmsfem/metrics.py",
        ROOT/"neural_operator/fem.py", ROOT/"neural_operator/training.py",
        ROOT/"neural_operator/sno/model.py", ROOT/"neural_operator/sno/basis.py",
        ROOT/"neural_operator/mesh_transfer.py", FINE_MESH, REFERENCE_MESH,
        *checkpoints.values(),
    ]
    before = {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths}
    if _sha256(FINE_MESH) != FINE_SHA256 or _sha256(REFERENCE_MESH) != REFERENCE_SHA256:
        raise ValueError("Authoritative mesh hashes differ")
    datasets = {band: load_dataset(HERE/"data"/band) for band in ("low", "high")}
    for band in ("low", "high"):
        directory = HERE/"data"/band
        expected = json.loads((directory/"dataset_info.json").read_text())["sha256"]
        for filename, digest in expected.items():
            path = directory/filename
            actual = _sha256(path)
            if actual != digest:
                raise ValueError(f"Dataset hash differs: {path}")
            before[str(path.relative_to(ROOT))] = actual
        for filename in ("sno.json", "baselines.json"):
            path = HERE/"results"/band/filename
            before[str(path.relative_to(ROOT))] = _sha256(path)
    ood_path = HERE/"results/high/extrapolation.json"
    before[str(ood_path.relative_to(ROOT))] = _sha256(ood_path)
    for key in ("coordinates", "triangles", "A_cells", "n_cells"):
        first = getattr(datasets["low"], key)
        second = getattr(datasets["high"], key)
        if not np.array_equal(first[:1] if key.endswith("cells") else first,
                              second[:1] if key.endswith("cells") else second):
            raise ValueError(f"Low/high fixed geometry or coefficients differ: {key}")
    predictors = {band: load_predictor(path) for band, path in checkpoints.items()}
    saved_sno = {band: _read_rows(HERE/"results"/band/"sno.json") for band in datasets}
    saved_gms = {band: _read_rows(HERE/"results"/band/"baselines.json") for band in datasets}
    saved_ood = _read_rows(ood_path)
    reference = datasets["low"]
    for band, predictor in predictors.items():
        if (not np.array_equal(predictor.coordinates, reference.coordinates)
                or not np.array_equal(predictor.triangles, reference.triangles)):
            raise ValueError(f"Frozen {band} checkpoint geometry differs from reference")
    x = y = np.linspace(-2., 2., 401)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    display_points = np.column_stack((xx.ravel(), yy.ravel()))
    reference_grid = P1Transfer(reference.coordinates, reference.triangles, display_points)
    source = HelmholtzSolver(FINE_MESH)
    source_grid = P1Transfer(source.coordinates, source.triangles, display_points)
    reference_points = np.einsum("qj,cjd->cqd", BARYCENTRIC,
                                 reference.coordinates[reference.triangles]).reshape(-1, 2)
    source_quadrature = P1Transfer(source.coordinates, source.triangles, reference_points)
    alpha = np.asarray([MATERIALS["notebook"][int(tag)][0] for tag in source.cell_tags])
    refractive = np.asarray([MATERIALS["notebook"][int(tag)][1] for tag in source.cell_tags])
    fields, cases = dict(x=x, y=y), []
    selection = (("low_k1", "low", 1.), ("low_k2", "low", 2.),
                 ("high_k3", "high", 3.), ("high_k4", "high", 4.))
    for identifier, band, target in selection:
        dataset = datasets[band]
        index = _select(dataset, target)
        k, theta = float(dataset.k0[index]), float(dataset.theta[index])
        if identifier == "low_k1" and (k != 1. or theta != 0.):
            raise ValueError("k=1,d=(1,0) anchor selection differs")
        if identifier in ("low_k2", "high_k4") and k != target:
            raise ValueError(f"Expected exact band endpoint {target}")
        direction = [float(np.cos(theta)), float(np.sin(theta))]
        sno_row = _saved_row(saved_sno[band], index)
        gms_row = _saved_row(saved_gms[band], index, "GMsFEM 20x20 Type 2 M=16")
        for row in (sno_row, gms_row):
            if row["k"] != k or row["theta"] != theta:
                raise ValueError("Saved case parameters differ from selected reference sample")
        native_reference = np.asarray(dataset.u_scat[index])
        native_sno = predictors[band].predict_dataset(dataset, np.asarray([index]))[0]
        checks = {"sno": _check_metrics(metrics_on_reference(dataset, index, native_sno),
                                        sno_row, identifier+" SNO")}
        problem = source.build_problem(alpha, refractive, k, theta)
        space = build_offline_space(problem, n_coarse=20, max_modes=16,
                                    basis_type=2, progress=None)
        native_gms, diagnostics = solve_reduced(problem, space, 16)
        # Metrics use source P1 directly at reference quadrature, exactly as
        # baselines.py. No coarse->reference-node->pixel chain is involved.
        checks["gms"] = _check_metrics(metrics_on_reference(
            dataset, index, np.zeros(len(dataset.coordinates), dtype=complex),
            quadrature_prediction=source_quadrature.apply(native_gms)),
            gms_row, identifier+" GMsFEM")
        fields[identifier+"_reference_scattered"] = reference_grid.apply(native_reference).reshape(401, 401)
        fields[identifier+"_sno_scattered"] = reference_grid.apply(native_sno).reshape(401, 401)
        fields[identifier+"_gms_scattered"] = source_grid.apply(native_gms).reshape(401, 401)
        fields[identifier+"_incident"] = np.exp(1j*k*(display_points @ np.asarray(direction))).reshape(401, 401)
        ood_row = None
        if band == "high":
            native_ood = predictors["low"].predict_dataset(dataset, np.asarray([index]))[0]
            ood_row = _saved_row(saved_ood, index)
            checks["ood"] = _check_metrics(metrics_on_reference(dataset, index, native_ood),
                                            ood_row, identifier+" frozen low SNO")
            fields[identifier+"_sno_low_scattered"] = reference_grid.apply(native_ood).reshape(401, 401)
        cases.append(dict(id=identifier, band=band, index=index, k=k, theta=theta,
                          theta_deg=float(np.degrees(theta)), direction=direction,
                          model_features=_model_features(predictors[band], theta, k),
                          ood_model_features=_model_features(predictors["low"], theta, k) if band == "high" else None,
                          split="test", sno_metrics=sno_row, gms_metrics=gms_row,
                          ood_metrics=ood_row, checks=checks,
                          gms_solve_diagnostics=diagnostics,
                          source_mesh_dofs=len(source.coordinates),
                          reference_mesh_dofs=len(reference.coordinates)))
        print(f"Prepared {identifier}: index={index}, actual k={k:.12g}, theta={theta:.12g}; metric checks passed", flush=True)
    for name, array in fields.items():
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite display field {name}")
        if name not in ("x", "y") and (array.shape != (401, 401) or not np.iscomplexobj(array)):
            raise ValueError(f"Incorrect display field schema {name}")
    after = {relative: _sha256(ROOT/relative) for relative in before}
    if before != after:
        raise ValueError("A source, checkpoint, dataset or paired-result file changed during preparation")
    temporary = output/"fields.tmp.npz"
    np.savez_compressed(temporary, **fields)
    temporary.replace(output/"fields.npz")
    np.savez_compressed(output/"inputs.npz", coordinates=reference.coordinates,
                        triangles=reference.triangles, alpha_cells=reference.A_cells[0],
                        n_cells=reference.n_cells[0], alpha_nodes=reference.A[0],
                        n_nodes=reference.n[0], cell_tags=reference.cell_tags,
                        circles=reference.circles)
    metadata = dict(cases=cases, geometry=dict(
        domain=[[-2., 2.], [-2., 2.]], reference_dofs=len(reference.coordinates),
        reference_triangles=len(reference.triangles), gms_fine_dofs=len(source.coordinates),
        inclusion_count=len(reference.circles), circle_columns=["x", "y", "radius", "material_tag"],
        materials={str(tag): {"alpha": pair[0], "n": pair[1]} for tag, pair in MATERIALS["notebook"].items()},
        coefficient_representation="Exact elementwise DG0 in inputs.npz; nodal averages included only as optional model input context",
        grid_shape=[401, 401], grid_extent=[-2., 2., -2., 2.], array_axes=["y", "x"],
        interpolation="Each native complex P1 field evaluated directly on the common grid; incident wave evaluated analytically",
        saved_fields="Complex scattered fields; total field = scattered + incident",
        difference_convention="Compute complex difference before taking abs; amplitude difference is separately abs(approx)-abs(reference)",
        metric_protocol="Saved FEM quadrature metrics; not a display-pixel norm"),
        immutable_inputs_sha256=before, immutable_inputs_unchanged=True,
        generated_field_keys=sorted(fields), training_epochs=0,
        generation_seconds=perf_counter()-started)
    (output/"cases.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False,
                                               allow_nan=False)+"\n")
    print(f"Saved fields.npz, inputs.npz, cases.json in {output}; all immutable hashes unchanged", flush=True)
    return metadata


if __name__ == "__main__":
    prepare_visual_fields()
