"""Paired GMsFEM baselines on the original mesh, scored on the reference mesh.

Paired errors evaluate each field directly at reference quadrature points;
reference-node transfers are saved only as convenient visualization fields.
No mesh is generated and no coefficients are sampled here.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import splu

from gmsfem.fem import MATERIALS
from gmsfem.metrics import BARYCENTRIC, WEIGHTS, _summaries, relative_errors


ROOT = Path(__file__).resolve().parents[2]
FINE_MESH = ROOT / "gmsfem/meshes/fine_n20_h0p04.msh"
REFERENCE_MESH = ROOT / "gmsfem/meshes/reference_n20_h0p04.msh"
FINE_SHA256 = "a8af14fc965d4823fcb5be5d6f102e7225ff2d553b1c4b3a71325834e30807de"
REFERENCE_SHA256 = "9273efc7c0b500209ff847de4ee36d3790b93d64b09e67f62348318ed6b6cb47"


def _checked_mesh(path, expected):
    path = Path(path).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Exact reference experiment mesh checksum differs: {path}")
    return path


def _materials(tags, case="notebook"):
    table = MATERIALS[case]
    if not set(np.unique(tags)).issubset(table):
        raise ValueError("Mesh has unknown material tags")
    return tuple(np.asarray([table[int(tag)][component] for tag in tags])
                 for component in (0, 1))


def _strict_json(value):
    if isinstance(value, dict):
        return {str(k): _strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def metrics_on_reference(dataset, index, prediction, *, quadrature_prediction=None):
    """Shared SNO/GMsFEM scoring; all prefixed e_* values are percentages.

    Predictions and labels are scattered nodal P1 fields on dataset geometry.
    quadrature_prediction overrides prediction evaluation for source-mesh fields,
    avoiding an extra interpolation through reference nodes.
    For total-field quadrature, the incident wave is analytic, matching the
    GMsFEM tables. error_fem_l2 is a ratio, with the scattered FEM denominator.
    """
    theta, k = float(dataset.theta[index]), float(dataset.k0[index])
    direction = (float(np.cos(theta)), float(np.sin(theta)))
    vertices = dataset.coordinates[dataset.triangles]
    edges = vertices[:, 1:] - vertices[:, :1]
    areas = np.abs(edges[:, 0, 0]*edges[:, 1, 1]
                   - edges[:, 0, 1]*edges[:, 1, 0])/2
    true = np.asarray(dataset.u_scat[index])
    prediction = np.asarray(prediction)
    if prediction.shape != true.shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction must be a finite scattered field on reference nodes")
    aq = prediction[dataset.triangles] @ BARYCENTRIC.T if quadrature_prediction is None else np.asarray(quadrature_prediction).reshape(len(areas), -1)
    rq = true[dataset.triangles] @ BARYCENTRIC.T
    if aq.shape != rq.shape or not np.isfinite(aq).all():
        raise ValueError("Quadrature prediction shape differs from reference quadrature")
    points = np.einsum("qj,cjd->cqd", BARYCENTRIC, vertices)
    incident = np.exp(1j*k*(points @ np.asarray(direction)))
    weights = areas[:, None]*WEIGHTS[None, :]
    result = {f"{field}_{name}": value
              for field, a, r in (("scattered", aq, rq), ("total", aq+incident, rq+incident))
              for name, value in _summaries(a, r, weights).items()}
    result["error_fem_l2"] = result["scattered_e_complex_ref"]/100
    return result


class _ReducedSystem:
    """One frequency's reduced matrix/factorization, reused for all angles."""
    def __init__(self, problem, space, modes):
        self.P = space.prolongation(modes)
        self.PH = self.P.conjugate().T
        started = perf_counter()
        matrix = (self.PH @ (problem.operator @ self.P)).tocsc()
        self.matrix_projection_seconds = perf_counter()-started
        started = perf_counter()
        self.factor = splu(matrix)
        self.factorization_seconds = perf_counter()-started

    def solve(self, problem):
        started = perf_counter()
        reduced_rhs = self.PH @ problem.rhs
        rhs_seconds = perf_counter()-started
        started = perf_counter()
        coefficients = self.factor.solve(reduced_rhs)
        solve_seconds = perf_counter()-started
        started = perf_counter()
        solution = np.asarray(self.P @ coefficients).ravel()
        reconstruction_seconds = perf_counter()-started
        residual = np.linalg.norm(self.PH @ (problem.operator @ solution-problem.rhs)) / max(
            np.linalg.norm(reduced_rhs), 1e-30)
        if not np.isfinite(solution).all() or residual > 1e-7:
            raise RuntimeError(f"Reduced solve residual {residual:.3e}")
        return solution, dict(
            dofs_coarse=int(self.P.shape[1]), galerkin_residual=float(residual),
            matrix_projection_seconds=self.matrix_projection_seconds,
            rhs_projection_seconds=rhs_seconds,
            projection_seconds=self.matrix_projection_seconds+rhs_seconds,
            factorization_seconds=self.factorization_seconds,
            solve_seconds=solve_seconds,
            reconstruction_seconds=reconstruction_seconds)


def evaluate_baselines(dataset, output_dir, indices, *, coarse_grids=(10, 20),
                       basis_types=(1, 2), modes=16, reproduce=True,
                       save_fields=True, progress=print):
    """Evaluate paired test samples and write CSV/JSON plus optional fields.

    Returns {rows, metadata, fields, reproduction}. Baselines use the original
    17549-DOF FEM mesh; dataset labels must use the exact 53246-DOF reference.
    Reused offline/projection/factorization timings are repeated on case rows
    for visibility; metadata records their true one-time totals separately.
    """
    from neural_operator.fem import HelmholtzSolver
    from neural_operator.mesh_transfer import P1Transfer
    from gmsfem.offline import build_offline_space

    selected = np.asarray(indices)
    if selected.ndim != 1 or not len(selected) or selected.dtype.kind not in "iu":
        raise ValueError("indices must be a nonempty integer vector")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("indices must not contain duplicates")
    if np.any(selected < 0) or np.any(selected >= len(dataset.k0)):
        raise ValueError("sample index outside dataset")
    if np.any(dataset.split[selected] != 2):
        raise ValueError("Paired comparison requires held-out test samples")
    if not coarse_grids or any(n not in (10, 20) for n in coarse_grids):
        raise ValueError("Table-compatible coarse_grids must be drawn from (10,20)")
    if not basis_types or any(t not in (1, 2) for t in basis_types):
        raise ValueError("basis_types must be drawn from (1,2)")
    if isinstance(modes, bool) or int(modes) != modes or modes < 1:
        raise ValueError("modes must be a positive integer")
    coarse_grids, basis_types = tuple(sorted(set(coarse_grids))), tuple(sorted(set(basis_types)))
    modes = int(modes)
    fine_path = _checked_mesh(FINE_MESH, FINE_SHA256)
    reference_path = _checked_mesh(REFERENCE_MESH, REFERENCE_SHA256)
    source = HelmholtzSolver(fine_path)
    reference = HelmholtzSolver(reference_path)
    if (not np.array_equal(reference.coordinates, dataset.coordinates)
            or not np.array_equal(reference.triangles, dataset.triangles)):
        raise ValueError("Dataset must use the exact saved reference mesh and DOF ordering")
    expected_a, expected_n = _materials(reference.cell_tags)
    for name, expected in (("A_cells", expected_a), ("n_cells", expected_n)):
        values = getattr(dataset, name, None)
        if values is None or any(not np.array_equal(values[i], expected) for i in selected):
            raise ValueError(f"Dataset {name} does not match notebook reference coefficients")
    alpha, refractive = _materials(source.cell_tags)
    started = perf_counter()
    transfer = P1Transfer(source.coordinates, source.triangles, dataset.coordinates)
    quad_points = np.einsum("qj,cjd->cqd", BARYCENTRIC, dataset.coordinates[dataset.triangles])
    quad_transfer = P1Transfer(source.coordinates, source.triangles, quad_points.reshape(-1, 2))
    transfer_setup_seconds = perf_counter()-started
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, fields, type1_spaces, build_counts = [], {}, {}, {}
    totals = dict(offline_seconds=0., matrix_projection_seconds=0.,
                  factorization_seconds=0., assembly_seconds=0.)
    if save_fields:
        for nc in coarse_grids:
            for kind in basis_types:
                fields[f"gmsfem_n{nc}_type{kind}_M{modes}"] = np.empty(
                    (len(selected), len(dataset.coordinates)), dtype=np.complex128)
        fields["fine_h"] = np.empty((len(selected), len(dataset.coordinates)), dtype=np.complex128)
    positions = {int(index): j for j, index in enumerate(selected)}
    for k in np.unique(dataset.k0[selected]):
        frequency_indices = selected[dataset.k0[selected] == k]
        first = int(frequency_indices[0])
        started = perf_counter()
        base_problem = source.build_problem(alpha, refractive, float(k), float(dataset.theta[first]))
        assembly_seconds = perf_counter()-started
        totals["assembly_seconds"] += assembly_seconds
        started = perf_counter()
        fine_factor = splu(base_problem.operator.tocsc())
        fine_factorization_seconds = perf_counter()-started
        systems, spaces = {}, {}
        for nc in coarse_grids:
            for kind in basis_types:
                key = (nc, kind)
                if kind == 1 and nc in type1_spaces:
                    space = type1_spaces[nc]
                else:
                    space = build_offline_space(base_problem, n_coarse=nc,
                                                max_modes=modes, basis_type=kind,
                                                progress=None)
                    count_key = f"n{nc}_type{kind}"
                    build_counts[count_key] = build_counts.get(count_key, 0)+1
                    totals["offline_seconds"] += space.build_seconds
                    if kind == 1:
                        type1_spaces[nc] = space
                spaces[key] = space
                system = _ReducedSystem(base_problem, space, modes)
                systems[key] = system
                totals["matrix_projection_seconds"] += system.matrix_projection_seconds
                totals["factorization_seconds"] += system.factorization_seconds
        for raw_index in frequency_indices:
            index = int(raw_index)
            theta = float(dataset.theta[index])
            direction = (float(np.cos(theta)), float(np.sin(theta)))
            started = perf_counter()
            rhs = source.rhs(alpha, refractive, float(k), theta)
            rhs_seconds = perf_counter()-started
            problem = replace(base_problem, rhs=rhs, direction=direction,
                              incident=np.exp(1j*float(k)*(source.coordinates @ np.asarray(direction))))
            common = dict(index=index, sample_index=index, frequency_id=int(dataset.frequency_id[index]),
                          k=float(k), theta=theta, dofs_fine=len(source.coordinates),
                          dofs_reference=len(dataset.coordinates), assembly_seconds=assembly_seconds,
                          rhs_assembly_seconds=rhs_seconds)
            started = perf_counter()
            fine = fine_factor.solve(rhs)
            fine_solve_seconds = perf_counter()-started
            started = perf_counter()
            mapped_fine = transfer.apply(fine)
            quadrature_fine = quad_transfer.apply(fine)
            transfer_seconds = perf_counter()-started
            rows.append(dict(common, method="FEM h=0.04", n_coarse=0, basis_type=0, modes=0,
                             dofs_coarse=len(fine), offline_seconds=0., projection_seconds=0.,
                             factorization_seconds=fine_factorization_seconds,
                             solve_seconds=fine_solve_seconds, transfer_seconds=transfer_seconds,
                             **metrics_on_reference(dataset, index, mapped_fine, quadrature_prediction=quadrature_fine)))
            if save_fields:
                fields["fine_h"][positions[index]] = mapped_fine
            for (nc, kind), system in systems.items():
                solution, diagnostics = system.solve(problem)
                started = perf_counter()
                mapped = transfer.apply(solution)
                quadrature_solution = quad_transfer.apply(solution)
                transfer_seconds = perf_counter()-started
                rows.append(dict(common, method=f"GMsFEM {nc}x{nc} Type {kind} M={modes}",
                                 n_coarse=nc, basis_type=kind, modes=modes,
                                 offline_seconds=spaces[nc, kind].build_seconds,
                                 transfer_seconds=transfer_seconds, **diagnostics,
                                 **metrics_on_reference(dataset, index, mapped, quadrature_prediction=quadrature_solution)))
                if save_fields:
                    fields[f"gmsfem_n{nc}_type{kind}_M{modes}"][positions[index]] = mapped
        if progress:
            progress(f"GMsFEM paired test k={k:g}: {len(frequency_indices)} directions", flush=True) if progress is print else progress(
                f"GMsFEM paired test k={k:g}: {len(frequency_indices)} directions")
    metadata = dict(fine_mesh=str(fine_path), fine_mesh_sha256=FINE_SHA256,
                    reference_mesh=str(reference_path), reference_mesh_sha256=REFERENCE_SHA256,
                    case="notebook", materials={str(tag): list(pair) for tag, pair in MATERIALS["notebook"].items()},
                    sample_indices=selected.tolist(), coarse_grids=list(coarse_grids),
                    basis_types=list(basis_types), modes=modes, offline_build_counts=build_counts,
                    unique_frequency_count=len(np.unique(dataset.k0[selected])),
                    transfer_setup_seconds=transfer_setup_seconds, unique_cost_totals=totals,
                    comparison_protocol="Original fine P1 evaluated directly at reference-mesh quadrature points via a cached sparse transfer; analytic total incident wave. Reference-node fields are visualization output only.",
                    error_units="error_fem_l2 is a ratio; all scattered_e_* and total_e_* are percent",
                    timing_protocol="Offline once per grid for Type1, per k/grid for Type2; matrix projection and factorization once per k/grid/type. Reused costs repeat in per-case rows; unique totals are separate.")
    _write_csv(output/"baselines.csv", rows)
    (output/"baselines.json").write_text(json.dumps(_strict_json(dict(rows=rows, metadata=metadata)), indent=2, allow_nan=False)+"\n")
    if save_fields:
        fields["sample_indices"] = selected
        np.savez_compressed(output/"baseline_fields.npz", **fields)
    reproduction = reproduce_saved_tables(output/"table_reproduction", progress=progress) if reproduce else None
    return dict(rows=rows, metadata=metadata, fields=fields, reproduction=reproduction)


def reproduce_saved_tables(output_dir, *, case="notebook", modes=(1, 2, 4, 6, 8, 12, 16),
                           percentage_point_tolerance=1e-6,
                           absolute_l2_tolerance=1e-8, progress=print):
    """Re-solve k=1,theta=0 and report measured differences from saved CSVs.

    Uses the original UFL assembler and original metric functions, independently
    of the faster paired-test solver. Missing saved rows are reported as errors.
    """
    from gmsfem.fem import assemble_problem, solve_fine
    from gmsfem.metrics import errors_on_reference
    from gmsfem.offline import build_offline_space, solve_reduced

    for name, tolerance in (("percentage_point_tolerance", percentage_point_tolerance),
                            ("absolute_l2_tolerance", absolute_l2_tolerance)):
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(f"{name} must be positive and finite")
    fine = assemble_problem(_checked_mesh(FINE_MESH, FINE_SHA256), case=case, k=1., direction=(1., 0.))
    reference = assemble_problem(_checked_mesh(REFERENCE_MESH, REFERENCE_SHA256), case=case, k=1., direction=(1., 0.))
    fine_values, fine_info = solve_fine(fine)
    reference_values, reference_info = solve_fine(reference)
    saved_root = ROOT/"gmsfem/results"/case
    with (saved_root/"errors.csv").open() as stream:
        saved = {(int(r["n_coarse"]), int(r["basis_type"]), int(r["modes"])): r for r in csv.DictReader(stream)}
    with (saved_root/"reference_errors.csv").open() as stream:
        saved_reference = {r["label"]: r for r in csv.DictReader(stream)}
    differences, actual_rows, reference_rows = [], [], []

    def compare(actual, expected, table, label):
        for name, value in actual.items():
            if not (name.startswith("e_") or name.startswith("l2_error_") or name.startswith("scattered_e_") or name.startswith("scattered_l2_error_")):
                continue
            old = float(expected[name])
            differences.append(dict(table=table, label=label, metric=name,
                                    saved=old, recomputed=float(value),
                                    signed_difference=float(value)-old,
                                    absolute_difference=abs(float(value)-old)))

    def refined_row(label, values):
        row = dict(label=label, **errors_on_reference(fine, values, reference, reference_values, field="total"))
        reference_rows.append(row)
        compare(row, saved_reference[label], "reference_errors.csv", label)

    refined_row("МКЭ h", fine_values)
    for nc in (10, 20):
        for kind in (1, 2):
            space = build_offline_space(fine, n_coarse=nc, max_modes=max(modes), basis_type=kind, progress=None)
            for m in modes:
                values, diagnostics = solve_reduced(fine, space, m)
                row = dict(n_coarse=nc, basis_type=kind, modes=m, **diagnostics,
                           **relative_errors(fine, values, fine_values, field="total"))
                row.update({"scattered_"+key: value for key, value in relative_errors(
                    fine, values, fine_values, field="scattered").items()})
                actual_rows.append(row)
                compare(row, saved[nc, kind, m], "errors.csv", f"n{nc}_type{kind}_M{m}")
                if m == 16:
                    refined_row(f"GMsFEM {nc}×{nc}, Type {kind}, M=16", values)
            if progress:
                progress(f"Saved-table reproduction: {nc}x{nc} Type {kind}")
    all_finite = all(np.isfinite(row[key]) for row in differences
                     for key in ("saved", "recomputed", "absolute_difference"))
    max_percentage_difference = max(row["absolute_difference"] for row in differences
                                    if row["metric"].startswith(("e_", "scattered_e_")))
    max_absolute_l2_difference = max(row["absolute_difference"] for row in differences
                                    if row["metric"].startswith(("l2_error_", "scattered_l2_error_")))
    passed = (all_finite and max_percentage_difference <= percentage_point_tolerance
              and max_absolute_l2_difference <= absolute_l2_tolerance)
    result = dict(case=case, k=1., theta=0., fine_diagnostics=fine_info,
                  reference_diagnostics=reference_info, compared_values=len(differences),
                  max_absolute_difference=max(row["absolute_difference"] for row in differences),
                  max_error_percentage_point_difference=max_percentage_difference,
                  max_absolute_l2_difference=max_absolute_l2_difference,
                  percentage_point_tolerance=float(percentage_point_tolerance),
                  tolerance_percentage_points=float(percentage_point_tolerance),
                  absolute_l2_tolerance=float(absolute_l2_tolerance),
                  all_finite=bool(all_finite), passed=bool(passed),
                  protocol="Original table mesh, UFL assembly, quadrature and denominator conventions; differences are measured, never replaced by saved values")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output/"recomputed_errors.csv", actual_rows)
    _write_csv(output/"recomputed_reference_errors.csv", reference_rows)
    _write_csv(output/"differences.csv", differences)
    (output/"summary.json").write_text(json.dumps(_strict_json(result), indent=2, allow_nan=False)+"\n")
    if not passed:
        raise RuntimeError(f"Saved GMsFEM tables failed reproduction tolerances; inspect {output/'summary.json'}")
    return result
