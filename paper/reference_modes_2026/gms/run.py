"""Paired P1 GMsFEM calculations for the new reference-mode experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import splu

from experiments.reference_sno_gmsfem.baselines import (
    FINE_MESH, FINE_SHA256, REFERENCE_MESH, REFERENCE_SHA256,
    _ReducedSystem, _checked_mesh, _materials, _write_csv,
)
from experiments.reference_sno_gmsfem.dataset import _save_npz, _write_json
from gmsfem.fem import assemble_problem
from gmsfem.metrics import BARYCENTRIC, WEIGHTS
from gmsfem.offline import build_offline_space, solve_reduced
from neural_operator.data import mass_matrix
from neural_operator.fem import HelmholtzSolver
from neural_operator.mesh_transfer import P1Transfer

MODES = (4, 8, 16)
METHODS = ((0, 0),) + tuple((kind, m) for kind in (1, 2) for m in MODES)
N_COARSE = 20
RESIDUAL_TOLERANCE = 1e-8
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inputs(root):
    path = root / "data" / "queries.npz"
    with np.load(path, allow_pickle=False) as queries:
        k = queries["k"] if "k" in queries.files else queries["k0"]
        theta, split = queries["theta"], queries["split"]
    selected = np.flatnonzero(split == 2)
    frequencies, counts = np.unique(k[selected], return_counts=True)
    if (len(frequencies) != 16 or any(counts != 8)
            or not np.all(np.isfinite(k[selected])) or np.any(k[selected] <= 0)
            or not np.all(np.isfinite(theta[selected]))
            or len(np.unique(np.column_stack((k[selected], theta[selected])), axis=0))
            != len(selected)):
        raise ValueError("Expected 16 held-out frequencies with eight distinct directions each")
    return k, theta, selected, frequencies


def residuals(operator, values, rhs):
    errors = np.linalg.norm(operator @ values - rhs, axis=0) / np.linalg.norm(rhs, axis=0)
    if not np.all(np.isfinite(errors)) or np.max(errors) > RESIDUAL_TOLERANCE:
        raise ArithmeticError(f"Bad linear system residual: {np.max(errors):g}")
    return errors


def norm2(mass, rows):
    return np.real(np.einsum("ij,ij->i", rows.conj(), (mass @ rows.T).T))


def method_name(kind, modes):
    if kind == 0:
        return "FEM h=0.04"
    return f"GMsFEM 20x20 Type {kind} M={modes}"


def cache_path(cache, index):
    return cache / f"frequency_{index:02d}.npz"


def load_cache(path, indices, n_fine):
    with np.load(path, allow_pickle=False) as archive:
        actual_indices = archive["indices"]
        values, errors = archive["values"], archive["residuals"]
        seconds = archive["seconds"]
    if (not np.array_equal(indices, actual_indices)
            or values.shape != (len(METHODS), len(indices), n_fine)
            or errors.shape != values.shape[:2]
            or not np.isfinite(values).all()
            or not np.isfinite(errors).all() or np.any(errors > RESIDUAL_TOLERANCE)):
        raise ValueError(f"Invalid cached solutions: {path}")
    return values, errors, seconds


def calculate(root, cache):
    k, theta, selected, frequencies = inputs(root)
    source = HelmholtzSolver(_checked_mesh(FINE_MESH, FINE_SHA256))
    alpha, n = _materials(source.cell_tags)
    type1 = None
    for group, frequency in enumerate(frequencies):
        started = perf_counter()
        indices = selected[k[selected] == frequency]
        path = cache_path(cache, group)
        if path.exists():
            load_cache(path, indices, len(source.coordinates))
            print(f"GMsFEM {group + 1}/{len(frequencies)} k={frequency:.7g}: cached", flush=True)
            continue
        angles = theta[indices]
        problem = source.build_problem(alpha, n, frequency, angles[0])
        rhs = np.column_stack([problem.rhs, *(source.rhs(alpha, n, frequency, angle)
                                              for angle in angles[1:])])
        fine = splu(problem.operator.tocsc()).solve(rhs)
        solutions, diagnostics = [fine.T], [residuals(problem.operator, fine, rhs)]
        if type1 is None:
            type1 = build_offline_space(problem, n_coarse=N_COARSE,
                                        max_modes=max(MODES), basis_type=1, progress=None)
        type2 = build_offline_space(problem, n_coarse=N_COARSE,
                                    max_modes=max(MODES), basis_type=2, progress=None)
        for space in (type1, type2):
            for m in MODES:
                system = _ReducedSystem(problem, space, m)
                reduced_rhs = system.PH @ rhs
                solution = system.P @ system.factor.solve(reduced_rhs)
                projected = system.PH @ (problem.operator @ solution - rhs)
                errors = np.linalg.norm(projected, axis=0) / np.linalg.norm(reduced_rhs, axis=0)
                if not np.all(np.isfinite(solution)) or not np.all(np.isfinite(errors)) or np.max(errors) > RESIDUAL_TOLERANCE:
                    raise ArithmeticError(f"Invalid Galerkin reconstruction: Type {space.basis_type}, M={m}")
                solutions.append(solution.T)
                diagnostics.append(errors)
        seconds = perf_counter() - started
        _save_npz(path, indices=indices, values=np.asarray(solutions),
                  residuals=np.asarray(diagnostics), seconds=seconds)
        print(f"GMsFEM {group + 1}/{len(frequencies)} k={frequency:.7g}: eight directions, "
              f"{seconds:.1f}s", flush=True)


def independent_checks(source, reference, coordinates, triangles, transfer,
                       mass, labels, k, theta, index, values):
    direction = (float(np.cos(theta[index])), float(np.sin(theta[index])))
    fine_ufl = assemble_problem(source.mesh_path, case="notebook", k=float(k[index]), direction=direction)
    if not np.array_equal(fine_ufl.coordinates, source.coordinates):
        raise ValueError("Independent fine assembly coordinate order differs")
    fine_direct = splu(fine_ufl.operator.tocsc()).solve(fine_ufl.rhs)
    fine_replay = float(np.linalg.norm(fine_direct - values[0]) / np.linalg.norm(fine_direct))
    ref_ufl = assemble_problem(reference.mesh_path, case="notebook", k=float(k[index]), direction=direction)
    if not np.array_equal(ref_ufl.coordinates, coordinates):
        raise ValueError("Independent reference assembly coordinate order differs")
    ref_direct = splu(ref_ufl.operator.tocsc()).solve(ref_ufl.rhs)
    ref_replay = float(np.sqrt(norm2(mass, (ref_direct - labels[index])[None])
                               / norm2(mass, ref_direct[None]))[0])
    offline = build_offline_space(fine_ufl, n_coarse=N_COARSE, max_modes=max(MODES),
                                  basis_type=2, progress=None)
    reduced_direct, _ = solve_reduced(fine_ufl, offline, max(MODES))
    reduced_replay = float(np.linalg.norm(reduced_direct - values[-1]) / np.linalg.norm(reduced_direct))
    vertices = coordinates[triangles]
    area = np.abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1])) / 2
    quad = np.einsum("qj,cjd->cqd", BARYCENTRIC, vertices)
    direct = P1Transfer(source.coordinates, source.triangles, quad.reshape(-1, 2))
    reference_quad = labels[index, triangles] @ BARYCENTRIC.T
    weights = area[:, None] * WEIGHTS
    rows = []
    for (kind, m), field in zip(METHODS, values):
        mapped = transfer.apply(field)
        nodal_error = np.sqrt((norm2(mass, (mapped - labels[index])[None])
                               / norm2(mass, labels[index][None]))[0])
        mapped_quad = mapped[triangles] @ BARYCENTRIC.T
        direct_quad = direct.apply(field).reshape(quad.shape[:2])
        denominator = np.sum(weights * np.abs(reference_quad) ** 2)
        mapped_quad_error = np.sqrt(np.sum(weights * np.abs(mapped_quad - reference_quad) ** 2) / denominator)
        direct_quad_error = np.sqrt(np.sum(weights * np.abs(direct_quad - reference_quad) ** 2) / denominator)
        if abs(mapped_quad_error - nodal_error) > 1e-12:
            raise ArithmeticError("Mass matrix and quadrature disagree for transferred P1 field")
        rows.append(dict(method=method_name(kind, m), nodal_error_percent=100 * nodal_error,
                         direct_quadrature_error_percent=100 * direct_quad_error,
                         signed_difference_pp=100 * (nodal_error - direct_quad_error)))
    if max(fine_replay, ref_replay, reduced_replay) > 1e-7:
        raise ArithmeticError(f"Independent UFL replay mismatch: {fine_replay}, {ref_replay}, {reduced_replay}")
    return dict(index=int(index), k=float(k[index]), theta=float(theta[index]),
                independent_fine_relative_nodal_difference=fine_replay,
                independent_reference_relative_mass_difference=ref_replay,
                independent_type2_m16_relative_nodal_difference=reduced_replay,
                metric_crosscheck=rows)


def score(root, cache):
    data = root / "data"
    if not (data / "complete.json").is_file():
        raise FileNotFoundError("The reference FEM generator has not completed")
    k, theta, selected, frequencies = inputs(root)
    with np.load(data / "reference_problem.npz", allow_pickle=False) as geometry:
        coordinates, triangles = geometry["coordinates"], geometry["triangles"]
        alpha, n = _materials(geometry["cell_tags"])
        if (not np.array_equal(geometry["A_cells"].reshape(-1), alpha)
                or not np.array_equal(geometry["n_cells"].reshape(-1), n)):
            raise ValueError("Reference labels must be for the notebook medium")
    source = HelmholtzSolver(_checked_mesh(FINE_MESH, FINE_SHA256))
    reference = HelmholtzSolver(_checked_mesh(REFERENCE_MESH, REFERENCE_SHA256))
    if not (np.array_equal(reference.coordinates, coordinates)
            and np.array_equal(reference.triangles, triangles)):
        raise ValueError("Reference label coordinate order or cell connectivity changed")
    transfer = P1Transfer(source.coordinates, source.triangles, coordinates)
    mass = mass_matrix(coordinates, triangles)
    labels = np.load(data / "fields.npy", mmap_mode="r", allow_pickle=False)
    if labels.shape != (len(k), len(coordinates)) or not np.iscomplexobj(labels):
        raise ValueError("Expected one native complex label per query")
    rows, timings = [], []
    for group, frequency in enumerate(frequencies):
        indices = selected[k[selected] == frequency]
        values, diagnostics, seconds = load_cache(cache_path(cache, group), indices, len(source.coordinates))
        if group == len(frequencies) // 2:
            replay = independent_checks(source, reference, coordinates, triangles, transfer,
                                        mass, labels, k, theta, indices[0], values[:, 0])
        timings.append(float(seconds))
        truth = labels[indices]
        denominators = norm2(mass, truth)
        if not np.isfinite(denominators).all() or np.any(denominators <= 0):
            raise ArithmeticError("Invalid reference FEM field energy")
        for (kind, m), fields, errors in zip(METHODS, values, diagnostics):
            mapped = transfer.apply(fields)
            percentages = 100 * np.sqrt(norm2(mass, mapped - truth) / denominators)
            dofs = (N_COARSE + 1) ** 2 * m if kind else len(source.coordinates)
            for i, percentage, residual in zip(indices, percentages, errors):
                rows.append(dict(index=int(i), k=float(k[i]), theta=float(theta[i]),
                                 method=method_name(kind, m), basis_type=kind, modes=m,
                                 n_coarse=N_COARSE if kind else 0, dofs_complex=dofs,
                                 dofs_real=2 * dofs, error_percent=float(percentage),
                                 relative_linear_residual=float(residual)))
    summaries = []
    for kind, m in METHODS:
        group_rows = [row for row in rows if (row["basis_type"], row["modes"]) == (kind, m)]
        errors = np.array([row["error_percent"] for row in group_rows])
        if len(group_rows) != len(selected) or not np.isfinite(errors).all():
            raise ArithmeticError("Missing or nonfinite case errors")
        summaries.append(dict(method=method_name(kind, m), basis_type=kind, modes=m,
                              n_coarse=group_rows[0]["n_coarse"], dofs_complex=group_rows[0]["dofs_complex"],
                              dofs_real=group_rows[0]["dofs_real"], count=len(errors),
                              mean_percent=float(errors.mean()), median_percent=float(np.median(errors)),
                              p95_percent=float(np.quantile(errors, .95)), max_percent=float(errors.max()),
                              max_relative_linear_residual=max(row["relative_linear_residual"] for row in group_rows)))
    output = root / "gms"
    _write_csv(output / "per_case.csv", rows)
    _write_csv(output / "summary.csv", summaries)
    hashes = {name: digest(output / name) for name in ("per_case.csv", "summary.csv")}
    _write_json(output / "verification.json", dict(
        passed=True, test_indices=selected.tolist(), test_frequencies=frequencies.tolist(),
        query_sha256=digest(data / "queries.npz"), reference_problem_sha256=digest(data / "reference_problem.npz"),
        fine_mesh_sha256=FINE_SHA256, reference_mesh_sha256=REFERENCE_SHA256,
        reference_dofs=len(coordinates), residual_tolerance=RESIDUAL_TOLERANCE,
        score="100 * sqrt((u_mapped-u_ref)^H M_ref (u_mapped-u_ref) / (u_ref^H M_ref u_ref)) for the scattered field.",
        transfer="Evaluate the source-mesh P1 field at the reference nodes, then use consistent P1 mass on the reference mesh.",
        timing_seconds_by_frequency=timings, calculation_seconds=sum(timings),
        independent_replay=replay, output_sha256=hashes))
    print(f"Verified {len(rows)} cases and wrote {output / 'summary.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "calculate", "score"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    query = root / "data" / "queries.npz"
    solver_source = Path(__file__)
    key = digest(query)[:16] + "-" + digest(solver_source)[:16]
    cache = root / "gms" / ".array_cache" / key
    cache.mkdir(parents=True, exist_ok=True)
    if args.stage in ("calculate", "all"):
        calculate(root, cache)
    if args.stage in ("score", "all"):
        score(root, cache)


if __name__ == "__main__":
    main()
