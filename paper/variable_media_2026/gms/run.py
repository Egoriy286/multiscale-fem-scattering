"""Generate independent paired GMsFEM baselines for the variable-medium paper."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import splu

from experiments.reference_sno_gmsfem.baselines import (
    FINE_MESH, FINE_SHA256, REFERENCE_MESH, REFERENCE_SHA256,
    _ReducedSystem, _checked_mesh, _write_csv,
)
from experiments.reference_sno_gmsfem.dataset import _save_npz, _write_json
from gmsfem.fem import MATERIALS, assemble_problem
from gmsfem.offline import build_offline_space, solve_reduced
from neural_operator.data import mass_matrix
from neural_operator.fem import HelmholtzSolver
from neural_operator.mesh_transfer import P1Transfer
from paper.reference_modes_2026.gms.run import (
    METHODS, MODES, N_COARSE, RESIDUAL_TOLERANCE,
    digest, method_name, norm2, residuals,
)

ROOT = Path(__file__).resolve().parents[1]
MATERIAL_KEYS = ("alpha1", "alpha2", "n1", "n2")
REPLAY_TOLERANCE = 1e-7


def load_inputs(root):
    path = root / "data" / "queries.npz"
    with np.load(path, allow_pickle=False) as archive:
        queries = {name: archive[name] for name in ("k", "theta", "medium", "split", *MATERIAL_KEYS)}
    selected = np.flatnonzero(queries["split"] == 2)
    media = np.unique(queries["medium"][selected])
    if len(selected) != 128 or len(media) != 16:
        raise ValueError("Expected 128 test cases in 16 media")
    selected_media = queries["medium"][selected]
    if np.isin(selected_media, queries["medium"][queries["split"] != 2]).any():
        raise ValueError("A test medium appears outside the test set")
    groups, signatures = [], []
    for medium in media:
        indices = selected[selected_media == medium]
        angles = queries["theta"][indices]
        material = np.stack([queries[key][indices] for key in MATERIAL_KEYS], axis=1)
        if (len(indices) != 8 or len(np.unique(angles)) != 8
                or not np.isfinite(angles).all() or not np.all(queries["k"][indices] == 2)
                or not np.all(material == material[0]) or not np.isfinite(material).all()
                or np.any(material <= 0)):
            raise ValueError(f"Inconsistent test group {medium}")
        groups.append(indices)
        signatures.append(tuple(material[0]))
    known = np.column_stack([queries[key][queries["split"] != 2] for key in MATERIAL_KEYS])
    known_signatures = {tuple(row) for row in known}
    if len(set(signatures)) != len(media) or known_signatures.intersection(signatures):
        raise ValueError("Duplicated physical parameters across the medium split")
    return queries, groups


def material_table(queries, row):
    a1, a2, n1, n2 = (float(queries[key][row]) for key in MATERIAL_KEYS)
    return {1: (1., 1.), 2: (a1, n1), 3: (a1, n1), 4: (a2, n2)}


def coefficients(tags, table):
    if set(np.unique(tags)) != set(table):
        raise ValueError("Unexpected physical markers on the FEM mesh")
    return tuple(np.array([table[int(tag)][j] for tag in tags]) for j in (0, 1))


def cache_path(cache, j):
    return cache / f"medium_{j:02d}.npz"


def read_case(path, indices, n_fine):
    with np.load(path, allow_pickle=False) as archive:
        old_indices = archive["indices"]
        solutions = archive["solutions"]
        errors = archive["residuals"]
        seconds = float(archive["seconds"])
    if (not np.array_equal(indices, old_indices)
            or solutions.shape != (len(METHODS), 8, n_fine) or errors.shape != solutions.shape[:2]
            or not np.isfinite(solutions).all() or not np.isfinite(errors).all()
            or np.max(errors) > RESIDUAL_TOLERANCE or not 0 < seconds < np.inf):
        raise ValueError(f"Invalid cached solves at {path}")
    return solutions, errors, seconds


def solve_group(source, queries, indices):
    alpha, n = coefficients(source.cell_tags, material_table(queries, indices[0]))
    k = float(queries["k"][indices[0]])
    theta = queries["theta"][indices]
    problem = source.build_problem(alpha, n, k, theta[0])
    rhs = np.column_stack([problem.rhs, *(source.rhs(alpha, n, k, angle) for angle in theta[1:])])
    fine = splu(problem.operator.tocsc()).solve(rhs)
    solutions, errors = [fine.T], [residuals(problem.operator, fine, rhs)]
    for kind in (1, 2):
        space = build_offline_space(problem, n_coarse=N_COARSE,
                                    max_modes=max(MODES), basis_type=kind, progress=None)
        for m in MODES:
            system = _ReducedSystem(problem, space, m)
            reduced_rhs = system.PH @ rhs
            solution = system.P @ system.factor.solve(reduced_rhs)
            projected = system.PH @ (problem.operator @ solution - rhs)
            rel = np.linalg.norm(projected, axis=0) / np.linalg.norm(reduced_rhs, axis=0)
            if not np.isfinite(solution).all() or not np.isfinite(rel).all() or np.max(rel) > RESIDUAL_TOLERANCE:
                raise ArithmeticError(f"Invalid Galerkin reconstruction, Type {kind}, M={m}")
            solutions.append(solution.T)
            errors.append(rel)
    return np.asarray(solutions), np.asarray(errors)


def calculate(cache, queries, groups):
    source = HelmholtzSolver(_checked_mesh(FINE_MESH, FINE_SHA256))
    for j, indices in enumerate(groups):
        path = cache_path(cache, j)
        medium = queries["medium"][indices[0]]
        if path.exists():
            read_case(path, indices, len(source.coordinates))
            print(f"GMsFEM {j + 1}/{len(groups)}, medium {medium}: cached", flush=True)
            continue
        started = perf_counter()
        solutions, errors = solve_group(source, queries, indices)
        seconds = perf_counter() - started
        _save_npz(path, indices=indices, solutions=solutions, residuals=errors, seconds=seconds)
        print(f"GMsFEM {j + 1}/{len(groups)}, medium {medium}: eight directions, {seconds:.1f}s", flush=True)


def relative_nodal(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def replay_fem(source, reference, mass, angles, table, k, truth, approximation, row, medium):
    # The existing independent UFL assembler takes a key in its material table.
    # Insert this physical medium only for this process and only during assembly.
    name = "variable_media_2026_replay"
    if name in MATERIALS:
        raise RuntimeError("Temporary verification material key already exists")
    direction = (float(np.cos(angles)), float(np.sin(angles)))
    MATERIALS[name] = table
    try:
        fine = assemble_problem(source.mesh_path, case=name, k=k, direction=direction)
        ref = assemble_problem(reference.mesh_path, case=name, k=k, direction=direction)
    finally:
        del MATERIALS[name]
    for independent, same in ((fine, source), (ref, reference)):
        if not (np.array_equal(independent.coordinates, same.coordinates)
                and np.array_equal(independent.triangles, same.triangles)):
            raise ValueError("Independent UFL mesh ordering differs")
    fine_u = splu(fine.operator.tocsc()).solve(fine.rhs)
    ref_u = splu(ref.operator.tocsc()).solve(ref.rhs)
    checks = {
        "fine_fem_relative_nodal_difference": relative_nodal(approximation[0], fine_u),
        "reference_relative_mass_difference": float(
            np.sqrt(norm2(mass, (ref_u - truth)[None]) / norm2(mass, ref_u[None]))[0]),
    }
    for kind in (1, 2):
        space = build_offline_space(fine, n_coarse=N_COARSE,
                                    max_modes=max(MODES), basis_type=kind, progress=None)
        value, _ = solve_reduced(fine, space, max(MODES))
        checks[f"type{kind}_m16_relative_nodal_difference"] = relative_nodal(
            approximation[METHODS.index((kind, max(MODES)))], value)
    if not all(0 <= x < REPLAY_TOLERANCE for x in checks.values()):
        raise ArithmeticError(f"Independent UFL replay mismatch on row {row}: {checks}")
    return dict(row=int(row), medium=int(medium), k=k, theta=angles, **checks)


def score(root, cache, queries, groups):
    data = root / "data"
    if not (data / "complete.json").is_file():
        raise FileNotFoundError("Reference FEM generation has not completed")
    source = HelmholtzSolver(_checked_mesh(FINE_MESH, FINE_SHA256))
    reference = HelmholtzSolver(_checked_mesh(REFERENCE_MESH, REFERENCE_SHA256))
    with np.load(data / "reference_problem.npz", allow_pickle=False) as geometry:
        coordinates, triangles = geometry["coordinates"], geometry["triangles"]
        if not np.array_equal(reference.cell_tags, geometry["cell_tags"]):
            raise ValueError("Reference cell markers differ")
    if not (np.array_equal(reference.coordinates, coordinates)
            and np.array_equal(reference.triangles, triangles)):
        raise ValueError("Reference coordinates or connectivity differ")
    transfer = P1Transfer(source.coordinates, source.triangles, coordinates)
    mass = mass_matrix(coordinates, triangles)
    labels = np.load(data / "fields.npy", mmap_mode="r", allow_pickle=False)
    if labels.shape != (len(queries["k"]), len(coordinates)) or not np.iscomplexobj(labels):
        raise ValueError("Expected one complex reference field per manifest row")
    output = root / "gms"
    rows, replays, times = [], [], []
    for j, indices in enumerate(groups):
        solutions, residual, seconds = read_case(cache_path(cache, j), indices, len(source.coordinates))
        times.append(seconds)
        truth = labels[indices]
        den = norm2(mass, truth)
        if not np.isfinite(den).all() or np.any(den <= 0):
            raise ArithmeticError("Invalid reference energy")
        if j in (0, len(groups) // 2, len(groups) - 1):
            i = indices[0]
            replays.append(replay_fem(source, reference, mass,
                           float(queries["theta"][i]), material_table(queries, i),
                           float(queries["k"][i]), truth[0], solutions[:, 0],
                           i, queries["medium"][i]))
        for (kind, m), fields, res in zip(METHODS, solutions, residual):
            mapped = transfer.apply(fields)
            errors = 100 * np.sqrt(norm2(mass, mapped - truth) / den)
            if not np.isfinite(errors).all():
                raise ArithmeticError("Nonfinite FEM L2 score")
            dofs = (N_COARSE + 1) ** 2 * m if kind else len(source.coordinates)
            for i, error, rel in zip(indices, errors, res):
                common = {name: float(queries[name][i]) for name in ("k", "theta", *MATERIAL_KEYS)}
                rows.append(dict(index=int(i), medium=int(queries["medium"][i]), **common,
                                 method=method_name(kind, m), basis_type=kind, modes=m,
                                 n_coarse=N_COARSE if kind else 0,
                                 dofs_complex=dofs, dofs_real=2 * dofs, error_percent=float(error),
                                 relative_linear_residual=float(rel)))
    summaries = []
    for kind, m in METHODS:
        cases = [row for row in rows if (row["basis_type"], row["modes"]) == (kind, m)]
        errors = np.array([row["error_percent"] for row in cases])
        if len(cases) != 128:
            raise ValueError("A GMsFEM method is missing test cases")
        keys = ("method", "basis_type", "modes", "n_coarse", "dofs_complex", "dofs_real")
        summaries.append(dict({key: cases[0][key] for key in keys}, count=len(cases),
                              mean_percent=float(errors.mean()), median_percent=float(np.median(errors)),
                              p95_percent=float(np.quantile(errors, .95)), max_percent=float(errors.max()),
                              max_relative_linear_residual=max(row["relative_linear_residual"] for row in cases)))
    _write_csv(output / "per_case.csv", rows)
    _write_csv(output / "summary.csv", summaries)
    checksums = {name: digest(output / name) for name in ("per_case.csv", "summary.csv")}
    _write_json(output / "verification.json", dict(
        passed=True, query_sha256=digest(data / "queries.npz"),
        material_assignment={"tags_2_3": ["alpha1", "n1"], "tag_4": ["alpha2", "n2"], "tag_1": [1., 1.]},
        reference_problem_sha256=digest(data / "reference_problem.npz"),
        fine_mesh_sha256=FINE_SHA256, reference_mesh_sha256=REFERENCE_SHA256,
        reference_dofs=len(coordinates), independent_ufl_replays=replays, times_seconds=times,
        calculation_seconds=sum(times), output_sha256=checksums, residual_tolerance=RESIDUAL_TOLERANCE,
        score="Complex scattered relative L2 percent, source P1 evaluated at reference nodes, consistent reference P1 mass.",
        test_indices=np.sort(np.concatenate(groups)).tolist(), case_count=len(rows)))
    print(f"Verified {len(rows)} paired cases and wrote {output / 'summary.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "calculate", "score"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    queries, groups = load_inputs(root)
    cache_key = digest(root / "data" / "queries.npz")[:16] + "-" + digest(Path(__file__))[:16]
    cache = root / "gms" / ".array_cache" / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "calculate"):
        calculate(cache, queries, groups)
    if args.stage in ("all", "score"):
        score(root, cache, queries, groups)


if __name__ == "__main__":
    main()
