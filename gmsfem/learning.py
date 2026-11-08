"""Two reproducible learning exercises, using the production GMsFEM modules.

    python3 -m gmsfem.learning verify
    python3 -m gmsfem.learning trace --output /tmp/gmsfem-lesson

``verify`` reads stored arrays and recomputes metrics without Gmsh/DOLFINx.
``trace`` solves a small 2D scattering study and exposes a central patch.
See docs/learning_path.md for the mathematical interpretation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent


def p1_l2_norm(values, triangles, areas):
    """Exact complex P1 L² norm, independently of the error quadrature.

    On a triangle: ∫|v|² = |T|/12 * (sum |v_i|² + |sum v_i|²).
    This remains valid for complex values and nonuniform triangles.
    """
    local = np.asarray(values)[triangles]
    return float(np.sqrt(np.sum(areas / 12 * (
        np.sum(np.abs(local)**2, axis=1) + np.abs(np.sum(local, axis=1))**2))))


def verify_saved(directory: str | Path) -> dict:
    """Check stored metrics, dimensions and spectra against fields.npz.

    Recomputed metrics verify serialization and interpretation of a saved
    study. They cannot establish the PDE residual because A_h and P are not
    persisted. The exact P1 numerator is a separate quadrature cross-check.
    Changed source hashes are reported as provenance information, not as a
    numerical failure of an old result.
    """
    from .metrics import relative_errors

    directory = Path(directory).resolve()
    summary = json.loads((directory / "summary.json").read_text())
    config = summary["config"]
    if config.get("error_field") != "total":
        raise ValueError("Этот проверяющий пример ожидает error_field=total.")
    with (directory / "errors.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(summary["rows"]):
        raise ValueError("Число строк errors.csv и summary.json различается.")
    for saved, csv_row in zip(summary["rows"], csv_rows):
        if set(saved) != set(csv_row):
            raise ValueError("Столбцы errors.csv и summary.json различаются.")
        for name, value in saved.items():
            if value is None:
                if np.isfinite(float(csv_row[name])):
                    raise ValueError(f"Ожидалась неопределенная метрика {name}.")
            elif isinstance(value, (int, float)):
                np.testing.assert_allclose(float(csv_row[name]), value, rtol=1e-12, atol=1e-12,
                                           err_msg=f"errors.csv: {name}")
            elif csv_row[name] != str(value):
                raise ValueError(f"errors.csv: неверное значение {name}.")
    with np.load(directory / "fields.npz", allow_pickle=False) as arrays:
        xy, triangles = arrays["coordinates"], arrays["triangles"]
        if xy.ndim != 2 or xy.shape[1] != 2 or triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("Ожидается двумерная треугольная сетка.")
        if not np.issubdtype(triangles.dtype, np.integer) or triangles.min() < 0 or triangles.max() >= len(xy):
            raise ValueError("Некорректные индексы треугольников.")
        vertices = xy[triangles]
        edges = vertices[:, 1:] - vertices[:, :1]
        areas = np.abs(edges[:, 0, 0]*edges[:, 1, 1] - edges[:, 0, 1]*edges[:, 1, 0])/2
        if np.any(areas <= 0) or not np.isclose(areas.sum(), 16., rtol=0, atol=1e-9):
            raise ValueError("Неверная площадь или вырожденные элементы.")
        if summary["fine"]["dofs"] != len(xy) or summary["fine"]["triangles"] != len(triangles):
            raise ValueError("Размеры fine в summary.json и fields.npz не совпадают.")
        problem = SimpleNamespace(coordinates=xy, triangles=triangles, areas=areas,
                                  direction=tuple(config["direction"]), k=config["k"])
        fine = arrays["fine_scattered"]
        if fine.shape != (len(xy),) or not np.all(np.isfinite(fine)):
            raise ValueError("Неверное fine-поле.")
        incident = np.exp(1j*config["k"]*(xy @ np.asarray(config["direction"])))
        np.testing.assert_allclose(arrays["incident_nodes"], incident, rtol=1e-12, atol=1e-12)
        max_metric_difference, max_mass_difference = 0., 0.
        selections = []
        for row in summary["rows"]:
            nc, kind, modes = row["n_coarse"], row["basis_type"], row["modes"]
            expected_dofs = (nc+1)**2*modes
            if row["dofs_coarse"] != expected_dofs or row["dofs_fine"] != len(xy):
                raise ValueError("Размер coarse не равен (n_coarse+1)^2*M.")
            np.testing.assert_allclose(row["compression"], len(xy)/expected_dofs, rtol=1e-12)
            solution = arrays[f"ms_n{nc}_type{kind}_M{modes}"]
            if solution.shape != fine.shape or not np.all(np.isfinite(solution)):
                raise ValueError("Неверное multiscale-поле.")
            exact_error = p1_l2_norm(solution-fine, triangles, areas)
            np.testing.assert_allclose(row["l2_error_complex"], exact_error, rtol=2e-10, atol=2e-12)
            max_mass_difference = max(max_mass_difference, abs(exact_error-row["l2_error_complex"]))
            for field, prefix in (("total", ""), ("scattered", "scattered_")):
                computed = relative_errors(problem, solution, fine, field=field)
                for name, value in computed.items():
                    saved = row[prefix+name]
                    if saved is None:
                        if np.isfinite(value):
                            raise ValueError(f"Неожиданный null: {prefix+name}")
                    else:
                        np.testing.assert_allclose(saved, value, rtol=2e-9, atol=2e-10,
                                                   err_msg=f"n={nc}, Type={kind}, M={modes}, {prefix+name}")
                        max_metric_difference = max(max_metric_difference, abs(saved-value))
            if nc == max(config["coarse_grids"]) and kind == 2 and modes in (1, 4, 16):
                selections.append({name: row[name] for name in (
                    "n_coarse", "basis_type", "modes", "dofs_coarse", "e_complex_ref",
                    "e_abs_ref", "scattered_e_complex_ref")})
    with np.load(directory / "spectra.npz", allow_pickle=False) as spectra:
        for nc in config["coarse_grids"]:
            for kind in config["basis_types"]:
                values = spectra[f"n{nc}_type{kind}"]
                expected_shape = ((nc+1)**2, max(config["modes"]))
                if values.shape != expected_shape or not np.all(np.isfinite(values)):
                    raise ValueError("Неверные размер/значения спектров.")
                if values.min() < -1e-8 or np.any(np.diff(values, axis=1) < -1e-8):
                    raise ValueError("Спектр отрицательный или не отсортирован.")
                if kind == 1 and np.max(np.abs(values[:, 0])) > 1e-8:
                    raise ValueError("В Type 1 отсутствует нулевая мода.")
    changed = []
    for name, expected in summary.get("source_sha256", {}).items():
        source = ROOT / name
        if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            changed.append(name)
    return dict(directory=str(directory), case=config["case"], k=config["k"],
                fine_dofs=len(xy), triangles=len(triangles), area=float(areas.sum()),
                rows_verified=len(summary["rows"]),
                max_metric_absolute_difference=float(max_metric_difference),
                max_exact_P1_error_difference=float(max_mass_difference),
                selected_rows=selections, changed_source_files=changed,
                scope="Saved fields/metrics/spectra consistency; PDE and Galerkin residuals not recomputed.")


def trace_study(output: str | Path) -> dict:
    """Run the canonical quick study and expose one central vertex patch."""
    from .experiment import run_study, save_study
    from .fem import local_matrices, solve_dirichlet

    output = Path(output).resolve()
    study = run_study(h=.1, coarse_grids=(4,), modes=(1, 2, 4), mesh_dir=output / "meshes")
    output = save_study(study, output)
    saved_summary = json.loads((output / "summary.json").read_text())
    p = study.problem
    patches = {}
    for (_, kind), space in study.spaces.items():
        preview = space.preview
        local = local_matrices(p, preview["cells"])
        operator = local.K if kind == 1 else local.K-p.k**2*local.Mn
        snapshots, residual = solve_dirichlet(operator, local.boundary, np.eye(len(local.boundary)))
        mass_snapshot = snapshots.T @ (local.Salpha @ snapshots)
        vertices = (space.n_coarse+1)**2
        patches[f"type{kind}"] = dict(
            vertex=int(preview["vertex"]), fine_dofs=len(local.dofs),
            interior_dofs=len(local.dofs)-len(local.boundary),
            boundary_dofs=len(local.boundary),
            snapshot_shape=list(snapshots.shape),
            spectral_shape=list(mass_snapshot.shape),
            kept_modes=space.max_modes,
            eigenvalues=space.eigenvalues[preview["vertex"]].tolist(),
            first_mode_relative_range=float(np.ptp(preview["modes"][:, 0]) /
                                            np.max(np.abs(preview["modes"][:, 0]))),
            snapshot_sum_defect=float(np.max(np.abs(snapshots.sum(axis=1)-1))),
            snapshot_dirichlet_defect=float(np.max(np.abs(snapshots[local.boundary]-np.eye(len(local.boundary))))),
            snapshot_backward_error=float(residual),
            P_shape=[p.ndofs, vertices*space.max_modes],
            diagnostics=space.diagnostics)
    trace = dict(config=study.config, environment=saved_summary["environment"],
                 source_sha256=saved_summary["source_sha256"], fine_dofs=p.ndofs, patches=patches,
                 rows=[{name: row[name] for name in (
                     "basis_type", "modes", "dofs_coarse", "e_complex_ref",
                     "scattered_e_complex_ref", "galerkin_residual")} for row in study.rows],
                 note="No refined-mesh or exterior-boundary convergence study is performed by this lesson.")
    (output / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2)+"\n")
    return trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="Пересчитать метрики сохраненной серии без FEM solve")
    verify.add_argument("directory", nargs="?", type=Path, default=ROOT / "results" / "notebook")
    trace = commands.add_parser("trace", help="Решить quick-пример и раскрыть размеры центрального patch")
    trace.add_argument("--output", type=Path, default=Path("/tmp/gmsfem-lesson"))
    args = parser.parse_args(argv)
    result = verify_saved(args.directory) if args.command == "verify" else trace_study(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
