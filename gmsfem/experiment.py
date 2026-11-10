"""Воспроизводимые серии GMsFEM и отдельный более мелкий МКЭ-эталон.

run_study не читает готовые таблицы и не подгоняет ошибки. Каждый результат
получен решением P^H A_h P; fine solution используется только в метриках.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from .fem import FineProblem, MATERIALS, assemble_problem, solve_fine
from .geometry import generate_mesh
from .metrics import errors_on_reference, relative_errors
from .offline import OfflineSpace, build_offline_space, solve_reduced


ROOT = Path(__file__).resolve().parent


@dataclass
class Study:
    problem: FineProblem
    fine: np.ndarray
    fine_diagnostics: dict
    spaces: dict[tuple[int, int], OfflineSpace]
    solutions: dict[tuple[int, int, int], np.ndarray]
    rows: list[dict]
    config: dict
    elapsed_seconds: float


@dataclass
class ReferenceComparison:
    problem: FineProblem
    fine: np.ndarray
    diagnostics: dict
    rows: list[dict]


def _positive_integers(values, name):
    values = tuple(values)
    if not values or any(isinstance(v, bool) or int(v) != v or v < 1 for v in values):
        raise ValueError(f"{name} должен содержать положительные целые числа.")
    return tuple(sorted(set(int(v) for v in values)))


def run_study(
    *,
    case: str = "notebook",
    k: float = 1.0,
    h: float = .04,
    coarse_grids: tuple[int, ...] = (10, 20),
    modes: tuple[int, ...] = (1, 2, 4, 6, 8, 12, 16),
    basis_types: tuple[int, ...] = (1, 2),
    direction: tuple[float, float] = (1.0, 0.0),
    mesh_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = print,
) -> Study:
    """Общая fine mesh для всех H и Type; h=целевой шаг Gmsh, не max диаметра.

    Геометрия разбивается на lcm(coarse_grids) квадратов вдоль оси: так,
    например, одна fine mesh согласована одновременно с 10×10 и 20×20.
    """
    started = perf_counter()
    coarse_grids = _positive_integers(coarse_grids, "coarse_grids")
    modes = _positive_integers(modes, "modes")
    basis_types = _positive_integers(basis_types, "basis_types")
    if not set(basis_types) <= {1, 2}:
        raise ValueError("Доступны только basis_types 1 и 2.")
    if case not in MATERIALS:
        raise ValueError(f"Неизвестный case={case!r}.")
    n_geo = math.lcm(*coarse_grids)
    if n_geo > 80:
        raise ValueError("lcm(coarse_grids)>80: выберите согласованные крупные сетки, например (10,20).")
    mesh_dir = Path(mesh_dir) if mesh_dir is not None else ROOT / "meshes"
    h_name = format(h, ".8g").replace(".", "p")
    mesh_path = generate_mesh(mesh_dir / f"fine_n{n_geo}_h{h_name}.msh", n_coarse=n_geo, h=h)
    problem = assemble_problem(mesh_path, case=case, k=k, direction=direction)
    fine, fine_diagnostics = solve_fine(problem)
    if progress:
        progress(f"МКЭ: {problem.ndofs} DOF, {len(problem.triangles)} треугольников, residual={fine_diagnostics['relative_residual']:.2e}")
    spaces, solutions, rows = {}, {}, []
    for nc in coarse_grids:
        for basis_type in basis_types:
            space = build_offline_space(problem, n_coarse=nc, max_modes=max(modes), basis_type=basis_type, progress=progress)
            spaces[nc, basis_type] = space
            for M in modes:
                solution, diagnostics = solve_reduced(problem, space, M)
                total_errors = relative_errors(problem, solution, fine, field="total")
                scattered_errors = relative_errors(problem, solution, fine, field="scattered")
                row = dict(case=case, k=float(k), n_coarse=nc, basis_type=basis_type, modes=M,
                           field="total", offline_seconds=space.build_seconds, **diagnostics, **total_errors)
                row.update({f"scattered_{key}": value for key, value in scattered_errors.items()})
                rows.append(row)
                solutions[nc, basis_type, M] = solution
                if progress:
                    progress(f"  M={M:2d}, DOF={diagnostics['dofs_coarse']:5d}: e_Re={total_errors['e_real_paper']:.3f}%, "
                             f"e_Im={total_errors['e_imag_paper']:.3f}%, e_|u|={total_errors['e_abs_paper']:.3f}%")
    config = dict(case=case, k=float(k), h=float(h), coarse_grids=list(coarse_grids),
                  modes=list(modes), basis_types=list(basis_types), direction=list(direction),
                  geometry_subdivisions=n_geo, mesh_path=str(mesh_path), mesh_dir=str(mesh_dir.resolve()),
                  method="CG1 GMsFEM; A=alpha*I; all boundary-delta snapshots; S_alpha mass",
                  error_field="total", paper_denominator="multiscale", ref_denominator="fine",
                  quadrature="7-point Dunavant degree 5, analytic incident field")
    return Study(problem, fine, fine_diagnostics, spaces, solutions, rows, config, perf_counter()-started)


def compare_refined(study: Study, *, progress: Callable[[str], None] | None = print) -> ReferenceComparison:
    """В финале построить независимую reference mesh h/2 и решить обычный МКЭ."""
    cfg = study.config
    h_name = format(cfg["h"], ".8g").replace(".", "p")
    mesh_path = generate_mesh(Path(cfg["mesh_dir"]) / f"reference_n{cfg['geometry_subdivisions']}_h{h_name}.msh",
                              n_coarse=cfg["geometry_subdivisions"], h=cfg["h"], reference=True)
    problem = assemble_problem(mesh_path, case=cfg["case"], k=cfg["k"], direction=tuple(cfg["direction"]))
    fine, diagnostics = solve_fine(problem)
    if problem.ndofs <= study.problem.ndofs:
        raise RuntimeError("Reference mesh должна иметь больше DOF, чем базовая fine mesh.")
    if progress:
        progress(f"Независимый МКЭ h/2: {problem.ndofs} DOF, {len(problem.triangles)} треугольников")
    rows = []
    candidates = [("МКЭ h", None, study.fine)]
    M = max(cfg["modes"])
    candidates += [(f"GMsFEM {nc}×{nc}, Type {t}, M={M}", (nc, t, M), study.solutions[nc, t, M])
                   for nc in cfg["coarse_grids"] for t in cfg["basis_types"]]
    for label, key, values in candidates:
        metrics = errors_on_reference(study.problem, values, problem, fine, field="total")
        row = {"label": label, "n_coarse": key[0] if key else None,
               "basis_type": key[1] if key else None, "modes": key[2] if key else None,
               "dofs": (key[0]+1)**2*key[2] if key else study.problem.ndofs, **metrics}
        rows.append(row)
        if progress:
            progress(f"  {label}: e_complex,ref={metrics['e_complex_ref']:.3f}%, e_abs,ref={metrics['e_abs_ref']:.3f}%")
    diagnostics.update({"mesh_path": str(mesh_path), "target_h": cfg["h"]/2,
                        "dofs": problem.ndofs, "triangles": len(problem.triangles)})
    return ReferenceComparison(problem, fine, diagnostics, rows)


def _finite_json(value):
    """JSON strict: не определённые relative errors сохраняются как null, не NaN."""
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_study(study: Study, output_dir: str | Path, reference: ReferenceComparison | None = None) -> Path:
    """Сохранить реальные метрики, поля, спектры, конфигурацию и читаемый отчёт."""
    import dolfinx
    import gmsh
    import scipy
    from petsc4py import PETSc

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(study.rows[0]))
        writer.writeheader()
        writer.writerows(study.rows)
    arrays = {"coordinates": study.problem.coordinates, "triangles": study.problem.triangles,
              "material_tags": study.problem.material_tags, "fine_scattered": study.fine,
              "incident_nodes": study.problem.incident}
    for (nc, t, M), values in study.solutions.items():
        arrays[f"ms_n{nc}_type{t}_M{M}"] = values
    if reference:
        arrays.update(reference_coordinates=reference.problem.coordinates, reference_triangles=reference.problem.triangles,
                      reference_scattered=reference.fine)
        with (output / "reference_errors.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(reference.rows[0]))
            writer.writeheader()
            writer.writerows(reference.rows)
    else:
        # These files belong to the optional reference stage. Retaining them
        # would attribute an older reference calculation to this new study.
        (output / "reference_errors.csv").unlink(missing_ok=True)
        (output / "reference_comparison.png").unlink(missing_ok=True)
    np.savez_compressed(output / "fields.npz", **arrays)
    np.savez_compressed(output / "spectra.npz", **{f"n{nc}_type{t}": s.eigenvalues for (nc, t), s in study.spaces.items()})
    sources = list(ROOT.glob("*.py")) + list((ROOT / "geo").glob("*.geo"))
    summary = dict(config=study.config, fine=dict(dofs=study.problem.ndofs, triangles=len(study.problem.triangles),
                    assembly_seconds=study.problem.assembly_seconds, element_assembly_error=study.problem.element_assembly_error,
                    **study.fine_diagnostics),
                   elapsed_seconds=study.elapsed_seconds, rows=study.rows,
                   offline={f"n{nc}_type{t}": dict(build_seconds=s.build_seconds, **s.diagnostics)
                            for (nc, t), s in study.spaces.items()},
                   reference=dict(diagnostics=reference.diagnostics, rows=reference.rows) if reference else None,
                   environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                    dolfinx=dolfinx.__version__, gmsh=gmsh.__version__, petsc=PETSc.Sys.getVersion(),
                                    scalar_type=str(np.dtype(PETSc.ScalarType))),
                   source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output / "summary.json").write_text(json.dumps(_finite_json(summary), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    write_report(study, reference, output)
    return output


def write_report(study: Study, reference: ReferenceComparison | None, output: Path):
    """Refresh only the Markdown report, including figures already rendered."""
    cfg = study.config
    lines = ["# Численный эксперимент GMsFEM", "",
             f"Preset `{cfg['case']}`, k={cfg['k']:g}, d={tuple(cfg['direction'])}, A=αI. Геометрия: 24 включения исходных notebooks.", "",
             f"Тонкая CG1-сетка: **{study.problem.ndofs} DOF**, {len(study.problem.triangles)} треугольников; целевой h={cfg['h']:g}.",
             "Все варианты этой серии используют один fine-оператор и одинаковую контрастную RHS.", "",
             "Сравнивается **полное поле** u=u_i+u_s. Таблица: интегральные L²-ошибки в процентах, знаменатель — multiscale, как на стр. 9 статьи.",
             "В errors.csv дополнительно сохранены reference-нормированные ошибки, комплексная ошибка и ошибки рассеянного поля.", "",
             "| Крупная сетка | Type | M | DOF | Re, % | Im, % | Модуль, % | Complex/ref, % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in study.rows:
        lines.append(f"| {row['n_coarse']}×{row['n_coarse']} | {row['basis_type']} | {row['modes']} | {row['dofs_coarse']} | "
                     f"{row['e_real_paper']:.4f} | {row['e_imag_paper']:.4f} | {row['e_abs_paper']:.4f} | {row['e_complex_ref']:.4f} |")
    if (output / "convergence.png").exists():
        lines += ["", "![Сходимость по числу локальных мод](convergence.png)"]
    for filename, caption in (("fields_total.png", "Полное поле: МКЭ и GMsFEM"),
                              ("fields_dg_style.png", "Компоненты рассеянного поля и модуль полного: вид исходного DG notebook")):
        if (output / filename).exists():
            lines += ["", f"![{caption}]({filename})"]
    lines += ["", "## Время и проверки", "",
              f"Сборка fine-форм: {study.problem.assembly_seconds:.3f} s; решение fine-системы: {study.fine_diagnostics['solve_seconds']:.3f} s.",
              f"Относительная невязка МКЭ: {study.fine_diagnostics['relative_residual']:.2e}; независимая ошибка P1/UFL stiffness: {study.problem.element_assembly_error:.2e}.",
              "Время offline включает χ, snapshots и спектры; projection и online solve измеряются отдельно в CSV. Ускорение полного расчёта не выводится из одного времени online solve.", "",
              "| Крупная сетка | Type | Offline, s | Snapshot residual | Eigen residual | Отклонение суммы χ от 1 |", "|---|---:|---:|---:|---:|---:|"]
    for (nc, t), space in study.spaces.items():
        diag = space.diagnostics
        lines.append(f"| {nc}×{nc} | {t} | {space.build_seconds:.3f} | {diag['max_snapshot_residual']:.2e} | {diag['max_eigen_residual']:.2e} | {diag['partition_sum_defect']:.2e} |")
    lines += ["", "Type 2 использует Helmholtz-продолжения: сумма χ внутри крупной ячейки не обязана равняться единице и не нормируется.", ""]
    if reference:
        lines += ["## Последний этап: отдельная эталонная сетка МКЭ", "",
                  f"На той же CAD-геометрии заново построена сетка с целевым h/2={reference.diagnostics['target_h']:g}: "
                  f"**{reference.problem.ndofs} DOF**, {len(reference.problem.triangles)} треугольников.",
                  "Обычный CG1 МКЭ решён независимо. Оба поля оцениваются в точках квадратуры reference-сетки с кусочно-линейной интерполяцией исходного поля.",
                  "Эта проверка содержит ошибку редукции, fine-дискретизации и полигонализации окружностей. Более мелкий МКЭ — численный эталон, а не точное решение.", "",
                  "| Метод | DOF | Re/ref, % | Im/ref, % | Модуль/ref, % | Complex/ref, % |", "|---|---:|---:|---:|---:|---:|"]
        for row in reference.rows:
            lines.append(f"| {row['label']} | {row['dofs']} | {row['e_real_ref']:.4f} | {row['e_imag_ref']:.4f} | {row['e_abs_ref']:.4f} | {row['e_complex_ref']:.4f} |")
        if (output / "reference_comparison.png").exists():
            lines += ["", "![Сопоставление с независимым эталоном МКЭ](reference_comparison.png)"]
    lines += ["", "## Сопоставление со статьёй", "",
              "Реализованы уравнения (9)–(17): два типа snapshots/χ, спектральная масса Sα, наименьшие собственные значения и глобальная CG-проекция.",
              "Это адаптация к 24 включениям проекта. Статья не публикует координаты исходной геометрии; её fine-сетки имеют около 100 тысяч DOF. Числа таблиц напрямую не воспроизводятся.",
              "Коэффициенты notebooks и статьи различаются; preset необходимо учитывать. Исходные notebooks содержат интерфейсный дефект RHS; здесь используется согласованная слабая форма.",
              "Ссылка: [Kalachikova et al., JCAM 424 (2023), 114977](https://doi.org/10.1016/j.cam.2022.114977). Выводы и ограничения метода: [методическая документация](../../docs/method.md).", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
