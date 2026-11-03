"""Научные рисунки сеток, локального базиса, комплексных полей и ошибок.

Все функции возвращают Figure и не вызывают show. Поля в аргументах —
рассеянные; падающая волна добавляется только при field='total'.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.tri import Triangulation
import numpy as np

from .geometry import inclusion_circles

if TYPE_CHECKING:
    from .fem import FineProblem
    from .offline import OfflineSpace


_STYLE = {"font.family": "DejaVu Sans", "font.size": 10,
          "axes.titlesize": 11, "axes.labelsize": 10}
_ZOOM = (-.8, .9, -.95, .95)


def _tri(problem):
    xy = problem.coordinates
    return Triangulation(xy[:, 0], xy[:, 1], problem.triangles)


def _axes(ax, *, zoom=False):
    ax.set_aspect("equal")
    bounds = _ZOOM if zoom else (-2, 2, -2, 2)
    ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="x", ylabel="y")


def _circles(ax, *, color="black", lw=.45):
    for x, y, radius, _ in inclusion_circles():
        ax.add_patch(Circle((x, y), radius, fill=False, edgecolor=color,
                            linewidth=lw, alpha=.7))


def _coarse(ax, count):
    for line in np.linspace(-2, 2, count + 1):
        ax.axhline(line, color="#32465a", lw=.55, alpha=.7)
        ax.axvline(line, color="#32465a", lw=.55, alpha=.7)


def plot_meshes(problem: FineProblem, n_coarse: int = 10,
                reference: FineProblem | None = None) -> Figure:
    """Материалы и крупная сетка; увеличенные тонкая и reference-сетки."""
    with plt.rc_context(_STYLE):
        cols = 3 if reference is not None else 2
        fig, axes = plt.subplots(1, cols, figsize=(5.1*cols, 5.1),
                                 layout="constrained", squeeze=False)
        axes = axes[0]
        cmap = ListedColormap(["#edf1f5", "#e8ae47", "#7eb3c9", "#ac739f"])
        artist = axes[0].tripcolor(_tri(problem), facecolors=problem.material_tags,
                                  cmap=cmap, norm=BoundaryNorm(np.arange(.5, 5), 4),
                                  shading="flat", rasterized=True)
        _coarse(axes[0], n_coarse)
        _circles(axes[0])
        _axes(axes[0])
        axes[0].set_title(f"Материалы; H = {4/n_coarse:g}")
        bar = fig.colorbar(artist, ax=axes[0], shrink=.76, pad=.025, ticks=[1, 2, 3, 4])
        bar.ax.set_yticklabels(["фон", "r = 0.02", "r = 0.05", "r = 0.10"])
        for ax, mesh, title in [(axes[1], problem, "Тонкая сетка")]+(
                [(axes[2], reference, "Независимая reference-сетка")] if reference is not None else []):
            ax.triplot(_tri(mesh), color="#495569", lw=.27, rasterized=True)
            _circles(ax, lw=.8)
            _coarse(ax, n_coarse)
            _axes(ax, zoom=True)
            ax.set_title(f"{title}\n{mesh.ndofs:,} DOF; {len(mesh.triangles):,} треугольников")
        return fig


def plot_basis(problem: FineProblem, space: OfflineSpace) -> Figure:
    """Показать χ, до трёх локальных базисов и спектр выбранной окрестности."""
    preview = space.preview
    dofs = np.asarray(preview["dofs"])
    cells = np.asarray(preview["cells"])
    xy = problem.coordinates[dofs]
    local_tri = np.searchsorted(dofs, problem.triangles[cells])
    triangulation = Triangulation(xy[:, 0], xy[:, 1], local_tri)
    values = [(r"Разбиение $\chi_i$", np.asarray(preview["chi"]))]
    values += [(rf"Базис $\chi_i\psi_{{{j+1}}}$", preview["basis"][:, j])
               for j in range(min(3, preview["basis"].shape[1]))]
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        for ax, (title, value) in zip(axes.ravel()[:4], values):
            bound = max(float(np.max(np.abs(value))), 1e-15)
            artist = ax.tripcolor(triangulation, np.real(value), shading="gouraud",
                                  cmap="RdBu_r", vmin=-bound, vmax=bound, rasterized=True)
            ax.set(title=title, xlabel="x", ylabel="y")
            ax.set_aspect("equal")
            ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
            ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
            _circles(ax)
            fig.colorbar(artist, ax=ax, shrink=.85, pad=.025)
        for ax in axes.ravel()[len(values):4]:
            ax.set_visible(False)
        spectrum = axes[1, 1]
        eig = np.asarray(space.eigenvalues)
        modes = np.arange(1, eig.shape[1] + 1)
        # Нулевое Neumann-значение имеет погрешность roundoff и на log не отображается.
        positive = np.where(eig > 1e-10, eig, np.nan)
        for row in positive:
            spectrum.semilogy(modes, row, color="#94a3b8", alpha=.25, lw=.7)
        spectrum.semilogy(modes, positive[int(preview["vertex"])], "o-",
                          color="#244d78", label="Выбранная окрестность")
        spectrum.set(title="Локальные собственные значения", xlabel="Номер моды",
                     ylabel=r"$\lambda_j$ (нулевая мода исключена)")
        spectrum.grid(True, which="both", alpha=.25)
        spectrum.legend(fontsize=8)
        axes[1, 2].axis("off")
        axes[1, 2].text(.05, .8,
                       f"Type {space.basis_type}\nH = {4/space.n_coarse:g}\n"
                       f"Вершина: {preview['vertex']}\n"
                       f"Локальных DOF: {len(dofs):,}\n"
                       f"Граничных snapshots: {preview['snapshots']}\n"
                       f"Мод на окрестность: {space.max_modes}",
                       transform=axes[1, 2].transAxes, va="top", linespacing=1.8)
        return fig


def plot_fields(problem: FineProblem, fine: np.ndarray,
                approximations: list[tuple[str, np.ndarray]], *, field: str = "total") -> Figure:
    """Сравнить Re/Im/модуль с единой цветовой шкалой в каждом столбце.

    paper_mixed повторяет явно подписанный вид DG notebook: Re(u_s), Im(u_s),
    |u_i+u_s|. Для слабого рассеяния шкала модуля охватывает реальный диапазон,
    иначе диапазон около единицы визуально исчезает на шкале от нуля.
    """
    if field not in ("total", "scattered", "paper_mixed"):
        raise ValueError("field должен быть total, scattered или paper_mixed")
    named = [(f"МКЭ · {problem.ndofs:,} DOF", fine)] + list(approximations)
    transforms = (np.real, np.imag, np.abs)
    titles = (r"Re $u^s$", r"Im $u^s$", r"$|u^i+u^s|$") if field == "paper_mixed" else ("Re u", "Im u", "|u|")
    triangulation = _tri(problem)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(len(named), 3, figsize=(13, 3.6*len(named)),
                                 squeeze=False, layout="constrained")
        for col, (transform, title) in enumerate(zip(transforms, titles)):
            offset = problem.incident if field == "total" or (field == "paper_mixed" and col == 2) else 0
            transformed = [transform(np.asarray(value)+offset) for _, value in named]
            maximum = max(float(np.max(np.abs(value))) for value in transformed)
            maximum = max(maximum, 1e-15)
            lower = min(float(np.min(value)) for value in transformed) if col == 2 else -maximum
            if maximum-lower < 1e-12:
                lower, maximum = lower-1e-12, maximum+1e-12
            norm = Normalize(lower, maximum)
            for row, values in enumerate(transformed):
                ax = axes[row, col]
                artist = ax.tripcolor(triangulation, values, shading="gouraud",
                                      cmap="viridis" if col == 2 else "RdBu_r", norm=norm,
                                      rasterized=True)
                _axes(ax)
                _circles(ax)
                ax.set_title(f"{named[row][0]}\n{title}")
            fig.colorbar(artist, ax=axes[:, col].tolist(), shrink=.8, pad=.025)
        fig.suptitle({"total": "Полное поле", "scattered": "Рассеянное поле",
                      "paper_mixed": "Вид исходного DG notebook: компоненты рассеянного поля и модуль полного"}[field])
        return fig


def plot_convergence(rows: list[dict]) -> Figure:
    """Ошибки статьи (%) по числу мод; линии различают H и тип базиса."""
    if not rows:
        raise ValueError("Нет результатов для графика сходимости")
    groups = sorted({(int(row["n_coarse"]), int(row["basis_type"])) for row in rows})
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        colors = {n: plt.get_cmap("tab10")(j) for j, n in enumerate(sorted({n for n, _ in groups}))}
        titles = (r"Re $u^s$", r"Im $u^s$", r"$|u^i+u^s|$") if rows[0].get("field") == "paper_mixed" else ("Re u", "Im u", "|u|")
        for ax, key, title in zip(axes, ("e_real_paper", "e_imag_paper", "e_abs_paper"), titles):
            for n, kind in groups:
                subset = sorted((r for r in rows if int(r["n_coarse"]) == n and int(r["basis_type"]) == kind),
                                key=lambda r: r["modes"])
                x = [r["modes"] for r in subset]
                y = np.array([r[key] for r in subset], dtype=float)
                y[~np.isfinite(y) | (y <= 0)] = np.nan
                ax.plot(x, y, marker="o" if kind == 1 else "s",
                        linestyle="-" if kind == 1 else "--", color=colors[n],
                        label=f"H = {4/n:g}; Type {kind}", markersize=4)
            ax.set(yscale="log", title=title, xlabel="Мод на окрестность, M",
                   ylabel="Относительная L² ошибка, %")
            ax.grid(True, which="both", alpha=.25)
            ax.legend(fontsize=8)
        return fig


def plot_refined_comparison(problem: FineProblem, fine: np.ndarray, ms: np.ndarray,
                            reference: FineProblem, reference_values: np.ndarray,
                            label: str) -> Figure:
    """Независимая reference: модули полного поля, ошибки рассеянного и сетки.

    Ошибки показываются в узлах основной сетки после P1-переноса reference.
    Это иллюстрация; интегральные ошибки вычисляются отдельно в metrics.
    """
    from .metrics import P1Evaluator

    ref_at_source = P1Evaluator(reference, reference_values)(problem.coordinates)
    totals = (np.abs(fine + problem.incident), np.abs(ms + problem.incident),
              np.abs(reference_values + reference.incident))
    errors = (np.abs(fine-ref_at_source), np.abs(ms-ref_at_source))
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout="constrained")
        lower, upper = min(float(v.min()) for v in totals), max(float(v.max()) for v in totals)
        if upper-lower < 1e-12:
            lower, upper = lower-1e-12, upper+1e-12
        norm = Normalize(lower, upper)
        for ax, mesh, values, title in zip(axes[0], (problem, problem, reference), totals,
                                         ("МКЭ", label, "Reference")):
            artist = ax.tripcolor(_tri(mesh), values, shading="gouraud", cmap="viridis", norm=norm,
                                  rasterized=True)
            _axes(ax)
            _circles(ax)
            ax.set_title(title + r": $|u_{\mathrm{total}}|$")
        fig.colorbar(artist, ax=axes[0].tolist(), shrink=.85, pad=.025)
        norm = Normalize(0, max(max(float(v.max()) for v in errors), 1e-15))
        for ax, values, title in zip(axes[1, :2], errors, ("МКЭ − reference", f"{label} − reference")):
            artist = ax.tripcolor(_tri(problem), values, shading="gouraud", cmap="magma", norm=norm,
                                  rasterized=True)
            _axes(ax)
            _circles(ax, color="white")
            ax.set_title(f"{title}\nМодуль ошибки рассеянного поля")
        fig.colorbar(artist, ax=axes[1, :2].tolist(), shrink=.85, pad=.025)
        ax = axes[1, 2]
        ax.triplot(_tri(reference), color="#d18c41", lw=.25, alpha=.55, label="Reference")
        ax.triplot(_tri(problem), color="#30485f", lw=.4, alpha=.8, label="Основная")
        _axes(ax, zoom=True)
        _circles(ax)
        ax.set_title("Сопоставление сеток: увеличение")
        ax.legend(fontsize=8)
        return fig


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Сохранить PNG (160 dpi) с полями для подписей и вернуть абсолютный путь."""
    path = Path(path).expanduser().resolve().with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    return path
