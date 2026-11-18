"""Russian scientific figures from verified scores; no training or PDE solves."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.reference_sno_gmsfem.dataset import _sha256
from experiments.reference_sno_gmsfem.run import write_json
from .comparison import validate_shared_results

SNO = "SVD-SNO shared"
FEM = "FEM h=0.04"
LABELS = {
    SNO: "Единая SVD-SNO", FEM: "FEM h=0,04",
    **{f"GMsFEM {n}x{n} Type {kind} M=16": f"GMsFEM {n}×{n}, тип {kind}, M=16"
       for n in (10, 20) for kind in (1, 2)},
}
COLORS = dict(zip(LABELS, ("#2563eb", "#64748b", "#a78bfa", "#0f766e", "#d97706", "#e85d3f")))
FIELDS = (("total", "Полное поле"), ("scattered", "Рассеянное поле"))


def paired_errors(rows, first, second, field):
    """Pair by exact query identity, never by file order or sorted error values."""
    tables = []
    for method in (first, second):
        selected = [r for r in rows if r["method"] == method]
        table = {r["index"]: r for r in selected}
        if not table or len(table) != len(selected):
            raise ValueError("Missing or duplicate method queries")
        tables.append(table)
    a, b = tables
    if a.keys() != b.keys():
        raise ValueError("Methods have different test queries")
    indices = sorted(a)
    for index in indices:
        if any(a[index][key] != b[index][key] for key in ("k", "theta")):
            raise ValueError("Paired physical queries differ")
    key = f"{field}_e_complex_ref"
    x = np.asarray([a[i][key] for i in indices])
    y = np.asarray([b[i][key] for i in indices])
    if not np.isfinite([x, y]).all() or np.any(x < 0) or np.any(y < 0):
        raise ValueError("Errors must be finite and nonnegative")
    return x, y, np.asarray([a[i]["k"] for i in indices])


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelcolor": "#334155", "text.color": "#0f172a",
        "axes.edgecolor": "#cbd5e1", "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "#e2e8f0",
        "grid.alpha": .8, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.frameon": False,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def _save(fig, output, stem):
    import matplotlib.pyplot as plt
    fig.savefig(output / f"{stem}.png", dpi=175)
    fig.savefig(output / f"{stem}.svg")
    plt.close(fig)


def _accuracy(rows, best, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8), layout="constrained")
    for axis, (field, title) in zip(axes, FIELDS):
        for method in (SNO, best, FEM):
            selected = [r for r in rows if r["method"] == method]
            ks = np.unique([r["k"] for r in selected])
            values = [np.asarray([r[f"{field}_e_complex_ref"] for r in selected if r["k"] == k]) for k in ks]
            mean = np.asarray([v.mean() for v in values])
            axis.plot(ks, mean, "o-" if method != FEM else "--", color=COLORS[method],
                      label=LABELS[method], linewidth=2.2, markersize=3)
            if method != FEM:
                axis.fill_between(ks, [np.percentile(v, 10) for v in values],
                                  [np.percentile(v, 90) for v in values],
                                  color=COLORS[method], alpha=.12, linewidth=0)
        axis.set(yscale="log", xlim=(.1, 4), xlabel="Волновое число k",
                 ylabel="Комплексная относительная L²-ошибка, %", title=title)
        axis.grid(which="major")
        axis.legend(loc="upper left")
    fig.suptitle("Одна SVD-SNO на всём диапазоне k ∈ [0,1; 4,0]\n"
                 "Линия — среднее по направлениям; заливка — 10–90-й перцентили, не доверительный интервал",
                 fontsize=12, fontweight="bold")
    _save(fig, output, "accuracy_by_k")


def _methods(summary, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ordered = sorted(summary, key=lambda r: r["total_mean_percent"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, layout="constrained")
    y = np.arange(len(ordered))
    for axis, (field, title) in zip(axes, FIELDS):
        for position, item in enumerate(ordered):
            mean, p95 = item[f"{field}_mean_percent"], item[f"{field}_p95_percent"]
            color = COLORS[item["method"]]
            axis.hlines(position, mean, p95, color=color, alpha=.45, linewidth=3)
            axis.scatter(mean, position, color=color, s=65, zorder=3)
            axis.scatter(p95, position, edgecolor=color, facecolor="white", s=55, marker="D", zorder=3)
            for value, offset in ((mean, 9), (p95, -15)):
                axis.annotate(f"{value:.2f}%", (value, position), xytext=(0, offset),
                              textcoords="offset points", ha="center", fontsize=9, color=color)
        axis.set(xscale="log", xlabel="Ошибка, % · логарифмическая шкала", title=title)
        axis.grid(axis="x")
        axis.margins(x=.2, y=.14)
    axes[0].set_yticks(y, [LABELS[r["method"]] for r in ordered])
    axes[0].invert_yaxis()
    axes[1].legend(handles=[
        Line2D([], [], color="#334155", marker="o", linestyle="none", label="Среднее"),
        Line2D([], [], color="#334155", marker="D", markerfacecolor="white", linestyle="none", label="95-й перцентиль"),
    ], loc="lower left")
    fig.suptitle("Все методы на одних и тех же 424 тестовых полях", fontsize=14, fontweight="bold")
    _save(fig, output, "methods_comparison")


def _distributions(rows, best, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    statistics = {}
    for col, (field, title) in enumerate(FIELDS):
        axis = axes[0, col]
        for method in (SNO, best, FEM):
            errors = np.sort([r[f"{field}_e_complex_ref"] for r in rows if r["method"] == method])
            axis.step(np.maximum(errors, 1e-10), np.arange(1, len(errors) + 1) / len(errors) * 100,
                      where="post", color=COLORS[method], linewidth=2, label=LABELS[method])
        axis.axhline(95, color="#94a3b8", linewidth=1, linestyle=":")
        axis.set(xscale="log", xlabel="Ошибка, %", ylabel="Доля тестовых полей, %",
                 title=f"{title}: распределение ошибок", ylim=(0, 101))
        axis.legend(fontsize=8)
        axis.grid()
        x, y, k = paired_errors(rows, SNO, best, field)
        wins = int(np.count_nonzero(x < y))
        statistics[field] = {"samples": len(x), "sno_better": wins,
                             "gmsfem_better": int(np.count_nonzero(y < x)),
                             "ties": int(np.count_nonzero(x == y))}
        axis = axes[1, col]
        dots = axis.scatter(np.maximum(x, 1e-10), np.maximum(y, 1e-10), c=k, cmap="viridis",
                            norm=Normalize(.1, 4), s=16, alpha=.75, edgecolors="none", rasterized=True)
        low = max(min(x.min(), y.min()) * .7, 1e-10)
        high = max(x.max(), y.max()) * 1.4
        axis.plot([low, high], [low, high], color="#94a3b8", linestyle="--", linewidth=1)
        axis.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
                 xlabel="Ошибка SVD-SNO, %", ylabel="Ошибка GMsFEM, %",
                 title=f"{title}: попарное сравнение")
        axis.text(.04, .96, f"SVD-SNO точнее: {wins} из {len(x)}\n"
                  "Выше диагонали — точнее SVD-SNO", transform=axis.transAxes,
                  va="top", fontsize=9, bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
        axis.grid()
    fig.colorbar(dots, ax=axes[1, :], label="Волновое число k", shrink=.85, pad=.025)
    fig.suptitle("Разброс качества: средняя ошибка не определяет победителя каждой задачи\n"
                 f"Сравнение с {LABELS[best]}", fontsize=12, fontweight="bold")
    _save(fig, output, "error_distributions")
    return statistics


def _training(history, metrics, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    epochs = np.arange(1, len(history["train"]) + 1)
    for key, label, color in (("train", "Обучение", "#2563eb"), ("val", "Validation", "#e85d3f")):
        axes[0].plot(epochs, np.asarray(history[key]) * 100, color=color, label=label, linewidth=1.5)
    best = metrics["best_epoch"]
    axes[0].axvline(best, color="#64748b", linestyle="--", label=f"Выбрана эпоха {best}")
    axes[0].scatter(best, history["val"][best - 1] * 100, color="#e85d3f", s=45, zorder=4)
    axes[0].set(yscale="log", xlabel="Эпоха", ylabel="Ошибка рассеянного поля, %",
                title="Согласованная FEM L² loss")
    axes[0].legend(fontsize=9)
    axes[1].step(epochs, history["lr"], where="post", color="#0f766e", linewidth=2)
    axes[1].set(yscale="log", xlabel="Эпоха", ylabel="Learning rate", title="Шаг оптимизатора")
    for axis in axes:
        axis.grid()
    fig.suptitle("Обучение одной SVD-SNO: 500 эпох, выбор checkpoint по validation",
                 fontsize=13, fontweight="bold")
    _save(fig, output, "training")


def build_visuals(data_dir, run_dir, results_dir, *, validation=None):
    import matplotlib
    matplotlib.use("Agg")
    results = Path(results_dir)
    if validation is None:
        validation = validate_shared_results(data_dir, run_dir, results)
    if validation.get("passed") is not True:
        raise ValueError("Figures require verified results")
    comparison = json.loads((results / "comparison.json").read_text())
    history = json.loads((results / "sno_history.json").read_text())
    metrics = json.loads((results / "sno_metrics.json").read_text())
    rows, summary = comparison["rows"], comparison["summary"]
    canonical = json.loads((results / "sno.json").read_text())["rows"] + json.loads(
        (results / "baselines.json").read_text())["rows"]
    from .report import summarize_rows
    if rows != canonical or summary != summarize_rows(canonical):
        raise ValueError("Plot data differ from verified query scores")
    best = min((r for r in summary if r["method"].startswith("GMsFEM")),
               key=lambda r: r["total_mean_percent"])["method"]
    output = results / "visuals"
    output.mkdir(parents=True, exist_ok=True)
    _style()
    _accuracy(rows, best, output)
    _methods(summary, output)
    paired = _distributions(rows, best, output)
    _training(history, metrics, output)
    lines = ["# Графики общей SVD-SNO и GMsFEM", "",
        "Все графики построены по проверенным результатам 424 общих test-полей "
        "одной фиксированной среды. Модель и POD-базисы общие для k∈[0.1,4]. "
        "Графики не требуют нового обучения или FEM-решений.", "",
        "## Ошибки во всём диапазоне k", "",
        "Линии — среднее по направлениям на каждой удержанной частоте. Заливка — "
        "10–90-й перцентили по этим направлениям; это не доверительный интервал. "
        "Сравнение с лучшим GMsFEM по средней ошибке полного поля и с FEM h=0.04. "
        "Reference для всех ошибок — FEM h=0.02.", "",
        "![Ошибка в зависимости от k](accuracy_by_k.png)", "",
        "[SVG для экспорта](accuracy_by_k.svg)", "",
        "## Средняя ошибка и хвост распределения", "",
        "Круг — среднее, ромб — 95-й перцентиль. Шкала ошибок логарифмическая. "
        "В ошибку GMsFEM входят редукция и различие его fine-сетки с reference.", "",
        "![Сравнение всех методов](methods_comparison.png)", "",
        "[SVG для экспорта](methods_comparison.svg)", "",
        "## Распределение и попарное сравнение", "",
        "Верхние графики показывают долю полей с ошибкой не выше заданной. "
        "Нижние сопоставляют ошибки на одних и тех же (k,θ); цвет обозначает k. "
        "Выше диагонали SVD-SNO точнее GMsFEM. Обобщённое среднее не означает "
        "превосходства на каждом отдельном запросе.", "",
        "![Распределение ошибок](error_distributions.png)", "",
        "[SVG для экспорта](error_distributions.svg)", "",
        "## Обучение", "", "![Кривые обучения](training.png)", "",
        "[SVG для экспорта](training.svg)", "",
        "## Физические поля и пространственная ошибка", "",
        "Re, Im, модуль и абсолютная комплексная разность рассеянного поля. "
        "Цветовые шкалы согласованы между методами для каждой компоненты. "
        "Визуализация использует nodal fields; табличные GMsFEM-метрики вычислены "
        "прямо в reference-квадратуре. Выбраны заранее проверенные k=1 и k=4, "
        "а не примеры с минимальной ошибкой.", "",
        "![Поля при k=1](../fields_k1.png)", "",
        "![Поля при k=4](../fields_k4.png)", "",
        "[Полный численный отчёт](../REPORT.md) · [Данные сравнения](../comparison.csv)", ""]
    (output / "README.md").write_text("\n".join(lines))
    stems = ("accuracy_by_k", "methods_comparison", "error_distributions", "training")
    write_json(output / "manifest.json", dict(
        passed=True, test_samples=validation["test_samples"], baseline_method=best,
        paired_statistics=paired, generator_sha256=_sha256(Path(__file__)),
        source_sha256={name: _sha256(results / name) for name in
                       ("comparison.json", "sno_history.json", "sno_metrics.json", "verification.json")},
        files={f"{stem}.{extension}": _sha256(output / f"{stem}.{extension}")
               for stem in stems for extension in ("png", "svg")}))
    print(f"Saved 4 PNG + 4 SVG figures and gallery: {output}", flush=True)
