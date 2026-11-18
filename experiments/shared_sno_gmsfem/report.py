"""Measured paired comparison, training curves and actual field visualizations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.reference_sno_gmsfem.dataset import load_dataset
from experiments.reference_sno_gmsfem.run import write_csv, write_json
from .comparison import validate_shared_results


def summarize_rows(rows):
    result = []
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        item = {"method": method, "samples": len(selected)}
        for field in ("total", "scattered"):
            values = np.array([row[f"{field}_e_complex_ref"] for row in selected])
            item.update({f"{field}_{key}_percent": float(value) for key, value in (
                ("mean", values.mean()), ("median", np.median(values)),
                ("p95", np.percentile(values, 95)), ("max", values.max()))})
        result.append(item)
    return result


def _field_plots(data, run_dir, results_dir, scores):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    mesh = mtri.Triangulation(*data.coordinates.T, triangles=data.triangles)
    with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as archive:
        prediction = archive["prediction"]
        position = {int(index): i for i, index in enumerate(archive["indices"])}
    with np.load(results_dir / "verification" / "baseline_fields.npz", allow_pickle=False) as archive:
        for field_position, index in enumerate(archive["sample_indices"]):
            if data.k0[index] not in (1., 4.):
                continue
            true = data.u_scat[index]
            sno = prediction[position[int(index)]]
            gms = archive["gmsfem_n20_type2_M16"][field_position]
            fig, axes = plt.subplots(4, 3, figsize=(12, 14), constrained_layout=True)
            for row, (label, transform) in enumerate((
                ("Re(u_s)", np.real), ("Im(u_s)", np.imag), ("Abs(u_s)", np.abs),
            )):
                values = [transform(value) for value in (true, sno, gms)]
                limit = max(np.max(np.abs(value)) for value in values)
                for col, value in enumerate(values):
                    image = axes[row, col].tripcolor(
                        mesh, value, shading="gouraud", cmap="viridis" if row == 2 else "RdBu_r",
                        vmin=0 if row == 2 else -limit, vmax=limit)
                    axes[row, col].set_title(f"{('FEM reference', 'SVD-SNO shared', 'GMsFEM 20x20 Type2')[col]}: {label}")
                    fig.colorbar(image, ax=axes[row, col], shrink=.7)
            axes[3, 0].axis("off")
            sno_error = scores["SVD-SNO shared", int(index)]["scattered_e_complex_ref"]
            gms_error = scores["GMsFEM 20x20 Type 2 M=16", int(index)]["scattered_e_complex_ref"]
            axes[3, 0].text(.05, .7, f"k={data.k0[index]:g}\ntheta={data.theta[index]:.6f}\n"
                           f"Complex relative L2:\nSVD-SNO: {sno_error:.4f}%\nGMsFEM: {gms_error:.4f}%",
                           transform=axes[3, 0].transAxes, va="top")
            errors = [np.abs(sno - true), np.abs(gms - true)]
            limit = max(np.max(value) for value in errors)
            for col, value in enumerate(errors, start=1):
                image = axes[3, col].tripcolor(mesh, value, shading="gouraud",
                                               cmap="magma", vmin=0, vmax=limit)
                axes[3, col].set_title("Abs(complex prediction error), nodal view")
                fig.colorbar(image, ax=axes[3, col], shrink=.7)
            for axis in axes.flat:
                axis.set_aspect("equal")
            fig.suptitle("One shared model on k in [0.1,4]; scattered fields")
            fig.savefig(results_dir / f"fields_k{data.k0[index]:g}.png", dpi=140)
            plt.close(fig)


def build_report(data_dir, run_dir, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir, run_dir, results_dir = map(Path, (data_dir, run_dir, results_dir))
    validation = validate_shared_results(data_dir, run_dir, results_dir)
    metrics = json.loads((results_dir / "sno_metrics.json").read_text())
    history = json.loads((results_dir / "sno_history.json").read_text())
    baseline = json.loads((results_dir / "baselines.json").read_text())
    rows = json.loads((results_dir / "sno.json").read_text())["rows"] + baseline["rows"]
    summary = summarize_rows(rows)
    intervals = []
    for lower, upper in ((.1, 1.), (1., 2.), (2., 3.), (3., 4.)):
        selected = [r for r in rows if (lower <= r["k"] if lower == .1 else lower < r["k"])
                    and r["k"] <= upper]
        intervals.extend(dict(k_lower=lower, k_upper=upper, **r) for r in summarize_rows(selected))
    write_json(results_dir / "comparison.json", dict(summary=summary, intervals=intervals, rows=rows))
    write_csv(results_dir / "comparison.csv", rows)
    write_csv(results_dir / "summary.csv", summary)
    write_json(results_dir / "validation.json", validation)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        ks = np.unique([r["k"] for r in selected])
        for axis, field in zip(axes, ("total", "scattered")):
            values = [np.mean([r[f"{field}_e_complex_ref"] for r in selected if r["k"] == k])
                      for k in ks]
            axis.semilogy(ks, np.maximum(values, 1e-10), ".-", label=method,
                          linewidth=2 if method == "SVD-SNO shared" else 1, markersize=3)
    for axis, field in zip(axes, ("Total field", "Scattered field")):
        axis.set(xlabel="k", ylabel=f"{field}: complex relative L2, %", xlim=(.1, 4))
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.savefig(results_dir / "errors_by_k.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key in ("train", "val"):
        axes[0].semilogy(np.arange(1, len(history[key]) + 1), np.array(history[key]) * 100, label=key)
    axes[0].axvline(metrics["best_epoch"], color="gray", linestyle="--", label="selected epoch")
    axes[0].set(xlabel="Epoch", ylabel="Scattered consistent FEM L2, %")
    axes[0].legend()
    axes[1].semilogy(np.arange(1, len(history["lr"]) + 1), history["lr"])
    axes[1].set(xlabel="Epoch", ylabel="Learning rate")
    for axis in axes:
        axis.grid(alpha=.25)
    fig.savefig(results_dir / "training.png", dpi=170)
    plt.close(fig)
    scores = {(r["method"], r["index"]): r for r in rows}
    _field_plots(load_dataset(data_dir), run_dir, results_dir, scores)

    sno = next(r for r in summary if r["method"] == "SVD-SNO shared")
    gm = [r for r in summary if r["method"].startswith("GMsFEM")]
    best = min(gm, key=lambda r: r["total_mean_percent"])
    winner = sno if sno["total_mean_percent"] < best["total_mean_percent"] else best
    floor = metrics["consistent_projection_floor_fem_l2"]
    lines = [
        "# Одна SVD-SNO на k ∈ [0.1,4.0] и сравнение с GMsFEM", "",
        "**Обучение и независимая проверка выполнены.** Используется один checkpoint, "
        "одна нормировка k и общие train-only POD-базисы Re/Im. "
        "Частотного выбора модели или декодера нет.", "",
        f"На общем test из {validation['test_samples']} полей средняя complex L2 ошибка полного поля "
        f"SVD-SNO равна **{sno['total_mean_percent']:.4f}%**, лучшего GMsFEM "
        f"({best['method']}) — **{best['total_mean_percent']:.4f}%**. "
        f"По этой метрике точнее **{winner['method']}** в данном запуске.", "",
        "| Метод | Test-полей | Полное: mean, % | Полное: p95, % | Полное: max, % | Рассеянное: mean, % | Рассеянное: p95, % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(f"| {item['method']} | {item['samples']} | {item['total_mean_percent']:.4f} | "
                     f"{item['total_p95_percent']:.4f} | {item['total_max_percent']:.4f} | "
                     f"{item['scattered_mean_percent']:.4f} | {item['scattered_p95_percent']:.4f} |")
    lines += ["", "## Данные и физическая задача", "",
        "Объединены существующие **2568 FEM-полей / 320 уникальных k**: "
        "train 1712/214 частот, validation 432/54, test 424/52. "
        "Все прежние splits сохранены. Группа k=2 содержит 16 различных направлений "
        "и целиком принадлежит test. Частоты образуют union прежних сеток, а не новый "
        "равномерный linspace. Никаких новых обучающих FEM-решений не потребовалось.", "",
        "Среда фиксирована: Ω=[−2,2]², 24 включения, фон α=n=1, "
        "во включениях A=0.2I и n=10. Падающая волна exp(ik d·x), d=(cosθ,sinθ). "
        "Комплексный P1-оператор Kα−k²Mn−ikB; поглощающее условие первого порядка. "
        "Reference: h=0.02, 53 246 узлов, 105 690 треугольников. "
        "Проверяется интерполяция по k и θ одной среды, а не перенос на новые материалы.", "",
        "## Архитектура и фактическое обучение", "",
        "SVD-SNO v2: tokens [c_A,j,c_n,j,cosθ,sinθ,scaled k], lift 5→48, "
        "6 блоков Z←Z+GELU(ZWlocal+b+BZWchannel), общая голова 48→2, "
        "обратная train-нормировка и два POD-декодера. "
        "Материал неизменен, поэтому r_A=r_n=0 и независимые входы фактически (k,θ).", "",
        f"Фактические POD-ранги: `{metrics['ranks']}`; "
        f"обучаемых параметров **{metrics['parameters']}**. "
        "Rank cap=96; ранг выбирался только на validation с порогом 0.1% среднего "
        f"projection error; порог достигнут: **{metrics['rank_target_attained']}**. "
        f"Consistent decoder floor на test: mean **{100*floor['mean']:.4f}%**.", "",
        "AdamW, lr=0.002, weight decay=1e-5, batch=32, seed=42, CPU threads=2; "
        "gradient clipping=1. ReduceLROnPlateau: factor=0.5, patience=20, min_lr=1e-5. "
        "Loss — относительная комплексная consistent FEM L2 рассеянного поля, "
        "точно вычисляемая в POD-координатах с учётом decoder floor.", "",
        f"Выполнено **{metrics['completed_epochs']} эпох**, выбран checkpoint эпохи "
        f"**{metrics['best_epoch']}** по validation. Причина остановки: `{metrics['stop_reason']}`. "
        f"Подготовка: {metrics['preprocessing_seconds']:.2f} с; обучение: "
        f"{metrics['training_seconds']:.2f} с. Test не использовался для выбора ранга, "
        "эпохи или гиперпараметров. Один seed не определяет разброс между обучениями.", "",
        "## Согласованность сравнения", "",
        "GMsFEM: исходная fine-сетка h=0.04, 17 549 узлов; крупные сетки 10×10 и 20×20, "
        "Type 1/2, по 16 локальных мод на крупную вершину (1936/7056 coarse DOF). "
        "Тип 1 имеет эллиптические snapshots, Type 2 — частотно-зависимые Helmholtz snapshots. "
        "Поэтому в ошибку GMsFEM входят и редукция, и отличия его исходной fine-сетки.", "",
        "Все 2120 baseline-строк перенесены из проверенных исходных запусков по точным "
        "(source, index, k, θ), с проверкой SHA256 полей, геометрии и настроек. "
        "Метрики не интерполировались. Дополнительно заново решены FEM h=0.04 и "
        "GMsFEM 20×20 Type 2 M=16 при k=0.1,1,2,4; совпадение с сохранёнными "
        "метриками проверено с допуском 1e-6. "
        "Все 424 предсказания SNO повторно получены после загрузки checkpoint и сверены.", "",
        "Комплексные ошибки вычисляются в одних reference-квадратурных точках; для полного "
        "поля падающая волна задаётся аналитически. GMsFEM оценивается прямо из своего "
        "P1-пространства в этих точках. Промежуточные nodal fields используются только "
        "для рисунков. Ошибки в таблице — проценты относительно соответствующего "
        "reference-поля; полное и рассеянное поля имеют разные знаменатели.", "",
        "Baseline timings относятся к двум историческим запускам; Type1 и setup при k=2 "
        "выполнялись в них раздельно. Здесь не заявляется измеренное ускорение единого "
        "GMsFEM workflow относительно SNO. FEM-reference имеет собственную ошибку "
        "дискретизации и аппроксимации внешней границы.", "",
        "## Ошибки по подынтервалам одной модели", "",
        "Эти интервалы используются только для анализа; веса и POD-декодеры общие.", "",
        "| Интервал k | Метод | Полей | Полное mean, % | Рассеянное mean, % |",
        "|---|---|---:|---:|---:|",
    ]
    for item in intervals:
        if item["method"] in ("SVD-SNO shared", best["method"]):
            bracket = "[" if item["k_lower"] == .1 else "("
            lines.append(f"| {bracket}{item['k_lower']:g},{item['k_upper']:g}] | {item['method']} | "
                         f"{item['samples']} | {item['total_mean_percent']:.4f} | "
                         f"{item['scattered_mean_percent']:.4f} |")
    lines += ["", "![Ошибки по волновому числу](errors_by_k.png)", "",
              "![Обучение одной модели](training.png)", "",
              "![Поля при k=1](fields_k1.png)", "",
              "![Поля при k=4](fields_k4.png)", "",
              "[Полные per-query данные](comparison.csv) · [JSON](comparison.json) · "
              "[Протокол](protocol.json) · [Проверка](validation.json) · "
              "[Независимый reload/FEM-контроль](verification.json)", ""]
    (results_dir / "REPORT.md").write_text("\n".join(lines))
    from .visualize import build_visuals
    build_visuals(data_dir, run_dir, results_dir, validation=validation)
    with (results_dir / "REPORT.md").open("a") as stream:
        stream.write("\n[Галерея графиков на русском и SVG для экспорта](visuals/README.md)\n")
    print(json.dumps(summary, indent=2), flush=True)
