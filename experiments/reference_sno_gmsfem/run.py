"""Reproducible stages: data, train, baselines, report (one process per stage)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BANDS = ("low", "high")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def training_config(args):
    from .training import TrainConfig
    return TrainConfig(method="svd", epochs=args.epochs, batch_size=32,
                       learning_rate=0.002, width=48, depth=6, rank=96,
                       input_rank=32, sno_architecture_version=2,
                       target_projection_error=0.001, pod_method="randomized",
                       patience=100, min_epochs=150, seed=42, threads=2, device="cpu")


def evaluation_rows(dataset, predictor, indices, method):
    """The SAME quadrature and percent convention as the GMsFEM tables."""
    from gmsfem.metrics import relative_errors
    xy = dataset.coordinates[dataset.triangles]
    edges = xy[:, 1:] - xy[:, :1]
    areas = np.abs(edges[:, 0, 0]*edges[:, 1, 1] - edges[:, 0, 1]*edges[:, 1, 0]) / 2
    problem = SimpleNamespace(coordinates=dataset.coordinates,
                              triangles=dataset.triangles, areas=areas, case="notebook")
    rows = []
    # One field at a time bounds peak memory on the 53,246-node mesh.
    for position, index in enumerate(indices):
        problem.k = float(dataset.k0[index])
        problem.direction = (float(np.cos(dataset.theta[index])), float(np.sin(dataset.theta[index])))
        start = perf_counter()
        pred = predictor.predict_dataset(dataset, np.asarray([index]))[0]
        seconds = perf_counter() - start
        row = dict(method=method, index=int(index), k=problem.k,
                   theta=float(dataset.theta[index]), inference_seconds=seconds)
        for field in ("total", "scattered"):
            row.update({f"{field}_{key}": float(value) for key, value in
                        relative_errors(problem, pred, dataset.u_scat[index], field=field).items()})
        rows.append(row)
        if position % 40 == 0:
            print(f"{method}: metrics {position + 1}/{len(indices)}", flush=True)
    return rows


def record_environment():
    import dolfinx
    import gmsh
    import scipy
    import torch
    from petsc4py import PETSc
    value = dict(python=platform.python_version(), machine=platform.machine(),
                 numpy=np.__version__, scipy=scipy.__version__,
                 torch=str(torch.__version__), dolfinx=dolfinx.__version__, gmsh=gmsh.__version__,
                 petsc=list(PETSc.Sys.getVersion()), scalar_type=str(np.dtype(PETSc.ScalarType)),
                 threads=2, training_preprocessing_blas_threads=2, fem_blas_threads=1, device="cpu")
    if value["scalar_type"] != "complex128":
        raise RuntimeError("Complex PETSc is required")
    value["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for folder in (HERE, ROOT / "neural_operator", ROOT / "gmsfem")
                              for p in sorted(folder.rglob("*.py"))}
    write_json(HERE / "results" / "environment.json", value)
    for band in BANDS:
        for name in ("config", "dataset_info", "provenance"):
            path = HERE / "data" / band / f"{name}.json"
            if path.exists():
                write_json(HERE / "results" / band / f"dataset_{name}.json", json.loads(path.read_text()))


def run_data(args, bands):
    from .dataset import generate_dataset
    for band in bands:
        generate_dataset(HERE / "data" / band, band=band, n_frequencies=args.frequencies,
                         n_directions=args.directions, seed=2026)
        for name in ("config", "dataset_info", "provenance"):
            write_json(HERE / "results" / band / f"dataset_{name}.json",
                       json.loads((HERE / "data" / band / f"{name}.json").read_text()))


def run_train(args, bands):
    from .dataset import load_dataset
    from .training import run_experiment
    for band in bands:
        dataset = load_dataset(HERE / "data" / band)
        predictor, metrics, history = run_experiment(dataset, HERE / "runs" / band,
                                                     training_config(args))
        write_json(HERE / "results" / band / "sno_metrics.json", metrics)
        write_json(HERE / "results" / band / "sno_history.json", history)
        rows = evaluation_rows(dataset, predictor, dataset.split_indices()[2], f"SNO {band}")
        write_csv(HERE / "results" / band / "sno.csv", rows)
        write_json(HERE / "results" / band / "sno.json", {"rows": rows})


def run_extrapolation():
    from .dataset import load_dataset
    from neural_operator.training import load_predictor
    dataset = load_dataset(HERE / "data" / "high")
    predictor = load_predictor(HERE / "runs" / "low" / "model.pt")
    test = dataset.split_indices()[2]
    test = test[dataset.k0[test] > 2]
    rows = evaluation_rows(dataset, predictor, test, "SNO low extrapolation")
    write_csv(HERE / "results" / "high" / "extrapolation.csv", rows)
    write_json(HERE / "results" / "high" / "extrapolation.json", {"rows": rows})


def run_baselines(bands):
    from .dataset import load_dataset
    from .baselines import evaluate_baselines
    for band in bands:
        dataset = load_dataset(HERE / "data" / band)
        evaluate_baselines(dataset, HERE / "results" / band,
                           dataset.split_indices()[2], coarse_grids=(10, 20),
                           basis_types=(1, 2), modes=16, save_fields=False)


def report():
    from .report_validation import validate_results
    validation = validate_results(HERE / "results", HERE / "data")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    all_rows, aggregates, training_notes, timing_notes = [], [], [], []
    lines = ["# Результаты SNO и GMsFEM на эталонной среде", "",
             "Ошибки ниже — проценты, знаменатель — эталонное поле. "
             "Оба метода проверены на одинаковых удержанных частотах и направлениях; "
             "полное поле интегрируется с аналитической падающей волной.", "",
             "| Диапазон / метод | Число полей | Полное: mean, % | Полное: p95, % | Рассеянное: mean, % | Рассеянное: p95, % |",
             "|---|---:|---:|---:|---:|---:|"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for col, band in enumerate(BANDS):
        folder = HERE / "results" / band
        rows = []
        for name in ("sno.json", "baselines.json", "extrapolation.json"):
            path = folder / name
            if path.exists():
                value = json.loads(path.read_text())
                rows.extend(value["rows"] if isinstance(value, dict) else value)
        if not rows:
            continue
        for row in rows:
            row["band"] = band
        all_rows.extend(rows)
        methods = list(dict.fromkeys(row["method"] for row in rows))
        for method in methods:
            selected = [r for r in rows if r["method"] == method]
            total = np.array([r["total_e_complex_ref"] for r in selected])
            scat = np.array([r["scattered_e_complex_ref"] for r in selected])
            item = dict(band=band, method=method, samples=len(selected),
                        total_mean_percent=float(total.mean()), total_p95_percent=float(np.percentile(total, 95)),
                        total_max_percent=float(total.max()), scattered_mean_percent=float(scat.mean()),
                        scattered_p95_percent=float(np.percentile(scat, 95)), scattered_max_percent=float(scat.max()))
            aggregates.append(item)
            lines.append(f"| {band} / {method} | {len(selected)} | {total.mean():.4f} | {np.percentile(total,95):.4f} | {scat.mean():.4f} | {np.percentile(scat,95):.4f} |")
            for axis, metric in zip(axes[:, col], ("total_e_complex_ref", "scattered_e_complex_ref")):
                ks = sorted(set(r["k"] for r in selected))
                means = [np.mean([r[metric] for r in selected if r["k"] == k]) for k in ks]
                axis.semilogy(ks, np.maximum(means, 1e-10), ".-", label=method, markersize=3)
        axes[0, col].set_title(f"K = {'[0.1, 2]' if band == 'low' else '[2, 4]'}")
        for axis, field in zip(axes[:, col], ("Total field", "Scattered field")):
            axis.set_xlabel("k")
            axis.set_ylabel(f"{field}: relative complex L2, %")
            axis.grid(alpha=.25)
            axis.legend(fontsize=6)
        metric_path = folder / "sno_metrics.json"
        if metric_path.exists():
            m = json.loads(metric_path.read_text())
            training_notes.extend(["", f"{band}: SNO — {m['parameters']} параметров, POD-ранги {m['ranks']}, "
                          f"эпох {m['completed_epochs']}, лучшая {m['best_epoch']}, остановка {m['stop_reason']}; "
                          f"POD floor (test scattered mean) = {100*m['projection_fem_l2']['mean']:.4f}%. "
                          f"Validation-порог POD достигнут: {m['rank_target_attained']}. "
                          f"Подготовка {m['preprocessing_seconds']:.1f} с, обучение {m['training_seconds']:.1f} с.", ""])
            sno_rows = [r for r in rows if r["method"] == f"SNO {band}"]
            timing_notes += [f"| {band} / SNO | {m['preprocessing_seconds']:.2f} | {m['training_seconds']:.2f} | — | "
                             f"{1000*np.mean([r['inference_seconds'] for r in sno_rows]):.3f} |"]
        for method in methods:
            if method.startswith("SNO"):
                continue
            selected = [r for r in rows if r["method"] == method]
            first_by_k = {r["k"]: r for r in selected}
            offline = (selected[0].get("offline_seconds", 0.) if selected[0].get("basis_type") == 1
                       else sum(r.get("offline_seconds", 0.) for r in first_by_k.values()))
            setup = sum(r.get("matrix_projection_seconds", 0.) + r.get("factorization_seconds", 0.)
                        for r in first_by_k.values())
            online = np.mean([r.get("rhs_projection_seconds", 0.) + r.get("solve_seconds", 0.)
                              + r.get("reconstruction_seconds", 0.) + r.get("transfer_seconds", 0.)
                              for r in selected])
            timing_notes += [f"| {band} / {method} | {offline:.2f} | — | {setup:.2f} | {1000*online:.3f} |"]
    (HERE / "results").mkdir(exist_ok=True)
    fig.savefig(HERE / "results" / "errors_by_k.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, band in zip(axes, BANDS):
        path = HERE / "results" / band / "sno_history.json"
        if path.exists():
            h = json.loads(path.read_text())
            for key in ("train", "val"):
                axis.semilogy(np.arange(1, len(h[key])+1), np.array(h[key])*100, label=key)
            axis.set(title=band, xlabel="Epoch", ylabel="Scattered FEM L2 loss, %")
            axis.legend()
            axis.grid(alpha=.25)
    fig.savefig(HERE / "results" / "training.png", dpi=180)
    plt.close(fig)
    lines += ["", "## Какая модель точнее в этом запуске", ""]
    for band in BANDS:
        sno = next(a for a in aggregates if a["band"] == band and a["method"] == f"SNO {band}")
        best = min((a for a in aggregates if a["band"] == band and a["method"].startswith("GMsFEM")),
                   key=lambda a: a["total_mean_percent"])
        winner = sno if sno["total_mean_percent"] < best["total_mean_percent"] else best
        lines += [f"{band}: по средней комплексной ошибке полного поля точнее **{winner['method']}**. "
                  f"SNO: {sno['total_mean_percent']:.4f}%; лучший GMsFEM ({best['method']}): "
                  f"{best['total_mean_percent']:.4f}%. Это вывод для удержанных частот, "
                  "одной среды и указанных настроек; времени обучения он не учитывает.", ""]
    ood = next(a for a in aggregates if a["method"] == "SNO low extrapolation")
    lines += [f"Замороженная low-модель на **k>2**: средняя ошибка полного поля "
              f"**{ood['total_mean_percent']:.4f}%**, рассеянного — {ood['scattered_mean_percent']:.4f}%. "
              "Успех отдельно обученной high-модели нельзя трактовать как успешную экстраполяцию low-модели.", "",
              "GMsFEM сохраняет исходную fine-сетку h=0.04, а SNO обучена на reference h=0.02. "
              "Строка `FEM h=0.04` показывает вклад этого различия. При фиксированных 16 модах "
              "в high-диапазоне возможны большие ошибки редукции даже при очень малой невязке "
              "Галеркина; малая невязка не доказывает близость к reference FEM.", ""]
    lines += ["## Точка исходной таблицы: k = 1, θ = 0", "",
              "В последнем столбце — сохранённое сравнение GMsFEM с более мелким FEM-эталоном. "
              "Основная старая таблица относительно fine h=0.04 использует другой знаменатель-эталон; "
              "она отдельно воспроизведена в `table_reproduction/`.", "",
              "| Метод | Re/ref, % | Im/ref, % | Модуль/ref, % | Complex/ref, % | Сохранённый complex/ref, % |",
              "|---|---:|---:|---:|---:|---:|"]
    with (ROOT / "gmsfem/results/notebook/reference_errors.csv").open() as stream:
        saved_anchor = {r["label"]: float(r["e_complex_ref"]) for r in csv.DictReader(stream)}
    for row in all_rows:
        if row["band"] != "low" or row["k"] != 1. or row["theta"] != 0.:
            continue
        label = None
        if row["method"] == "FEM h=0.04":
            label = "МКЭ h"
        elif row["method"].startswith("GMsFEM"):
            n, t = row["n_coarse"], row["basis_type"]
            label = f"GMsFEM {n}×{n}, Type {t}, M=16"
        old = f"{saved_anchor[label]:.6f}" if label else "—"
        lines += [f"| {row['method']} | {row['total_e_real_ref']:.6f} | {row['total_e_imag_ref']:.6f} | "
                  f"{row['total_e_abs_ref']:.6f} | {row['total_e_complex_ref']:.6f} | {old} |"]
    lines += training_notes + ["", "## Вычислительная стоимость", "",
              "| Диапазон / метод | POD либо offline, с | Обучение, с | Проекция матриц + LU, с | Online на поле, мс |",
              "|---|---:|---:|---:|---:|"] + timing_notes + ["",
              "SNO online измерен по одному полю: кодирование входа, сеть и восстановление на reference-узлах. "
              "GMsFEM online включает проекцию RHS, решение, восстановление и межсеточный перенос "
              "как в reference-узлы, так и в точки квадратуры для последующей оценки; "
              "сборка физической RHS и интегрирование ошибок сюда не включены. "
              "Offline/LU учтены один раз при фактическом переиспользовании. "
              "Torch использует 2 потока, BLAS в FEM — 1; измерения выполнены при параллельной работе этапов. "
              "Времена не являются изолированным аппаратным benchmark и не задают равный бюджет. "
              "Стоимость SNO в таблице относится к окончательному запуску; предварительный validation-scout "
              "с постоянным шагом (остановлен до test) — дополнительный расход, описанный в `validation_decision.json`. "
              "Генерация FEM-меток для обучения учитывается отдельно в `dataset_dataset_info.json`; "
              "для GMsFEM обучающие метки не требуются.", "",
              "![Ошибка по k](errors_by_k.png)", "", "![Обучение](training.png)", "",
              "`SNO low extrapolation` — замороженная модель из [0.1,2] на тестовых k>2. "
              "`SNO high` — отдельная модель, обученная внутри [2,4]. Это разные проверки.", "",
              "Лимит окончательного обучения — 500 эпох. Этот опыт сравнивает конкретные сохранённые модели; "
              "минимально возможная ошибка SNO при другом бюджете или настройках здесь не установлена.", "",
              "Подробные гиперпараметры: [MODEL.md](../MODEL.md). "
              "Проверка исходной таблицы k=1 сохранена в `table_reproduction/` в подпапках диапазонов. "
              "16 мод GMsFEM задаются на каждую крупную вершину и не равны глобальному POD-рангу SNO.", ""]
    write_csv(HERE / "results" / "comparison.csv", all_rows)
    write_json(HERE / "results" / "comparison.json", {"summary": aggregates, "rows": all_rows})
    write_json(HERE / "results" / "validation.json", validation)
    (HERE / "results" / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("data", "train", "baselines", "extrapolate", "report", "environment"))
    parser.add_argument("--band", choices=(*BANDS, "both"), default="both")
    parser.add_argument("--frequencies", type=int, default=160)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    bands = BANDS if args.band == "both" else (args.band,)
    if args.stage == "data":
        run_data(args, bands)
    elif args.stage == "train":
        run_train(args, bands)
    elif args.stage == "baselines":
        run_baselines(bands)
    elif args.stage == "extrapolate":
        run_extrapolation()
    elif args.stage == "report":
        report()
    elif args.stage == "environment":
        record_environment()


if __name__ == "__main__":
    main()
