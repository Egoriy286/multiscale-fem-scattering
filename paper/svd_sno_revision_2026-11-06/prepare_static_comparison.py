"""Compare fixed-medium randomized SVD-SNO and FNO saved runs.

Run from the repository root inside dolfinx-complex. No training or FEM solves.
Both methods use the sample selected by the existing SNO median-error rule;
FNO errors do not influence sample selection. Use --band low for the completed
low-band pair; --band all also requires a completed high-band FNO checkpoint.
"""
from __future__ import annotations

import csv
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch

from prepare_static_predictions import (
    COMPONENTS, IMAGES, REPOSITORY, relative, sha256, summarize, verify_band,
    scattering_components, load_predictor,
)

LABELS = {"svd": "SVD-SNO", "fno": "FNO"}
BANDS = ("low", "high")
EXPECTED_INDICES = {"low": 2211, "high": 6077}


def verify_fno(band, dataset, index, sno_record):
    run = REPOSITORY / f"neural_operator/randomized/runs/{band}_frequency_static_1000x8/fno"
    checkpoint_path = run / "model.pt"
    metrics_path = run / "metrics.json"
    predictions_path = run / "test_predictions.npz"
    for path in (checkpoint_path, metrics_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required completed static FNO artifact is missing: {path}")
    metrics = json.loads(metrics_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if (checkpoint["method"] != "fno" or metrics["method"] != "fno"
            or metadata["dataset_fingerprint"] != sno_record["dataset_fingerprint"]
            or metrics["dataset_fingerprint"] != sno_record["dataset_fingerprint"]):
        raise ValueError(f"{band}: FNO method or static dataset fingerprint mismatch")
    # Device may differ because the current FNO run uses host MPS. Architecture,
    # seed, optimizer, batching and stopping settings must otherwise agree.
    expected_config = dict(sno_record["config"], method="fno", device=metrics["config"]["device"])
    if metrics["config"] != expected_config or metadata["config"] != expected_config:
        raise ValueError(f"{band}: FNO configuration differs from SNO beyond method/device")
    for key in ("method", "field", "metric", "split_strategy", "physical_media",
                "train_samples", "val_samples", "test_samples", "split_groups",
                "completed_epochs", "best_epoch", "test_fem_l2"):
        if metadata[key] != metrics[key]:
            raise ValueError(f"{band}: FNO checkpoint/metrics disagree on {key}")
    if (metrics["field"] != "scattered" or metrics["metric"] != "consistent_complex_FEM_L2"
            or metrics["split_strategy"] != "frequency" or metrics["physical_media"] != 1
            or [metrics[k] for k in ("train_samples", "val_samples", "test_samples")]
            != [5360, 1360, 1280]
            or metrics["split_groups"] != {"train": 670, "validation": 170, "test": 160}):
        raise ValueError(f"{band}: FNO field, norm or split metadata mismatch")
    predictor = load_predictor(checkpoint_path)
    if (not np.array_equal(predictor.coordinates, dataset.coordinates)
            or not np.array_equal(predictor.triangles, dataset.triangles)):
        raise ValueError(f"{band}: FNO mesh or node ordering mismatch")
    test = dataset.split_indices()[2]
    position = int(np.flatnonzero(test == index)[0])
    with np.load(predictions_path, allow_pickle=False) as archive:
        if (not np.array_equal(archive["indices"], test)
                or archive["prediction"].shape != (1280, len(dataset.coordinates))
                or archive["fem_l2"].shape != (1280,)):
            raise ValueError(f"{band}: FNO saved test field shape or indices mismatch")
        saved = archive["prediction"]
        errors = dataset.relative_l2(dataset.u_scat[test], saved)
        if not np.all(np.isfinite(errors)):
            raise ValueError(f"{band}: nonfinite FNO test errors")
        np.testing.assert_allclose(errors, archive["fem_l2"], rtol=2e-6, atol=1e-9)
        for key, value in summarize(errors).items():
            np.testing.assert_allclose(value, metrics["test_fem_l2"][key], rtol=2e-6, atol=1e-9)
        prediction = predictor.predict_dataset(dataset, [index])[0]
        np.testing.assert_allclose(prediction, saved[position], rtol=2e-5, atol=2e-6)
        parity = float(np.max(np.abs(prediction - saved[position])))
    sample_error = float(dataset.relative_l2(dataset.u_scat[index], prediction))
    files = {"checkpoint": checkpoint_path, "metrics": metrics_path,
             "saved_predictions": predictions_path}
    record = {
        "method": "fno", "run": relative(run),
        "files": {key: {"path": relative(path), "sha256": sha256(path)}
                  for key, path in files.items()},
        "dataset_fingerprint": metrics["dataset_fingerprint"],
        "config": metrics["config"], "completed_epochs": metrics["completed_epochs"],
        "training_versions": metrics.get("versions", {}), "verification_inference_device": "cpu",
        "best_epoch": metrics["best_epoch"],
        "test_complex_consistent_fem_l2": metrics["test_fem_l2"],
        "mean_test_error_percent": 100 * metrics["test_fem_l2"]["mean"],
        "sample_error_percent": 100 * sample_error,
        "sample_complex_consistent_fem_l2": sample_error,
        "checkpoint_vs_saved_prediction_max_abs_difference": parity,
        "checks": {
            "dataset_checkpoint_metrics_fingerprints_match": True,
            "checkpoint_metrics_config_and_experiment_metadata_match": True,
            "same_config_as_sno_except_method_and_device": True,
            "geometry_and_node_order_match": True,
            "all_1280_test_indices_match_saved_split": True,
            "all_1280_saved_errors_match_recomputed_consistent_fem_l2": True,
            "test_metric_summaries_match_saved_predictions": True,
            "checkpoint_prediction_matches_saved_prediction": True,
            "sample_index_selected_by_sno_only": True,
        },
    }
    return prediction, record


def verify_comparison(band):
    dataset, index, sno_prediction, sno_record = verify_band(band)
    if index != EXPECTED_INDICES[band]:
        raise ValueError(f"{band}: SNO median sample changed; review sample selection before publishing")
    fno_prediction, fno_record = verify_fno(band, dataset, index, sno_record)
    sno_record["method"] = "svd"
    sno_record["sample_error_percent"] = sno_record["sample"]["error_percent"]
    sno_record["sample_complex_consistent_fem_l2"] = sno_record["sample"]["complex_consistent_fem_l2"]
    sno_record["training_versions"] = json.loads(
        (REPOSITORY / sno_record["files"]["metrics"]["path"]).read_text()).get("versions", {})
    sno_record["verification_inference_device"] = "cpu"
    record = {
        "band": band, "dataset": sno_record["dataset"],
        "dataset_fingerprint": sno_record["dataset_fingerprint"],
        "samples": sno_record["samples"], "physical_media": sno_record["physical_media"],
        "nodes": sno_record["nodes"], "triangles": sno_record["triangles"],
        "frequency_groups": sno_record["frequency_groups"],
        "directions_per_frequency": sno_record["directions_per_frequency"],
        "direction_sampling": sno_record["direction_sampling"],
        "split_counts": sno_record["split_counts"], "split_strategy": sno_record["split_strategy"],
        "observed_k_range": sno_record["observed_k_range"], "sample": sno_record["sample"],
        "selection_depends_on": "SNO saved test error only; FNO does not influence sample selection",
        "configuration_differences": {
            key: {"svd": sno_record["config"][key], "fno": fno_record["config"][key]}
            for key in sno_record["config"]
            if sno_record["config"][key] != fno_record["config"][key]},
        "hardware_comparison": "Training devices may differ; no runtime or throughput ranking is made. Verification inference uses CPU for both checkpoints.",
        "models": {"svd": sno_record, "fno": fno_record},
    }
    return dataset, index, {"svd": sno_prediction, "fno": fno_prediction}, record


def draw_comparison(dataset, index, predictions, record, methods, name):
    truth = scattering_components(dataset.u_scat[index])
    components = {method: scattering_components(value) for method, value in predictions.items()}
    columns = 1 + 2 * len(methods)
    fig, axes = plt.subplots(3, columns, figsize=(13.8 if len(methods) == 2 else 8.3, 8.0),
                             layout="constrained")
    fig.get_layout_engine().set(h_pad=.045, w_pad=.045, hspace=.10, wspace=.03)
    tri = mtri.Triangulation(*dataset.coordinates.T, dataset.triangles)
    color_scales = {}
    for row, (key, label) in enumerate(COMPONENTS):
        limit = max(float(np.abs(truth[key]).max()),
                    *(float(np.abs(value[key]).max()) for value in components.values()), 1e-12)
        error_limit = max(*(float(np.abs(value[key] - truth[key]).max())
                            for value in components.values()), 1e-12)
        cmap, lower = ("viridis", 0) if key == "abs" else ("RdBu_r", -limit)
        color_scales[key] = {"field_min": lower, "field_max": limit,
                             "absolute_error_min": 0, "absolute_error_max": error_limit}
        panels = [(truth[key], "FEM", cmap, lower, limit)]
        for method in methods:
            panels += [(components[method][key], LABELS[method], cmap, lower, limit),
                       (np.abs(components[method][key] - truth[key]),
                        f"Ошибка {LABELS[method]}", "magma", 0, error_limit)]
        for col, (values, title, palette, lo, hi) in enumerate(panels):
            ax = axes[row, col]
            artist = ax.tripcolor(tri, values, shading="gouraud", cmap=palette,
                                  vmin=lo, vmax=hi, rasterized=True)
            ax.set_aspect("equal")
            ax.set(xlim=(-2, 2), ylim=(-2, 2), xticks=(-2, 0, 2), yticks=(-2, 0, 2))
            ax.set_title(title, fontsize=11, pad=5)
            ax.tick_params(labelsize=9, length=3, width=.6)
            ax.set_xlabel("$x$", fontsize=11, labelpad=1)
            ax.set_ylabel(label + "\n$y$" if col == 0 else "$y$", fontsize=11, labelpad=2)
            colorbar = fig.colorbar(artist, ax=ax, fraction=.046, pad=.025, aspect=24,
                                   format=FuncFormatter(lambda value, _: f"{value:.2g}"))
            colorbar.ax.tick_params(labelsize=9, width=.6, length=2)
            colorbar.outline.set_linewidth(.5)
            for spine in ax.spines.values():
                spine.set_linewidth(.6)
    sample = record["sample"]
    errors = "; ".join(f'{LABELS[method]} {record["models"][method]["sample_error_percent"]:.3f}%'
                       for method in methods)
    means = "; ".join(f'{LABELS[method]} {record["models"][method]["mean_test_error_percent"]:.3f}%'
                      for method in methods)
    fig.suptitle(f'Фиксированная среда, {record["band"]}: test #{index}, '
                 f'$k={sample["k0"]:.4f}$, $\\theta={sample["theta_radians"]:.4f}$\n'
                 f'Комплексная FEM $L^2$, пример: {errors}\n'
                 f'Средняя по 1280 test-полям: {means}', fontsize=12)
    outputs = []
    for extension in ("pdf", "png"):
        path = IMAGES / f"{name}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        outputs.append({"path": relative(path), "sha256": sha256(path)})
    plt.close(fig)
    return {"files": outputs, "methods": list(methods), "color_scales": color_scales}


def summary_rows(provenance):
    rows = []
    for band in provenance["bands"]:
        record = provenance["bands"][band]
        for method in ("svd", "fno"):
            model = record["models"][method]
            row = {"band": band, "method": method, "dataset_fingerprint": record["dataset_fingerprint"],
                   "samples": record["samples"], "test_samples": 1280,
                   "sample_index": record["sample"]["index"],
                   "sample_error_percent": model["sample_error_percent"],
                   "training_device": model["config"]["device"],
                   "training_torch_version": model["training_versions"].get("torch", "unknown"),
                   "training_numpy_version": model["training_versions"].get("numpy", "unknown"),
                   "completed_epochs": model["completed_epochs"], "best_epoch": model["best_epoch"],
                   "checkpoint": model["files"]["checkpoint"]["path"]}
            row.update({f"test_{key}_error_percent": 100 * value
                        for key, value in model["test_complex_consistent_fem_l2"].items()})
            rows.append(row)
    return rows


def write_manifest(provenance):
    lines = ["# Статические SVD-SNO и FNO: проверенное сравнение", "",
             "Текущие PyTorch-модели используют одну фиксированную среду в каждой полосе: "
             "8000 полей = 1000 частот x 8 направлений на частоту. Восемь различных направлений "
             "выбираются отдельно для каждой частоты, общей угловой сетки нет. "
             "Train/validation/test: 5360/1360/1280 полей, 670/170/160 непересекающихся частот. "
             "Проверены одинаковые dataset fingerprints, mesh и порядок узлов обеих моделей.", "",
             "Пример выбирается исключительно по SVD-SNO: ближайшая к медиане ошибка среди всех "
             "1280 сохранённых test-полей, первая позиция при равенстве. FNO не влияет на выбор; "
             "выбор не минимизирует ошибку одной из архитектур. Этот выбор типичен по ошибке SNO, "
             "но не гарантирует медианную ошибку FNO. Средние ошибки используют все test-поля.", ""]
    for band in provenance["bands"]:
        r = provenance["bands"][band]
        s = r["sample"]
        lines += [f"## {band}", "", f"- Датасет: `{r['dataset']}`.",
                  f"- Fingerprint: `{r['dataset_fingerprint']}`.",
                  f"- Общий test index: {s['index']}; k={s['k0']:.12f}; theta={s['theta_radians']:.12f} рад."]
        for method in ("svd", "fno"):
            m = r["models"][method]
            lines += [f"- {LABELS[method]}: пример **{m['sample_error_percent']:.6f}%**; "
                      f"средняя test **{m['mean_test_error_percent']:.6f}%**; "
                      f"эпох {m['completed_epochs']}, лучшая {m['best_epoch']}; "
                      f"training device `{m['config']['device']}`, versions `{m['training_versions']}`; "
                      f"checkpoint `{m['files']['checkpoint']['path']}`."]
        lines += [f"- `prediction_static_{band}_comparison_components.pdf/.png`: обзор 3x5 "
                  "(FEM, SVD-SNO, ошибка SVD-SNO, FNO, ошибка FNO).",
                  f"- `prediction_static_{band}_comparison_sno_components.pdf/.png` и "
                  f"`prediction_static_{band}_comparison_fno_components.pdf/.png`: "
                  "отдельные 3x3 рисунки для статьи с теми же шкалами.", ""]
    lines += ["Строки: Re, Im, Abs рассеянного поля. Модуль вычисляется из реконструированных Re/Im. "
              "Ошибки компонент: abs(Re(pred)-Re(FEM)), abs(Im(pred)-Im(FEM)), "
              "abs(abs(pred)-abs(FEM)); последняя не является модулем комплексной ошибки. "
              "Цветовые шкалы полей общие для FEM и обеих моделей в каждой строке; "
              "шкалы абсолютных ошибок общие для обеих моделей. Диапазоны полные, без отсечения. "
              "Рисунки используют исходные FEM-треугольники.", "",
              "Обе модели проверены по fingerprint, configuration и metadata checkpoint, split, "
              "mesh, всем 1280 сохранённым test indices, пересчитанным consistent complex FEM L2 "
              "ошибкам и mean/median/p95/max; новый инференс показанного примера сверен с архивом. "
              "Configuration совпадает между моделями за исключением method и, возможно, device. "
              "Точные config и версии обучения сохранены отдельно для каждой модели; "
              "устройства обучения могут отличаться (CPU/MPS), поэтому сравнение времени и "
              "производительности не проводится. Проверочный инференс обоих checkpoint выполняется на CPU. "
              "Сравнение относится "
              "к этим обученным моделям и настройкам, а не устанавливает универсальный рейтинг архитектур.", "",
              "Подробные SHA256, конфигурации, проверки и численные шкалы: "
              "`prediction_static_comparison_provenance.json`. Метрики всех четырёх моделей: "
              "`prediction_static_comparison_summary.json` и `.csv` (ошибки в процентах).", "",
              "```sh", "docker compose exec -T dolfinx-complex python "
              "paper/svd_sno_revision_2026-11-06/prepare_static_comparison.py --band "
              + (next(iter(provenance["bands"])) if len(provenance["bands"]) == 1 else "all"), "```", "",
              "Скрипт не обучает модели и не выполняет FEM solves."]
    (IMAGES / "prediction_static_comparison_manifest.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", choices=("low", "high", "all"), default="all")
    args = parser.parse_args()
    bands = BANDS if args.band == "all" else (args.band,)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 300, "axes.unicode_minus": False})
    # Complete all numerical checks before authoring any deliverable.
    verified = {band: verify_comparison(band) for band in bands}
    provenance = {
        "experiment": "Current PyTorch fixed-medium randomized SVD-SNO versus FNO",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_epochs_performed_by_this_script": 0, "fem_solves_performed_by_this_script": 0,
        "generator": {"path": relative(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "versions": {"numpy": np.__version__, "torch": str(torch.__version__),
                     "matplotlib": matplotlib.__version__}, "bands": {},
    }
    IMAGES.mkdir(parents=True, exist_ok=True)
    for band, (dataset, index, predictions, record) in verified.items():
        outputs = {}
        for suffix, methods in (("components", ("svd", "fno")),
                                ("sno_components", ("svd",)), ("fno_components", ("fno",))):
            name = f"prediction_static_{band}_comparison_{suffix}"
            outputs[suffix] = draw_comparison(dataset, index, predictions, record, methods, name)
        record["outputs"] = outputs
        provenance["bands"][band] = record
    rows = summary_rows(provenance)
    (IMAGES / "prediction_static_comparison_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    (IMAGES / "prediction_static_comparison_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (IMAGES / "prediction_static_comparison_summary.csv").write_text(stream.getvalue())
    write_manifest(provenance)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
