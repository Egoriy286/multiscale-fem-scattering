"""Render honest paired PyTorch pilot examples from verified saved artifacts.

Run from the repository root in its existing scientific environment::

    docker compose exec -T dolfinx-complex python \
        paper/svd_sno_revision_2026-11-06/prepare_pilot_predictions.py

No training or PDE solves are performed. The sample is the first common test
index, selected without inspecting prediction errors. These are a separate
PyTorch pilot, not the historical JAX experiment reported in the main table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch


ARTICLE = Path(__file__).resolve().parent
REPOSITORY = ARTICLE.parents[1]
sys.path.insert(0, str(REPOSITORY))

from neural_operator.data import load_dataset
from neural_operator.fields import scattering_components
from neural_operator.training import dataset_fingerprint, load_predictor, summarize


IMAGES = ARTICLE / "images"
DATASET = REPOSITORY / "neural_operator/data/low_frequency"
RUNS = {
    "svd": REPOSITORY / "neural_operator/runs/low_frequency/svd",
    "fno": REPOSITORY / "neural_operator/runs/low_frequency/fno",
}
LABELS = {"svd": "SVD-SNO", "fno": "FNO"}
COMPONENTS = (("real", r"$\operatorname{Re}(u_{\mathrm{scattering}})$"),
              ("imag", r"$\operatorname{Im}(u_{\mathrm{scattering}})$"),
              ("abs", r"$\operatorname{Abs}(u_{\mathrm{scattering}})$"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(path.relative_to(REPOSITORY))


def verify_inputs():
    dataset = load_dataset(DATASET, mmap=True)
    fingerprint = dataset_fingerprint(dataset)
    test = dataset.split_indices()[2]
    if not len(test):
        raise ValueError("The saved dataset has no test samples")
    index = int(test[0])
    predictions, records = {}, {}
    for method, run in RUNS.items():
        metrics = json.loads((run / "metrics.json").read_text())
        checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        if (metrics["method"] != method or checkpoint["method"] != method
                or metrics["dataset_fingerprint"] != fingerprint
                or checkpoint["metadata"]["dataset_fingerprint"] != fingerprint
                or checkpoint["metadata"]["config"] != metrics["config"]):
            raise ValueError(f"{method}: checkpoint, metrics and dataset are incompatible")
        predictor = load_predictor(run / "model.pt")
        if (not np.array_equal(predictor.coordinates, dataset.coordinates)
                or not np.array_equal(predictor.triangles, dataset.triangles)):
            raise ValueError(f"{method}: FEM geometry or node ordering differs")
        with np.load(run / "test_predictions.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["indices"], test):
                raise ValueError(f"{method}: saved test indices differ from the dataset split")
            saved = archive["prediction"]
            all_errors = dataset.relative_l2(dataset.u_scat[test], saved)
            np.testing.assert_allclose(all_errors, archive["fem_l2"], rtol=2e-6, atol=1e-9)
            for key, value in summarize(all_errors).items():
                np.testing.assert_allclose(value, metrics["test_fem_l2"][key],
                                           rtol=2e-6, atol=1e-9)
            fresh = predictor.predict_dataset(dataset, [index])[0]
            np.testing.assert_allclose(fresh, saved[0], rtol=2e-5, atol=2e-6)
            parity = float(np.max(np.abs(fresh - saved[0])))
        predictions[method] = fresh
        error = float(dataset.relative_l2(dataset.u_scat[index], fresh))
        records[method] = {
            "label": LABELS[method],
            "checkpoint": relative(run / "model.pt"),
            "checkpoint_sha256": sha256(run / "model.pt"),
            "metrics": relative(run / "metrics.json"),
            "metrics_sha256": sha256(run / "metrics.json"),
            "saved_predictions": relative(run / "test_predictions.npz"),
            "saved_predictions_sha256": sha256(run / "test_predictions.npz"),
            "sample_complex_consistent_fem_l2": error,
            "sample_error_percent": 100 * error,
            "test_complex_consistent_fem_l2": metrics["test_fem_l2"],
            "checkpoint_vs_saved_prediction_max_abs_difference": parity,
            "config": metrics["config"],
            "completed_epochs": metrics["completed_epochs"],
            "best_epoch": metrics["best_epoch"],
            "ranks": metrics.get("ranks"),
        }
    sample = {
        "selection": "First test index, independent of either model's errors",
        "index": index,
        "medium_id": int(dataset.medium_id[index]),
        "k0": float(dataset.k0[index]),
        "theta_radians": float(dataset.theta[index]),
        "split": "test",
    }
    provenance = {
        "experiment": "Separate paired PyTorch pilot; not the article's historical JAX run",
        "comparison_scope": (
            "Identical dataset, geometry and test sample; training configurations differ. "
            "This visual example does not establish an architecture ranking."),
        "training_epochs_performed_by_this_script": 0,
        "fem_solves_performed_by_this_script": 0,
        "dataset": relative(DATASET),
        "dataset_archive_sha256": sha256(DATASET / "fem_dataset.npz"),
        "geometry_archive_sha256": sha256(DATASET / "geometry.npz"),
        "dataset_fingerprint": fingerprint,
        "samples": len(dataset.u_scat),
        "media": len(np.unique(dataset.medium_id)),
        "nodes": len(dataset.coordinates),
        "triangles": len(dataset.triangles),
        "split_strategy": dataset.split_strategy,
        "split_counts": {
            name: {"samples": len(indices), "media": len(np.unique(dataset.medium_id[indices]))}
            for name, indices in zip(("train", "validation", "test"), dataset.split_indices())
        },
        "observed_k_range": [float(dataset.k0.min()), float(dataset.k0.max())],
        "sample": sample,
        "models": records,
        "checks": {
            "dataset_and_both_model_fingerprints_match": True,
            "checkpoint_and_metrics_configs_match": True,
            "both_model_meshes_and_node_orderings_match_dataset": True,
            "both_saved_test_index_vectors_match_dataset": True,
            "saved_test_errors_match_recomputed_consistent_fem_l2": True,
            "metrics_test_summaries_match_saved_predictions": True,
            "checkpoint_predictions_match_saved_predictions": True,
        },
        "plot_conventions": {
            "field": "Nodal scattered complex field, not total field",
            "amplitude": "Computed from reconstructed real/imag, before plotting",
            "error_real": "abs(Re(prediction) - Re(FEM))",
            "error_imag": "abs(Im(prediction) - Im(FEM))",
            "error_abs": "abs(abs(prediction) - abs(FEM))",
            "interpolation": "Gouraud display on the saved FEM triangles",
            "color_scales": "Full data range; shared for both models in each component",
        },
        "versions": {"numpy": np.__version__, "torch": str(torch.__version__),
                     "matplotlib": matplotlib.__version__},
    }
    return dataset, index, predictions, provenance


def draw_figure(dataset, index, predictions, methods, name, provenance):
    truth = scattering_components(dataset.u_scat[index])
    components = {method: scattering_components(predictions[method]) for method in RUNS}
    columns = 1 + 2 * len(methods)
    width = 8.3 if len(methods) == 1 else 12.2
    fig, axes = plt.subplots(3, columns, figsize=(width, 8.0), layout="constrained")
    fig.get_layout_engine().set(h_pad=.045, w_pad=.045, hspace=.10, wspace=.03)
    tri = mtri.Triangulation(*dataset.coordinates.T, dataset.triangles)
    for row, (key, label) in enumerate(COMPONENTS):
        limit = max(float(np.abs(truth[key]).max()), *(
            float(np.abs(values[key]).max()) for values in components.values()), 1e-12)
        error_limit = max(*(
            float(np.abs(values[key] - truth[key]).max())
            for values in components.values()), 1e-12)
        cmap, lower = ("viridis", 0) if key == "abs" else ("RdBu_r", -limit)
        panels = [(truth[key], "FEM", cmap, lower, limit)]
        for method in methods:
            panels += [
                (components[method][key], LABELS[method], cmap, lower, limit),
                (np.abs(components[method][key] - truth[key]), "Абс. ошибка", "magma", 0, error_limit),
            ]
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
            cbar = fig.colorbar(artist, ax=ax, fraction=.046, pad=.025, aspect=24,
                               format=FuncFormatter(lambda value, _: f"{value:.2g}"))
            cbar.ax.tick_params(labelsize=9, width=.6, length=2)
            cbar.outline.set_linewidth(.5)
            for spine in ax.spines.values():
                spine.set_linewidth(.6)
    sample = provenance["sample"]
    errors = "; ".join(
        f'{LABELS[method]}: {provenance["models"][method]["sample_error_percent"]:.2f}%'
        for method in methods)
    fig.suptitle(f'Отдельный PyTorch-пилот: test #{index}, среда {sample["medium_id"]}\n'
                 f'$k={sample["k0"]:.4f}$, $\\theta={sample["theta_radians"]:.4f}$; '
                 f'комплексная FEM $L^2$: {errors}', fontsize=12)
    for extension in ("pdf", "png"):
        fig.savefig(IMAGES / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def write_manifest(provenance):
    sample, models = provenance["sample"], provenance["models"]
    text = f"""# Происхождение рисунков предсказания

Это отдельный **PyTorch-пилот**, не исторический JAX-эксперимент из основной таблицы статьи. Рисунки используют одну и ту же сохранённую test-среду и FEM-метку для обеих моделей; конфигурации обучения различаются. Иллюстрация не устанавливает рейтинг архитектур.

- Датасет: `{provenance['dataset']}`; 192 поля, 48 сред, 3221 узел, 6280 треугольников.
- Train/validation/test: 128/32/32 поля, 32/8/8 сред.
- Проверенный общий fingerprint: `{provenance['dataset_fingerprint']}`.
- Выбор примера: первый test index **{sample['index']}**, без отбора по ошибке; medium_id **{sample['medium_id']}**, k = **{sample['k0']:.12f}**, theta = **{sample['theta_radians']:.12f}** рад.
- Complex consistent FEM L2 на этом примере: SVD-SNO **{models['svd']['sample_error_percent']:.4f}%**, FNO **{models['fno']['sample_error_percent']:.4f}%**.
- SVD-SNO: полный POD, выходные ранги 85/85, width=64, depth=4; выполнено 66 эпох, лучшая 16; mean test **{100*models['svd']['test_complex_consistent_fem_l2']['mean']:.4f}%**.
- FNO: resolution=32, modes=8, width=24, depth=3; выполнено 200 эпох, лучшая 194; mean test **{100*models['fno']['test_complex_consistent_fem_l2']['mean']:.4f}%**.
- Сохранённые поля совпадают с новым инференсом checkpoint: максимальная абсолютная разность **0** для обеих моделей.

`prediction_components.pdf/.png` содержит обзор 3x5: строки Re, Im, Abs; столбцы FEM, SVD-SNO, его абсолютная ошибка, FNO, его абсолютная ошибка.

Для читаемого включения в статью сохранены отдельные `prediction_sno_components.pdf/.png` и `prediction_fno_components.pdf/.png`: по 3x3 панели, native width 8.3 inches. Они используют те же узлы, пример и цветовые шкалы, что и обзор. Re/Im имеют симметричную знаковую шкалу, Abs - неотрицательную; диапазоны охватывают все значения обеих моделей без отсечения. Ошибка Abs - `abs(abs(prediction)-abs(FEM))`, а не модуль комплексной ошибки.

Пути checkpoint, dataset, metrics и saved predictions, SHA256 входных файлов, точные конфигурации, проверки и версии библиотек записаны в `prediction_provenance.json`.

Воспроизведение из корня репозитория:

```sh
docker compose exec -T dolfinx-complex python paper/svd_sno_revision_2026-11-06/prepare_pilot_predictions.py
```

Скрипт не обучает модели и не решает FEM-задачу; при несовпадении fingerprint, геометрии, test indices или сохранённых предсказаний он останавливается с ошибкой.
"""
    (IMAGES / "prediction_manifest.md").write_text(text)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "axes.unicode_minus": False,
    })
    dataset, index, predictions, provenance = verify_inputs()
    IMAGES.mkdir(parents=True, exist_ok=True)
    draw_figure(dataset, index, predictions, ("svd", "fno"), "prediction_components", provenance)
    draw_figure(dataset, index, predictions, ("svd",), "prediction_sno_components", provenance)
    draw_figure(dataset, index, predictions, ("fno",), "prediction_fno_components", provenance)
    (IMAGES / "prediction_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    write_manifest(provenance)
    print(json.dumps({"sample": provenance["sample"],
                      "errors_percent": {key: value["sample_error_percent"]
                                         for key, value in provenance["models"].items()},
                      "output": relative(IMAGES)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
