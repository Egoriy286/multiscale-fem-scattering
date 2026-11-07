"""Render current fixed-medium SVD-SNO results from saved artifacts only.

From the repository root::

    docker compose exec -T dolfinx-complex python \
        paper/svd_sno_revision_2026-11-06/prepare_static_predictions.py

No training or FEM solves. Each band uses the saved test sample nearest to the
median complex FEM L2 error (first occurrence on ties), not the best sample.
These PyTorch models are separate from both the
historical JAX experiment and the archived variable-medium pilot figures.
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
EXPECTED_MEANS = {"low": .009613946273391481, "high": .05537525831383929}
EXPECTED_SPLITS = ((5360, 670), (1360, 170), (1280, 160))
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


def verify_band(band):
    directory = REPOSITORY / f"neural_operator/data/static_{band}_frequency_1000x8"
    run = REPOSITORY / f"neural_operator/randomized/runs/{band}_frequency_static_1000x8/svd"
    dataset = load_dataset(directory, mmap=True)
    fingerprint = dataset_fingerprint(dataset)
    metrics = json.loads((run / "metrics.json").read_text())
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    if (dataset.split_strategy != "frequency" or len(np.unique(dataset.medium_id)) != 1
            or len(dataset.u_scat) != 8000 or metrics["method"] != "svd"
            or metrics["dataset_fingerprint"] != fingerprint
            or checkpoint["metadata"]["dataset_fingerprint"] != fingerprint
            or checkpoint["metadata"]["config"] != metrics["config"]):
        raise ValueError(f"{band}: static data, metrics or checkpoint mismatch")
    # Verify the full frequency grouping, rather than inferring 1000 x 8 from
    # the sample count. Directions are sampled separately at each frequency.
    frequencies, counts = np.unique(dataset.frequency_id, return_counts=True)
    if (len(frequencies) != 1000 or len(np.unique(dataset.k0)) != 1000
            or np.any(counts != 8)
            or any(len(np.unique(dataset.theta[dataset.frequency_id == group])) != 8
                   for group in frequencies)):
        raise ValueError(f"{band}: expected 1000 distinct frequencies with 8 distinct directions each")
    splits = dataset.split_indices()
    for indices, (samples, groups) in zip(splits, EXPECTED_SPLITS):
        if (len(indices) != samples
                or len(np.unique(dataset.frequency_id[indices])) != groups):
            raise ValueError(f"{band}: frequency split counts differ from the documented experiment")
    for key in ("method", "field", "metric", "split_strategy", "physical_media",
                "train_samples", "val_samples", "test_samples", "split_groups",
                "completed_epochs", "best_epoch", "ranks", "test_fem_l2"):
        if checkpoint["metadata"][key] != metrics[key]:
            raise ValueError(f"{band}: checkpoint and metrics disagree on {key}")
    if (checkpoint["method"] != "svd" or metrics["field"] != "scattered"
            or metrics["metric"] != "consistent_complex_FEM_L2"
            or metrics["config"]["pod_method"] != "randomized"):
        raise ValueError(f"{band}: expected randomized SVD-SNO scattered-field FEM L2 results")
    np.testing.assert_allclose(metrics["test_fem_l2"]["mean"], EXPECTED_MEANS[band], rtol=1e-12)
    predictor = load_predictor(run / "model.pt")
    if (not np.array_equal(predictor.coordinates, dataset.coordinates)
            or not np.array_equal(predictor.triangles, dataset.triangles)
            or predictor.bases["A"].rank != 0 or predictor.bases["n"].rank != 0):
        raise ValueError(f"{band}: mesh, node order or static input ranks differ")
    test = splits[2]
    if len(test) != 1280:
        raise ValueError(f"{band}: expected 1280 held-out test fields")
    with np.load(run / "test_predictions.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["indices"], test):
            raise ValueError(f"{band}: saved test indices do not match current split")
        saved = archive["prediction"]
        errors = dataset.relative_l2(dataset.u_scat[test], saved)
        np.testing.assert_allclose(errors, archive["fem_l2"], rtol=2e-6, atol=1e-9)
        for key, value in summarize(errors).items():
            np.testing.assert_allclose(value, metrics["test_fem_l2"][key], rtol=2e-6, atol=1e-9)
        # Selection uses the archived errors exactly; argmin takes the first tie.
        saved_errors = archive["fem_l2"]
        position = int(np.argmin(np.abs(saved_errors - np.median(saved_errors))))
        index = int(test[position])
        prediction = predictor.predict_dataset(dataset, [index])[0]
        np.testing.assert_allclose(prediction, saved[position], rtol=2e-5, atol=2e-6)
        parity = float(np.max(np.abs(prediction - saved[position])))
    sample_error = float(dataset.relative_l2(dataset.u_scat[index], prediction))
    files = {
        "checkpoint": run / "model.pt", "metrics": run / "metrics.json",
        "saved_predictions": run / "test_predictions.npz",
        "dataset_archive": directory / "fem_dataset.npz",
        "geometry_archive": directory / "geometry.npz",
    }
    record = {
        "band": band,
        "dataset": relative(directory),
        "run": relative(run),
        "files": {name: {"path": relative(path), "sha256": sha256(path)}
                  for name, path in files.items()},
        "dataset_fingerprint": fingerprint,
        "samples": len(dataset.u_scat), "physical_media": len(np.unique(dataset.medium_id)),
        "frequency_groups": len(frequencies),
        "directions_per_frequency": 8,
        "direction_sampling": "Eight distinct directions sampled separately for each frequency; not one shared angle grid",
        "nodes": len(dataset.coordinates), "triangles": len(dataset.triangles),
        "split_strategy": dataset.split_strategy,
        "split_counts": {
            name: {"samples": len(indices), "frequency_groups": len(np.unique(dataset.frequency_id[indices]))}
            for name, indices in zip(("train", "validation", "test"), splits)
        },
        "observed_k_range": [float(dataset.k0.min()), float(dataset.k0.max())],
        "sample": {"selection": "Nearest to the median saved complex FEM L2 error across all 1280 test fields; first occurrence on ties",
                   "selection_expression": "np.argmin(np.abs(saved_errors - np.median(saved_errors)))",
                   "position_in_test": position,
                   "index": index, "medium_id": int(dataset.medium_id[index]),
                   "frequency_id": int(dataset.frequency_id[index]),
                   "k0": float(dataset.k0[index]), "theta_radians": float(dataset.theta[index]),
                   "complex_consistent_fem_l2": sample_error, "error_percent": 100 * sample_error},
        "test_complex_consistent_fem_l2": metrics["test_fem_l2"],
        "mean_test_error_percent": 100 * metrics["test_fem_l2"]["mean"],
        "config": metrics["config"], "ranks": metrics["ranks"],
        "completed_epochs": metrics["completed_epochs"], "best_epoch": metrics["best_epoch"],
        "checkpoint_vs_saved_prediction_max_abs_difference": parity,
        "checks": {
            "dataset_checkpoint_metrics_fingerprints_match": True,
            "checkpoint_metrics_configs_match": True,
            "checkpoint_metrics_experiment_metadata_match": True,
            "frequency_groups_1000_with_8_distinct_directions_each": True,
            "frequency_split_counts_5360_1360_1280": True,
            "frequency_groups_disjoint_between_splits": True,
            "geometry_and_node_order_match": True,
            "physical_media_one_input_ranks_zero": True,
            "all_1280_test_indices_match_saved_split": True,
            "all_1280_saved_errors_match_recomputed_consistent_fem_l2": True,
            "test_metric_summaries_match_saved_predictions": True,
            "checkpoint_prediction_matches_saved_prediction": True,
        },
    }
    return dataset, index, prediction, record


def audit_fno_checkpoints(static_fingerprints):
    """Record why old local FNO results cannot serve as static comparisons."""
    candidates = []
    for path in sorted((REPOSITORY / "neural_operator").rglob("model.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("method") != "fno":
            continue
        fingerprint = checkpoint.get("metadata", {}).get("dataset_fingerprint")
        candidates.append({"path": relative(path), "sha256": sha256(path),
                           "dataset_fingerprint": fingerprint,
                           "matches_current_static_dataset": fingerprint in static_fingerprints})
    return {"scope": "All model.pt checkpoints below neural_operator/",
            "matching_static_checkpoint_available": any(
                item["matches_current_static_dataset"] for item in candidates),
            "candidates": candidates}


def draw_band(dataset, index, prediction, record):
    reference, predicted = scattering_components(dataset.u_scat[index]), scattering_components(prediction)
    tri = mtri.Triangulation(*dataset.coordinates.T, dataset.triangles)
    fig, axes = plt.subplots(3, 3, figsize=(8.3, 8.0), layout="constrained")
    fig.get_layout_engine().set(h_pad=.045, w_pad=.045, hspace=.10, wspace=.03)
    for row, (key, label) in enumerate(COMPONENTS):
        limit = max(float(np.abs(reference[key]).max()), float(np.abs(predicted[key]).max()), 1e-12)
        error = np.abs(predicted[key] - reference[key])
        error_limit = max(float(error.max()), 1e-12)
        cmap, lower = ("viridis", 0) if key == "abs" else ("RdBu_r", -limit)
        panels = ((reference[key], "FEM", cmap, lower, limit),
                  (predicted[key], "SVD-SNO", cmap, lower, limit),
                  (error, "Абс. ошибка", "magma", 0, error_limit))
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
    fig.suptitle(f'Фиксированная среда, {record["band"]}: пример около медианы test\n'
                 f'#{index}, $k={sample["k0"]:.4f}$, $\\theta={sample["theta_radians"]:.4f}$\n'
                 f'Комплексная FEM $L^2$: пример {sample["error_percent"]:.3f}%; '
                 f'средняя test {record["mean_test_error_percent"]:.3f}%', fontsize=12)
    name = f'prediction_static_{record["band"]}_components'
    for extension in ("pdf", "png"):
        fig.savefig(IMAGES / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def write_manifest(provenance):
    fno_note = ("Аудит локальных checkpoint обнаружил FNO для текущих статических данных; "
                "данные рисунки показывают только SVD-SNO."
                if provenance["fno_checkpoint_audit"]["matching_static_checkpoint_available"] else
                "На момент генерации подходящего обученного FNO checkpoint для этих статических данных в проекте нет.")
    lines = ["# Текущие статические SVD-SNO: происхождение рисунков", "",
             "Рисунки используют **одну фиксированную среду**, по отдельному randomized SVD-SNO checkpoint на каждую полосу. Это текущие PyTorch-результаты с разбиением по частотам; они отличаются от исторического JAX-эксперимента статьи и от архивного пилота с 48 средами. " + fno_note, "",
             "В каждой полосе: 8000 полей = 1000 частот x 8 направлений на частоту; направления выбираются отдельно для каждой частоты, общей сетки из 8 углов нет. Train/validation/test = 5360/1360/1280 полей (670/170/160 непересекающихся частот). Physical media = 1; входные POD-ранги A,n = 0. Выбран пример, ближайший к медиане сохранённой complex FEM L2 ошибки по всем 1280 test-полям: `np.argmin(abs(saved_errors - median(saved_errors)))`, при равенстве берётся первая позиция. Это типичный по величине ошибки пример, не лучший. Средняя test-ошибка и ошибка показанного примера подписаны раздельно.", ""]
    for band, record in provenance["bands"].items():
        sample = record["sample"]
        lines += [f"## {band}", "",
                  f"- Датасет: `{record['dataset']}`.",
                  f"- Checkpoint: `{record['files']['checkpoint']['path']}`.",
                  f"- Fingerprint: `{record['dataset_fingerprint']}`.",
                  f"- Пример: test index {sample['index']}, frequency_id {sample['frequency_id']}, medium_id {sample['medium_id']}, k={sample['k0']:.12f}, theta={sample['theta_radians']:.12f} рад.",
                  f"- Ошибка примера: **{sample['error_percent']:.6f}%**; средняя по всем 1280 test-полям: **{record['mean_test_error_percent']:.6f}%**, complex consistent FEM L2.",
                  f"- Выполнено эпох: {record['completed_epochs']}; лучшая: {record['best_epoch']}; ранги: `{record['ranks']}`.",
                  f"- `prediction_static_{band}_components.pdf/.png`: строки Re/Im/Abs, столбцы FEM / SVD-SNO / абсолютная ошибка соответствующей компоненты.", ""]
    lines += ["Модуль вычисляется из реконструированных Re и Im. Ошибка Abs - `abs(abs(prediction)-abs(FEM))`, а не модуль комплексной ошибки. Для FEM и модели используется общая полная цветовая шкала в каждой строке, без отсечения экстремумов. Поле отображается на исходных FEM-треугольниках.", "",
              "Проверены fingerprint модели и данных, config и metadata, геометрия и порядок узлов, 1000 частот с 8 направлениями, непересекающиеся frequency splits и их размеры, все 1280 test indices, ошибки и их summaries, совпадение нового инференса с сохранённым полем. Все локальные model.pt проверены на наличие подходящего FNO по dataset fingerprint. SHA256 файлов, sample/mean errors, конфигурации, проверки и аудит FNO: `prediction_static_provenance.json`.", "",
              "Воспроизведение из корня репозитория:", "", "```sh",
              "docker compose exec -T dolfinx-complex python paper/svd_sno_revision_2026-11-06/prepare_static_predictions.py",
              "```", "", "Скрипт не выполняет обучение или FEM solves."]
    (IMAGES / "prediction_static_manifest.md").write_text("\n".join(lines) + "\n")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 300, "axes.unicode_minus": False})
    IMAGES.mkdir(parents=True, exist_ok=True)
    provenance = {
        "experiment": "Current PyTorch randomized SVD-SNO on one fixed physical medium",
        "split": "Held-out frequencies; not new media",
        "training_epochs_performed_by_this_script": 0,
        "fem_solves_performed_by_this_script": 0,
        "bands": {},
        "versions": {"numpy": np.__version__, "torch": str(torch.__version__),
                     "matplotlib": matplotlib.__version__},
        "generator": {"path": relative(Path(__file__).resolve()),
                      "sha256": sha256(Path(__file__).resolve())},
    }
    for band in ("low", "high"):
        dataset, index, prediction, record = verify_band(band)
        draw_band(dataset, index, prediction, record)
        provenance["bands"][band] = record
        print(json.dumps({"band": band, "sample": record["sample"],
                          "mean_test_error_percent": record["mean_test_error_percent"],
                          "ranks": record["ranks"], "completed_epochs": record["completed_epochs"],
                          "best_epoch": record["best_epoch"]}, ensure_ascii=False), flush=True)
    provenance["fno_checkpoint_audit"] = audit_fno_checkpoints(
        {record["dataset_fingerprint"] for record in provenance["bands"].values()})
    provenance["fno_static_checkpoint_available"] = provenance["fno_checkpoint_audit"]["matching_static_checkpoint_available"]
    (IMAGES / "prediction_static_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    write_manifest(provenance)


if __name__ == "__main__":
    main()
