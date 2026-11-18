"""Regenerate the manual's reference-mesh illustrations from the saved low-band run."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from experiments.reference_sno_gmsfem.dataset import (
    ANCHOR_FIELDS,
    REFERENCE_MESH,
    REFERENCE_SHA256,
    load_dataset,
)
from neural_operator.training import load_predictor
from svd_sno_manual.paraview_colormap import plot_nodal_field

OUT = HERE / "assets"
EXPERIMENT = ROOT / "experiments/reference_sno_gmsfem"
BAND = "low"
RUN = EXPERIMENT / "runs" / BAND
DATA = EXPERIMENT / "data" / BAND
CHECKPOINT = RUN / "model.pt"
PREDICTIONS = RUN / "test_predictions.npz"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": "#17324d",
    "text.color": "#17324d",
    "axes.titleweight": "normal",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_geometry(dataset, predictor):
    """Check the source MSH and compare the native FEM array order exactly."""
    mesh_hash = sha256(REFERENCE_MESH)
    if mesh_hash != REFERENCE_SHA256:
        raise ValueError("Authoritative GMsFEM reference mesh checksum differs")
    np.testing.assert_array_equal(predictor.coordinates, dataset.coordinates)
    np.testing.assert_array_equal(predictor.triangles, dataset.triangles)
    with np.load(ANCHOR_FIELDS, allow_pickle=False) as anchor:
        np.testing.assert_array_equal(dataset.coordinates, anchor["reference_coordinates"])
        np.testing.assert_array_equal(dataset.triangles, anchor["reference_triangles"])
    return mesh_hash


def read_prediction_row(position, array_shape):
    """Read one complex field from the compressed NPY member in bounded memory."""
    with ZipFile(PREDICTIONS) as archive, archive.open("prediction.npy") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        if version not in readers:
            raise ValueError(f"Unsupported prediction NPY header version {version}")
        shape, fortran_order, dtype = readers[version](stream)
        if shape != array_shape or fortran_order or dtype.kind != "c":
            raise ValueError("Expected C-order complex predictions on the reference mesh")
        row_bytes = shape[1] * dtype.itemsize
        stream.seek(position * row_bytes, 1)
        return np.frombuffer(stream.read(row_bytes), dtype=dtype, count=shape[1])


def save_figure(fig, name, *, png=True):
    fig.savefig(OUT / f"{name}.pdf", dpi=300)
    if png:
        fig.savefig(OUT / f"{name}.png", dpi=180)
    plt.close(fig)


def draw_cover(triangulation, mean, correction, prediction):
    fields = (mean, correction, prediction)
    limit = max(float(np.max(np.abs(values))) for values in fields)
    scale = Normalize(vmin=-limit, vmax=limit)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.85), layout="constrained")
    titles = ("Среднее поле", "Модальная поправка", "Прогноз")
    for ax, values, title in zip(axes, fields, titles):
        image = plot_nodal_field(ax, triangulation, values, scale)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, pad=5)
    bar = fig.colorbar(image, ax=axes, orientation="horizontal", shrink=.62,
                       aspect=42, pad=.09)
    bar.locator = MaxNLocator(nbins=5, symmetric=True)
    bar.update_ticks()
    bar.outline.set_visible(False)
    save_figure(fig, "cover_reconstruction", png=False)


def draw_fields(triangulation, true, prediction):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.1), layout="constrained")
    components = (("Re", np.real), ("Im", np.imag), ("Модуль", np.abs))
    for row, (label, component) in enumerate(components):
        reference, predicted = component(true), component(prediction)
        limit = max(np.max(np.abs(reference)), np.max(np.abs(predicted)))
        sequential = component is np.abs
        norm = Normalize(vmin=0 if sequential else -limit, vmax=limit)
        error = np.abs(predicted - reference)
        error_norm = Normalize(vmin=0, vmax=error.max())
        for col, (values, scale) in enumerate((
            (reference, norm), (predicted, norm), (error, error_norm),
        )):
            ax = axes[row, col]
            image = plot_nodal_field(ax, triangulation, values, scale)
            fig.colorbar(image, ax=ax, shrink=.78, pad=.02)
            ax.set(aspect="equal", xticks=[-2, 0, 2], yticks=[-2, 0, 2])
            if row == 0:
                ax.set_title(("FEM", "SVD-SNO", "Ошибка")[col])
            if col == 0:
                ax.set_ylabel(f"{label}\n$y$")
            if row == 2:
                ax.set_xlabel("$x$")
    save_figure(fig, "field_comparison")


def draw_modes(triangulation, bases):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65), layout="constrained")
    for row, name in enumerate(("real", "imag")):
        basis = bases[name]
        fields = (basis.mean, basis.modes[:, 0], basis.modes[:, 1])
        for col, values in enumerate(fields):
            limit = np.max(np.abs(values))
            ax = axes[row, col]
            image = plot_nodal_field(ax, triangulation, values, Normalize(-limit, limit))
            fig.colorbar(image, ax=ax, shrink=.75, pad=.02)
            ax.set(aspect="equal", xticks=[-2, 0, 2], yticks=[-2, 0, 2],
                   title=("Среднее", "Мода 1", "Мода 2")[col])
            if col == 0:
                ax.set_ylabel(("Re", "Im")[row])
    save_figure(fig, "pod_modes")


def draw_training(history, metrics):
    fig, (ax, bars_ax) = plt.subplots(1, 2, figsize=(7.2, 3.05), layout="constrained")
    epochs = np.arange(1, len(history["train"]) + 1)
    ax.semilogy(epochs, 100 * np.array(history["train"]), color="#007f82", lw=1.25,
                label="Обучение")
    ax.semilogy(epochs, 100 * np.array(history["val"]), color="#b46e1b", lw=1.25,
                label="Валидация")
    best_epoch = metrics["best_epoch"]
    ax.axvline(best_epoch, color="#17324d", ls="--", lw=.8)
    ax.set(xlabel="Эпоха", ylabel=r"Средняя ошибка в $L^2$, %")
    ax.legend(frameon=False)
    ax.grid(alpha=.15)

    positions = np.arange(3)
    metric_names = ("consistent_projection_floor_fem_l2", "projection_fem_l2", "test_fem_l2")
    means = 100 * np.array([metrics[name]["mean"] for name in metric_names])
    bars = bars_ax.bar(positions, means, color=["#7bbdb7", "#82a1ba", "#17324d"], width=.6)
    bars_ax.set_xticks(positions, ["Проекция\nв норме $M$", "Проекция\nв норме $W$", "Сеть"])
    bars_ax.set(ylabel=r"Ошибка на тесте в $L^2$, %", ylim=(0, 1.17 * means.max()))
    bars_ax.bar_label(bars, labels=[f"{value:.4f}".replace(".", ",") for value in means],
                      padding=3, fontsize=8)
    save_figure(fig, "training_and_floor")


def draw_spectrum(bases):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    for name, color, component in (("real", "#007f82", "Re"), ("imag", "#b46e1b", "Im")):
        basis = bases[name]
        mode_numbers = np.arange(1, basis.rank + 1)
        axes[0].semilogy(mode_numbers, basis.singular_values / basis.singular_values[0],
                        "-o", ms=3, color=color, label=component)
        axes[1].plot(mode_numbers, 100 * basis.energy, "-o", ms=3, color=color, label=component)
    axes[0].set_ylabel(r"$\sigma_j/\sigma_1$")
    axes[1].set_ylabel("Доля энергии обучающих снимков, %")
    for ax in axes:
        ax.set_xlabel("Номер моды $j$")
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    save_figure(fig, "spectrum")


def collect_forward_trace(predictor, features):
    model = predictor.model
    with torch.inference_mode():
        tokens = model.coefficient_tokens(torch.as_tensor(features))
        hidden = torch.nn.functional.gelu(model.lift(tokens))
        stages = [dict(stage="tokens", shape=list(tokens.shape)),
                  dict(stage="lift", shape=list(hidden.shape),
                       row_spread=float((hidden - hidden[:, :1]).abs().max()))]
        for number, block in enumerate(model.blocks, start=1):
            hidden = block(hidden)
            stages.append(dict(stage=f"block{number}", shape=list(hidden.shape),
                               row_spread=float((hidden - hidden[:, :1]).abs().max())))
        output = model.project(hidden[:, :model.output_modes])
        stages.append(dict(stage="project", shape=list(output.shape)))
        stats = predictor.preprocessing["output_stats"]
        std = torch.as_tensor(stats["std"], dtype=output.dtype)
        mean = torch.as_tensor(stats["mean"], dtype=output.dtype)
        coefficients = output * std + mean
        decoded = predictor.decode(output)[0].numpy()
    trace = dict(k_stats=predictor.preprocessing["k_stats"], feature_vector=features.tolist(),
                 first_token=tokens[0, 0].tolist(), stages=stages,
                 normalized_first3=output[0, :3].tolist(),
                 output_mean_first3=mean[:3].tolist(), output_std_first3=std[:3].tolist(),
                 physical_coeff_first3=coefficients[0, :3].tolist())
    return trace, coefficients[0].numpy(), decoded[:, 0] + 1j * decoded[:, 1]


def main():
    OUT.mkdir(exist_ok=True)
    dataset = load_dataset(DATA)
    predictor = load_predictor(CHECKPOINT)
    mesh_hash = verify_geometry(dataset, predictor)
    metrics = json.loads((RUN / "metrics.json").read_text())
    history = json.loads((RUN / "history.json").read_text())
    with np.load(PREDICTIONS, allow_pickle=False) as archive:
        indices, errors = archive["indices"], archive["fem_l2"]
    np.testing.assert_array_equal(indices, dataset.split_indices()[2])
    median_error = float(np.median(errors))
    np.testing.assert_allclose(median_error, metrics["test_fem_l2"]["median"], rtol=1e-12, atol=0)
    position = int(np.argmin(np.abs(errors - median_error)))
    index = int(indices[position])
    saved = read_prediction_row(position, (len(indices), len(dataset.coordinates)))
    true = np.array(dataset.u_scat[index])
    theta, k = float(dataset.theta[index]), float(dataset.k0[index])
    query = (dataset.A[index], dataset.n[index], theta, k)
    prediction = predictor.predict(*query)
    np.testing.assert_allclose(prediction, saved, rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(dataset.relative_l2(true, saved), errors[position], rtol=1e-8, atol=0)
    error = float(dataset.relative_l2(true, prediction))
    del dataset

    trace, coefficients, decoded = collect_forward_trace(predictor, predictor.encode(*query))
    np.testing.assert_array_equal(decoded, prediction)
    basis = predictor.bases["real"]
    correction = basis.modes @ coefficients[:basis.rank, 0]
    np.testing.assert_allclose(basis.mean + correction, prediction.real, rtol=1e-5, atol=2e-6)
    trace.update(run=str(RUN.relative_to(ROOT)), sample_index=index, sample_position=position, k=k,
                 theta=theta, architecture_version=predictor.model.architecture_version)
    (OUT / "forward_trace.json").write_text(json.dumps(trace, indent=2) + "\n")

    triangulation = mtri.Triangulation(*predictor.coordinates.T, predictor.triangles)
    draw_cover(triangulation, basis.mean, correction, prediction.real)
    draw_fields(triangulation, true, prediction)
    draw_modes(triangulation, predictor.bases)
    draw_training(history, metrics)
    draw_spectrum(predictor.bases)

    source_files = [
        "experiments/reference_sno_gmsfem/dataset.py",
        "experiments/reference_sno_gmsfem/training.py",
        "neural_operator/sno/basis.py",
        "neural_operator/sno/model.py",
        "neural_operator/_layers.py",
        "neural_operator/training.py",
        "neural_operator/losses.py",
        "neural_operator/data.py",
        "neural_operator/fem.py",
        "neural_operator/run.py",
        "neural_operator/mesh_inference.py",
        "experiments/gms_sno_100/model.py",
        "experiments/gms_sno_100/pod.py",
        "experiments/mscale_sno_100/training.py",
        "experiments/mscale_sno_100/METHOD.md",
        "experiments/highk_svd_gmsfem/training.py",
        "experiments/svd_sno_mesh_stability_2026_11_16/README.md",
    ]
    sources = {path: sha256(ROOT / path) for path in source_files}
    evidence = dict(
        date="2026-11-18",
        band=BAND,
        run=str(RUN.relative_to(ROOT)),
        dataset=str(DATA.relative_to(ROOT)),
        checkpoint_sha256=sha256(CHECKPOINT),
        test_predictions_sha256=sha256(PREDICTIONS),
        reference_mesh=str(REFERENCE_MESH.relative_to(ROOT)),
        reference_mesh_sha256=mesh_hash,
        gmsfem_reference_fields=str(ANCHOR_FIELDS.relative_to(ROOT)),
        gmsfem_reference_fields_sha256=sha256(ANCHOR_FIELDS),
        reference_coordinates_exact_match=True,
        reference_triangles_exact_match=True,
        sample_selection="argmin(abs(test_fem_l2 - median(test_fem_l2))); illustration only",
        sample_index=index,
        sample_position=position,
        k=k,
        theta=theta,
        nodes=len(predictor.coordinates),
        triangles=len(predictor.triangles),
        relative_error_saved=float(errors[position]),
        median_relative_error_saved=median_error,
        relative_error_replayed=error,
        max_prediction_replay_difference=float(np.max(np.abs(prediction - saved))),
        rank_validation=predictor.preprocessing["rank_validation"],
        metrics=metrics,
        sources_sha256=sources,
    )
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    keys = ("sample_index", "k", "theta", "nodes", "triangles", "relative_error_replayed",
            "max_prediction_replay_difference")
    print(json.dumps({key: evidence[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
