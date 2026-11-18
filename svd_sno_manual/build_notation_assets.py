"""Trace the real-valued reconstruction at the vertices of a GMsFEM reference cell."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.reference_sno_gmsfem.dataset import ANCHOR_FIELDS, load_dataset
from neural_operator.training import load_predictor
from svd_sno_manual.paraview_colormap import RAINBOW, plot_nodal_field

EXPERIMENT = ROOT / "experiments/reference_sno_gmsfem"
OUTPUT = Path(__file__).parent / "assets"


def draw_element(dataset, element):
    xy, triangles = dataset.coordinates, dataset.triangles
    vertices = xy[triangles[element]]
    center = vertices.mean(axis=0)
    radius = 2.5 * np.ptp(vertices, axis=0).max()
    occupied = dataset.n_cells[0] != 1
    colors = ListedColormap(["#e1edef", "#316d70"])
    limits = [center - radius, center + radius]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, (whole, zoom) = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    whole.tripcolor(*xy.T, triangles, facecolors=occupied.astype(float), cmap=colors,
                    vmin=0, vmax=1, rasterized=True)
    whole.add_patch(Rectangle(limits[0], 2*radius, 2*radius, fill=False,
                              color="#bd4a19", linewidth=1.4))
    whole.set_title("Среда с 24 включениями")
    whole.set(xticks=[-2, 0, 2], yticks=[-2, 0, 2])
    visible = np.all(np.abs(xy[triangles] - center) <= 2*radius, axis=(1, 2))
    local = triangles[visible]
    zoom.tripcolor(*xy.T, local, facecolors=occupied[visible].astype(float),
                   cmap=colors, vmin=0, vmax=1)
    zoom.triplot(*xy.T, local, lw=.65, color="#819699")
    zoom.fill(*vertices.T, color="#f1bc82", edgecolor="#8c4016", linewidth=1.3)
    for label, point in zip("ABC", vertices):
        offset = (point-center)/np.linalg.norm(point-center)
        zoom.scatter(*point, s=15, color="#733812", zorder=5)
        zoom.annotate(label, point, 10*offset, textcoords="offset points", ha="center",
                      va="center", fontweight="bold", color="#723410")
    zoom.set(xlim=(limits[0][0], limits[1][0]), ylim=(limits[0][1], limits[1][1]),
             xticks=[-.02, 0, .02, .04], yticks=[-.02, 0, .02, .04])
    zoom.set_title("Фрагмент эталонной сетки")
    for ax in (whole, zoom):
        ax.set_aspect("equal")
        ax.set(xlabel="$x$", ylabel="$y$")
    fig.savefig(OUTPUT/"notation_example.pdf", dpi=270)
    fig.savefig(OUTPUT/"notation_example.png", dpi=180)
    plt.close(fig)


def draw_operators(model):
    block = model.model.blocks[0]
    weights = block.modes.weight.detach().numpy()
    change = 0.1 * model.bases["real"].modes[:, 0]
    fig, (modal, spatial) = plt.subplots(1, 2, figsize=(7.2, 2.65), layout="constrained")
    limit = np.abs(weights).max()
    image = modal.imshow(weights, extent=(.5, 25.5, 25.5, .5), interpolation="nearest",
                         cmap=RAINBOW, vmin=-limit, vmax=limit)
    modal.add_patch(Rectangle((1.5, .5), 1, 1, fill=False, edgecolor="black", linewidth=1.1))
    modal.set(xlabel="Входная позиция j", ylabel="Выходная позиция i",
              title="Матрица $B_1$", xticks=[1, 10, 20, 25], yticks=[1, 10, 20, 25])
    fig.colorbar(image, ax=modal, shrink=.83)
    limit = np.abs(change).max()
    triangulation = mtri.Triangulation(*model.coordinates.T, model.triangles)
    field = plot_nodal_field(spatial, triangulation, change, Normalize(-limit, limit))
    spatial.set(xlabel="$x$", ylabel="$y$", title="Изменение поля при $\\Delta c_1=0{,}1$",
                xticks=[-2, 0, 2], yticks=[-2, 0, 2], aspect="equal")
    fig.colorbar(field, ax=spatial, shrink=.83)
    fig.savefig(OUTPUT/"operator_matrices.pdf", dpi=270)
    fig.savefig(OUTPUT/"operator_matrices.png", dpi=220)
    plt.close(fig)


def main():
    data = load_dataset(EXPERIMENT / "data/low")
    model = load_predictor(EXPERIMENT / "runs/low/model.pt")
    with np.load(ANCHOR_FIELDS) as anchor:
        np.testing.assert_array_equal(data.coordinates, anchor["reference_coordinates"])
        np.testing.assert_array_equal(data.triangles, anchor["reference_triangles"])
    np.testing.assert_array_equal(data.coordinates, model.coordinates)
    np.testing.assert_array_equal(data.triangles, model.triangles)
    with np.load(EXPERIMENT/"runs/low/test_predictions.npz") as saved:
        errors = saved["fem_l2"]
        sample = int(saved["indices"][np.argmin(np.abs(errors-np.median(errors)))])
    center_node = np.argmin(np.linalg.norm(data.coordinates, axis=1))
    element = int(np.flatnonzero((data.triangles == center_node).any(axis=1))[0])
    nodes = data.triangles[element]
    features = model.encode(data.A[sample], data.n[sample], data.theta[sample], data.k0[sample])
    with torch.inference_mode():
        output = model.model(torch.as_tensor(features))
        prediction = model.decode(output)[0, :, 0].numpy()
        stats = model.preprocessing["output_stats"]
        real_coefficients = (output[0] * torch.as_tensor(stats["std"], dtype=output.dtype)
                             + torch.as_tensor(stats["mean"], dtype=output.dtype))[:, 0].numpy()
    basis = model.bases["real"]
    contributions = basis.modes[nodes] * real_coefficients
    np.testing.assert_allclose(basis.mean[nodes] + contributions.sum(axis=1),
                               prediction[nodes], atol=2e-6, rtol=0)
    # A unit change in one input entry isolates one column of the linear block.
    block = model.model.blocks[0]
    with torch.inference_mode():
        impulse = torch.zeros((1, model.model.token_modes, model.model.width))
        impulse[0, 1, 0] = 1
        weight_b = block.modes.weight[0, 1].item()
        weight_channel = block.channels.weight[0, 0].item()
        nonlocal_weight = block.modes(block.channels(impulse).transpose(1, 2))[0, 0, 0].item()
    np.testing.assert_allclose(nonlocal_weight, weight_b * weight_channel, atol=1e-7)
    record = dict(sample=sample, element=element, nodes=nodes.tolist(),
                  coordinates=data.coordinates[nodes].tolist(), mean=basis.mean[nodes].tolist(),
                  mode_one=basis.modes[nodes, 0].tolist(), coefficient_one=float(real_coefficients[0]),
                  first_term=contributions[:, 0].tolist(),
                  other_terms=contributions[:, 1:].sum(axis=1).tolist(),
                  prediction=prediction[nodes].tolist(), fem=data.u_scat[sample,nodes].real.tolist(),
                  block_one_impulse=dict(mode_weight=weight_b, channel_weight=weight_channel,
                                         response=nonlocal_weight))
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/"notation_example.json").write_text(json.dumps(record, indent=2)+"\n")
    draw_element(data, element)
    draw_operators(model)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
