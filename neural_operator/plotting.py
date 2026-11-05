"""Shared scientific figures for the experiment notebooks."""

from __future__ import annotations

import html
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def display_table(rows, columns=None):
    from IPython.display import HTML, display
    if not rows:
        return
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    def format_value(value):
        return f"{value:.5g}" if isinstance(value, (float, np.floating)) else str(value)
    header = "".join(f"<th>{html.escape(key)}</th>" for key in columns)
    body = "".join("<tr>"+"".join(f"<td>{html.escape(format_value(row.get(key, '')))}</td>"
                                  for key in columns)+"</tr>" for row in rows)
    display(HTML(f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"))


def plot_history(history):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name in ("train", "val"):
        ax.semilogy(np.arange(1, len(history[name])+1), history[name], label=name)
    ax.set(xlabel="Эпоха", ylabel="Средняя относительная FEM L2 ошибка")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    return fig


def plot_prediction(dataset, index, prediction, title=""):
    triangulation = mtri.Triangulation(*dataset.coordinates.T, dataset.triangles)
    true = dataset.u_scat[index]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), constrained_layout=True)
    real_limit = max(float(np.abs(true.real).max()), float(np.abs(prediction.real).max()), 1e-12)
    amplitude_limit = max(float(np.abs(true).max()), float(np.abs(prediction).max()), 1e-12)
    items = [(true.real, "Re $u^s$: FEM", "RdBu_r", -real_limit, real_limit),
             (prediction.real, "Re $u^s$: prediction", "RdBu_r", -real_limit, real_limit),
             (np.abs(prediction), "$|u^s|$: prediction", "viridis", 0, amplitude_limit),
             (np.abs(prediction-true), "$|u^s_{pred}-u^s_{FEM}|$", "magma", 0, None)]
    for ax, (values, label, cmap, vmin, vmax) in zip(axes, items):
        artist = ax.tripcolor(triangulation, values, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set(title=label, aspect="equal", xlabel="x", ylabel="y")
        fig.colorbar(artist, ax=ax, shrink=.78)
    fig.suptitle(title or f"k={dataset.k0[index]:.3f}; medium={dataset.medium_id[index]}")
    return fig


def plot_svd(curves, spectra):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), constrained_layout=True)
    reference = "full" if "real_full" in spectra else "randomized"
    for field in ("A", "n", "real", "imag"):
        singular = spectra[f"{field}_{reference}"]
        if len(singular):
            axes[0].semilogy(np.arange(1, len(singular)+1), singular/singular[0], label=field)
            axes[1].plot(np.arange(1, len(singular)+1), spectra[f"{field}_{reference}_energy"], label=field)
    for method in dict.fromkeys(row["method"] for row in curves):
        chosen = [r for r in curves if r["method"] == method]
        axes[2].semilogy([r["rank"] for r in chosen], [r["mean_fem_l2"] for r in chosen], label=method+" mean")
        axes[2].semilogy([r["rank"] for r in chosen], [r["p95_fem_l2"] for r in chosen], ":", label=method+" p95")
    axes[0].set(xlabel="r", ylabel=r"$\sigma_r/\sigma_1$")
    axes[1].set(xlabel="r", ylabel="Доля полной centered train energy", ylim=(0, 1.02))
    axes[2].set(xlabel="r", ylabel="Validation FEM L2: реконструкция")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    return fig
