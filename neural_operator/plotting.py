"""Shared scientific figures for the experiment notebooks."""

from __future__ import annotations

import html
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from .fields import scattering_components


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
    """Compare Re, Im and Abs of u_scattering with their FEM references."""
    triangulation = mtri.Triangulation(*dataset.coordinates.T, dataset.triangles)
    true = scattering_components(dataset.u_scat[index])
    predicted = scattering_components(prediction)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for row, (name, label) in enumerate((("real", r"Re($u_{\mathrm{scattering}}$)"),
                                        ("imag", r"Im($u_{\mathrm{scattering}}$)"),
                                        ("abs", r"Abs($u_{\mathrm{scattering}}$)"))):
        reference, estimate = true[name], predicted[name]
        limit = max(float(np.abs(reference).max()), float(np.abs(estimate).max()), 1e-12)
        cmap = "viridis" if name == "abs" else "RdBu_r"
        vmin = 0 if name == "abs" else -limit
        items = ((reference, f"{label}: FEM", cmap, vmin, limit),
                 (estimate, f"{label}: prediction", cmap, vmin, limit),
                 (np.abs(estimate-reference), f"{label}: absolute error", "magma", 0, None))
        for ax, (values, heading, colors, lower, upper) in zip(axes[row], items):
            artist = ax.tripcolor(triangulation, values, shading="gouraud", cmap=colors,
                                  vmin=lower, vmax=upper)
            ax.set(title=heading, aspect="equal", xlabel="x", ylabel="y")
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
