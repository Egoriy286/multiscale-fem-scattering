"""Build article figures from the preserved geometry and saved training log.

No PDE solves, training, or execution of notebook cells are performed.
Training lines connect only the 51 points present in the archived text log.
"""
from pathlib import Path
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 220,
})


def save_figure(fig, name):
    fig.savefig(ROOT / "images" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "images" / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    (ROOT / "images").mkdir(exist_ok=True)
    circles = json.loads((ROOT / "evidence/candidate_circles.json").read_text())
    fig, ax = plt.subplots(figsize=(4.1, 4.1), constrained_layout=True)
    for circle in circles:
        ax.add_patch(Circle((circle["x"], circle["y"]), circle["r"],
                            facecolor="0.88", edgecolor="0.15", linewidth=0.7))
    ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal", xlabel="$x$", ylabel="$y$")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(direction="out", width=0.7)
    save_figure(fig, "candidate_geometry")

    with (ROOT / "evidence/training_log.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    assert len(rows) == 51 and epochs[0] == 1 and epochs[-1] == 500
    fig, ax = plt.subplots(figsize=(6.9, 3.7), constrained_layout=True)
    ax.semilogy(epochs, [100 * float(row["train_nodal_l2"]) for row in rows],
                color="0.1", linewidth=1.3, label="Обучение")
    ax.semilogy(epochs, [100 * float(row["validation_nodal_l2"]) for row in rows],
                color="0.4", linewidth=1.3, linestyle="--", label="Валидация")
    ax.set(xlim=(1, 500), ylim=(6, 85), xlabel="Эпоха",
           ylabel="Средняя узловая ошибка, %")
    ax.yaxis.set_major_locator(FixedLocator([7, 10, 20, 40, 80]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(which="major", color="0.9", linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "training_history")


if __name__ == "__main__":
    main()
