"""Draw reproducible SNO architecture and matrix-operation teaching figures.

Run from the repository root in the existing Python/Matplotlib environment:
    python3 paper/svd_sno_revision_2026-11-06/prepare_architecture.py

All matrix entries are explicitly illustrative, not trained model weights.
"""

from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
DOC_IMAGES = HERE.parents[1] / "neural_operator" / "docs" / "images"
INK = "#172b3a"
DATA = "#e9f3fa"
WEIGHT_W = "#fff0d8"
WEIGHT_B = "#efe9f8"
FIELD = "#e7f3eb"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "text.color": INK, "pdf.fonttype": 42,
                     "svg.fonttype": "none"})


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    return fig, ax


def label(ax, x, y, text, size=12, **kwargs):
    style = {"ha": "center", "va": "center", **kwargs}
    return ax.text(x, y, text, fontsize=size, **style)


def arrow(ax, start, end, **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.1, color=INK, **kwargs))


def box(ax, x, y, width, height, text, color=DATA, size=12):
    ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                               boxstyle="round,pad=0.03,rounding_size=0.08",
                               linewidth=.8, edgecolor="#aab7c2", facecolor=color))
    label(ax, x, y, text, size=size, linespacing=1.5)


def matrix(ax, x, y, values, name, fill=DATA, cell=.47, size=13, label_y=None):
    """Draw a numerical matrix centered at x,y; use visible row/column cells."""
    values = np.asarray(values)
    h, w = np.array(values.shape) * cell
    label(ax, x, y+h/2+.25 if label_y is None else label_y, name, size=13)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            left, bottom = x-w/2+j*cell, y+h/2-(i+1)*cell
            ax.add_patch(Rectangle((left, bottom), cell, cell, facecolor=fill,
                                   edgecolor="#afbec8", linewidth=.65))
            label(ax, left+cell/2, bottom+cell/2, f"{values[i,j]:g}", size=size)


def save(fig, name):
    images = HERE / "images"
    images.mkdir(exist_ok=True)
    DOC_IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(images / f"{name}.pdf", facecolor="white")
    fig.savefig(images / f"{name}.png", dpi=200, facecolor="white")
    shutil.copyfile(images / f"{name}.png", DOC_IMAGES / f"{name}.png")
    plt.close(fig)


def hidden_mixing():
    z = np.array([[1, 2], [3, 4], [5, 6]])
    w = np.array([[1, 1], [0, 1]])
    bmat = np.array([[1, 0, 0], [.5, .5, 0], [0, 0, 1]])
    bias = np.array([0, 1])
    np.testing.assert_array_equal(z @ w, [[1, 3], [3, 7], [5, 11]])
    np.testing.assert_array_equal(bmat @ z, [[1, 2], [2, 3], [5, 6]])
    np.testing.assert_array_equal(bmat @ (z @ w + bias), [[1, 4], [2, 6], [5, 12]])

    fig, ax = canvas(9, 8.8)
    label(ax, 4.5, 8.48, "Z — скрытые данные; W и B — обучаемые матрицы", size=15)
    label(ax, 4.5, 8.04, "Учебный пример: 3 моды × 2 скрытых канала", size=12)
    matrix(ax, 2.0, 6.93, z, r"$Z\in\mathbb{R}^{3\times2}$", cell=.44)
    for i in range(3):
        label(ax, 1.36, 7.37-i*.44, f"мода {i+1}", size=10, ha="right")
    label(ax, 5.8, 7.16, "Строка: одна позиция POD-моды.\nСтолбец: один скрытый канал.\nZ меняется при новом входе.", size=12, linespacing=1.7)
    label(ax, 5.8, 6.24, "Скрытые каналы ещё не являются Re и Im.", size=11)

    label(ax, .45, 5.76, "1. W смешивает каналы: умножение справа", size=13, ha="left")
    matrix(ax, 1.55, 4.48, z, "Z", cell=.43, label_y=5.35)
    label(ax, 2.62, 4.48, "×", size=20)
    matrix(ax, 3.65, 4.48, w, "W: 2 × 2", WEIGHT_W, cell=.43, label_y=5.35)
    label(ax, 4.73, 4.48, "=", size=20)
    matrix(ax, 5.85, 4.48, z @ w, "ZW: 3 × 2", cell=.43, label_y=5.35)
    label(ax, 7.65, 4.48, "Каждая строка\nобрабатывается\nотдельно.", size=11, linespacing=1.6)

    label(ax, .45, 3.63, "2. B смешивает моды: умножение слева", size=13, ha="left")
    matrix(ax, 1.55, 2.34, bmat, "B: 3 × 3", WEIGHT_B, cell=.43, size=12, label_y=3.20)
    label(ax, 2.70, 2.34, "×", size=20)
    matrix(ax, 3.65, 2.34, z, "Z", cell=.43, label_y=3.20)
    label(ax, 4.73, 2.34, "=", size=20)
    matrix(ax, 5.85, 2.34, bmat @ z, "BZ: 3 × 2", cell=.43, label_y=3.20)
    label(ax, 7.65, 2.34, "Строка 2 стала\nсредним строк\n1 и 2.", size=11, linespacing=1.6)

    label(ax, 4.5, 1.38, "В текущем блоке: сначала W, затем B, GELU и остаточная связь", size=12)
    label(ax, 4.5, .93, r"$Z_{\rm next}=Z+\mathrm{GELU}\!\left(B(ZW+\mathbf{1}b^T)\right)$", size=17)
    label(ax, 4.5, .34, "Числа условные. B — веса сети; пространственный POD-базис обозначен Φ.", size=11)
    save(fig, "sno_hidden_mixing")


def architecture():
    fig, ax = canvas(9, 10.9)
    label(ax, 4.5, 10.55, "SVD-SNO: архитектура от входа до поля", size=16)
    label(ax, 2.30, 10.04, "Исторический эксперимент статьи · JAX", size=12)
    label(ax, 6.70, 10.04, "Текущий код проекта · PyTorch", size=12)
    columns = [(2.30, True), (6.70, False)]
    for x, historical in columns:
        if historical:
            text = "POD A, n + параметры волны\nМатрица X: 71 × 5\nВходные моды дополняются нулями"
        else:
            text = "POD A, n + параметры волны\n" + r"Вектор $q$: $d=r_A+r_n+3$"
        box(ax, x, 9.30, 4.05, .92, text, size=10.6)
        arrow(ax, (x, 8.81), (x, 8.47))
        text = "Linear: 5 → 64, затем GELU" if historical else "Linear: d → R·H, затем reshape"
        box(ax, x, 8.12, 4.05, .65, text, size=12)
        arrow(ax, (x, 7.77), (x, 7.41))
        text = "Z⁽⁰⁾: 71 мод × 64 скрытых канала" if historical else "Z⁽⁰⁾: R мод × H скрытых каналов"
        box(ax, x, 7.06, 4.05, .65, text, size=11.5)
        arrow(ax, (x, 6.70), (x, 6.37))
        if historical:
            text = ("4 остаточных блока\n"
                    "Локальная и модальная ветви параллельны\n"
                    r"$T=ZW_{\rm loc}+\mathbf{1}b_{\rm loc}^T$" "\n"
                    r"$S=BZW_{\rm ch}+\mathbf{1}b_{\rm ch}^T$" "\n"
                    r"$Z\leftarrow Z+\mathrm{GELU}(T+S)$")
        else:
            text = ("L остаточных блоков\n"
                    "Каналы и моды смешиваются последовательно\n"
                    r"$T=ZW+\mathbf{1}b^T$" "\n"
                    r"$S=BT$" "\n"
                    r"$Z\leftarrow Z+\mathrm{GELU}(S)$")
        box(ax, x, 5.28, 4.05, 2.10, text, color=WEIGHT_B, size=11.2)
        arrow(ax, (x, 4.20), (x, 3.88))
        text = "Linear: 64 → 2; коэффициенты 71 × 2" if historical else "Linear: H → 2; коэффициенты R × 2"
        box(ax, x, 3.53, 4.05, .65, text, size=11.3)
        arrow(ax, (x, 3.17), (x, 2.79))

    box(ax, 4.50, 2.34, 8.42, .84,
        "Каждая модель использует собственные нормировки и POD-декодеры\n"
        r"$\widehat u_R=\mu_R+\Phi_R\widehat c_R,\qquad\widehat u_I=\mu_I+\Phi_I\widehat c_I$",
        color=FIELD, size=12)
    arrow(ax, (4.5, 1.88), (4.5, 1.54))
    box(ax, 4.5, 1.04, 8.42, .94,
        "Re(u_scattering)      Im(u_scattering)      Abs(u_scattering)\n"
        r"$\mathrm{Abs}=\sqrt{\widehat u_R^{\,2}+\widehat u_I^{\,2}}$",
        color=FIELD, size=13)
    label(ax, 4.5, .22, "Z — данные одного запроса; B, W и b — веса слоя; Φ — фиксированный POD-базис.", size=10.4)
    save(fig, "sno_architecture")


if __name__ == "__main__":
    hidden_mixing()
    architecture()
    print("Saved SNO architecture/mixing PDF+PNG in", HERE / "images")
    print("Copied guide PNGs to", DOC_IMAGES)
