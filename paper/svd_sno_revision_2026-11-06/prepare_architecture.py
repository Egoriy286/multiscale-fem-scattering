#!/usr/bin/env python3
"""Draw only the current SVD-SNO version 2 used in the article."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PAPER = Path(__file__).resolve().parent

def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.set(xlim=(0, 9), ylim=(0, 10)); ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    rows = [
        (9.0, 'Одна фиксированная среда; параметры k и θ\nTrain-only POD: r_A = r_n = 0, r_Re = r_Im = 10'),
        (7.7, 'Матрица C: 10 × 5\n[0, 0, cos θ, sin θ, нормированное k] в каждой строке'),
        (6.4, 'Общий Linear 5 → 24 и GELU\nСкрытые данные Z: 10 модальных позиций × 24 канала'),
        (4.7, 'Три остаточных блока версии 2\nЛокальная ветвь: T = Z W_loc + 1 b_locᵀ\nМодальная ветвь: S = B Z W_ch\nZ_next = Z + GELU(T + S)'),
        (2.8, 'Linear 24 → 2\nДенормировка коэффициентов Re и Im'),
        (1.35, 'Два POD-декодера в 3221 FEM-узел\nRe = μ_Re + Φ_Re c_Re; Im = μ_Im + Φ_Im c_Im\nAbs = √(Re² + Im²)'),
    ]
    for i, (y, text) in enumerate(rows):
        height = 1.45 if i in (3, 5) else 1.0
        ax.add_patch(FancyBboxPatch((.4, y-height/2), 8.2, height,
            boxstyle='round,pad=.03', facecolor='#e9f3fa' if i < 4 else '#e7f3eb',
            edgecolor='#9aafbd', lw=1))
        ax.text(4.5, y, text, ha='center', va='center', linespacing=1.5)
        if i:
            previous_y = rows[i-1][0]
            previous_h = 1.45 if i-1 in (3, 5) else 1.0
            ax.add_patch(FancyArrowPatch((4.5, previous_y-previous_h/2-.04),
                (4.5, y+height/2+.04), arrowstyle='-|>', mutation_scale=13, color='#17344a'))
    ax.text(4.5, .22, 'Z зависит от запроса; B и W — обучаемые веса; Φ — фиксированные POD-моды.',
            ha='center', va='center', fontsize=10)
    (PAPER/'images').mkdir(exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(PAPER/f'images/sno_v2_architecture.{extension}', dpi=200)
    plt.close(fig)

if __name__ == '__main__':
    main()
