"""Draw the current 32x32, width=24, depth=3 FNO inference pipeline.

The diagram follows neural_operator/fno/model.py and Predictor.encode/decode.
It describes the actual static-run configuration, not learned numerical weights.
"""
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

HERE = Path(__file__).resolve().parent
DOC_IMAGES = HERE.parents[1] / 'neural_operator' / 'docs' / 'images'
INK = '#172b3a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'text.color': INK, 'pdf.fonttype': 42})


def main():
    fig, ax = plt.subplots(figsize=(8, 10))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 8), ylim=(0, 10)); ax.axis('off')

    def label(x, y, value, size=11, **kw):
        return ax.text(x, y, value, ha='center', va='center', fontsize=size,
                       linespacing=1.35, **kw)

    def box(x, y, w, h, value, fill='#e9f3fa', size=11):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                     boxstyle='round,pad=0.025,rounding_size=0.065',
                     facecolor=fill, edgecolor='#aab7c2', linewidth=.9))
        label(x, y, value, size)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                     mutation_scale=11, linewidth=1.1, color=INK))

    def route(points):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=1.1)
        arrow(points[-2], points[-1])

    def plus(x, y):
        ax.add_patch(Circle((x, y), .13, facecolor='white', edgecolor=INK, linewidth=1.05))
        label(x, y, '+', 14)

    label(4, 9.75, 'FNO: от семи входных каналов до FEM-поля', 15, weight='bold')
    box(4, 9.13, 7.15, .78,
        'Равномерная сетка: 32 × 32 × 7\n'
        r'$[\widetilde A,\widetilde n,\cos\theta,\sin\theta,\widetilde k,\widetilde x,\widetilde y]$'
        '\nНормировки по train; координаты в [−1, 1]', size=11.7)
    arrow((4, 8.71), (4, 8.60))
    box(4, 8.32, 6.2, .47, 'Linear в каждой точке: 7 → 24; поле 32 × 32 × 24', size=11.5)
    arrow((4, 8.06), (4, 7.92))
    box(4, 7.64, 6.2, .47, 'Нулевой padding: +8 справа и снизу → 40 × 40 × 24', size=11.3)
    arrow((4, 7.38), (4, 6.56))

    ax.add_patch(FancyBboxPatch((.38, 3.01), 7.24, 4.16,
                 boxstyle='round,pad=0.02,rounding_size=0.10', facecolor='#faf9fc',
                 edgecolor='#b4a9c7', linewidth=1, zorder=-1))
    label(2.25, 6.96, 'Остаточный блок × 3\nРазмер: 40 × 40 × 24', 11.0,
          weight='bold')
    label(3.78, 6.56, r'$X$', 13)
    ax.add_patch(Circle((4, 6.52), .035, facecolor=INK))
    route([(4, 6.52), (2.24, 6.52), (2.24, 6.30)])
    route([(4, 6.52), (5.57, 6.52), (5.57, 5.81)])
    route([(4, 6.52), (7.18, 6.52), (7.18, 3.26), (4.16, 3.26)])
    label(6.55, 3.48, 'skip: X', 11.1)

    box(2.24, 6.05, 3.28, .43, 'rFFT2: 40 × 40 → 40 × 21', fill='#efe9f8', size=11.2)
    arrow((2.24, 5.80), (2.24, 5.76))
    box(2.24, 5.25, 3.28, .94,
        'Два набора комплексных весов\n'
        'q₁: 0…7 и −8…−1; q₂: 0…7\n'
        '24 → 24 на каждой частоте\n'
        'Остальные частоты выхода — нули', fill='#efe9f8', size=10.65)
    arrow((2.24, 4.75), (2.24, 4.64))
    box(2.24, 4.38, 3.28, .43, 'irFFT2: 40 × 21 → 40 × 40', fill='#efe9f8', size=11.05)
    box(5.57, 5.25, 2.72, 1.04,
        'Локальная ветвь\nConv 1 × 1: 24 → 24\nВ каждой точке сетки', fill='#fff0d8', size=11.2)

    route([(2.24, 4.13), (2.24, 3.99), (3.84, 3.99)])
    route([(5.57, 4.70), (5.57, 3.99), (4.16, 3.99)])
    label(1.99, 4.00, 'S', 11.5)
    label(4.95, 4.15, 'L', 11.5)
    plus(4, 3.99)
    arrow((4, 3.85), (4, 3.83))
    box(4, 3.64, 1.15, .32, 'GELU', fill='#e7f3eb', size=11.3)
    arrow((4, 3.45), (4, 3.40)); plus(4, 3.26)
    label(2.1, 3.26, r'$X_{\rm next}=X+\mathrm{GELU}(S+L)$', 11.2)
    arrow((4, 3.10), (4, 2.96))

    box(4, 2.70, 7.15, .44, 'Crop: удалить padding → 32 × 32 × 24', size=11.5)
    arrow((4, 2.45), (4, 2.32))
    box(4, 2.04, 7.15, .50,
        'В каждой точке: Linear 24 → 24 → GELU → Linear 24 → 2\n'
        'Выход на сетке: 32 × 32 × 2 (нормированные Re, Im)', size=11.1)
    arrow((4, 1.76), (4, 1.65))
    box(4, 1.32, 7.15, .60,
        'Билинейная интерполяция: сетка → 3221 FEM-узел\n'
        r'$\widehat u_c(x_j)=\mu_c(x_j)+s_c\,[D_{\rm bilinear}(Y)]_{j,c},\quad c\in\{R,I\}$',
        fill='#e7f3eb', size=11.3)
    arrow((4, .99), (4, .85))
    box(4, .56, 7.15, .50,
        r'$\mathrm{Re}(u_{\rm scattering})\qquad\mathrm{Im}(u_{\rm scattering})\qquad'
        r'\mathrm{Abs}=\sqrt{\widehat u_R^2+\widehat u_I^2}$',
        fill='#e7f3eb', size=12)
    label(4, .14, 'В статическом опыте A и n фиксированы; все семь каналов сохраняются.', 10.7)

    images = HERE / 'images'; images.mkdir(exist_ok=True)
    DOC_IMAGES.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(images / f'fno_architecture.{extension}', dpi=220, facecolor='white')
    shutil.copyfile(images / 'fno_architecture.png', DOC_IMAGES / 'fno_architecture.png')
    plt.close(fig)
    print(images / 'fno_architecture.pdf')
    print(images / 'fno_architecture.png')
    print(DOC_IMAGES / 'fno_architecture.png')


if __name__ == '__main__':
    main()
