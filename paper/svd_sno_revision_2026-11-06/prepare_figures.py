#!/usr/bin/env python3
"""Draw the current fixed medium from saved DG0 arrays; no FEM solve."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parents[1]

def main():
    data = REPO/'neural_operator/data/static_low_frequency_1000x8'
    with np.load(data/'geometry.npz') as geometry, np.load(data/'fem_dataset.npz') as fields:
        circles = geometry['etal_circles']
        tags = geometry['cell_tags']
        active = []
        for cx, cy, radius, tag in circles:
            mask = tags == int(tag)
            a, n = np.unique(fields['A_cells'][0, mask]), np.unique(fields['n_cells'][0, mask])
            assert len(a) == len(n) == 1
            if a[0] != 1 or n[0] != 1:
                active.append((float(cx), float(cy), float(radius), float(a[0]), float(n[0])))
    assert len(active) == 3
    fig, ax = plt.subplots(figsize=(5.5, 5.2), layout='constrained')
    for j, (cx, cy, radius, _, _) in enumerate(active, start=1):
        ax.add_patch(Circle((cx, cy), radius, facecolor=plt.cm.viridis(.18+j*.22), edgecolor='black'))
        ax.annotate(str(j), (cx, cy), (cx+.2, cy+.15),
                    arrowprops=dict(arrowstyle='-', lw=.7), fontsize=11)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), xlabel='x', ylabel='y', aspect='equal')
    (PAPER/'images').mkdir(exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(PAPER/f'images/static_medium_geometry.{extension}', dpi=200)
    plt.close(fig)
    print(active)

if __name__ == '__main__':
    main()
