from argparse import ArgumentParser
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpi4py import MPI
from dolfinx import io, mesh
from dolfinx.io import gmsh as gmsh_io


parser = ArgumentParser(description='Лабораторная 6: окрестности вершин грубой сетки.')
parser.add_argument('--fine', type=int, default=4, help='Число мелких отрезков на стороне крупного квадрата.')
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
args = parser.parse_args()
if args.fine < 1 or MPI.COMM_WORLD.size != 1:
    parser.error('Нужны --fine >= 1 и запуск в одном MPI-процессе.')
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)


def boundary_tags(msh, vertex=None):
    """Внешняя граница: 1; для ячейки — метки 1, 2, 3 относительно A."""
    msh.topology.create_connectivity(1, 2)
    facets = mesh.exterior_facet_indices(msh.topology)
    values = np.ones(len(facets), dtype=np.int32)
    if vertex is not None:
        xy = mesh.compute_midpoints(msh, 1, facets)
        values[:] = 3
        values[np.isclose(xy[:, 0], vertex[0])] = 1
        values[np.isclose(xy[:, 1], vertex[1])] = 2
    tags = mesh.meshtags(msh, 1, facets, values)
    tags.name = 'boundary'
    return tags


def save(msh, path, *tags):
    """Один XDMF-файл с сеткой и её метками, без отдельного HDF5-файла."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.XDMFFile(msh.comm, path, 'w', encoding=io.XDMFFile.Encoding.ASCII) as file:
        file.write_mesh(msh)
        for tag in tags:
            file.write_meshtags(tag, msh.geometry)


# Геометрия уже содержит общие линии всех крупных квадратов, как в лабораторной 4.
gmsh.initialize()
try:
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.open(str(Path(__file__).resolve().parents[1] / 'laba4' / 'mesh.geo'))
    n = int(gmsh.parser.getNumber('n')[0])
    h = 1 / (n * args.fine)
    gmsh.option.setNumber('Mesh.MeshSizeMin', h)
    gmsh.option.setNumber('Mesh.MeshSizeMax', h)
    # Одна физическая группа нужна только для переноса всей сетки в DOLFINx.
    gmsh.model.addPhysicalGroup(2, [tag for dim, tag in gmsh.model.getEntities(2)], 1)
    gmsh.model.mesh.generate(2)
    msh = gmsh_io.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2).mesh
finally:
    gmsh.finalize()

# Номера крупных квадратов определяем по координатам центров треугольников.
cells = np.arange(msh.topology.index_map(2).size_local, dtype=np.int32)
ij = (n * mesh.compute_midpoints(msh, 2, cells)[:, :2]).astype(np.int32)
numbers = ij[:, 0] + n * ij[:, 1]
blocks = mesh.meshtags(msh, 2, cells, numbers)
blocks.name = 'blocks'
save(msh, out / 'mesh.xdmf', blocks)

example = []
total_cells = 0
for j in range(n + 1):
    for i in range(n + 1):
        vertex = np.array([i, j]) / n
        neighbors = [p + n*q for q in (j-1, j) for p in (i-1, i)
                     if 0 <= p < n and 0 <= q < n]
        folder = out / f'omega_{i + (n+1)*j:03d}'
        selected = cells[np.isin(numbers, neighbors)]
        omega = mesh.create_submesh(msh, 2, selected)[0]
        save(omega, folder / 'omega.xdmf', boundary_tags(omega))

        for number in neighbors:
            part = mesh.create_submesh(msh, 2, blocks.find(number))[0]
            tags = boundary_tags(part, vertex)
            save(part, folder / f'cell_{number:03d}.xdmf', tags)
            total_cells += 1
            if i == j == n // 2:
                example.append((part, tags))

# Общая сетка и одна внутренняя окрестность с метками из рисунка задания.
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
xy = msh.geometry.x
for ax in axes:
    ax.triplot(xy[:, 0], xy[:, 1], msh.geometry.dofmaps[0], color='black', lw=0.3)
    ax.set(aspect='equal', xlabel='x', ylabel='y')
for value in np.linspace(0, 1, n + 1):
    axes[0].axvline(value, color='royalblue', lw=1)
    axes[0].axhline(value, color='royalblue', lw=1)
axes[0].set(xlim=(0, 1), ylim=(0, 1), title=f'Грубая сетка {n}×{n} и мелкие треугольники')
colors = np.array(['', 'seagreen', 'deepskyblue', 'crimson'])
for part, tags in example:
    edges = mesh.entities_to_geometry(part, 1, tags.indices)
    axes[1].add_collection(LineCollection(part.geometry.x[edges, :2], colors=colors[tags.values], lw=2))
center = (n // 2) / n
axes[0].plot(center, center, 'ro', ms=4)
axes[1].plot(center, center, 'ro', ms=4)
axes[1].annotate('A', (center, center), xytext=(5, 5), textcoords='offset points')
axes[1].set(xlim=(center-1/n, center+1/n), ylim=(center-1/n, center+1/n), title='Ячейки окрестности A')
for value in (1, 2, 3):
    axes[1].plot([], [], color=colors[value], lw=2, label=str(value))
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
fig.tight_layout()
fig.savefig(out / 'mesh.png', dpi=150)
plt.close(fig)

print(f'Грубая сетка: {n}×{n}; треугольников: {len(cells)}; h = {h:g}')
print(f'Окрестностей: {(n+1)**2}; сохранённых ячеек с локальными метками: {total_cells}')
print(f'Результаты: {out}')
