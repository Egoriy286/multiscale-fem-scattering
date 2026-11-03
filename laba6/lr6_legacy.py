from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

from dolfin import *
import gmsh
import meshio
import numpy as np


parser = ArgumentParser(description='Лабораторная 6: DOLFIN 2019, последовательности PVD.')
parser.add_argument('--fine', type=int, default=4)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results_legacy')
args = parser.parse_args()
if args.fine < 1 or MPI.size(MPI.comm_world) != 1:
    parser.error('Нужны --fine >= 1 и запуск в одном MPI-процессе.')
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
set_log_level(LogLevel.ERROR)


def boundary_tags(msh, vertex=None):
    """Метка 1 у окрестности; у квадрата — 1, 2, 3 относительно A."""
    tags = MeshFunction('size_t', msh, 1, 0)
    AutoSubDomain(lambda x, on_boundary: on_boundary).mark(tags, 1)
    if vertex is not None:
        for facet in facets(msh):
            if tags[facet]:
                x = facet.midpoint()
                tags[facet] = 1 if near(x[0], vertex[0], 1e-10) else 2 if near(x[1], vertex[1], 1e-10) else 3
    tags.rename('boundary', 'boundary')
    return tags


# Та же геометрия Gmsh; промежуточные MSH/XML живут только до чтения сетки.
with TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.open(str(Path(__file__).resolve().parents[1] / 'laba4' / 'mesh.geo'))
        n = int(gmsh.parser.getNumber('n')[0])
        h = 1 / (n * args.fine)
        gmsh.option.setNumber('Mesh.MeshSizeMin', h)
        gmsh.option.setNumber('Mesh.MeshSizeMax', h)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(tmp / 'mesh.msh'))
    finally:
        gmsh.finalize()
    source = meshio.read(tmp / 'mesh.msh')
    meshio.write(tmp / 'mesh.xml', meshio.Mesh(source.points[:, :2],
                 [('triangle', source.get_cells_type('triangle'))]), file_format='dolfin-xml')
    msh = Mesh(str(tmp / 'mesh.xml'))

# Как в лабораторной 4: номер крупного квадрата по центру треугольника.
blocks = MeshFunction('size_t', msh, 2, 0)
for cell in cells(msh):
    x = cell.midpoint()
    blocks[cell] = int(n*x[0]) + n*int(n*x[1])
blocks.rename('blocks', 'blocks')
File(str(out / 'blocks.pvd')) << blocks
coarse_cells = [SubMesh(msh, blocks, number) for number in range(n*n)]
selected = MeshFunction('size_t', msh, 2, 0)

omega_file = File(str(out / 'neighborhoods.pvd'))
omega_boundary_file = File(str(out / 'neighborhood_boundaries.pvd'))
cell_file = File(str(out / 'cells.pvd'))
cell_boundary_file = File(str(out / 'cell_boundaries.pvd'))
submeshes, cell_submeshes, timeline = [], [], []
for j in range(n + 1):
    for i in range(n + 1):
        index = i + (n+1)*j
        vertex = np.array([i, j]) / n
        neighbors = [p + n*q for q in (j-1, j) for p in (i-1, i)
                     if 0 <= p < n and 0 <= q < n]
        selected.array()[:] = np.isin(blocks.array(), neighbors)
        omega = SubMesh(msh, selected, 1)
        tags = boundary_tags(omega)
        submeshes.append((omega, tags))
        omega_file << (omega, float(index))
        omega_boundary_file << (tags, float(index))

        for number in neighbors:
            part = coarse_cells[number]
            tags = boundary_tags(part, vertex)
            frame = len(cell_submeshes)
            cell_submeshes.append((part, tags))
            cell_file << (part, float(frame))
            cell_boundary_file << (tags, float(frame))
            timeline.append((frame, index, number))

np.savetxt(out / 'cells.csv', timeline, fmt='%d', delimiter=',',
           header='frame,vertex,block', comments='')
print(f'Грубая сетка: {n}×{n}; треугольников: {msh.num_cells()}; h = {h:g}')
print(f'Кадров окрестностей: {len(submeshes)}; кадров квадратов: {len(cell_submeshes)}')
print(f'Результаты PVD/VTU и соответствие кадров cells.csv: {out}')
