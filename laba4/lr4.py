from dolfin import *
import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path('results').mkdir(exist_ok=True)
set_log_level(LogLevel.ERROR)

# Строим сетку Gmsh.
gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 0)
gmsh.open('mesh.geo')
n = int(gmsh.parser.getNumber('n')[0])
gmsh.model.mesh.generate(2)
gmsh.write('mesh.msh')
gmsh.finalize()
source = meshio.read('mesh.msh')
triangles = source.get_cells_type('triangle')
meshio.write('mesh.xml', meshio.Mesh(source.points[:,:2], [('triangle', triangles)]),
             file_format='dolfin-xml')

mesh = Mesh('mesh.xml')

# Нумеруем квадратные области по координатам.
domains = MeshFunction('size_t', mesh, 2, 0)
for cell in cells(mesh):
    x = cell.midpoint()
    i = int(n*x[0])
    j = int(n*x[1])
    domains[cell] = i + n*j

File('results/blocks.pvd') << domains

# Сохраняем пары: подсетка и метки её границы.
Path('results/submeshes_2').mkdir(parents=True, exist_ok=True)

submeshes = []
submesh_file =File(f'results/submeshes_2/submesh.pvd')
facets_file = File(f'results/submeshes_2/facet.pvd')
for j in range(n):
    for i in range(n):
        number = i + n*j
        submesh = SubMesh(mesh, domains, number)

        facets = MeshFunction('size_t', submesh, 1, 0)
        AutoSubDomain(lambda x, on_boundary: on_boundary).mark(facets, 1)

        submeshes.append((submesh, facets))
        submesh_file  << submesh
        facets_file << facets

print('Подсеток:', len(submeshes))

# Визуализация: чёрная мелкая сетка и синие линии крупных блоков.
fig, ax = plt.subplots(figsize=(6,6))
ax.triplot(source.points[:,0], source.points[:,1], triangles, color='black', lw=0.3)
for v in np.linspace(0,1,n+1):
    ax.axvline(v, color='royalblue', lw=1)
    ax.axhline(v, color='royalblue', lw=1)
ax.set(xlim=(0,1), ylim=(0,1), xlabel='x', ylabel='y', title='Сетка 2D', aspect='equal')
plt.tight_layout()
plt.savefig('results/mesh.png', dpi=150)
plt.show()
