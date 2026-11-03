from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.io import XDMFFile
import numpy as np
import ufl
from ufl import (Measure, TestFunction, TrialFunction, grad, dot, avg, jump,
                 SpatialCoordinate, exp, as_vector, CellDiameter, FacetNormal, inner, div)
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import pyvista
import dolfinx
import matplotlib.pyplot as plt

# Инициализация MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Чтение сетки в параллельном режиме
with XDMFFile(comm, "mesh.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain, name="Grid")

# Создание связностей для топологии
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)

# Чтение меток граней
with XDMFFile(comm, "mt.xdmf", "r") as xdmf:
    ft = xdmf.read_meshtags(domain, name="Grid")

# Меры для интегрирования
dx = Measure("dx", domain=domain, subdomain_data=ct)
ds = Measure("ds", domain=domain, subdomain_data=ft)
dS = Measure("dS", domain=domain)

# Степень элемента
degree = 3

# Пространства функций
V = fem.functionspace(domain, ("DG", degree))
W0 = fem.functionspace(domain, ("DG", 0))

# Волновое число
k0 = 1.0
k = fem.Constant(domain, PETSc.ScalarType(k0 + 0j))

# Коэффициенты материала
A = fem.Function(W0, dtype=PETSc.ScalarType)
A.x.array[:] = 1.0 + 0j

# Находим локальные ячейки с меткой 2
local_cells = ct.indices[ct.values == 2]
if len(local_cells) > 0:
    A.x.array[local_cells] = 0.2 + 0j

n_coef = fem.Function(W0, dtype=PETSc.ScalarType)
n_coef.x.array[:] = 1.0 + 0j
if len(local_cells) > 0:
    n_coef.x.array[local_cells] = 10.0 + 0j

# Падающая волна u_i
x = SpatialCoordinate(domain)
d = as_vector((1.0, 0.0))
u_i = exp(1j * k * dot(x, d))

# Правая часть
f = (
    div((1.0 - A) * grad(u_i))
    + k**2 * (1.0 - n_coef) * u_i
)

# Пробная и тестовая функции
u = TrialFunction(V)
v = TestFunction(V)

# Геометрические величины
n = FacetNormal(domain)
h = CellDiameter(domain)
h_avg = (h('+') + h('-')) / 2.0
alpha = fem.Constant(domain, PETSc.ScalarType(10.0 * degree**2))

# Слабая форма (метод внутренних штрафов)
a = (
    inner(A * grad(u), grad(v)) * dx
    - k**2 * inner(n_coef * u, v) * dx
    - inner(avg(A * grad(u)), jump(v, n)) * dS
    - inner(jump(u, n), avg(A * grad(v))) * dS
    + alpha / h_avg * inner(jump(u, n), jump(v, n)) * dS
    - 1j * k * inner(u, v) * ds
)

L = inner(f, v) * dx

# Настройка решателя для параллельных вычислений
petsc_options={"ksp_type": "preonly", "pc_type": "lu"}

# Создание и решение задачи
problem = LinearProblem(
    a, L,
    petsc_options=petsc_options,
    bcs=[]
)

# Решение
uh = problem.solve()

# @title Обработка данных - ТОЛЬКО ДЛЯ ВИЗУАЛИЗАЦИИ
# Важно: каждая визуализация работает только с локальной частью данных

# Создаем функцию для полного поля
u_total_func = fem.Function(V, dtype=PETSc.ScalarType)

# Локально вычисляем полное поле: uh + u_i
# Для этого нужно вычислить u_i в точках локальных DOFs

# Получаем DOF координаты для локальной части
dof_coords = V.tabulate_dof_coordinates()
local_dofs = V.dofmap.index_map.size_local

# Вычисляем u_i в точках DOFs
u_i_at_dofs = np.exp(1j * k0 * dof_coords[:local_dofs, 0])

# Складываем локальные значения
u_total_local = uh.x.array[:local_dofs] + u_i_at_dofs
u_total_func.x.array[:local_dofs] = u_total_local

# Обновляем значения (важно для параллельных вычислений)
u_total_func.x.scatter_forward()

# Визуализация ЛОКАЛЬНОЙ части
pyvista.start_xvfb()
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)

# Получаем локальную сетку для визуализации
pyvista_cells, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)

# Получаем только локальные точки
local_points = geometry[:len(pyvista_cells)]

# Вычисляем полное поле для локальных точек
coords = local_points
u_i_vals_local = np.exp(1j * k0 * coords[:, 0])

# Получаем значения uh в локальных точках
# Для этого нужно интерполировать uh на точки сетки
# Проще использовать u_total_func, который уже содержит правильные значения

# Используем значения из функции
grid.point_data["u_abs"] = np.abs(u_total_func.x.array[:len(coords)])

# Визуализация локальной части
grid.set_active_scalars("u_abs")
p_abs = pyvista.Plotter(off_screen=True, window_size=[800, 600])
p_abs.add_text(f"Process {rank}: Local u_abs", 
               position="upper_edge", 
               font_size=14, 
               color="black")
p_abs.add_mesh(grid, cmap="jet", show_edges=False)
p_abs.view_xy()

# Каждый процесс сохраняет свою часть
figure = p_abs.screenshot(f"u_abs_local_rank_{rank}.png")
comm.Barrier()
print(f"Process {rank}: Локальная визуализация сохранена в u_abs_local_rank_{rank}.png")
# # Альтернатива: собираем все данные на процесс 0 для полной визуализации
# if rank == 0:
#     # Создаем список для сбора всех данных
#     all_points = []
#     all_values = []
# else:
#     all_points = None
#     all_values = None

# # Собираем локальные данные
# local_points_list = coords.tolist()
# local_values_list = grid.point_data["u_abs"].tolist()

# # Собираем размеры данных
# local_size = len(local_points_list)
# sizes = comm.gather(local_size, root=0)

# if rank == 0:
#     # Выделяем память для всех данных
#     total_size = sum(sizes)
#     all_points = np.zeros((total_size, 3))
#     all_values = np.zeros(total_size)
# else:
#     total_size = None

# # Подготавливаем смещения для gatherv
# if rank == 0:
#     displacements = np.cumsum([0] + [s * 3 for s in sizes[:-1]])
#     value_displacements = np.cumsum([0] + list(sizes[:-1]))
# else:
#     displacements = None
#     value_displacements = None

# # Собираем точки
# comm.Gatherv(coords.flatten(), 
#              [all_points, sizes, displacements, MPI.DOUBLE] if rank == 0 else None,
#              root=0)

# # Собираем значения
# comm.Gatherv(grid.point_data["u_abs"], 
#              [all_values, sizes, value_displacements, MPI.DOUBLE] if rank == 0 else None,
#              root=0)

# # На процессе 0 создаем полную визуализацию
# if rank == 0:
#     print(f"Параллельное вычисление завершено на {size} процессах")
#     print(f"Размерность пространства: {V.dofmap.index_map.size_global}")
#     print(f"Всего точек для визуализации: {len(all_points)}")
    
#     # Создаем полную сетку
#     full_grid = pyvista.PolyData(all_points)
#     full_grid.point_data["u_abs"] = all_values
    
#     # Визуализация полного поля
#     plotter_full = pyvista.Plotter(off_screen=True, window_size=[1000, 800])
#     plotter_full.add_text("Full Solution: |u(x,y)|", 
#                          position="upper_edge", 
#                          font_size=16, 
#                          color="black")
    
#     # Используем триангуляцию для создания поверхности
#     mesh_2d = full_grid.delaunay_2d()
    
#     plotter_full.add_mesh(mesh_2d, 
#                          scalars="u_abs", 
#                          cmap="jet", 
#                          show_edges=False,
#                          show_scalar_bar=True,
#                          scalar_bar_args={'title': '|u|'})
    
#     plotter_full.view_xy()
#     plotter_full.screenshot("u_abs_full.png")
    
#     # Выводим изображение
#     img = plt.imread("u_abs_full.png")
#     plt.figure(figsize=(12, 8))
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title(f"Полное решение |u(x,y)| (вычислено на {size} процессах)", 
#               fontsize=14, pad=20)
#     plt.tight_layout()
#     plt.show()
    
#     # Вывод статистики
#     print(f"\nСтатистика решения:")
#     print(f"  Min |u|: {np.min(all_values):.4f}")
#     print(f"  Max |u|: {np.max(all_values):.4f}")
#     print(f"  Mean |u|: {np.mean(all_values):.4f}")
#     print(f"  Std |u|: {np.std(all_values):.4f}")

# comm.Barrier()
# print(f"Process {rank}: Локальная визуализация сохранена в u_abs_local_rank_{rank}.png")