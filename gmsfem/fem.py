"""Тонкосеточный CG1-оператор Гельмгольца и матрицы локальных задач.

DOLFINx собирает глобальные формы, включая корректную контрастную правую
часть. Точные формулы треугольного P1-элемента используются для локальных
Neumann-матриц. Вырезать principal submatrix из глобальной stiffness нельзя:
в ней остались бы вклады элементов за пределами локальной области.

Реализация последовательная (COMM_SELF); SciPy работает с теми же номерами
степеней свободы, что Function.x.array в DOLFINx.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


MATERIALS = {
    "notebook": {1: (1.0, 1.0), 2: (0.2, 10.0), 3: (0.2, 10.0), 4: (0.2, 10.0)},
    # Fig. 7: маленькие красные → третий материал, большие голубые → первый.
    "paper_example1": {1: (1.0, 1.0), 2: (10.0, 0.2), 3: (5.0, 0.4), 4: (3.0, 0.8)},
    "paper_example2": {1: (1.0, 1.0), 2: (10.0, 0.2), 3: (10.0, 0.2), 4: (10.0, 0.2)},
    "homogeneous": {tag: (1.0, 1.0) for tag in (1, 2, 3, 4)},
}


@dataclass
class FineProblem:
    """Один комплексный CG1-оператор; triangles содержит номера DOF, не Gmsh nodes."""

    mesh: object
    V: object
    cell_tags: object
    facet_tags: object
    alpha_function: object
    n_function: object
    coordinates: np.ndarray
    triangles: np.ndarray
    material_tags: np.ndarray
    alpha: np.ndarray
    n: np.ndarray
    areas: np.ndarray
    element_K: np.ndarray
    element_M0: np.ndarray
    K: sparse.csr_matrix
    Mn: sparse.csr_matrix
    M0: sparse.csr_matrix
    Salpha: sparse.csr_matrix
    B: sparse.csr_matrix
    operator: sparse.csr_matrix
    rhs: np.ndarray
    incident: np.ndarray
    k: float
    direction: tuple[float, float]
    case: str
    assembly_seconds: float
    element_assembly_error: float

    @property
    def ndofs(self) -> int:
        return len(self.coordinates)

    def as_function(self, values: np.ndarray, name: str = "u_s"):
        """Вернуть DOLFINx Function в исходном пространстве, например для VTK."""
        from dolfinx import fem

        function = fem.Function(self.V, name=name)
        function.x.array[:] = values
        function.x.scatter_forward()
        return function


@dataclass
class LocalMatrices:
    dofs: np.ndarray
    boundary: np.ndarray
    K: sparse.csr_matrix
    Mn: sparse.csr_matrix
    Salpha: sparse.csr_matrix


def _petsc_to_csr(matrix, *, real: bool = True) -> sparse.csr_matrix:
    indptr, indices, values = matrix.getValuesCSR()
    result = sparse.csr_matrix((values.copy(), indices.copy(), indptr.copy()), shape=matrix.getSize())
    return result.real.tocsr() if real else result


def assemble_problem(
    mesh_path: str | Path,
    *,
    case: str = "notebook",
    k: float = 1.0,
    direction: tuple[float, float] = (1.0, 0.0),
) -> FineProblem:
    """Собрать K - k² M_n - i k B и RHS для u_s = u_total - exp(i k d·x).

    Включения заполнены материалом A=alpha*I. Граничный Robin применяется
    только к physical facets 11..14. Внутренние интерфейсы не являются ds.
    Никакие коэффициенты не дифференцируются: это существенно при скачках.
    """
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc
    from dolfinx.io import gmsh as gmshio
    import ufl

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("GMsFEM использует последовательный SciPy backend. Запустите без mpirun.")
    if np.dtype(PETSc.ScalarType).kind != "c":
        raise RuntimeError("Нужно ядро Python (DOLFINx Complex): PETSc.ScalarType=complex128.")
    if case not in MATERIALS:
        raise ValueError(f"Неизвестный case={case!r}; доступны {tuple(MATERIALS)}")
    if not np.isfinite(k) or k <= 0:
        raise ValueError("Волновое число k должно быть положительным и конечным.")
    d = np.asarray(direction, dtype=float)
    if d.shape != (2,) or not np.all(np.isfinite(d)) or not np.isclose(np.linalg.norm(d), 1.0):
        raise ValueError("direction должен быть единичным двумерным вектором.")

    started = perf_counter()
    data = gmshio.read_from_msh(str(Path(mesh_path).resolve()), MPI.COMM_SELF, gdim=2)
    domain, ct, ft = data.mesh, data.cell_tags, data.facet_tags
    if ct is None or ft is None:
        raise ValueError("В .msh необходимы physical surfaces и внешние physical curves.")
    V = fem.functionspace(domain, ("Lagrange", 1))
    W0 = fem.functionspace(domain, ("DG", 0))
    ncells = domain.topology.index_map(2).size_local
    material_tags = np.zeros(ncells, dtype=np.int32)
    material_tags[ct.indices] = ct.values
    if not set(np.unique(material_tags)).issubset(MATERIALS[case]) or np.any(material_tags == 0):
        raise ValueError("Каждый треугольник должен иметь material tag 1, 2, 3 или 4.")
    if set(np.unique(ft.values)) != {11, 12, 13, 14}:
        raise ValueError("Ожидаются только внешние facet tags 11..14.")

    alpha = np.array([MATERIALS[case][int(tag)][0] for tag in material_tags])
    n_coef = np.array([MATERIALS[case][int(tag)][1] for tag in material_tags])
    alpha_f, n_f = fem.Function(W0, name="alpha"), fem.Function(W0, name="n")
    # Номер DG0 DOF не обязан совпадать с номером ячейки.
    for cell in range(ncells):
        local_dof = W0.dofmap.cell_dofs(cell)
        alpha_f.x.array[local_dof] = alpha[cell]
        n_f.x.array[local_dof] = n_coef[cell]
    alpha_f.x.scatter_forward()
    n_f.x.scatter_forward()
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    boundary_mass = sum(ufl.inner(u, v) * ds(tag) for tag in (11, 12, 13, 14))
    expressions = [
        ufl.inner(alpha_f * ufl.grad(u), ufl.grad(v)) * dx,
        ufl.inner(n_f * u, v) * dx,
        ufl.inner(u, v) * dx,
        ufl.inner(alpha_f * u, v) * dx,
        boundary_mass,
    ]
    matrices = []
    for expression in expressions:
        matrix = fem_petsc.assemble_matrix(fem.form(expression))
        matrix.assemble()
        matrices.append(_petsc_to_csr(matrix))
        matrix.destroy()
    K, Mn, M0, Salpha, B = matrices
    x = ufl.SpatialCoordinate(domain)
    ui = ufl.exp(1j * k * (d[0] * x[0] + d[1] * x[1]))
    # -a_contrast(ui, v): после интегрирования по частям здесь остаётся
    # интерфейсный вклад, который теряется при div(DG0 * grad(ui))*dx.
    load = (
        ufl.inner((1 - alpha_f) * ufl.grad(ui), ufl.grad(v))
        + k**2 * ufl.inner((n_f - 1) * ui, v)
    ) * dx
    vector = fem_petsc.assemble_vector(fem.form(load))
    vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    rhs = vector.array.copy()
    vector.destroy()

    coords = V.tabulate_dof_coordinates()[:, :2].copy()
    triangles = np.array([V.dofmap.cell_dofs(c) for c in range(ncells)], dtype=np.int32)
    vertices = coords[triangles]
    mapping = np.concatenate((np.ones((ncells, 3, 1)), vertices), axis=2)
    signed_det = np.linalg.det(mapping)
    areas = np.abs(signed_det) / 2
    if np.any(areas <= 0):
        raise ValueError("В сетке есть вырожденный треугольник.")
    gradients = np.linalg.inv(mapping)[:, 1:, :].transpose(0, 2, 1)
    element_K = alpha[:, None, None] * areas[:, None, None] * np.einsum("cik,cjk->cij", gradients, gradients)
    element_M0 = areas[:, None, None] / 12 * (np.ones((3, 3)) + np.eye(3))[None, :, :]
    ndofs = len(coords)
    assembled_K = assemble_elements(triangles, element_K, ndofs)
    defect = sparse.linalg.norm(assembled_K - K) / max(sparse.linalg.norm(K), 1e-30)
    if defect > 1e-10:
        raise RuntimeError(f"Локальная P1-сборка не совпадает с DOLFINx: {defect:.3e}")
    incident = np.exp(1j * k * (coords @ d))
    return FineProblem(
        domain, V, ct, ft, alpha_f, n_f, coords, triangles, material_tags,
        alpha, n_coef, areas, element_K, element_M0,
        K, Mn, M0, Salpha, B, (K - k**2 * Mn - 1j * k * B).tocsr(),
        rhs, incident, float(k), (float(d[0]), float(d[1])), case,
        perf_counter() - started, float(defect),
    )


def assemble_elements(triangles: np.ndarray, values: np.ndarray, ndofs: int) -> sparse.csr_matrix:
    """Сложить матрицы 3×3 по таблице DOF выбранных P1-треугольников."""
    rows = np.repeat(triangles, 3, axis=1).ravel()
    cols = np.tile(triangles, (1, 3)).ravel()
    return sparse.coo_matrix((values.ravel(), (rows, cols)), shape=(ndofs, ndofs)).tocsr()


def local_matrices(problem: FineProblem, cells: np.ndarray) -> LocalMatrices:
    """Сборка только по элементам patch/cell; границу определяет кратность рёбер."""
    cells = np.asarray(cells, dtype=np.int32)
    if not len(cells):
        raise ValueError("Пустая локальная область: проверьте согласованность сетки.")
    global_triangles = problem.triangles[cells]
    dofs = np.unique(global_triangles)
    triangles = np.searchsorted(dofs, global_triangles)
    edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Немногообразная локальная сетка.")
    boundary = np.unique(unique_edges[counts == 1])
    return LocalMatrices(
        dofs, boundary,
        assemble_elements(triangles, problem.element_K[cells], len(dofs)),
        assemble_elements(triangles, problem.n[cells, None, None] * problem.element_M0[cells], len(dofs)),
        assemble_elements(triangles, problem.alpha[cells, None, None] * problem.element_M0[cells], len(dofs)),
    )


def solve_dirichlet(matrix: sparse.spmatrix, boundary: np.ndarray, values: np.ndarray):
    """Одна sparse LU-факторизация для всех граничных правых частей.

    Возвращает решение и относительную внутреннюю невязку. values имеет
    форму (число boundary DOF, число правых частей).
    """
    boundary = np.asarray(boundary, dtype=int)
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if len(boundary) != len(values):
        raise ValueError("Несогласованные Dirichlet DOF и values.")
    interior = np.setdiff1d(np.arange(matrix.shape[0]), boundary, assume_unique=True)
    result = np.zeros((matrix.shape[0], values.shape[1]), dtype=np.result_type(matrix.dtype, values.dtype))
    result[boundary] = values
    if not len(interior):
        return result, 0.0
    Aii = matrix.tocsr()[interior][:, interior].tocsc()
    rhs = -(matrix.tocsr()[interior][:, boundary] @ values)
    try:
        factor = splu(Aii)
        result[interior] = factor.solve(np.asarray(rhs, order="F"))
    except RuntimeError as exc:
        raise RuntimeError("Локальная Dirichlet-задача вырождена. Для Type 2 возможен резонанс; измените H или k.") from exc
    residual = Aii @ result[interior] - rhs
    # Backward error остаётся осмысленной и при почти нулевой rhs.
    scale = sparse.linalg.norm(Aii) * np.linalg.norm(result[interior]) + np.linalg.norm(rhs)
    defect = np.linalg.norm(residual) / max(scale, 1e-30)
    if not np.all(np.isfinite(result)) or defect > 1e-9:
        raise RuntimeError(f"Некорректное локальное решение, backward error={defect:.3e}.")
    return result, float(defect)


def solve_fine(problem: FineProblem) -> tuple[np.ndarray, dict]:
    """Эталон CG1 на данной тонкой сетке; возвращает scattered field и диагностику."""
    started = perf_counter()
    solution = splu(problem.operator.tocsc()).solve(problem.rhs)
    residual = np.linalg.norm(problem.operator @ solution - problem.rhs) / max(np.linalg.norm(problem.rhs), 1e-30)
    if not np.all(np.isfinite(solution)) or residual > 1e-8:
        raise RuntimeError(f"МКЭ не сошёлся: relative residual={residual:.3e}")
    return solution, {"solve_seconds": perf_counter() - started, "relative_residual": float(residual)}


def coarse_membership(problem: FineProblem, n_coarse: int) -> np.ndarray:
    """Номер крупного квадрата iy*n_coarse+ix; проверяет, что его рёбра разрешены."""
    if isinstance(n_coarse, bool) or int(n_coarse) != n_coarse or n_coarse < 1:
        raise ValueError("n_coarse должен быть положительным целым.")
    xy = problem.coordinates[problem.triangles]
    H = 4.0 / n_coarse
    centers = xy.mean(axis=1)
    ij = np.clip(np.floor((centers + 2) / H).astype(int), 0, n_coarse - 1)
    lower = -2 + H * ij
    tol = 2e-8
    if np.any(xy < lower[:, None, :] - tol) or np.any(xy > lower[:, None, :] + H + tol):
        raise ValueError("Fine mesh пересекает coarse edges. Постройте её из gmsfem/geo/scattering.geo.")
    membership = ij[:, 1] * n_coarse + ij[:, 0]
    if len(np.unique(membership)) != n_coarse**2:
        raise ValueError("В крупной сетке есть пустые ячейки.")
    return membership
