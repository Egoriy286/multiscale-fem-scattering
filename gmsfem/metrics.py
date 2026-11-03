"""Интегральные ошибки: Re, Im, |u| и комплексная L²-норма.

Семиточечная квадратура Dunavant степени 5; падающее поле вычисляется
аналитически в точках квадратуры. |u| берётся ПОСЛЕ интерполяции комплексного
поля. Это отличается от интеграла P1-интерполянта узловых модулей.
"""

from __future__ import annotations

import numpy as np

from .fem import FineProblem


BARYCENTRIC = np.array([
    [1/3, 1/3, 1/3],
    [.059715871789770, .470142064105115, .470142064105115],
    [.470142064105115, .059715871789770, .470142064105115],
    [.470142064105115, .470142064105115, .059715871789770],
    [.797426985353087, .101286507323456, .101286507323456],
    [.101286507323456, .797426985353087, .101286507323456],
    [.101286507323456, .101286507323456, .797426985353087],
])
WEIGHTS = np.array([.225, .132394152788506, .132394152788506, .132394152788506,
                    .125939180544827, .125939180544827, .125939180544827])


def _summaries(approx: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> dict:
    result = {}
    for name, transform in (("real", np.real), ("imag", np.imag), ("abs", np.abs), ("complex", lambda x: x)):
        a, b = transform(approx), transform(reference)
        numerator2 = float(np.sum(weights * np.abs(a-b)**2))
        norm_approx2 = float(np.sum(weights * np.abs(a)**2))
        norm_ref2 = float(np.sum(weights * np.abs(b)**2))
        result[f"l2_error_{name}"] = np.sqrt(max(0., numerator2))
        # При нулевой норме относительная ошибка математически не определена.
        for convention, denominator2 in (("paper", norm_approx2), ("ref", norm_ref2)):
            result[f"e_{name}_{convention}"] = (100*np.sqrt(max(0., numerator2)/denominator2)
                                                  if denominator2 > 1e-28 else float("nan"))
    return result


def relative_errors(problem: FineProblem, approx: np.ndarray, reference: np.ndarray, *, field: str = "total") -> dict:
    """Обе нормы из статьи и обычные reference-нормы, в процентах, на одной сетке."""
    if field not in ("total", "scattered"):
        raise ValueError("field должен быть total или scattered.")
    aq = np.asarray(approx)[problem.triangles] @ BARYCENTRIC.T
    rq = np.asarray(reference)[problem.triangles] @ BARYCENTRIC.T
    if field == "total":
        points = np.einsum("qj,cjd->cqd", BARYCENTRIC, problem.coordinates[problem.triangles])
        incident = np.exp(1j*problem.k*(points @ np.asarray(problem.direction)))
        aq, rq = aq+incident, rq+incident
    return _summaries(aq, rq, problem.areas[:, None]*WEIGHTS[None, :])


class P1Evaluator:
    """Кусочно-линейное поле на произвольной треугольной сетке в физических точках.

    Matplotlib TriFinder определяет содержащий треугольник; затем используется
    его точная барицентрическая интерполяция. Это не nearest-neighbour перенос.
    """

    def __init__(self, problem: FineProblem, values: np.ndarray):
        from matplotlib.tri import Triangulation

        self.triangulation = Triangulation(problem.coordinates[:, 0], problem.coordinates[:, 1], problem.triangles)
        self.finder = self.triangulation.get_trifinder()
        self.lower = problem.coordinates.min(axis=0)
        self.upper = problem.coordinates.max(axis=0)
        vertices = problem.coordinates[problem.triangles]
        mapping = np.concatenate((np.ones((*vertices.shape[:2], 1)), vertices), axis=2)
        self.coefficients = np.linalg.solve(mapping, np.asarray(values)[problem.triangles, None])[..., 0]

    def __call__(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        shape = points.shape[:-1]
        points = points.reshape(-1, 2)
        cells = self.finder(points[:, 0], points[:, 1])
        if np.any(cells < 0):
            # TriFinder иногда отвергает точные граничные вершины из-за
            # round-off. Сдвигается ТОЛЬКО точка поиска треугольника;
            # значение P1 вычисляется в исходной физической точке.
            missing = np.flatnonzero(cells < 0)
            delta = 1e-10 * max(float(np.max(self.upper-self.lower)), 1.)
            candidates = points[missing]
            in_bounds = np.all((candidates >= self.lower-delta) & (candidates <= self.upper+delta), axis=1)
            on_boundary = np.any((np.abs(candidates-self.lower) <= delta) | (np.abs(candidates-self.upper) <= delta), axis=1)
            eligible = missing[in_bounds & on_boundary]
            query = np.clip(points[eligible], self.lower+delta, self.upper-delta)
            cells[eligible] = self.finder(query[:, 0], query[:, 1])
        if np.any(cells < 0):
            raise ValueError("Точки сравнения вне исходной области; межсеточная ошибка не определена.")
        coefficients = self.coefficients[cells]
        return (coefficients[:, 0] + np.sum(coefficients[:, 1:]*points, axis=1)).reshape(shape)


def errors_on_reference(
    source: FineProblem, approx: np.ndarray, reference: FineProblem, reference_values: np.ndarray,
    *, field: str = "total", chunk_cells: int = 4096,
) -> dict:
    """Ошибка на независимой более мелкой reference mesh.

    Оба поля интегрируются в точках квадратуры REFERENCE-сетки. Это сравнение
    содержит ошибки редукции, P1-дискретизации и полигонализации окружностей.
    Оно не заменяет чистую проверку GMsFEM на общей тонкой сетке.
    """
    if field not in ("total", "scattered"):
        raise ValueError("field должен быть total или scattered.")
    if source.k != reference.k or source.case != reference.case or source.direction != reference.direction:
        raise ValueError("Для межсеточного сравнения необходимы одинаковые k, материалы и падающая волна.")
    evaluate = P1Evaluator(source, approx)
    # Массивы порядка 7*N_ref небольшие; геометрический поиск выполняется блоками.
    approx_chunks, ref_chunks, weight_chunks = [], [], []
    for first in range(0, len(reference.triangles), chunk_cells):
        cells = slice(first, first+chunk_cells)
        triangles = reference.triangles[cells]
        points = np.einsum("qj,cjd->cqd", BARYCENTRIC, reference.coordinates[triangles])
        aq = evaluate(points)
        rq = np.asarray(reference_values)[triangles] @ BARYCENTRIC.T
        if field == "total":
            incident = np.exp(1j*reference.k*(points @ np.asarray(reference.direction)))
            aq, rq = aq+incident, rq+incident
        approx_chunks.append(aq.ravel())
        ref_chunks.append(rq.ravel())
        weight_chunks.append((reference.areas[cells, None]*WEIGHTS[None, :]).ravel())
    return _summaries(np.concatenate(approx_chunks), np.concatenate(ref_chunks), np.concatenate(weight_chunks))
