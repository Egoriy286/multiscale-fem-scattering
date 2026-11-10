from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from experiments.reference_sno_gmsfem import baselines
from gmsfem.metrics import BARYCENTRIC, P1Evaluator, errors_on_reference


def _geometry(coords, triangles):
    vertices = coords[triangles]
    mapping = np.concatenate((np.ones((*vertices.shape[:2], 1)), vertices), axis=2)
    return SimpleNamespace(coordinates=coords, triangles=triangles,
                           areas=np.abs(np.linalg.det(mapping))/2,
                           k=1., direction=(1., 0.), case="notebook")


def test_direct_quadrature_preserves_original_piecewise_field():
    """Crossed diagonals expose the extra approximation in node-only transfer."""
    coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    original = _geometry(coords, np.array([[0, 1, 2], [0, 2, 3]]))
    reference = _geometry(coords, np.array([[0, 1, 3], [1, 2, 3]]))
    values = np.array([0., 0., 1.+1j, 0.])
    labels = np.full(4, 1.+.3j, dtype=complex)
    dataset = SimpleNamespace(coordinates=coords, triangles=reference.triangles,
                              theta=np.array([0.]), k0=np.array([1.]),
                              u_scat=labels[None])
    points = np.einsum("qj,cjd->cqd", BARYCENTRIC, coords[reference.triangles])
    quadrature = P1Evaluator(original, values)(points)
    actual = baselines.metrics_on_reference(dataset, 0, values,
                                            quadrature_prediction=quadrature)
    nodal = baselines.metrics_on_reference(dataset, 0, values)
    for field in ("scattered", "total"):
        expected = errors_on_reference(original, values, reference, labels, field=field)
        for metric, number in expected.items():
            assert actual[f"{field}_{metric}"] == pytest.approx(number, abs=1e-12)
    assert abs(actual["error_fem_l2"]-nodal["error_fem_l2"]) > .01


def test_baselines_cache_frequency_operators_and_type1_spaces(tmp_path, monkeypatch):
    import neural_operator.fem
    import gmsfem.offline

    coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    k = np.array([1., 1., 2., 2.])
    theta = np.array([0., .1, .2, .3])

    @dataclass
    class Problem:
        operator: object
        rhs: np.ndarray
        direction: tuple
        incident: np.ndarray

    class Solver:
        builds = 0

        def __init__(self, path):
            self.coordinates, self.triangles = coords, triangles
            self.cell_tags = np.array([1, 2])

        def rhs(self, a, n, frequency, angle):
            return np.full(4, frequency+1j*angle)

        def build_problem(self, a, n, frequency, angle):
            Solver.builds += 1
            return Problem(sparse.eye(4, format="csr", dtype=complex),
                           self.rhs(a, n, frequency, angle), (1., 0.), np.ones(4))

    calls = []

    def space(problem, *, n_coarse, max_modes, basis_type, progress):
        calls.append((n_coarse, basis_type))
        return SimpleNamespace(build_seconds=.1,
                               prolongation=lambda m: sparse.eye(4, format="csc"))

    monkeypatch.setattr(neural_operator.fem, "HelmholtzSolver", Solver)
    monkeypatch.setattr(gmsfem.offline, "build_offline_space", space)
    monkeypatch.setattr(baselines, "_checked_mesh", lambda path, expected: Path(path))
    dataset = SimpleNamespace(coordinates=coords, triangles=triangles,
                              k0=k, theta=theta, split=np.full(4, 2),
                              frequency_id=np.array([0, 0, 1, 1]),
                              A_cells=np.tile([1., .2], (4, 1)),
                              n_cells=np.tile([1., 10.], (4, 1)),
                              u_scat=np.stack([np.full(4, f+1j*t) for f, t in zip(k, theta)]))
    result = baselines.evaluate_baselines(dataset, tmp_path, np.arange(4),
                                         coarse_grids=(10,), modes=1,
                                         reproduce=False, save_fields=False,
                                         progress=None)
    assert Solver.builds == 2
    assert calls.count((10, 1)) == 1
    assert calls.count((10, 2)) == 2
    assert result["metadata"]["offline_build_counts"] == {"n10_type1": 1, "n10_type2": 2}
    assert result["metadata"]["unique_cost_totals"]["offline_seconds"] == pytest.approx(.3)
    assert len(result["rows"]) == 12
    assert max(r["scattered_e_complex_ref"] for r in result["rows"]) < 1e-12
    assert max(r["total_e_complex_ref"] for r in result["rows"]) < 1e-12
    assert (tmp_path/"baselines.csv").exists()
    assert (tmp_path/"baselines.json").exists()
    dataset.split[0] = 0
    with pytest.raises(ValueError, match="held-out"):
        baselines.evaluate_baselines(dataset, tmp_path, [0], reproduce=False)
