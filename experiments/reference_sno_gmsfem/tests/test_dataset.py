"""Small storage/split tests; no DOLFINx or production-size field allocation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reference_sno_gmsfem import dataset as module


@pytest.mark.parametrize("band", ["low", "high"])
def test_frequency_groups_and_reserved_queries(band):
    q = module.make_queries(band, 160, 8, 2026)
    assert len(q["frequencies"]) == (161 if band == "low" else 160)
    assert np.array_equal(q["theta"], module.make_queries(band, 160, 8, 2026)["theta"])
    for group in np.unique(q["frequency_id"]):
        rows = q["frequency_id"] == group
        assert len(np.unique(q["split"][rows])) == 1
        assert np.ptp(q["k0"][rows]) == 0
    for k in module.BANDS[band]:
        assert np.all(q["split"][q["k0"] == k] == 2)
    if band == "low":
        anchor = (q["k0"] == 1.) & (q["theta"] == 0.)
        assert anchor.sum() == 1
        assert q["split"][anchor].item() == 2
        assert np.all(q["split"][q["k0"] == 1.] == 2)
    else:
        assert q["k0"].min() == 2. and q["k0"].max() == 4.


def test_contrast_rhs_matches_full_quadrature():
    from neural_operator.fem import HelmholtzSolver
    rng = np.random.default_rng(42)
    solver = SimpleNamespace(coordinates=np.zeros((5, 2)),
                             triangles=np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]),
                             qpoints=rng.normal(size=(3, 4, 2)),
                             qweights=rng.uniform(size=4),
                             areas=rng.uniform(size=3),
                             phi=rng.uniform(size=(4, 3)),
                             gradients=rng.normal(size=(3, 3, 2)))
    alpha, n = np.array([1., .2, 1.]), np.array([1., 10., 1.])
    for k, theta in ((1., 0.), (4., .173)):
        assert np.array_equal(module._contrast_rhs(solver, alpha, n, k, theta),
                              HelmholtzSolver.rhs(solver, alpha, n, k, theta))
    assert np.array_equal(module._contrast_rhs(solver, np.ones(3), np.ones(3), 4., .3),
                          np.zeros(5, dtype=complex))


@pytest.fixture
def tiny_backend(monkeypatch):
    problem = dict(coordinates=np.array([[0., 0.], [1., 0.], [0., 1.]]),
                   triangles=np.array([[0, 1, 2]], dtype=np.int32),
                   cell_tags=np.array([2], dtype=np.int32), circles=np.array([[.2, .2, .1, 2.]]),
                   A=np.full((1, 3), .2), n=np.full((1, 3), 10.),
                   A_cells=np.array([[.2]]), n_cells=np.array([[10.]]),
                   anchor_scattered=np.array([1., 2., 3.], dtype=complex))

    class Solver:
        coordinates = problem["coordinates"]
        triangles = problem["triangles"]
        calls = 0
        fail_on = None

        def solve(self, alpha, n, k, theta):
            self.calls += 1
            if self.calls == self.fail_on:
                raise RuntimeError("simulated interruption")
            return np.arange(1., 4.) * k * np.exp(1j * theta)

    solver = Solver()

    def prepare(directory):
        (directory / "etal.msh").write_bytes(b"small test mesh")
        module._save_npz(directory / "reference_problem.npz", **problem)
        return solver, problem

    monkeypatch.setattr(module, "_reference_problem", prepare)
    import hashlib
    monkeypatch.setattr(module, "REFERENCE_SHA256", hashlib.sha256(b"small test mesh").hexdigest())
    return solver


def test_native_storage_is_compact_readonly_and_rejects_mismatched_recipe(tmp_path, tiny_backend):
    data = module.generate_dataset(tmp_path / "high", "high", 10, 2)
    assert isinstance(data.u_scat, np.memmap)
    assert data.u_scat.flags.writeable is False
    assert data.A.flags.writeable is False and data.A.strides[0] == 0
    assert data.u_scat.shape == (20, 3)
    with np.load(data.directory / "reference_problem.npz") as saved:
        assert saved["A"].shape == (1, 3) and saved["A_cells"].shape == (1, 1)
    with pytest.raises(ValueError, match="different configuration"):
        module.generate_dataset(data.directory, "low", 10, 2)
    # A completed, matching dataset performs no new FEM solves.
    calls = tiny_backend.calls
    module.generate_dataset(data.directory, "high", 10, 2)
    assert tiny_backend.calls == calls


def test_resume_reuses_only_committed_frequencies(tmp_path, tiny_backend):
    directory = tmp_path / "high"
    tiny_backend.fail_on = 3
    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.generate_dataset(directory, "high", 10, 2)
    assert not (directory / "dataset_info.json").exists()
    assert np.load(directory / "completed.npy").sum() == 1
    with pytest.raises(FileNotFoundError):
        module.load_dataset(directory)
    tiny_backend.fail_on = None
    result = module.generate_dataset(directory, "high", 10, 2)
    assert tiny_backend.calls == 21  # 20 successful rows and one failed call.
    assert np.all(np.isfinite(result.u_scat))
    assert json.loads((directory / "dataset_info.json").read_text())["resumed_frequency_groups"] == 1


def test_metadata_corruption_is_rejected(tmp_path, tiny_backend):
    directory = tmp_path / "high"
    module.generate_dataset(directory, "high", 10, 2)
    with (directory / "samples.npz").open("ab") as stream:
        stream.write(b"unexpected trailing bytes")
    with pytest.raises(ValueError, match="samples.npz checksum"):
        module.load_dataset(directory)
