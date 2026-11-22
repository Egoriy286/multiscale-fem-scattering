"""Generate new direction groups with shared LU on the project's P1 reference mesh."""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.sparse.linalg import splu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.multiscale_sample_study.dataset import make_queries
from experiments.reference_sno_gmsfem.dataset import _reference_problem, _save_npz, _sha256, _write_json

def generate():
    data = HERE / 'data'
    data.mkdir(exist_ok=True)
    groups = make_queries((50, 75, 100, 125), 16, 16, seed=20261122)
    k = np.repeat(groups['k0'], 8)
    # Offset the entire angle stencil, keeping the angular problem identical at every k.
    theta = np.tile((np.arange(8) + 0.5) * np.pi / 4, len(groups['k0']))
    manifest = dict(k=k, theta=theta, split=np.repeat(groups['split'], 8),
                    group=np.repeat(np.arange(len(groups['k0'])), 8))
    query_path = data / 'queries.npz'
    if query_path.exists():
        with np.load(query_path) as old:
            assert all(np.array_equal(old[key], val) for key, val in manifest.items())
    else:
        _save_npz(query_path, **manifest)
    if (data / 'complete.json').exists():
        hashes = json.loads((data / 'complete.json').read_text())['sha256']
        assert all(_sha256(data / name) == digest for name, digest in hashes.items())
        return
    solver, geom = _reference_problem(data)
    a, n = geom['A_cells'][0], geom['n_cells'][0]
    base = solver.build_problem(a, n, 1., 0.)
    dims = (len(k), len(solver.coordinates))
    dest = data / 'fields.npy'
    fields = (np.load(dest, mmap_mode='r+') if dest.exists() else
              np.lib.format.open_memmap(dest, mode='w+', dtype=np.complex128, shape=dims))
    assert fields.shape == dims and fields.dtype == np.complex128
    state = data / 'residuals.npz'
    residuals = np.load(state)['values'] if state.exists() else np.full(len(k), np.nan)
    for g, freq in enumerate(groups['k0']):
        rows = slice(8*g, 8*(g+1))
        if np.isfinite(residuals[rows]).all():
            assert np.isfinite(fields[rows]).all()
            continue
        mat = (base.K - freq**2 * base.Mn - 1j*freq*base.B).tocsc()
        rhs = np.column_stack([solver.rhs(a, n, freq, t) for t in theta[rows]])
        sol = splu(mat).solve(rhs)
        rel = np.linalg.norm(mat @ sol - rhs, axis=0) / np.linalg.norm(rhs, axis=0)
        if not np.isfinite(rel).all() or max(rel) > 1.e-8:
            raise ValueError(f'Linear residual at k={freq}: {rel}')
        fields[rows] = sol.T
        fields.flush()
        residuals[rows] = rel
        _save_npz(state, values=residuals)
        if g % 16 == 0:
            print(f'{g+1}/{len(groups["k0"])} frequencies', flush=True)
    names = ('queries.npz', 'fields.npy', 'reference_problem.npz', 'etal.msh', 'residuals.npz')
    _write_json(data/'complete.json', dict(sha256={n: _sha256(data/n) for n in names},
                samples=len(k), residual_max=float(max(residuals))))
if __name__ == '__main__':
    generate()
