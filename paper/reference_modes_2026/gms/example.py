"""Export the paper's visual comparison case from the GMsFEM frequency cache."""
from pathlib import Path

import numpy as np

from .run import (
    DEFAULT_ROOT, FINE_MESH, FINE_SHA256, METHODS, P1Transfer,
    HelmholtzSolver, _checked_mesh, _save_npz, cache_path, digest, inputs, load_cache,
)

ROW = 192


def export(root=DEFAULT_ROOT):
    k, theta, selected, frequencies = inputs(root)
    if ROW not in selected:
        raise ValueError("The paper's example row is missing from the test split")
    query = root / "data" / "queries.npz"
    solver_source = Path(__file__).with_name("run.py")
    key = digest(query)[:16] + "-" + digest(solver_source)[:16]
    cache = root / "gms" / ".array_cache" / key
    group = int(np.searchsorted(frequencies, k[ROW]))
    indices = selected[k[selected] == k[ROW]]
    source = HelmholtzSolver(_checked_mesh(FINE_MESH, FINE_SHA256))
    values, _, _ = load_cache(cache_path(cache, group), indices, len(source.coordinates))
    with np.load(root / "data" / "reference_problem.npz", allow_pickle=False) as geometry:
        transfer = P1Transfer(source.coordinates, source.triangles, geometry["coordinates"])
    field = transfer.apply(values[METHODS.index((2, 16)), np.searchsorted(indices, ROW)])
    _save_npz(root / "gms" / "example.npz", field=field, row=ROW, k=k[ROW], theta=theta[ROW])


if __name__ == "__main__":
    export()
