"""One mesh/material/solve/storage pipeline for static and variable recipes."""
import json
from pathlib import Path

import numpy as np

from ..data import Dataset, group_split, load_dataset
from .config import StaticDatasetConfig, config_dict, plan_dataset


def _draw_medium(solver, config, medium_id):
    """Keep the legacy draw order, including the frequency draw for static data."""
    rng = np.random.default_rng([config.seed, medium_id])
    k = rng.uniform(*config.k_range)
    alpha, n = np.ones(len(solver.triangles)), np.ones(len(solver.triangles))
    chosen = rng.choice(24, size=rng.integers(config.active_range[0], config.active_range[1]+1), replace=False)
    for cid in chosen:
        cells = solver.cell_tags == cid+2
        alpha[cells] = np.exp(rng.uniform(*np.log(config.a_range)))
        n[cells] = np.exp(rng.uniform(*np.log(config.n_range)))
    return alpha, n, k, rng


def _report(progress, static, completed, count, directions):
    if progress is not None:
        label, unit = ('Static FEM dataset', 'frequencies') if static else ('FEM dataset', 'media')
        message = f'{label}: {completed}/{count} {unit}, {completed*directions}/{count*directions} fields'
        if progress is print:
            print(message, flush=True)
        else:
            progress(message)


def generate_static_dataset(output_dir, config=StaticDatasetConfig(), progress=print):
    """Compatibility spelling for explicitly static generation."""
    if not isinstance(config, StaticDatasetConfig):
        raise TypeError('Expected StaticDatasetConfig')
    return generate_dataset(output_dir, config, progress)


def generate_dataset(output_dir, config=StaticDatasetConfig(), progress=print, *, _backend=None):
    """Generate either recipe; validate completely before touching disk or FEM."""
    plan = plan_dataset(config)
    static = isinstance(config, StaticDatasetConfig)
    directory = Path(output_dir).expanduser().resolve()
    cfg = config_dict(config)
    config_path = directory/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Output directory contains a different configuration or medium mode')
    if (directory/'fem_dataset.npz').exists():
        if not config_path.exists():
            raise ValueError('Existing dataset has no configuration')
        loader = load_dataset if _backend is None else _backend.load_dataset
        return loader(directory, mmap=True) if static else loader(directory)
    # The backend is resolved only after validation and the cache check. Passing
    # it explicitly keeps legacy fem-module monkeypatches working for callers.
    backend = _backend
    if backend is None:
        from .. import fem as backend
    from gmsfem.geometry import inclusion_circles
    directory.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2)+'\n')
    backend.generate_mesh(directory/'etal.msh', h=config.h, n_coarse=config.n_coarse)
    solver = backend.HelmholtzSolver(directory/'etal.msh')
    circles = inclusion_circles().copy()
    circles[:, 3] = np.arange(2, 26)
    np.savez_compressed(directory/'geometry.npz', dof_coords=solver.coordinates,
                        triangles=solver.triangles, cell_tags=solver.cell_tags,
                        etal_circles=circles)
    count, ns = plan['split_groups'], plan['samples']
    nn, nc = len(solver.coordinates), len(solver.triangles)
    material_rows = 1 if static else ns
    materials = {name: np.empty((material_rows, width), dtype=np.float64)
                 for name, width in (('A', nn), ('n', nn), ('A_cells', nc), ('n_cells', nc))}
    arrays = dict(u_scat=np.empty((ns, nn), dtype=np.complex128),
                  theta=np.empty(ns, dtype=np.float64),
                  k0=np.empty(ns, dtype=np.float64),
                  medium_id=np.empty(ns, dtype=np.int32))
    group_ids = np.repeat(np.arange(count, dtype=np.int32), config.n_directions)
    if static:
        frequencies = np.linspace(*config.k_range, count)
        alpha, n, _, _ = _draw_medium(solver, config, config.fixed_medium_id)
        arrays['frequency_id'] = group_ids
        arrays['medium_id'].fill(config.fixed_medium_id)
    _report(progress, static, 0, count, config.n_directions)
    for group in range(count):
        if static:
            k = frequencies[group]
            rng = np.random.default_rng([config.seed, group, 2])
        else:
            alpha, n, k, rng = _draw_medium(solver, config, group)
        first = group*config.n_directions
        rows = slice(first, first+config.n_directions)
        if not static or group == 0:
            material_slice = slice(0, 1) if static else rows
            materials['A'][material_slice] = solver.nodal_coefficients(alpha)
            materials['n'][material_slice] = solver.nodal_coefficients(n)
            materials['A_cells'][material_slice] = alpha
            materials['n_cells'][material_slice] = n
        arrays['k0'][rows] = k
        if not static:
            arrays['medium_id'][rows] = group
        offset = rng.uniform(0, 2*np.pi/config.n_directions)
        for direction, theta in enumerate(offset+2*np.pi*np.arange(config.n_directions)/config.n_directions):
            sample = first+direction
            arrays['theta'][sample] = theta
            arrays['u_scat'][sample] = solver.solve(alpha, n, k, theta)
        if group == 0 or (group+1)%25 == 0 or group+1 == count:
            _report(progress, static, group+1, count, config.n_directions)
    arrays['split'] = group_split(group_ids, config.split_fractions, config.seed)
    extra = dict(split_strategy=np.asarray('frequency')) if static else {}
    # Atomic replacement prevents an interrupted save becoming a valid cache.
    temporary = directory/'fem_dataset.tmp.npz'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **materials, **arrays, **extra)
    temporary.replace(directory/'fem_dataset.npz')
    if static:
        materials = {name: np.broadcast_to(value, (ns, value.shape[1]))
                     for name, value in materials.items()}
    return Dataset(**materials, **arrays, coordinates=solver.coordinates,
                   triangles=solver.triangles, cell_tags=solver.cell_tags,
                   circles=circles, directory=directory,
                   split_strategy=plan['split_strategy'])
