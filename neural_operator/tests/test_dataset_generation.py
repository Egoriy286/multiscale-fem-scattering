"""Recipe planning is cheap; one pipeline preserves the established experiment."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from neural_operator.datasets import (
    DatasetConfig, StaticDatasetConfig, generate_dataset, load_config,
    load_dataset, plan_dataset,
)


def test_planning_imports_no_fem_runtime():
    code = '''
import json, sys
from neural_operator.datasets import plan_dataset
plan = plan_dataset()
assert plan['medium_mode'] == 'static'
assert plan['samples'] == 8000 and plan['media'] == 1
assert plan['config']['fixed_medium_id'] == 0
assert plan['config']['seed'] == 2026
assert not {'neural_operator.fem', 'gmsh', 'dolfinx', 'torch'} & sys.modules.keys()
json.dumps(plan, allow_nan=False)
'''
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('config', [
    StaticDatasetConfig(n_frequencies=6, n_directions=2),
    DatasetConfig(n_media=6, n_directions=2),
])
def test_plan_matches_group_split_and_json_roundtrip(tmp_path, config):
    from neural_operator.data import group_split
    plan = plan_dataset(config)
    ids = np.repeat(np.arange(plan['split_groups']), config.n_directions)
    split = group_split(ids, config.split_fractions, config.seed)
    assert [part['samples'] for part in plan['splits']] == np.bincount(split, minlength=3).tolist()
    assert plan['samples'] == 12
    assert sum(part['groups'] for part in plan['splits']) == 6
    path = tmp_path/'config.json'
    path.write_text(json.dumps(plan['config']))
    assert load_config(path) == config
    assert load_config(tmp_path) == config


@pytest.mark.parametrize('change', [
    dict(n_directions=True), dict(n_directions=1.5), dict(n_coarse=0),
    dict(h=float('inf')), dict(h=True), dict(seed=-1), dict(seed=True),
    dict(k_range=(1., float('inf'))), dict(k_range=('1', '2')),
    dict(k_range=(2., 1.)), dict(a_range=(0., 1.)),
    dict(n_range=(1., float('nan'))), dict(active_range=(1.5, 2)),
    dict(active_range=(0, 25)), dict(active_range=(1,)),
    dict(split_fractions=(.5, .5)), dict(split_fractions=(.5, .5, .5)),
    dict(split_fractions=(float('nan'), .2, .8)),
    dict(split_fractions=('0.6', '0.2', '0.2')),
])
@pytest.mark.parametrize('config_type', [StaticDatasetConfig, DatasetConfig])
def test_invalid_recipes_have_no_disk_effects(tmp_path, config_type, change):
    destination = tmp_path/'must-not-exist'
    with pytest.raises(ValueError):
        generate_dataset(destination, replace(config_type(), **change), progress=None)
    assert not destination.exists()


@pytest.mark.parametrize('config', [
    StaticDatasetConfig(n_frequencies=2),
    StaticDatasetConfig(fixed_medium_id=-1),
    StaticDatasetConfig(k_range=(1., np.nextafter(1., 2.))),
    DatasetConfig(n_media=0),
])
def test_mode_specific_validation(config):
    with pytest.raises(ValueError):
        plan_dataset(config)


@pytest.mark.parametrize('static', [False, True])
def test_pipeline_rng_storage_and_legacy_hooks(tmp_path, monkeypatch, static):
    import neural_operator.fem as fem

    class TinySolver:
        coordinates = np.array([[0., 0.], [1., 0.], [0., 1.]])
        triangles = np.tile(np.array([[0, 1, 2]], dtype=np.int32), (25, 1))
        cell_tags = np.arange(1, 26, dtype=np.int32)

        def __init__(self, path):
            pass

        def nodal_coefficients(self, values):
            return np.full(3, np.mean(values))

        def solve(self, alpha, n, k, theta):
            return np.full(3, k+1j*theta)

    def mesh(path, **kwargs):
        Path(path).write_text('fixture mesh')

    monkeypatch.setattr(fem, 'generate_mesh', mesh)
    monkeypatch.setattr(fem, 'HelmholtzSolver', TinySolver)
    config = (StaticDatasetConfig(n_frequencies=6, n_directions=2) if static
              else DatasetConfig(n_media=3, n_directions=2))
    messages = []
    dataset = generate_dataset(tmp_path, config, progress=messages.append)
    # These direction hashes were captured from the two pre-refactor generators.
    expected_theta = ('1b35fd923772b569c9c5b578cd5de62b96696d8da3293b3849cb9db781ec1ea1' if static
                      else 'd82d0ccccdce9f6e7954ed6b71b3081706d3e4c7ff9fa219e6f023e7b1e68fad')
    assert hashlib.sha256(dataset.theta.tobytes()).hexdigest() == expected_theta
    alpha, n = dataset.A_cells[0], dataset.n_cells[0]
    assert np.flatnonzero(alpha != 1).tolist() == [9, 12, 15]
    assert np.array_equal(alpha[alpha != 1], [.6928657451928999, .8660424265918046, .7818095086607539])
    assert np.array_equal(n[n != 1], [2.59454354016658, 1.6962110559323502, 1.8445453608693927])
    assert messages[0].endswith(f'0/{len(dataset.k0)} fields')
    with np.load(tmp_path/'fem_dataset.npz') as archive:
        assert archive['A_cells'].shape[0] == (1 if static else len(dataset.k0))
        assert ('frequency_id' in archive) == static
        assert ('split_strategy' in archive) == static
    loaded = load_dataset(tmp_path)
    for name in ('A', 'n', 'A_cells', 'n_cells', 'u_scat', 'theta', 'k0', 'split'):
        assert np.array_equal(getattr(dataset, name), getattr(loaded, name))
    if static:
        assert dataset.A.strides[0] == 0
    assert not (tmp_path/'fem_dataset.tmp.npz').exists()

    def no_mesh(*args, **kwargs):
        raise AssertionError('A matching recipe must use the cache')

    monkeypatch.setattr(fem, 'generate_mesh', no_mesh)
    cached = fem.generate_dataset(tmp_path, config, progress=None)
    assert np.array_equal(cached.u_scat, dataset.u_scat)
    with pytest.raises(ValueError, match='different configuration'):
        generate_dataset(tmp_path, replace(config, seed=config.seed+1), progress=None)
