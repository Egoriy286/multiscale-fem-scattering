"""The static problem varies incident frequency and direction, never material."""
from dataclasses import replace
import json
import numpy as np
import pytest


@pytest.fixture(scope='module')
def static_pair(tmp_path_factory):
    pytest.importorskip('dolfinx')
    from neural_operator.fem import StaticDatasetConfig,generate_static_dataset
    config = StaticDatasetConfig(n_frequencies=6,n_directions=8,h=.3,n_coarse=2)
    low = generate_static_dataset(tmp_path_factory.mktemp('static_low'),config,progress=None)
    high = generate_static_dataset(tmp_path_factory.mktemp('static_high'),replace(config,k_range=(4.,8.)),progress=None)
    return low,high,config


def test_static_medium_matches_dynamic_and_both_bands(static_pair,tmp_path):
    from neural_operator.fem import DatasetConfig,generate_dataset
    low,high,config = static_pair
    dynamic = generate_dataset(tmp_path,DatasetConfig(n_media=1,n_directions=1,h=config.h,n_coarse=config.n_coarse),progress=None)
    for name in ('A','n','A_cells','n_cells'):
        a,b = getattr(low,name),getattr(high,name)
        assert np.array_equal(a,b)
        assert np.array_equal(a,np.broadcast_to(a[0],a.shape))
        assert np.array_equal(a[0],getattr(dynamic,name)[0])
        assert a.strides[0] == 0
    assert np.array_equal(low.coordinates,high.coordinates)
    assert np.array_equal(low.triangles,high.triangles)
    assert np.unique(low.medium_id).tolist() == [0]
    assert low.split_strategy == 'frequency'


def test_frequency_split_groups_all_directions(static_pair):
    low,_,config = static_pair
    assert np.array_equal(low.split_group_ids,low.frequency_id)
    assert np.array_equal(low.frequency_id,np.repeat(np.arange(config.n_frequencies),config.n_directions))
    assert np.array_equal(low.k0,np.repeat(np.linspace(*config.k_range,config.n_frequencies),config.n_directions))
    groups = []
    for indices in low.split_indices():
        groups.append(set(low.frequency_id[indices]))
    assert all(groups)
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    for fid in np.unique(low.frequency_id):
        mask = low.frequency_id == fid
        assert mask.sum() == config.n_directions
        assert len(np.unique(low.split[mask])) == 1
        theta = low.theta[mask]
        assert np.allclose(np.diff(theta),2*np.pi/config.n_directions)


def test_compact_storage_load_cache_and_resolve(static_pair,monkeypatch):
    from neural_operator.data import load_dataset
    import neural_operator.fem as module
    low,_,config = static_pair
    with np.load(low.directory/'fem_dataset.npz') as archive:
        for name in ('A','n','A_cells','n_cells'):
            assert archive[name].shape == (1,getattr(low,name).shape[1])
        assert archive['split_strategy'].item() == 'frequency'
    assert json.loads((low.directory/'config.json').read_text())['medium_mode'] == 'static'
    loaded = load_dataset(low.directory)
    for name in ('A','n','A_cells','n_cells','u_scat','theta','frequency_id','split'):
        assert np.array_equal(getattr(low,name),getattr(loaded,name))
    def unexpected_mesh(*args,**kwargs):
        raise AssertionError('Cached static dataset must not regenerate')
    monkeypatch.setattr(module,'generate_mesh',unexpected_mesh)
    cached = module.generate_static_dataset(low.directory,config,progress=None)
    assert np.array_equal(cached.u_scat,low.u_scat)
    solver = module.HelmholtzSolver(low.directory/'etal.msh')
    index = config.n_directions+3
    actual = solver.solve(low.A_cells[index],low.n_cells[index],low.k0[index],low.theta[index])
    assert low.relative_l2(low.u_scat[index],actual) < 1e-12


@pytest.mark.parametrize('change',[dict(n_frequencies=2),dict(k_range=(1.,1.)),dict(k_range=(0.,1.)),dict(n_directions=0)])
def test_static_config_invalid(tmp_path,change):
    from neural_operator.fem import StaticDatasetConfig,generate_static_dataset
    with pytest.raises(ValueError):
        generate_static_dataset(tmp_path,replace(StaticDatasetConfig(),**change),progress=None)
