"""Query separation, zero-copy views, and crash recovery without production FEM."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from neural_operator.data import Dataset
from experiments.multiscale_sample_study import dataset as module


def test_default_plan_nested_counts_and_physical_separation():
    plan = module.make_queries()
    assert len(plan['k0']) == 1200
    np.testing.assert_array_equal(np.bincount(plan['split']), [1000, 100, 100])
    assert len(np.unique(plan['k0'])) == 1200
    assert len(np.unique(plan['frequency_id'])) == 1200
    assert np.all(plan['theta'] == 0)
    assert np.all(plan['medium_id'] == 0)
    previous = set()
    for size in (400, 600, 800, 1000):
        train = plan['k0'][200:200+size]
        assert previous <= set(train)
        assert train.min() == .1 and train.max() == 4.
        # Each of 20 equal physical bins receives a near-uniform prefix.
        counts, _ = np.histogram(train, bins=np.linspace(.1, 4., 21))
        assert counts.max()-counts.min() <= 8
        assert np.max(np.diff(np.sort(train))) < 2.2*3.9/(size-1)
        assert not set(train) & set(plan['k0'][:200])
        previous = set(train)
    for values in (plan['k0'][:100], plan['k0'][100:200]):
        assert np.all(np.diff(values) > 0)
        assert values.min() > .1 and values.max() < 4.
    assert module.query_hash(plan) == module.query_hash(module.make_queries())
    assert module.query_hash(plan) != module.query_hash(module.make_queries(seed=9))


@pytest.mark.parametrize('settings', [dict(train_sizes=()), dict(train_sizes=(4, 4)),
    dict(train_sizes=(6, 4)), dict(train_sizes=(True,)), dict(n_test=0),
    dict(n_validation=1.5), dict(seed=-1)])
def test_invalid_plan(settings):
    with pytest.raises(ValueError):
        module.make_queries(**settings)


def _small_data():
    plan = module.make_queries((4, 6), 2, 2)
    fields = np.arange(30).reshape(10, 3).astype(np.complex128)
    return Dataset(A=np.ones_like(fields.real), n=np.ones_like(fields.real),
                   u_scat=fields, coordinates=np.array([[0., 0.], [1., 0.], [0., 1.]]),
                   triangles=np.array([[0, 1, 2]]),
                   **{name: plan[name] for name in ('k0', 'theta', 'medium_id', 'split', 'frequency_id')},
                   split_strategy='frequency')


def test_training_view_preserves_backing_and_original_ids():
    source = _small_data()
    small = module.training_view(source, 4)
    large = module.training_view(source, 6)
    assert np.shares_memory(small.u_scat, source.u_scat)
    assert np.shares_memory(small.A, source.A)
    np.testing.assert_array_equal(small.original_indices, np.arange(8))
    np.testing.assert_array_equal(small.frequency_id, source.frequency_id[:8])
    np.testing.assert_array_equal(small.split_indices()[0], [4, 5, 6, 7])
    np.testing.assert_array_equal(small.k0[:4], large.k0[:4])
    with pytest.raises(ValueError):
        module.training_view(source, 7)


def test_generation_resumes_only_uncommitted_queries_and_checks_hashes(tmp_path, monkeypatch):
    calls = []
    fail = [True]

    def solve(a, n, k, theta):
        calls.append(k)
        if fail[0] and len(calls) == 3:
            raise RuntimeError('synthetic interrupted solve')
        return np.full(3, complex(k, theta), dtype=np.complex128)

    def problem(directory):
        (directory/'etal.msh').write_bytes(b'synthetic mesh')
        module._save_npz(directory/'reference_problem.npz', marker=np.array([1]))
        return SimpleNamespace(coordinates=np.zeros((3, 2)), triangles=np.zeros((1, 3)), solve=solve), dict(A_cells=np.ones((1, 1)), n_cells=np.ones((1, 1)))

    monkeypatch.setattr(module, '_reference_problem', problem)
    monkeypatch.setattr(module, 'load_dataset', lambda path: path)
    settings = dict(train_sizes=(4, 6), n_validation=2, n_test=2)
    with pytest.raises(RuntimeError, match='interrupted'):
        module.generate_dataset(tmp_path, **settings)
    np.testing.assert_array_equal(np.load(tmp_path/'completed.npy'), [True, True]+[False]*8)
    first = np.load(tmp_path/'u_scat.npy', mmap_mode='r')[:2].copy()
    fail[0] = False
    module.generate_dataset(tmp_path, **settings)
    assert len(calls) == 11  # Ten distinct labels plus the interrupted attempt.
    np.testing.assert_array_equal(np.load(tmp_path/'u_scat.npy')[:2], first)
    assert np.load(tmp_path/'completed.npy').all()
    info = json.loads((tmp_path/'dataset_info.json').read_text())
    assert info['new_fem_labels'] == 10 and info['inherited_fem_labels'] == 0
    assert info['resumed_frequency_groups'] == 2
    module.generate_dataset(tmp_path, **settings)
    assert len(calls) == 11
    with pytest.raises(ValueError, match='recipe differs'):
        module.generate_dataset(tmp_path, **(settings | dict(seed=5)))
    labels = np.load(tmp_path/'u_scat.npy', mmap_mode='r+')
    labels[0, 0] += 1
    labels.flush()
    with pytest.raises(ValueError, match='checksum differs'):
        module.generate_dataset(tmp_path, **settings)
