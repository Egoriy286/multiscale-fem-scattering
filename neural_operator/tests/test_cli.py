"""The CLI must keep data planning cheap and route models to the right profile."""
import json

import numpy as np
import pytest

from neural_operator import settings
from neural_operator.data import inspect_dataset
from neural_operator.run import main


def test_plan_reports_split_without_generating(tmp_path, capsys, monkeypatch):
    import neural_operator.datasets as datasets

    def unexpected(*args, **kwargs):
        raise AssertionError('A dry run must not generate a mesh or dataset')

    monkeypatch.setattr(datasets, 'generate_dataset', unexpected)
    directory = tmp_path/'not-created'
    main(argv=['data', '--dataset', str(directory), '--band', 'high', '--dry-run',
               '--n-frequencies', '6', '--n-directions', '2'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['samples'] == 12
    assert plan['media'] == 1
    assert plan['k_range'] == [4., 8.]
    assert [row['samples'] for row in plan['splits']] == [8, 2, 2]
    assert not directory.exists()


def test_custom_range_requires_an_explicit_directory(capsys):
    with pytest.raises(SystemExit) as error:
        main(argv=['data', '--k-range', '2', '3', '--dry-run'])
    assert error.value.code == 2
    assert '--k-range requires --dataset' in capsys.readouterr().err


def test_inspect_never_reads_field_payload_or_creates_cache(tmp_path, monkeypatch, capsys):
    np.savez_compressed(tmp_path/'fem_dataset.npz',
                        u_scat=np.zeros((12, 3), dtype=complex),
                        A=np.ones((1, 3)), n=np.ones((1, 3)),
                        k0=np.repeat(np.linspace(1, 2, 6), 2),
                        medium_id=np.zeros(12, dtype=int),
                        frequency_id=np.repeat(np.arange(6), 2),
                        split=np.repeat([0, 0, 0, 0, 1, 2], 2),
                        split_strategy=np.asarray('frequency'))
    original = np.lib.npyio.NpzFile.__getitem__

    def metadata_only(self, key):
        assert key not in ('u_scat', 'A', 'n'), 'inspect unpacked a field'
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', metadata_only)
    main(argv=['inspect', '--dataset', str(tmp_path)])
    info = json.loads(capsys.readouterr().out)
    assert info['samples'] == 12
    assert info['stored_arrays']['A']['shape'] == [1, 3]
    assert [row['groups'] for row in info['splits']] == [4, 1, 1]
    assert inspect_dataset(tmp_path/'fem_dataset.npz') == info
    assert sorted(p.name for p in tmp_path.iterdir()) == ['fem_dataset.npz']


@pytest.mark.parametrize('pod', ['full', 'randomized'])
@pytest.mark.parametrize('band', ['low', 'high'])
def test_train_and_compare_use_the_same_profile(pod, band, monkeypatch):
    from neural_operator import data, training, experiments
    from neural_operator.randomized import settings as randomized
    profile = randomized if pod == 'randomized' else settings
    expected = getattr(profile, f'{band.upper()}_RUNS')/'svd'
    trained, compared = [], []
    monkeypatch.setattr(data, 'load_dataset', lambda *a, **k: object())

    def train(dataset, destination, config, **kwargs):
        trained.append((destination, config.pod_method))
        return None, {'test_fem_l2': {}}, None

    def compare(dataset, runs, destination):
        compared.append(runs['svd'])
        return [], [], []

    monkeypatch.setattr(training, 'run_experiment', train)
    monkeypatch.setattr(experiments, 'compare_methods', compare)
    options = ['--band', band, '--pod-method', pod, '--method', 'svd']
    main(argv=['train', *options])
    main(argv=['compare', *options])
    assert trained == [(expected, pod)]
    assert compared == [expected]


def test_custom_data_has_separate_default_runs(tmp_path, monkeypatch):
    from neural_operator import data, training
    destinations = []
    monkeypatch.setattr(data, 'load_dataset', lambda *a, **k: object())

    def train(dataset, destination, config, **kwargs):
        destinations.append(destination)
        return None, {'test_fem_l2': {}}, None

    monkeypatch.setattr(training, 'run_experiment', train)
    main(argv=['train', '--dataset', str(tmp_path/'small'), '--method', 'fno'])
    assert destinations == [settings.BASE/'runs'/'small'/'fno']
    assert destinations[0] != settings.LOW_RUNS/'fno'


def test_grid_resolution_and_mode_budget_reach_training(monkeypatch):
    from neural_operator import data, training
    captured = []
    monkeypatch.setattr(data, 'load_dataset', lambda *a, **k: object())

    def train(dataset, destination, config, **kwargs):
        captured.append((config.resolution, config.modes, config.sno_architecture_version))
        return None, {'test_fem_l2': {}}, None

    monkeypatch.setattr(training, 'run_experiment', train)
    main(argv=['train', '--method', 'chebyshev', '--resolution', '48', '--modes', '12'])
    assert captured == [(48, 12, 2)]


def test_compare_explicit_model_root_is_independent_of_reports(tmp_path, monkeypatch):
    from neural_operator import data, experiments
    calls = []
    monkeypatch.setattr(data, 'load_dataset', lambda *a, **k: object())

    def compare(dataset, runs, destination):
        calls.append((runs, destination))
        return [], [], []

    monkeypatch.setattr(experiments, 'compare_methods', compare)
    main(argv=['compare', '--method', 'fno', '--runs', str(tmp_path/'models'),
               '--output', str(tmp_path/'report')])
    assert calls == [({'fno': tmp_path/'models'/'fno'}, tmp_path/'report')]


def test_comparison_uses_saved_mesh_partition(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from dataclasses import asdict
    from neural_operator.datasets import StaticDatasetConfig
    from neural_operator import experiments
    config = StaticDatasetConfig(n_frequencies=6, n_coarse=2)
    (tmp_path/'config.json').write_text(json.dumps({**asdict(config), 'medium_mode': 'static'}))
    dataset = SimpleNamespace(directory=tmp_path, split_group_ids=np.array([0]),
                              split_indices=lambda: (np.array([]), np.array([]), np.array([0])))

    class ReachedClassicalSolver(Exception):
        pass

    def compare(dataset, index, *, n_coarse, **kwargs):
        assert n_coarse == 2
        raise ReachedClassicalSolver

    monkeypatch.setattr(experiments, 'compare_classical', compare)
    with pytest.raises(ReachedClassicalSolver):
        experiments.compare_methods(dataset, {}, tmp_path/'report')
