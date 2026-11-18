"""Pipeline failure propagation and concurrency, independent of Docker."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('fem_pipeline', Path(__file__).resolve().parents[1]/'scripts/pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def options(**kwargs):
    return argparse.Namespace(**(dict(experiment='mscale_fno_100', build=False,
        skip_tests=False, from_stage='prepare', epochs=300, seeds=[11, 29, 47]) | kwargs))


def test_failed_stage_prevents_later_stages_and_records_log(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    invoked = []

    def run(command, log):
        invoked.append(log.stem)
        log.write_text('diagnostic')
        return 17 if log.stem == 'train' else 0

    monkeypatch.setattr(pipeline, 'run_logged', run)
    with pytest.raises(RuntimeError, match='train failed'):
        pipeline.execute(options())
    assert invoked == ['up', 'environment', 'tests', 'prepare', 'train']
    manifest = json.loads((tmp_path/'experiments/mscale_fno_100/pipeline/latest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['stages'][-1]['returncode'] == 17
    assert (tmp_path/manifest['stages'][-1]['log']).read_text() == 'diagnostic'


def test_resume_at_verification_skips_training(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    stages = []
    monkeypatch.setattr(pipeline, 'run_logged', lambda command, log: stages.append(log.stem) or 0)
    pipeline.execute(options(from_stage='verify'))
    assert stages == ['up', 'environment', 'tests', 'verify', 'report']
    manifest = json.loads((tmp_path/'experiments/mscale_fno_100/pipeline/latest.json').read_text())
    assert manifest['status'] == 'passed'


def test_concurrent_pipeline_cannot_write_same_experiment(tmp_path, monkeypatch):
    import fcntl
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    directory = tmp_path/'experiments/mscale_fno_100/pipeline'
    directory.mkdir(parents=True)
    with (directory/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another pipeline'):
            pipeline.execute(options())


def test_sno_pipeline_forwards_training_seeds_and_epochs():
    command = pipeline.command_for('train', 'mscale_sno_100', 300, [11, 29, 47])
    assert 'experiments.mscale_sno_100.run' in command
    assert command[-6:] == ['--epochs', '300', '--seeds', '11', '29', '47']
    assert 'mscale_sno_100' in pipeline.EXPERIMENTS


def test_shared_sno_rejects_unsupported_seeds():
    with pytest.raises(ValueError, match='--seeds'):
        pipeline.command_for('train', 'shared_sno_gmsfem', 300, [11])


def test_pipeline_collects_both_multiscale_suites_without_module_name_collisions():
    command = pipeline.test_command('mscale_fno_100')
    assert '--import-mode=importlib' in command
    assert 'experiments/mscale_sno_100/tests' in command
