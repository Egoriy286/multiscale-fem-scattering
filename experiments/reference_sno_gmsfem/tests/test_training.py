"""Scheduler policy and safe completed-run reuse, without fitting a model."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from experiments.reference_sno_gmsfem import training


def test_plateau_reduces_after_patience_and_stops_at_minimum():
    config = training.TrainConfig()
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    scheduler = training._make_scheduler(optimizer, config)
    scheduler.step(1.)  # Establish the validation baseline.
    for _ in range(20):
        scheduler.step(.9995)  # Below the 0.1% meaningful-improvement threshold.
    assert optimizer.param_groups[0]["lr"] == .002
    scheduler.step(.9995)
    assert optimizer.param_groups[0]["lr"] == .001
    for _ in range(300):
        scheduler.step(1.)
    assert optimizer.param_groups[0]["lr"] == config.lr_scheduler_min_lr


def test_meaningful_validation_improvement_resets_scheduler_patience():
    config = training.TrainConfig(lr_scheduler_patience=1)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=.002)
    scheduler = training._make_scheduler(optimizer, config)
    for loss in (1., 1., .99, .99):
        scheduler.step(loss)
    assert optimizer.param_groups[0]["lr"] == .002
    scheduler.step(.99)
    assert optimizer.param_groups[0]["lr"] == .001


@pytest.mark.parametrize("changes", [
    {"lr_scheduler_factor": 1.}, {"lr_scheduler_factor": float("nan")},
    {"lr_scheduler_patience": -1}, {"lr_scheduler_patience": 1.5},
    {"lr_scheduler_min_lr": 0.}, {"lr_scheduler_min_lr": .1},
])
def test_invalid_scheduler_settings_rejected_before_any_training(tmp_path, changes):
    with pytest.raises(ValueError, match="lr_scheduler"):
        training.run_experiment(object(), tmp_path / "unused",
                                replace(training.TrainConfig(), **changes), verbose=False)
    assert not (tmp_path / "unused").exists()


def test_checkpoint_reuse_requires_all_scheduler_settings_and_fingerprint(tmp_path, monkeypatch):
    config = training.TrainConfig()
    (tmp_path / "model.pt").write_bytes(b"loader is mocked; no fitting")
    (tmp_path / "metrics.json").write_text(json.dumps({"config": asdict(config),
                                                       "dataset_fingerprint": "exact-data"}))
    (tmp_path / "history.json").write_text(json.dumps({"train": [1.], "val": [1.], "lr": [.002]}))
    monkeypatch.setattr(training, "dataset_fingerprint", lambda _: "exact-data")
    loaded = object()
    monkeypatch.setattr(training, "load_predictor", lambda *args, **kwargs: loaded)
    assert training.run_experiment(object(), tmp_path, config, verbose=False)[0] is loaded
    for changes in ({"lr_scheduler_factor": .25}, {"lr_scheduler_patience": 10},
                    {"lr_scheduler_min_lr": 1e-6}):
        with pytest.raises(ValueError, match="different data/settings"):
            training.run_experiment(object(), tmp_path, replace(config, **changes), verbose=False)
    monkeypatch.setattr(training, "dataset_fingerprint", lambda _: "changed-data")
    with pytest.raises(ValueError, match="different data/settings"):
        training.run_experiment(object(), tmp_path, config, verbose=False)
    # Historical fixed-rate configs do not silently inherit the new scheduler.
    old = asdict(config)
    for key in ("lr_scheduler_factor", "lr_scheduler_patience", "lr_scheduler_min_lr"):
        old.pop(key)
    (tmp_path / "metrics.json").write_text(json.dumps({"config": old,
                                                       "dataset_fingerprint": "exact-data"}))
    monkeypatch.setattr(training, "dataset_fingerprint", lambda _: "exact-data")
    with pytest.raises(ValueError, match="different data/settings"):
        training.run_experiment(object(), tmp_path, config, verbose=False)
