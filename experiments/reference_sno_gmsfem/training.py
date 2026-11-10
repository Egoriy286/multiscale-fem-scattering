"""Experiment-local SNO training with validation-driven learning-rate decay.

POD, predictor bundles, consistent FEM loss and inference reuse the core.
The small training loop is copied locally to keep the established core and
its historical runs unchanged. Training checkpoints are selected on val.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from neural_operator.training import (
    TrainConfig as CoreTrainConfig, Predictor, FEMLoss, PODBasis, PODFEMLoss,
    _svd_preprocessing, _grid_preprocessing, dataset_fingerprint,
    load_predictor, summarize, scattering_components,
)


@dataclass
class TrainConfig(CoreTrainConfig):
    lr_scheduler_factor: float = .5
    lr_scheduler_patience: int = 20
    lr_scheduler_min_lr: float = 1e-5


def _validate_config(config):
    if not isinstance(config, TrainConfig):
        raise TypeError("Use this experiment's TrainConfig, including its scheduler settings")
    factor, patience, minimum = (config.lr_scheduler_factor,
                                 config.lr_scheduler_patience,
                                 config.lr_scheduler_min_lr)
    if (isinstance(factor, bool) or not isinstance(factor, Real)
            or not np.isfinite(factor) or not 0 < factor < 1):
        raise ValueError("lr_scheduler_factor must be finite and strictly between 0 and 1")
    if isinstance(patience, bool) or not isinstance(patience, Integral) or patience < 0:
        raise ValueError("lr_scheduler_patience must be an integer >= 0")
    if (isinstance(minimum, bool) or not isinstance(minimum, Real)
            or not np.isfinite(minimum) or not 0 < minimum <= config.learning_rate):
        raise ValueError("lr_scheduler_min_lr must be finite, positive and <= learning_rate")


def _make_scheduler(optimizer, config):
    _validate_config(config)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience, threshold=1e-3,
        threshold_mode="rel", min_lr=config.lr_scheduler_min_lr)


def run_experiment(dataset, output_dir, config=None, *, retrain=False, verbose=True):
    """Reuse a matching completed run, or train it locally on first execution."""
    config = config or TrainConfig()
    _validate_config(config)
    directory = Path(output_dir)
    if not retrain and (directory/"model.pt").exists():
        metrics = json.loads((directory/"metrics.json").read_text())
        saved_config = dict(metrics["config"])
        # v1 SNO predated this option; it remains reusable only when requested
        # explicitly. For the other methods this SNO option has no effect.
        if config.method == "svd":
            saved_config.setdefault("sno_architecture_version", 1)
        else:
            saved_config["sno_architecture_version"] = config.sno_architecture_version
        if (saved_config != asdict(config)
                or metrics["dataset_fingerprint"] != dataset_fingerprint(dataset)):
            raise ValueError("Saved run uses different data/settings; change output_dir or set retrain=True.")
        history = json.loads((directory/"history.json").read_text())
        torch.set_num_threads(config.threads)
        if verbose:
            print(f"Loaded completed local {config.method} run: {directory}")
        return load_predictor(directory/"model.pt", device=config.device), metrics, history
    return train_model(dataset, directory, config, verbose=verbose)


def train_model(dataset, output_dir, config=None, *, verbose=True):
    """Fit preprocessing on train; select rank/epoch on val; evaluate test once."""
    config = config or TrainConfig()
    _validate_config(config)
    if config.method not in ("svd", "chebyshev", "fno"):
        raise ValueError("method must be svd, chebyshev, or fno")
    if min(config.epochs, config.batch_size, config.width, config.depth, config.threads) < 1:
        raise ValueError("epochs, batch_size, width, depth, and threads must be positive")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    train, val, test = dataset.split_indices()
    if min(len(train), len(val), len(test)) < 1:
        raise ValueError("Training requires nonempty train, validation, and test splits.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if verbose:
        print(f"Preparing {config.method}: {len(train)} train samples, POD={config.pod_method}", flush=True)
    if config.method == "svd":
        prep = _svd_preprocessing(dataset, config, train, val)
        input_ranks = [PODBasis.from_dict(prep["bases"][k]).rank for k in ("A", "n")]
        inputs = sum(input_ranks) + 3
        kwargs = {"input_dim": inputs, "output_modes": prep["rank"],
                  "width": config.width, "depth": config.depth,
                  "architecture_version": config.sno_architecture_version}
        if config.sno_architecture_version == 2:
            kwargs["input_ranks"] = input_ranks
    else:
        prep = _grid_preprocessing(dataset, train)
        prep["resolution"] = config.resolution
        kwargs = {"input_channels": 7, "modes": config.modes,
                  "width": config.width, "depth": config.depth}
        if config.method == "fno":
            kwargs["padding"] = config.padding
    predictor = Predictor(config.method, kwargs, prep, dataset.coordinates,
                          dataset.triangles, device=config.device)
    features = predictor.encode(dataset.A, dataset.n, dataset.theta, dataset.k0)
    # The feature transform is fixed. Test labels are never used by optimisation.
    x = torch.as_tensor(features, device=predictor.device)
    if config.method == "svd":
        loss_fn = PODFEMLoss(predictor, dataset, dataset.u_scat,
                             target_dtype=np.float32).to(predictor.device)

        def batch_loss(indices):
            return loss_fn(predictor.model(x[indices]), indices)

    else:
        target = torch.as_tensor(np.stack([dataset.u_scat.real, dataset.u_scat.imag], axis=-1),
                                 dtype=torch.float32, device=predictor.device)
        loss_fn = FEMLoss(dataset.coordinates, dataset.triangles).to(predictor.device)

        def batch_loss(indices):
            return loss_fn(predictor.decode(predictor.model(x[indices])), target[indices])
    preprocessing_seconds = perf_counter()-started
    if verbose:
        print(f"Preprocessing ready in {preprocessing_seconds:.2f}s; starting optimization", flush=True)
    optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=config.learning_rate,
                                 weight_decay=1e-5)
    scheduler = _make_scheduler(optimizer, config)
    history = {"train": [], "val": [], "lr": []}
    best_value, best_epoch, stale = float("inf"), 0, 0
    best_state = None
    started = perf_counter()
    stop_reason = "epoch_limit"
    for epoch in range(1, config.epochs+1):
        predictor.model.train()
        total = 0.
        order = rng.permutation(train)
        for first in range(0, len(order), config.batch_size):
            indices = torch.as_tensor(order[first:first+config.batch_size], device=predictor.device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite FEM L2 loss during training.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        predictor.model.eval()
        val_total = 0.
        with torch.inference_mode():
            for first in range(0, len(val), config.batch_size):
                j = torch.as_tensor(val[first:first+config.batch_size], device=predictor.device)
                val_total += float(batch_loss(j)) * len(j)
        validation = val_total/len(val)
        history["train"].append(total/len(train))
        history["val"].append(validation)
        # Record the LR actually used during this epoch, before the scheduler
        # chooses the learning rate for the next epoch.
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step(validation)
        if validation < best_value:
            best_value, best_epoch = validation, epoch
            best_state = copy.deepcopy(predictor.model.state_dict())
            stale = 0
        else:
            stale += 1
        if verbose and (epoch == 1 or epoch % 20 == 0):
            print(f"{config.method} {epoch:4d}: train={history['train'][-1]:.5f} "
                  f"val={validation:.5f} best={best_value:.5f}", flush=True)
        if config.patience > 0 and epoch >= config.min_epochs and stale >= config.patience:
            stop_reason = "validation_plateau"
            break
    training_seconds = perf_counter()-started
    predictor.model.load_state_dict(best_state)
    predictor.model.eval()
    # Warm up; timing below includes field encoding and FEM reconstruction.
    predictor.predict_dataset(dataset, test[:1])
    started = perf_counter()
    predictions = predictor.predict_dataset(dataset, test, batch_size=config.batch_size)
    inference_seconds = perf_counter()-started
    errors = dataset.relative_l2(dataset.u_scat[test], predictions)
    metrics = {
        "method": config.method, "field": "scattered", "metric": "consistent_complex_FEM_L2",
        "dataset_fingerprint": dataset_fingerprint(dataset), "config": asdict(config),
        "train_samples": len(train), "val_samples": len(val), "test_samples": len(test),
        "split_strategy": dataset.split_strategy,
        "physical_media": len(np.unique(dataset.medium_id)),
        "split_groups": {name: len(np.unique(dataset.split_group_ids[indices]))
                         for name, indices in zip(("train", "validation", "test"), (train, val, test))},
        "completed_epochs": len(history["train"]), "best_epoch": best_epoch,
        "best_val_fem_l2": best_value, "stop_reason": stop_reason,
        "parameters": sum(p.numel() * (2 if p.is_complex() else 1)
                          for p in predictor.model.parameters()),
        "parameter_count_convention": "real scalar degrees of freedom; complex weight counts as two",
        "preprocessing_seconds": preprocessing_seconds, "training_seconds": training_seconds,
        "test_batch_seconds": inference_seconds, "test_seconds_per_sample": inference_seconds/len(test),
        "test_fem_l2": summarize(errors),
        "training_loss_evaluation": ("consistent_FEM_POD_quadratic" if config.method == "svd"
                                      else "consistent_FEM_field_reconstruction"),
        "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__)},
        "lr_scheduler": {
            "name": "ReduceLROnPlateau", "mode": "min",
            "metric": "validation_consistent_complex_FEM_L2",
            "factor": config.lr_scheduler_factor,
            "patience": config.lr_scheduler_patience,
            "threshold": 1e-3, "threshold_mode": "rel",
            "min_lr": config.lr_scheduler_min_lr,
            "final_optimizer_lr": float(optimizer.param_groups[0]["lr"]),
            "history_lr_convention": "rate used during the recorded epoch; scheduler steps afterwards",
            "reductions_used_by_later_epochs": sum(
                after < before for before, after in zip(history["lr"], history["lr"][1:])),
        },
    }
    if config.method == "svd":
        br, bi = predictor.bases["real"], predictor.bases["imag"]
        projection = br.decode(br.encode(dataset.u_scat[test].real)) + 1j*bi.decode(bi.encode(dataset.u_scat[test].imag))
        metrics["projection_fem_l2"] = summarize(dataset.relative_l2(dataset.u_scat[test], projection))
        metrics["ranks"] = {k: b.rank for k, b in predictor.bases.items()}
        metrics["rank_target_attained"] = prep["rank_validation"]["target_attained"]
        metric_indices = torch.as_tensor(test, device=predictor.device)
        optimal_floor = torch.sqrt(loss_fn.floor_squared[metric_indices] /
                                   loss_fn.norm_squared[metric_indices].clamp_min(1e-24))
        metrics["consistent_projection_floor_fem_l2"] = summarize(optimal_floor.detach().cpu().numpy())
    predictor.save(output_dir / "model.pt", metadata=metrics)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2)+"\n")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2)+"\n")
    np.savez_compressed(output_dir / "test_predictions.npz", indices=test,
                        prediction=predictions, fem_l2=errors,
                        **{f"prediction_{name}": values
                           for name, values in scattering_components(predictions).items()})
    return predictor, metrics, history
