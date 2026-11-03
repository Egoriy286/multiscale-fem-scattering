"""One local training/inference path with a consistent complex FEM L2 loss."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .basis import PODBasis
from .grid import GridTransfer
from .models import ChebyshevSNO, FNO2d, SVDSNO


@dataclass
class TrainConfig:
    method: str = "svd"
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 2e-3
    width: int = 24
    depth: int = 3
    rank: int = 48
    input_rank: int = 32
    target_projection_error: float = .01
    pod_method: str = "full"
    resolution: int = 32
    modes: int = 8
    padding: int = 8
    patience: int = 50
    min_epochs: int = 50
    seed: int = 42
    threads: int = 2
    device: str = "cpu"


def _stats(values, axis=0):
    values = np.asarray(values, dtype=np.float64)
    mean, std = values.mean(axis=axis), values.std(axis=axis)
    threshold = max(float(np.max(std, initial=0)) * 1e-7, 1e-12)
    return {"mean": np.asarray(mean).tolist(),
            "std": np.where(std > threshold, std, 1.).tolist()}


def _normalize(values, stats):
    return (np.asarray(values) - np.asarray(stats["mean"])) / np.asarray(stats["std"])


def dataset_fingerprint(dataset):
    """Associate a result with the full data, geometry, and the saved split."""
    digest = hashlib.sha256()
    for name in ("A", "n", "u_scat", "theta", "k0", "medium_id", "split",
                 "coordinates", "triangles"):
        values = np.ascontiguousarray(getattr(dataset, name))
        digest.update(name.encode())
        digest.update(str((values.shape, values.dtype.str)).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


class FEMLoss(torch.nn.Module):
    """Exact P1 mass integration, with real/imag channels in the last axis."""

    def __init__(self, coordinates, triangles):
        super().__init__()
        vertices = np.asarray(coordinates)[triangles]
        edges = vertices[:, 1:] - vertices[:, :1]
        areas = np.abs(edges[:, 0, 0]*edges[:, 1, 1]
                       - edges[:, 0, 1]*edges[:, 1, 0]) / 2
        self.register_buffer("triangles", torch.as_tensor(triangles, dtype=torch.long))
        self.register_buffer("areas", torch.as_tensor(areas / 12, dtype=torch.float32))

    def squared_norm(self, values):
        v = values[:, self.triangles]
        return ((v.square().sum(dim=(2, 3)) + v.sum(dim=2).square().sum(dim=2))
                * self.areas).sum(dim=1)

    def forward(self, prediction, target):
        error = self.squared_norm(prediction-target)
        reference = self.squared_norm(target)
        # sqrt stabilisation affects only roundoff-level zero differences.
        return torch.sqrt(error.clamp_min(1e-24) / reference.clamp_min(1e-24)).mean()


class GridDecoder(torch.nn.Module):
    """Differentiable bilinear tensor-grid -> original FEM-node transfer."""

    def __init__(self, transfer):
        super().__init__()
        matrix = transfer.to_nodes_matrix.tocoo()
        self.register_buffer("rows", torch.as_tensor(matrix.row, dtype=torch.long))
        self.register_buffer("cols", torch.as_tensor(matrix.col, dtype=torch.long))
        self.register_buffer("weights", torch.as_tensor(matrix.data, dtype=torch.float32))
        self.node_count = matrix.shape[0]

    def forward(self, grid):
        flat = grid.reshape(grid.shape[0], -1, 2)
        values = flat[:, self.cols] * self.weights[None, :, None]
        result = grid.new_zeros((len(grid), self.node_count, 2))
        return result.index_add(1, self.rows, values)


def _svd_preprocessing(dataset, config, train, val):
    _, first = np.unique(dataset.medium_id[train], return_index=True)
    medium_rows = train[first]
    weights = dataset.lumped_weights
    bases = {}
    for name, fields, indices, rank in (
        ("A", dataset.A, medium_rows, config.input_rank),
        ("n", dataset.n, medium_rows, config.input_rank),
        ("real", dataset.u_scat.real, train, config.rank),
        ("imag", dataset.u_scat.imag, train, config.rank),
    ):
        bases[name] = PODBasis.fit(fields[indices], rank=rank, weights=weights,
                                  method=config.pod_method, seed=config.seed)
    available = max(bases["real"].rank, bases["imag"].rank)
    if available < 1:
        raise ValueError("Training scattered fields need at least one nonconstant mode.")
    # Each field keeps its own numerical rank, including a constant component.
    errors = []
    coefficients = {key: bases[key].encode(getattr(dataset.u_scat[val], key))
                    for key in ("real", "imag")}
    for rank in range(1, available+1):
        recon = []
        for key in ("real", "imag"):
            basis = bases[key]
            recon.append(basis.mean + coefficients[key][:, :rank] @ basis.modes[:, :rank].T)
        errors.append(float(np.mean(dataset.relative_l2(
            dataset.u_scat[val], recon[0]+1j*recon[1]))))
    attained = np.flatnonzero(np.asarray(errors) <= config.target_projection_error)
    chosen = int(attained[0]+1) if len(attained) else available
    for key in ("real", "imag"):
        state = bases[key].to_dict()
        state["modes"] = bases[key].modes[:, :chosen].tolist()
        state["singular_values"] = bases[key].singular_values[:chosen].tolist()
        # from_dict recomputes the cumulative fraction from total energy.
        if "energy" in state:
            state["energy"] = np.asarray(bases[key].energy)[:chosen].tolist()
        bases[key] = PODBasis.from_dict(state)
    features = np.concatenate((bases["A"].encode(dataset.A[train]),
                               bases["n"].encode(dataset.n[train])), axis=1)
    out = np.zeros((len(train), chosen, 2))
    for component, key in enumerate(("real", "imag")):
        basis = bases[key]
        out[:, :basis.rank, component] = basis.encode(getattr(dataset.u_scat[train], key))
    return {
        "bases": {key: value.to_dict() for key, value in bases.items()},
        "input_stats": _stats(features), "output_stats": _stats(out),
        "k_stats": _stats(dataset.k0[train]), "rank": chosen,
        "rank_validation": {"ranks": list(range(1, available+1)),
                            "mean_fem_l2": errors,
                            "target": config.target_projection_error,
                            "target_attained": bool(len(attained))},
    }


def _grid_preprocessing(dataset, train):
    mean = np.stack([dataset.u_scat[train].real.mean(axis=0),
                     dataset.u_scat[train].imag.mean(axis=0)], axis=-1)
    fields = np.stack([dataset.u_scat[train].real, dataset.u_scat[train].imag], axis=-1)
    scale = np.sqrt(np.mean((fields-mean)**2, axis=(0, 1)))
    return {"A_stats": _stats(dataset.A[train], axis=None),
            "n_stats": _stats(dataset.n[train], axis=None),
            "k_stats": _stats(dataset.k0[train]),
            "output_mean": mean.tolist(), "output_scale": np.maximum(scale, 1e-12).tolist()}


class Predictor:
    """A saved operator plus all transforms; inference needs no training cells."""

    def __init__(self, method, model_kwargs, preprocessing, coordinates, triangles,
                 state_dict=None, device="cpu"):
        self.method, self.model_kwargs = method, model_kwargs
        self.preprocessing = preprocessing
        self.coordinates, self.triangles = np.asarray(coordinates), np.asarray(triangles)
        self.device = torch.device(device)
        classes = {"svd": SVDSNO, "chebyshev": ChebyshevSNO, "fno": FNO2d}
        self.model = classes[method](**model_kwargs).to(self.device)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.transfer = None
        if method == "svd":
            self.bases = {key: PODBasis.from_dict(value)
                          for key, value in preprocessing["bases"].items()}
            self.output_mean = torch.as_tensor(np.stack([
                self.bases["real"].mean, self.bases["imag"].mean], axis=-1),
                dtype=torch.float32, device=self.device)
            padded_modes = np.zeros((len(self.coordinates), preprocessing["rank"], 2))
            for component, key in enumerate(("real", "imag")):
                basis = self.bases[key]
                padded_modes[:, :basis.rank, component] = basis.modes
            self.output_modes = torch.as_tensor(padded_modes, dtype=torch.float32, device=self.device)
        else:
            self.transfer = GridTransfer(self.coordinates, self.triangles,
                                         resolution=preprocessing["resolution"],
                                         kind="uniform" if method == "fno" else "chebyshev")
            self.decoder = GridDecoder(self.transfer).to(self.device)
            self.output_mean = torch.as_tensor(preprocessing["output_mean"],
                                               dtype=torch.float32, device=self.device)
            self.output_scale = torch.as_tensor(preprocessing["output_scale"],
                                                dtype=torch.float32, device=self.device)

    def encode(self, A, n, theta, k0):
        A, n = np.atleast_2d(A), np.atleast_2d(n)
        count = len(A)
        theta = np.broadcast_to(np.asarray(theta), (count,))
        k0 = np.broadcast_to(np.asarray(k0), (count,))
        if A.shape != n.shape or A.shape[1] != len(self.coordinates):
            raise ValueError("A and n must use the checkpoint FEM-node ordering.")
        k = _normalize(k0, self.preprocessing["k_stats"])
        globals_ = np.stack([np.cos(theta), np.sin(theta), k], axis=-1)
        if self.method == "svd":
            coeff = np.concatenate([self.bases["A"].encode(A), self.bases["n"].encode(n)], axis=1)
            values = np.concatenate([_normalize(coeff, self.preprocessing["input_stats"]), globals_], axis=1)
        else:
            Ag = self.transfer.to_grid(_normalize(A, self.preprocessing["A_stats"]))
            ng = self.transfer.to_grid(_normalize(n, self.preprocessing["n_stats"]))
            shape = (*Ag.shape, 3)
            g = np.broadcast_to(globals_[:, None, None, :], shape)
            xy = self.transfer.points.reshape(*Ag.shape[1:], 2)
            lo, hi = self.coordinates.min(axis=0), self.coordinates.max(axis=0)
            xy = 2*(xy-lo)/(hi-lo)-1
            coords = np.broadcast_to(xy, (*Ag.shape, 2))
            values = np.concatenate([Ag[..., None], ng[..., None], g, coords], axis=-1)
        return np.ascontiguousarray(values, dtype=np.float32)

    def decode(self, outputs):
        if self.method == "svd":
            stats = self.preprocessing["output_stats"]
            mean = torch.as_tensor(stats["mean"], dtype=outputs.dtype, device=outputs.device)
            std = torch.as_tensor(stats["std"], dtype=outputs.dtype, device=outputs.device)
            coefficients = outputs * std + mean
            return self.output_mean + torch.einsum("brc,nrc->bnc", coefficients, self.output_modes)
        return self.output_mean + self.decoder(outputs) * self.output_scale

    def predict(self, A, n, theta, k0, batch_size=32):
        single = np.ndim(A) == 1
        self.model.eval()
        features = self.encode(A, n, theta, k0)
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(features), batch_size):
                x = torch.as_tensor(features[start:start+batch_size], device=self.device)
                predictions.append(self.decode(self.model(x)).cpu().numpy())
        values = np.concatenate(predictions)
        result = values[..., 0] + 1j*values[..., 1]
        return result[0] if single else result

    def predict_dataset(self, dataset, indices=None, batch_size=32):
        if (not np.array_equal(dataset.triangles, self.triangles)
                or not np.array_equal(dataset.coordinates, self.coordinates)):
            raise ValueError("Dataset geometry/node ordering differs from the saved model.")
        indices = np.arange(len(dataset.A)) if indices is None else np.asarray(indices)
        return self.predict(dataset.A[indices], dataset.n[indices],
                            dataset.theta[indices], dataset.k0[indices], batch_size=batch_size)

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "method": self.method, "model_kwargs": self.model_kwargs,
                   "preprocessing": self.preprocessing,
                   "coordinates": torch.from_numpy(self.coordinates.copy()),
                   "triangles": torch.from_numpy(self.triangles.copy()),
                   "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                   "metadata": metadata or {}}
        temporary = path.with_suffix(path.suffix+".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)


def load_predictor(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported model checkpoint format.")
    torch.set_num_threads(checkpoint.get("metadata", {}).get("config", {}).get("threads", 2))
    return Predictor(checkpoint["method"], checkpoint["model_kwargs"],
                     checkpoint["preprocessing"], checkpoint["coordinates"].numpy(),
                     checkpoint["triangles"].numpy(), checkpoint["state_dict"], device=device)


def run_experiment(dataset, output_dir, config=None, *, retrain=False, verbose=True):
    """Reuse a matching completed run, or train it locally on first execution."""
    config = config or TrainConfig()
    directory = Path(output_dir)
    if not retrain and (directory/"model.pt").exists():
        metrics = json.loads((directory/"metrics.json").read_text())
        if (metrics["config"] != asdict(config)
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
    if config.method == "svd":
        prep = _svd_preprocessing(dataset, config, train, val)
        inputs = sum(PODBasis.from_dict(prep["bases"][k]).rank for k in ("A", "n")) + 3
        kwargs = {"input_dim": inputs, "output_modes": prep["rank"],
                  "width": config.width, "depth": config.depth}
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
    target = torch.as_tensor(np.stack([dataset.u_scat.real, dataset.u_scat.imag], axis=-1),
                             dtype=torch.float32, device=predictor.device)
    loss_fn = FEMLoss(dataset.coordinates, dataset.triangles).to(predictor.device)
    preprocessing_seconds = perf_counter()-started
    optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=config.learning_rate,
                                 weight_decay=1e-5)
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
            prediction = predictor.decode(predictor.model(x[indices]))
            loss = loss_fn(prediction, target[indices])
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
                val_total += float(loss_fn(predictor.decode(predictor.model(x[j])), target[j])) * len(j)
        validation = val_total/len(val)
        history["train"].append(total/len(train))
        history["val"].append(validation)
        history["lr"].append(optimizer.param_groups[0]["lr"])
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
        "completed_epochs": len(history["train"]), "best_epoch": best_epoch,
        "best_val_fem_l2": best_value, "stop_reason": stop_reason,
        "parameters": sum(p.numel() * (2 if p.is_complex() else 1)
                          for p in predictor.model.parameters()),
        "parameter_count_convention": "real scalar degrees of freedom; complex weight counts as two",
        "preprocessing_seconds": preprocessing_seconds, "training_seconds": training_seconds,
        "test_batch_seconds": inference_seconds, "test_seconds_per_sample": inference_seconds/len(test),
        "test_fem_l2": summarize(errors),
        "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__)},
    }
    if config.method == "svd":
        br, bi = predictor.bases["real"], predictor.bases["imag"]
        projection = br.decode(br.encode(dataset.u_scat[test].real)) + 1j*bi.decode(bi.encode(dataset.u_scat[test].imag))
        metrics["projection_fem_l2"] = summarize(dataset.relative_l2(dataset.u_scat[test], projection))
        metrics["ranks"] = {k: b.rank for k, b in predictor.bases.items()}
        metrics["rank_target_attained"] = prep["rank_validation"]["target_attained"]
    predictor.save(output_dir / "model.pt", metadata=metrics)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2)+"\n")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2)+"\n")
    np.savez_compressed(output_dir / "test_predictions.npz", indices=test,
                        prediction=predictions, fem_l2=errors)
    return predictor, metrics, history
