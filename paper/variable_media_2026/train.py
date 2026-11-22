"""Fit material-aware SVD-SNO on grouped fixed-frequency P1 FEM fields."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys
from time import monotonic, sleep

import numpy as np
from scipy.linalg import eigh
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from neural_operator.data import mass_matrix
from neural_operator.sno.model import SVDSNO

DATA = HERE / "data"
RUNS = HERE / "runs"
SIZES = (400, 600, 800, 1000)
RANKS = (8, 16, 24)
WIDTH, DEPTH, EPOCHS, BATCH, SEED = 48, 4, 300, 64, 2026
MATERIAL_COLUMNS = ("alpha1", "alpha2", "n1", "n2")


def write_csv(name, rows):
    path = HERE / name
    with path.with_suffix(".tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tmp").replace(path)


def squared_norm(values, mass):
    return np.sum(values.conj() * (mass @ values.T).T, axis=1).real


def validate_queries(q, fields, mass):
    row_count, nodes = fields.shape
    if (fields.dtype != np.complex128 or mass.shape != (nodes, nodes)
            or any(q[key].shape != (row_count,) for key in
                   ("split", "medium", "theta", "k", *MATERIAL_COLUMNS))):
        raise ValueError("Inconsistent FEM fields, geometry, or query axes")
    if set(q["split"]) != {0, 1, 2, 3} or not np.all(q["k"] == 2):
        raise ValueError("Expected four partitions at fixed k=2")
    if not np.issubdtype(q["medium"].dtype, np.integer):
        raise ValueError("Expected integer medium IDs")
    materials = np.column_stack([q[key] for key in MATERIAL_COLUMNS])
    if not np.isfinite(q["theta"]).all() or not np.isfinite(materials).all() or (materials <= 0).any():
        raise ValueError("Nonfinite angles or invalid materials")
    seen = {}
    for medium in np.unique(q["medium"]):
        rows = np.flatnonzero(q["medium"] == medium)
        partitions = np.unique(q["split"][rows])
        if (len(rows) != 8 or len(np.unique(q["theta"][rows])) != 8
                or len(partitions) != 1 or not np.all(materials[rows] == materials[rows[0]])):
            raise ValueError(f"Invalid grouping for medium {medium}")
        key = tuple(materials[rows[0]])
        if key in seen and seen[key] != partitions[0]:
            raise ValueError("Identical material lies in multiple partitions")
        seen[key] = partitions[0]
    train_rows = np.flatnonzero(q["split"] == 0)
    if len(train_rows) != SIZES[-1]:
        raise ValueError(f"Expected {SIZES[-1]} train fields")
    for count in SIZES:
        _, frequencies = np.unique(q["medium"][train_rows[:count]], return_counts=True)
        if not (frequencies == 8).all():
            raise ValueError(f"Train prefix {count} splits a medium")
    return train_rows


def fit_pod(fields, mass, train):
    """Separate train-centered Re/Im snapshot bases in the consistent FEM mass."""
    rank = max(RANKS)
    norm2 = np.concatenate([squared_norm(fields[i:i+24], mass)
                            for i in range(0, len(fields), 24)])
    if not np.isfinite(norm2).all() or (norm2 <= 0).any():
        raise ValueError("Relative error requires finite nonzero FEM fields")
    out = {"norm2": norm2, "train": train}
    for part in ("real", "imag"):
        snapshots = np.ascontiguousarray(getattr(fields[train], part))
        mean = snapshots.mean(axis=0)
        snapshots -= mean
        gram = np.empty((len(train), len(train)))
        for i in range(0, len(train), 64):
            gram[:, i:i+64] = snapshots @ (mass @ snapshots[i:i+64].T)
        eigenvalues, vectors = eigh((gram + gram.T) / 2,
                                    subset_by_index=(len(train)-rank, len(train)-1))
        if not eigenvalues[0] > 0:
            raise ValueError(f"Insufficient numerical POD rank in {part}")
        modes = snapshots.T @ (vectors[:, ::-1] / np.sqrt(eigenvalues[::-1]))
        modes *= np.sign(modes[np.abs(modes).argmax(axis=0), np.arange(rank)])
        np.testing.assert_allclose(modes.T @ (mass @ modes), np.eye(rank), atol=5e-7, rtol=0)
        coeff = np.empty((len(fields), rank))
        norm = np.empty(len(fields))
        for i in range(0, len(fields), 24):
            centered = getattr(fields[i:i+24], part) - mean
            weighted = (mass @ centered.T).T
            coeff[i:i+24] = weighted @ modes
            norm[i:i+24] = np.sum(weighted * centered, axis=1)
        out.update({f"{part}_mean": mean, f"{part}_modes": modes,
                    f"{part}_coeff": coeff, f"{part}_norm2": norm})
    return out


def coefficient_labels(pod, rank):
    coeff = np.stack([pod[f"{part}_coeff"][:, :rank] for part in ("real", "imag")], axis=-1)
    residual = pod["real_norm2"] + pod["imag_norm2"] - np.sum(coeff**2, axis=(1, 2))
    if np.min(residual / pod["norm2"]) < -1e-9:
        raise ArithmeticError("POD coefficient energy exceeds centered field energy")
    return coeff, np.maximum(residual, 0)


def input_features(q, train, use_materials):
    angle = q["theta"]
    wave = np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
    if use_materials:
        inputs = np.column_stack([q[key] for key in MATERIAL_COLUMNS])
        mean, std = inputs[train].mean(axis=0), inputs[train].std(axis=0)
        if (std <= 0).any():
            raise ValueError("All four material parameters require train variation")
        wave = np.column_stack(((inputs - mean) / std, wave))
    return torch.tensor(wave, dtype=torch.float32)


def make_model(rank, width, use_materials):
    return SVDSNO(7 if use_materials else 3, rank, width, DEPTH,
                  input_ranks=(2, 2) if use_materials else (0, 0))


def fit_model(q, pod, rank, width, use_materials):
    train = pod["train"]
    validation = np.flatnonzero(q["split"] == 1)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    network = make_model(rank, width, use_materials)
    x = input_features(q, train, use_materials)
    coeff, residual = coefficient_labels(pod, rank)
    target = torch.tensor(coeff, dtype=torch.float32)
    scale = target[train].std(dim=0, correction=0)
    residual = torch.tensor(residual, dtype=torch.float32)
    norm = torch.tensor(pod["norm2"], dtype=torch.float32)
    opt = torch.optim.AdamW(network.parameters(), lr=0.002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=0.00015)

    def loss(ids):
        difference = network(x[ids]) * scale - target[ids]
        return ((difference.square().sum((1, 2)) + residual[ids]) / norm[ids]).sqrt().mean()

    history = []
    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        network.train()
        for batch in torch.as_tensor(train)[torch.randperm(len(train), generator=generator)].split(BATCH):
            opt.zero_grad()
            objective = loss(batch)
            if not torch.isfinite(objective):
                raise ArithmeticError(f"Nonfinite objective at epoch {epoch}")
            objective.backward()
            opt.step()
        scheduler.step()
        network.eval()
        with torch.inference_mode():
            value = float(loss(validation))
        history.append(value)
        if value < best:
            best = value
            best_epoch = epoch
            weights = {key: tensor.detach().clone() for key, tensor in network.state_dict().items()}
    return dict(weights=weights, scale=scale, epoch=best_epoch, history=history,
                val=best, rank=rank, width=width, use_materials=use_materials)


def predict(q, pod, checkpoint, rows):
    rank = checkpoint["rank"]
    network = make_model(rank, checkpoint["width"], checkpoint["use_materials"]).eval()
    network.load_state_dict(checkpoint["weights"])
    with torch.inference_mode():
        x = input_features(q, pod["train"], checkpoint["use_materials"])
        return (network(x[rows]) * checkpoint["scale"]).numpy().astype(np.float64)


def decode(pod, coeff):
    rank = coeff.shape[1]
    real, imag = [coeff[:, :, index] @ pod[f"{part}_modes"][:, :rank].T + pod[f"{part}_mean"]
                  for index, part in enumerate(("real", "imag"))]
    return real + 1j * imag


def test_scores(q, fields, mass, pod, checkpoint, model_name):
    rows = np.flatnonzero(q["split"] == 2)
    pred = predict(q, pod, checkpoint, rows)
    target, residual = coefficient_labels(pod, checkpoint["rank"])
    norm = pod["norm2"][rows]
    error = np.sqrt((np.sum((pred-target[rows])**2, axis=(1, 2))+residual[rows])/norm)
    floor = np.sqrt(residual[rows]/norm)
    for start in range(0, len(rows), 16):
        block = slice(start, start+16)
        labels = fields[rows[block]]
        for values, expected in ((pred[block], error[block]), (target[rows[block]], floor[block])):
            direct = np.sqrt(squared_norm(decode(pod, values) - labels, mass) / squared_norm(labels, mass))
            np.testing.assert_allclose(direct, expected, atol=5e-8, rtol=1e-7)
    metadata = dict(n=len(pod["train"]), rank=checkpoint["rank"], model=model_name, width=checkpoint["width"],
                    depth=DEPTH, parameters=sum(v.numel() for v in checkpoint["weights"].values()))
    summary = dict(**metadata, epoch=checkpoint["epoch"], val_pct=100*checkpoint["val"],
                   mean_pct=100*error.mean(), median_pct=100*np.median(error),
                   p95_pct=100*np.percentile(error, 95), projection_pct=100*floor.mean())
    cases = [dict(**metadata, row=row, medium=q["medium"][row], theta=q["theta"][row],
                  error_pct=100*e, projection_pct=100*f) for row, e, f in zip(rows, error, floor)]
    return summary, cases


def sensitivity(q, fields, mass, pod, checkpoint, control):
    indices = np.flatnonzero(q["split"] == 3)
    groups = np.unique(q["medium"][indices])
    if len(groups) != 3:
        raise ValueError("Sensitivity requires base, alpha perturbation and n perturbation")
    rows = [indices[q["medium"][indices] == group] for group in groups]
    rows = [group[np.argsort(q["theta"][group])] for group in rows]
    base, alpha, n = rows
    for observed, changed in ((alpha, "alpha1"), (n, "n1")):
        np.testing.assert_allclose(q["theta"][base], q["theta"][observed], atol=1e-12, rtol=0)
        if any((not np.all(q[key][observed] != q[key][base])) if key == changed
               else (not np.array_equal(q[key][observed], q[key][base])) for key in MATERIAL_COLUMNS):
            raise ValueError(f"Sensitivity pair does not isolate {changed}")
    truth = fields[np.concatenate(rows)].reshape(3, 8, -1)
    records = []
    for model, selected in (("materials", checkpoint), ("wave_only", control)):
        pred = decode(pod, predict(q, pod, selected, np.concatenate(rows))).reshape(3, 8, -1)
        base_truth = squared_norm(truth[0], mass)
        base_pred = squared_norm(pred[0], mass)
        for i, factor in ((1, "alpha1"), (2, "n1")):
            truth_change = truth[i] - truth[0]
            pred_change = pred[i] - pred[0]
            norm_truth = squared_norm(truth_change, mass)
            norm_pred = squared_norm(pred_change, mass)
            norm_error = squared_norm(pred_change - truth_change, mass)
            truth_rel = np.sqrt(norm_truth / base_truth)
            pred_rel = np.sqrt(norm_pred / base_pred)
            error = np.sqrt(norm_error / norm_truth)
            if not np.isfinite(np.stack((truth_rel, pred_rel, error))).all():
                raise ArithmeticError("Nonfinite sensitivity result")
            records.append(dict(model=model, n=len(pod["train"]), rank=selected["rank"], width=selected["width"],
                                changed=factor, angles=8, base_medium=groups[0], perturbed_medium=groups[i],
                                base_value=q[factor][base[0]], perturbed_value=q[factor][rows[i][0]],
                                true_change_pct=100*truth_rel.mean(), predicted_change_pct=100*pred_rel.mean(),
                                change_error_pct=100*error.mean()))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", type=int, default=0, help="Seconds to wait for data/complete.json")
    args = parser.parse_args()
    marker = DATA / "complete.json"
    deadline = monotonic() + args.wait
    while not marker.exists() and monotonic() < deadline:
        sleep(5)
    if not marker.exists():
        raise FileNotFoundError(marker)
    torch.set_num_threads(2)
    RUNS.mkdir(exist_ok=True)
    signature = hashlib.sha256(marker.read_bytes() + Path(__file__).read_bytes()).hexdigest()[:16]
    with np.load(DATA / "queries.npz") as q, np.load(DATA / "reference_problem.npz") as geometry:
        fields = np.load(DATA / "fields.npy", mmap_mode="r")
        mass = mass_matrix(geometry["coordinates"], geometry["triangles"])
        all_train = validate_queries(q, fields, mass)
        summaries, cases, selected = [], [], None
        for n in SIZES:
            pod = fit_pod(fields, mass, all_train[:n])
            checkpoints = []

            def cached_run(rank, width, use_materials):
                name = "materials" if use_materials else "wave_only"
                path = RUNS / f"{signature}_{name}_n{n}_r{rank}_w{width}.pt"
                if path.exists():
                    return torch.load(path, map_location="cpu")
                result = fit_model(q, pod, rank, width, use_materials)
                torch.save(result, path.with_suffix(".tmp"))
                path.with_suffix(".tmp").replace(path)
                print(f"{name} N={n} R={rank} w={width} validation={100*result['val']:.5f}%", flush=True)
                return result

            for rank in RANKS:
                checkpoints.append(cached_run(rank, WIDTH, True))
            if n == SIZES[-1]:
                selected = min(checkpoints, key=lambda result: result["val"])
                control = cached_run(selected["rank"], WIDTH, False)
                checkpoints.append(cached_run(16, 24, True))
                checkpoints.append(control)
                write_csv("sensitivity.csv", sensitivity(q, fields, mass, pod, selected, control))
                np.savez(RUNS / f"{signature}_pod_n{n}.npz", **pod)
            for checkpoint in checkpoints:
                name = "materials" if checkpoint["use_materials"] else "wave_only"
                summary, details = test_scores(q, fields, mass, pod, checkpoint, name)
                summaries.append(summary)
                cases.extend(details)
        write_csv("sno_cases.csv", cases)
        write_csv("sno_summary.csv", summaries)


if __name__ == "__main__":
    main()
