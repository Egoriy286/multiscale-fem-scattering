"""Train material-aware SVD-SNO on scaled scattered fields v=u_s/k."""
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
sys.path.insert(0, str(HERE.parents[2]))
from neural_operator.data import mass_matrix
from neural_operator.sno.model import SVDSNO

DATA, RUNS = HERE.parent / "data", HERE / "runs"
SIZES, RANKS = (400, 600, 800, 1000), (24, 48, 72)
WIDTH, DEPTH, EPOCHS, BATCH, SEED = 48, 4, 300, 64, 2026
MATERIALS = ("alpha1", "alpha2", "n1", "n2")


def save_csv(name, rows):
    path = HERE / name
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def squared_norm(values, mass):
    return np.sum(values.conj() * (mass @ values.T).T, axis=1).real


def partitions(q, fields, mass):
    row_count, nodes = fields.shape
    if (fields.dtype != np.complex128 or mass.shape != (nodes, nodes)
            or any(q[key].shape != (row_count,) for key in
                   ("k", "theta", "medium", "split", *MATERIALS))):
        raise ValueError("Inconsistent fields and query axes.")
    if any(not np.isfinite(q[key]).all() for key in ("k", "theta", *MATERIALS)):
        raise ValueError("Nonfinite query coordinates.")
    if np.min(q["k"]) < .05 or np.max(q["k"]) != 4 or set(q["split"]) != {0, 1, 2}:
        raise ValueError("Expected three splits and positive frequencies in [0.05, 4].")
    if not np.issubdtype(q["medium"].dtype, np.integer):
        raise ValueError("Medium IDs must be integers.")
    material_values = np.column_stack([q[key] for key in MATERIALS])
    if np.min(material_values) <= 0:
        raise ValueError("Material coefficients must be positive.")
    media_by_split = [set(q["medium"][q["split"] == label]) for label in range(3)]
    if tuple(map(len, media_by_split)) != (100, 12, 12) or sum(map(len, media_by_split)) != len(set.union(*media_by_split)):
        raise ValueError("Unexpected counts or medium leakage across splits.")
    parameter_groups = set()
    for group in np.unique(q["medium"]):
        rows = np.flatnonzero(q["medium"] == group)
        pairs = np.column_stack((q["k"][rows], q["theta"][rows]))
        params = material_values[rows]
        if (len(rows) != 10 or len(np.unique(pairs, axis=0)) != 10
                or np.unique(pairs[:, 0], return_counts=True)[1].tolist() != [2]*5
                or not np.all(params == params[0]) or tuple(params[0]) in parameter_groups):
            raise ValueError(f"Unexpected query grid or material for medium {group}.")
        parameter_groups.add(tuple(params[0]))
    split = tuple(np.flatnonzero(q["split"] == label) for label in range(3))
    for size in SIZES:
        if not np.all(np.unique(q["medium"][split[0][:size]], return_counts=True)[1] == 10):
            raise ValueError(f"Training prefix {size} cuts a medium.")
    return split


def fit_pod(fields, k, mass, train):
    """Separate weighted snapshot PODs of the real and imaginary parts of u_s/k."""
    norm2 = np.concatenate([squared_norm(fields[i:i+24], mass)
                            for i in range(0, len(fields), 24)]) / k**2
    if min(norm2) <= 0 or not np.isfinite(norm2).all():
        raise ValueError("Relative loss requires nonzero, finite reference fields.")
    result = dict(norm2=norm2, train=train)
    for part in ("real", "imag"):
        snapshots = np.ascontiguousarray(getattr(fields[train], part))
        snapshots /= k[train, None]
        mean = snapshots.mean(axis=0)
        snapshots -= mean
        gram = np.empty((len(train), len(train)))
        for i in range(0, len(train), 64):
            gram[:, i:i+64] = snapshots @ (mass @ snapshots[i:i+64].T)
        rank = max(RANKS)
        eig, vectors = eigh((gram+gram.T)/2, subset_by_index=(len(train)-rank, len(train)-1))
        if eig[0] <= eig[-1]*1e-12:
            raise ValueError(f"Numerical POD rank for {part} is below {rank}.")
        modes = snapshots.T @ (vectors[:, ::-1] / np.sqrt(eig[::-1]))
        modes *= np.sign(modes[np.abs(modes).argmax(axis=0), np.arange(rank)])
        np.testing.assert_allclose(modes.T @ (mass @ modes), np.eye(rank), rtol=0, atol=1e-7)
        coeff = np.empty((len(fields), rank))
        energy = np.empty(len(fields))
        for i in range(0, len(fields), 24):
            block = slice(i, i+24)
            centered = getattr(fields[block], part) / k[block, None] - mean
            weighted = (mass @ centered.T).T
            coeff[block] = weighted @ modes
            energy[block] = np.sum(weighted * centered, axis=1)
        result.update({f"{part}_mean": mean, f"{part}_modes": modes,
                       f"{part}_coeff": coeff, f"{part}_energy": energy})
    return result


def targets(pod, rank):
    coeff = np.stack([pod[f"{part}_coeff"][:, :rank] for part in ("real", "imag")], axis=-1)
    residual = pod["real_energy"]+pod["imag_energy"]-np.sum(coeff**2, axis=(1, 2))
    if np.min(residual / pod["norm2"]) < -1e-9:
        raise ArithmeticError("Invalid POD residual.")
    return coeff, np.maximum(residual, 0)


def feature_values(q):
    return np.column_stack([q[key] for key in (*MATERIALS, "k")])


def features(q, center, scale):
    standardized = (feature_values(q)-center)/scale
    angle = q["theta"]
    return torch.tensor(np.column_stack((standardized[:, :4], np.cos(angle),
                                         np.sin(angle), standardized[:, 4])), dtype=torch.float32)


def make_model(rank):
    return SVDSNO(7, rank, WIDTH, DEPTH, input_ranks=(2, 2), architecture_version=2)


def train_model(q, pod, validation, rank):
    train = pod["train"]
    torch.manual_seed(SEED)
    order = torch.Generator().manual_seed(SEED)
    model = make_model(rank)
    parameters = feature_values(q)[train]
    center, scale = parameters.mean(axis=0), parameters.std(axis=0)
    if np.min(scale) <= 0:
        raise ValueError("Expected train variation in all standardized input channels.")
    x = features(q, center, scale)
    coeff, residual = targets(pod, rank)
    coeff_scale = torch.tensor(coeff[train].std(axis=0), dtype=torch.float32)
    coeff = torch.tensor(coeff, dtype=torch.float32)
    residual = torch.tensor(residual, dtype=torch.float32)
    norm = torch.tensor(pod["norm2"], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=.00015)

    def loss(rows):
        difference = model(x[rows])*coeff_scale-coeff[rows]
        return ((difference.square().sum((1, 2))+residual[rows])/norm[rows]).sqrt().mean()

    train = torch.as_tensor(train)
    validation = torch.as_tensor(validation)
    best, history = float("inf"), []
    for epoch in range(1, EPOCHS+1):
        model.train()
        for rows in train[torch.randperm(len(train), generator=order)].split(BATCH):
            opt.zero_grad()
            value = loss(rows)
            if not torch.isfinite(value):
                raise ArithmeticError("Nonfinite SVD-SNO training error.")
            value.backward()
            opt.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            value = float(loss(validation))
        history.append(value)
        if value < best:
            best = value
            best_epoch = epoch
            weights = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    return dict(rank=rank, weights=weights, center=torch.tensor(center),
                scale=torch.tensor(scale), coeff_scale=coeff_scale,
                val=best, epoch=best_epoch, history=history,
                parameters=sum(parameter.numel() for parameter in model.parameters()))


def decode(pod, coeff):
    rank = coeff.shape[1]
    real, imag = [coeff[:, :, i] @ pod[f"{part}_modes"][:, :rank].T + pod[f"{part}_mean"]
                  for i, part in enumerate(("real", "imag"))]
    return real + 1j * imag


def restore_field(q, pod, checkpoint, rows, model=None):
    if model is None:
        model = make_model(checkpoint["rank"]).eval()
        model.load_state_dict(checkpoint["weights"])
    x = features(q, checkpoint["center"].numpy(), checkpoint["scale"].numpy())
    with torch.inference_mode():
        coeff = (model(x[rows])*checkpoint["coeff_scale"]).numpy().astype(np.float64)
    return q["k"][rows, None]*decode(pod, coeff), coeff


def evaluate(q, fields, pod, mass, test, checkpoint):
    prediction, predicted_coeff = restore_field(q, pod, checkpoint, test)
    target, residual = targets(pod, checkpoint["rank"])
    errors = np.sqrt((np.sum((predicted_coeff-target[test])**2, axis=(1, 2))
                      +residual[test])/pod["norm2"][test])
    for i in range(0, len(test), 16):
        rows = test[i:i+16]
        label = fields[rows]
        direct = np.sqrt(squared_norm(prediction[i:i+16]-label, mass)/squared_norm(label, mass))
        np.testing.assert_allclose(direct, errors[i:i+16], atol=1e-8, rtol=1e-7)
    n = len(pod["train"])
    errors *= 100
    summary = dict(n=n, parameters=checkpoint["parameters"], mean_pct=errors.mean(),
                   median_pct=np.median(errors), p95_pct=np.percentile(errors, 95),
                   val_pct=100*checkpoint["val"])
    detail = [dict(n=n, row=row, error_pct=error) for row, error in zip(test, errors)]
    if n == SIZES[-1]:
        indices = np.flatnonzero(q["k"][test] == 4)
        if indices.size == 0:
            raise ValueError("Missing k=4 testing example.")
        index = indices[0]
        np.savez(HERE / "example.npz", field=prediction[index], row=test[index], n=n)
        model = make_model(checkpoint["rank"]).eval()
        model.load_state_dict(checkpoint["weights"])
        zero_query = {key: q[key][test[index:index+1]].copy() for key in q.files}
        zero_query["k"][:] = 0
        zero_field, _ = restore_field(zero_query, pod, checkpoint, np.arange(1), model)
        np.testing.assert_array_equal(zero_field, np.zeros_like(zero_field))
    return summary, detail


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
    digest = hashlib.sha256(marker.read_bytes()+Path(__file__).read_bytes()).hexdigest()[:16]
    with np.load(DATA / "queries.npz") as q, np.load(DATA / "reference_problem.npz") as geometry:
        fields = np.load(DATA / "fields.npy", mmap_mode="r")
        mass = mass_matrix(geometry["coordinates"], geometry["triangles"])
        train, validation, test = partitions(q, fields, mass)
        summaries, cases, rank_trials = [], [], []
        for n in SIZES:
            pod = fit_pod(fields, q["k"], mass, train[:n])
            candidates = []
            for rank in RANKS:
                path = RUNS / f"{digest}_n{n}_r{rank}.pt"
                if path.exists():
                    checkpoint = torch.load(path, map_location="cpu")
                else:
                    checkpoint = train_model(q, pod, validation, rank)
                    torch.save(checkpoint, path.with_suffix(".tmp"))
                    path.with_suffix(".tmp").replace(path)
                print(f"N={n} R={rank} validation={100*checkpoint['val']:.6f}%", flush=True)
                candidates.append(checkpoint)
            best = min(candidates, key=lambda candidate: candidate["val"])
            rank_trials.extend(dict(n=n, rank=c["rank"], width=WIDTH, depth=DEPTH,
                                    parameters=c["parameters"], epoch=c["epoch"], val_pct=100*c["val"],
                                    selected=(c is best)) for c in candidates)
            summary, details = evaluate(q, fields, pod, mass, test, best)
            summaries.append(summary)
            cases.extend(details)
            path = RUNS / f"{digest}_pod_n{n}.npz"
            np.savez(path.with_suffix(".tmp.npz"), **pod)
            path.with_suffix(".tmp.npz").replace(path)
            print(f"N={n} selected R={best['rank']} test mean={summary['mean_pct']:.6f}%", flush=True)
        save_csv("validation.csv", rank_trials)
        save_csv("summary.csv", summaries)
        save_csv("cases.csv", cases)


if __name__ == "__main__":
    main()
