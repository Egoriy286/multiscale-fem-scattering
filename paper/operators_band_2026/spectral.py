"""Train matched Fourier-cosine and Chebyshev coefficient-space SNOs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import monotonic, sleep

import numpy as np
from scipy.linalg import cholesky, cho_solve
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from neural_operator.data import mass_matrix

SIZES = (400, 600, 800, 1000)
MODES, WIDTH, DEPTH, EPOCHS, BATCH, SEED = 16, 24, 4, 150, 64, 2026
FEATURES = ("alpha1", "alpha2", "n1", "n2", "k", "cos(theta)", "sin(theta)")


def spectral_basis(coordinates, family, modes):
    """Return tensor products in the order (first coordinate, second coordinate)."""
    t = np.clip(np.asarray(coordinates) / 2, -1, 1)
    if family == "fourier":
        vectors = np.cos(np.pi * (t[:, :, None] + 1) / 2 * np.arange(modes))
    else:
        vectors = np.polynomial.chebyshev.chebvander(t, modes - 1)
    return (vectors[:, 0, :, None] * vectors[:, 1, None, :]).reshape(len(t), modes**2)


def projection(basis, mass, fields, k):
    weighted = mass @ basis
    gram = basis.T @ weighted
    gram = (gram + gram.T) / 2
    lower = np.tril(cholesky(gram, lower=True))
    coeff = np.empty((len(k), basis.shape[1], 2))
    norm2 = np.empty(len(k))
    floor2 = np.empty(len(k))
    for start in range(0, len(k), 32):
        ix = slice(start, start + 32)
        scaled = fields[ix] / k[ix, None]
        values = np.stack((scaled.real, scaled.imag), axis=-1)
        for component in range(2):
            rhs = weighted.T @ values[:, :, component].T
            coeff[ix, :, component] = cho_solve((lower, True), rhs).T
        norm2[ix] = squared_norm(scaled, mass)
        difference = scaled - decode(basis, coeff[ix])
        floor2[ix] = squared_norm(difference, mass)
        lhs = np.sum((np.swapaxes(coeff[ix], 1, 2) @ lower)**2, axis=(1, 2))
        np.testing.assert_allclose(floor2[ix] + lhs, norm2[ix], rtol=5e-10, atol=1e-20)
    if np.any(norm2 <= 0) or not np.all(np.isfinite(norm2)):
        raise ValueError("The relative loss requires finite nonzero solutions")
    return coeff, lower, norm2, floor2


def squared_norm(field, mass):
    return np.sum(field.conj() * (mass @ field.T).T, axis=1).real


def decode(basis, coeff):
    return (coeff[:, :, 0] + 1j * coeff[:, :, 1]) @ basis.T


class MixingBlock(nn.Module):
    """Residual form of the coefficient and channel mixing in SNO Eq. (22)."""
    def __init__(self):
        super().__init__()
        self.x_modes = nn.Linear(MODES, MODES, bias=False)
        self.y_modes = nn.Linear(MODES, MODES, bias=False)
        self.channels = nn.Linear(WIDTH, WIDTH, bias=False)
        self.bias = nn.Parameter(torch.zeros(MODES, MODES, WIDTH))
        nn.init.eye_(self.x_modes.weight)
        nn.init.eye_(self.y_modes.weight)

    def forward(self, z):
        mixed = self.x_modes(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        mixed = self.y_modes(mixed.transpose(2, 3)).transpose(2, 3)
        return z + nn.functional.gelu(self.channels(mixed) + self.bias)


class SpectralSNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = nn.Linear(len(FEATURES), MODES**2 * WIDTH)
        self.blocks = nn.Sequential(*(MixingBlock() for _ in range(DEPTH)))
        self.readout = nn.Linear(WIDTH, 2)

    def forward(self, x):
        z = self.lift(x).reshape(-1, MODES, MODES, WIDTH)
        return self.readout(self.blocks(nn.functional.gelu(z))).reshape(-1, MODES**2, 2)


def train(inputs, coeff, lower, norm2, floor2, train_ids, validation, checkpoint):
    torch.manual_seed(SEED)
    shuffle = torch.Generator().manual_seed(SEED)
    model = SpectralSNO()
    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    coeff = torch.as_tensor(coeff, dtype=torch.float32)
    lower = torch.as_tensor(lower, dtype=torch.float32)
    norm2 = torch.as_tensor(norm2, dtype=torch.float32)
    floor2 = torch.as_tensor(floor2, dtype=torch.float32)
    scale = coeff[train_ids].square().mean(dim=0).sqrt().clamp_min(1e-8)
    optim = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS, eta_min=0.00015)

    def loss(ids):
        delta = (model(inputs[ids]) * scale - coeff[ids]).transpose(1, 2) @ lower
        return ((delta.square().sum(dim=(1, 2)) + floor2[ids]) / norm2[ids]).sqrt().mean()

    if checkpoint.is_file():
        result = torch.load(checkpoint, map_location="cpu")
    else:
        result = None
        history = []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for batch in torch.as_tensor(train_ids)[torch.randperm(len(train_ids), generator=shuffle)].split(BATCH):
                optim.zero_grad()
                objective = loss(batch)
                if not torch.isfinite(objective):
                    raise ArithmeticError(f"Nonfinite loss at epoch {epoch}")
                objective.backward()
                optim.step()
            schedule.step()
            model.eval()
            with torch.inference_mode():
                value = float(loss(validation))
            history.append(value)
            if result is None or value < result["val"]:
                result = dict(val=value, epoch=epoch,
                              weights={k: v.detach().clone() for k, v in model.state_dict().items()})
            if epoch % 50 == 0:
                print(f"N={len(train_ids)} epoch {epoch}: val={100*value:.4f}%", flush=True)
        result.update(history=history, scale=scale)
        temp = checkpoint.with_suffix(".tmp")
        torch.save(result, temp)
        temp.replace(checkpoint)
    model.load_state_dict(result["weights"])
    model.eval()
    with torch.inference_mode():
        pred = (model(inputs) * result["scale"]).numpy().astype(np.float64)
    return pred, sum(p.numel() for p in model.parameters()), result


def write_rows(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("fourier", "chebyshev"))
    parser.add_argument("--wait", type=int, default=1800)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    data = HERE / "data"
    marker = data / "complete.json"
    deadline = monotonic() + args.wait
    while not marker.is_file() and monotonic() < deadline:
        sleep(5)
    if not marker.is_file():
        raise FileNotFoundError(marker)
    from hashlib import sha256
    signature = sha256(marker.read_bytes() + Path(__file__).read_bytes()).hexdigest()[:16]
    torch.set_num_threads(args.threads)
    out = HERE / args.family
    out.mkdir(exist_ok=True)
    with np.load(data / "queries.npz") as manifest:
        q = dict(manifest)
    with np.load(data / "reference_problem.npz") as geom:
        xy, tri = geom["coordinates"], geom["triangles"]
    fields = np.load(data / "fields.npy", mmap_mode="r")
    train_all, validation, test = (np.flatnonzero(q["split"] == s) for s in (0, 1, 2))
    if (len(train_all), len(validation), len(test)) != (1000, 120, 120):
        raise ValueError("Unexpected split sizes")
    group_sets = [set(q["medium"][ids]) for ids in (train_all, validation, test)]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Overlapping media across partitions")
    if np.any(q["k"] <= 0) or fields.shape != (len(q["k"]), len(xy)):
        raise ValueError("Inconsistent FEM targets and wave parameters")
    mass = mass_matrix(xy, tri)
    basis = spectral_basis(xy, args.family, MODES)
    coeff, lower, norm2, floor2 = projection(basis, mass, fields, q["k"])
    values = np.column_stack([*(q[k] for k in FEATURES[:5]), np.cos(q["theta"]), np.sin(q["theta"])])
    summary, cases = [], []
    example_id = test[np.flatnonzero(q["k"][test] == 4)[0]]
    for n in SIZES:
        ids = train_all[:n]
        groups, counts = np.unique(q["medium"][ids], return_counts=True)
        if len(groups) != n // 10 or np.any(counts != 10):
            raise ValueError("Training prefix splits a medium")
        center, std = values[ids].mean(axis=0), values[ids].std(axis=0)
        x = (values - center) / np.where(std > 1e-12, std, 1)
        pred, params, result = train(x, coeff, lower, norm2, floor2, ids, validation,
                                     out / f"n{n}_{signature}.pt")
        delta = np.swapaxes(pred[test] - coeff[test], 1, 2) @ lower
        score = 100 * np.sqrt((np.sum(delta**2, axis=(1, 2)) + floor2[test]) / norm2[test])
        predicted = decode(basis, pred[test]) * q["k"][test, None]
        labels = fields[test]
        direct = 100 * np.sqrt(squared_norm(predicted - labels, mass) / squared_norm(labels, mass))
        np.testing.assert_allclose(score, direct, rtol=5e-8, atol=1e-7)
        summary.append(dict(n=n, parameters=params, mean_pct=score.mean(), median_pct=np.median(score),
                            p95_pct=np.quantile(score, 0.95), val_pct=100*result["val"],
                            projection_pct=100*np.sqrt(floor2[test]/norm2[test]).mean(),
                            epoch=result["epoch"]))
        cases.extend(dict(n=n, row=row, error_pct=error) for row, error in zip(test, score))
        if n == SIZES[-1]:
            field = predicted[np.flatnonzero(test == example_id)[0]]
            np.savez(out / "example.npz", field=field, row=example_id, k=q["k"][example_id],
                     theta=q["theta"][example_id])
        write_rows(out / "summary.csv", summary)
        write_rows(out / "cases.csv", cases)
        print(args.family, summary[-1], flush=True)
    config = dict(family=args.family, input_features=FEATURES, modes_per_axis=MODES, width=WIDTH,
                  depth=DEPTH, seed=SEED, epochs=EPOCHS, batch_size=BATCH,
                  adamw=dict(lr=.002, eta_min=.00015, weight_decay=1e-5), sizes=SIZES,
                  targets="complex scattered field divided by k", signature=signature,
                  basis_condition=float(np.linalg.cond(lower @ lower.T)))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
