"""Replay the selected N=1000 model on the held-out low-positive-k queries."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

if __package__:
    from . import train
else:
    import train


def main():
    torch.set_num_threads(2)
    with (train.HERE / "validation.csv").open(newline="") as stream:
        candidates = csv.DictReader(stream)
        selected, = (row for row in candidates if row["n"] == "1000" and row["selected"] == "True")
    digest = hashlib.sha256((train.DATA / "complete.json").read_bytes()
                            + Path(train.__file__).read_bytes()).hexdigest()[:16]
    checkpoint = torch.load(train.RUNS / f"{digest}_n1000_r{selected['rank']}.pt", map_location="cpu")
    with (np.load(train.DATA / "nearzero.npz") as data,
          np.load(train.DATA / "reference_problem.npz") as geometry,
          np.load(train.RUNS / f"{digest}_pod_n1000.npz") as pod):
        queries = {name: data[name] for name in ("k", "theta", *train.MATERIALS)}
        k, theta, labels = queries["k"], queries["theta"], data["fields"]
        expected_k = np.repeat([.001, .01, .025], 2)
        if (labels.shape != (6, len(geometry["coordinates"]))
                or not np.array_equal(np.sort(k), expected_k)
                or any(values.shape != (6,) for values in queries.values())
                or not np.isfinite(labels).all()):
            raise ValueError("Unexpected held-out near-zero FEM queries or labels.")
        mass = train.mass_matrix(geometry["coordinates"], geometry["triangles"])
        prediction, _ = train.restore_field(queries, pod, checkpoint, np.arange(len(k)))
        reference = train.squared_norm(labels, mass)
        if not np.all(reference > 0):
            raise ValueError("Relative error requires nonzero near-zero reference fields.")
        error = 100*np.sqrt(train.squared_norm(prediction-labels, mass)/reference)
        train.save_csv("nearzero.csv", [dict(row=row, k=k[row], theta=theta[row],
                                           error_pct=error[row]) for row in range(len(k))])


if __name__ == "__main__":
    main()
