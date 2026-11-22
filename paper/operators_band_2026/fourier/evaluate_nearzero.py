"""Evaluate both frozen spectral SNOs on the preregistered near-zero queries."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral import (
    HERE, FEATURES, MODES, SpectralSNO, decode, mass_matrix,
    spectral_basis, squared_norm, write_rows,
)


def features(query):
    return np.column_stack([*(query[key] for key in FEATURES[:5]),
                            np.cos(query["theta"]), np.sin(query["theta"])])


def main():
    torch.set_num_threads(2)
    with np.load(HERE / "data" / "queries.npz") as q:
        train = features(q)[q["split"] == 0]
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    with np.load(HERE / "data" / "reference_problem.npz") as mesh:
        coordinates = mesh["coordinates"]
        mass = mass_matrix(coordinates, mesh["triangles"])
    with np.load(HERE / "data" / "nearzero.npz") as extra:
        k, theta, labels = extra["k"], extra["theta"], extra["fields"]
        inputs = (features(extra) - center) / np.where(scale > 1e-12, scale, 1)
    if len(k) != 6 or np.any(k <= 0) or labels.shape != (len(k), len(coordinates)):
        raise ValueError("Unexpected near-zero diagnostic queries")
    for family in ("fourier", "chebyshev"):
        out = HERE / family
        signature = json.loads((out / "config.json").read_text())["signature"]
        checkpoint = torch.load(out / f"n1000_{signature}.pt", map_location="cpu")
        model = SpectralSNO().eval()
        model.load_state_dict(checkpoint["weights"])
        with torch.inference_mode():
            coeff = (model(torch.tensor(inputs, dtype=torch.float32)) * checkpoint["scale"]).numpy()
        pred = decode(spectral_basis(coordinates, family, MODES), coeff) * k[:, None]
        error = 100 * np.sqrt(squared_norm(pred - labels, mass) / squared_norm(labels, mass))
        if not np.all(np.isfinite(error)):
            raise ArithmeticError(f"Invalid diagnostic errors for {family}")
        write_rows(out / "nearzero.csv",
                   [dict(row=i, k=ki, theta=ti, error_pct=e)
                    for i, (ki, ti, e) in enumerate(zip(k, theta, error, strict=True))])


if __name__ == "__main__":
    main()
