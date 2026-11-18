"""Audit real saved operators without fitting models or solving new PDEs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.reference_sno_gmsfem.dataset import load_dataset
from neural_operator.training import load_predictor
from neural_operator.losses import PODFEMLoss
from neural_operator.chebyshev.model import chebyshev_matrices

OUT = Path(__file__).resolve().parent / "evidence"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    experiment = ROOT / "experiments/shared_sno_gmsfem"
    results = experiment / "results"
    data = load_dataset(experiment / "data")
    predictor = load_predictor(experiment / "runs/shared/model.pt")
    train, val, test = data.split_indices()
    groups = [set(data.k0[i]) for i in (train, val, test)]
    assert not any(groups[i] & groups[j] for i in range(3) for j in range(i))
    assert (len(train), len(val), len(test)) == (1712, 432, 424)
    assert predictor.model.architecture_version == 2
    assert predictor.bases["A"].rank == predictor.bases["n"].rank == 0
    assert predictor.bases["real"].rank == predictor.bases["imag"].rank == 61
    verification = json.loads((results / "verification.json").read_text())
    assert sha(experiment / "runs/shared/model.pt") == verification["checkpoint_sha256"]
    assert sha(experiment / "data/u_scat.npy") == verification["source_label_sha256"]
    for filename, expected in verification["score_sha256"].items():
        assert sha(results / filename) == expected
    protocol = json.loads((results / "protocol.json").read_text())
    source_parity = {name: sha(ROOT / name) == expected
                     for name, expected in protocol["source_sha256"].items()}
    assert all(source_parity.values()), source_parity
    print("Checked source, labels, split, checkpoint hashes; reloading 424 predictions", flush=True)
    prediction = predictor.predict_dataset(data, test, batch_size=32)
    with np.load(experiment / "runs/shared/test_predictions.npz") as saved:
        assert np.array_equal(saved["indices"], test)
        archived = saved["prediction"]
        reload_error = data.relative_l2(archived, prediction)
    scores = json.loads((results / "sno.json").read_text())["rows"]
    by_index = {r["index"]: r for r in scores}
    assert len(by_index) == len(test)
    measured = data.relative_l2(data.u_scat[test], prediction) * 100
    expected = np.array([by_index[int(i)]["scattered_e_complex_ref"] for i in test])
    metric_difference = float(np.max(np.abs(measured - expected)))
    assert reload_error.max() < 2e-5
    assert metric_difference < 2e-4  # percentage points, permits CPU/Torch differences
    chosen = test[np.linspace(0, len(test) - 1, 12).astype(int)]
    labels = data.u_scat[chosen]
    loss = PODFEMLoss(predictor, data, labels, target_dtype=np.float32)
    features = torch.from_numpy(predictor.encode(data.A[chosen], data.n[chosen],
                                                data.theta[chosen], data.k0[chosen]))
    with torch.inference_mode():
        outputs = predictor.model(features)
        quadratic = float(loss(outputs, torch.arange(len(chosen))))
        decoded = predictor.decode(outputs).numpy()
    rounded = labels.real.astype(np.float32).astype(float) + 1j * labels.imag.astype(np.float32).astype(float)
    direct = float(np.mean(data.relative_l2(rounded, decoded[..., 0] + 1j * decoded[..., 1])))
    assert abs(quadratic - direct) < 2e-7
    transform_errors = {}
    for n in (2, 9, 16, 32):
        a, s = chebyshev_matrices(n, dtype=torch.float64)
        error = float((a @ s - torch.eye(n, dtype=torch.float64)).abs().max())
        assert error < 2e-13
        transform_errors[str(n)] = error
    # A deterministic physical sample check, independent of model test errors.
    with np.load(results / "verification/baseline_fields.npz") as baseline:
        indices = baseline["sample_indices"]
        selected = np.array([j for j, i in enumerate(indices) if data.k0[i] in (1., 4.)])
        indices = indices[selected]
        positions = np.array([int(np.flatnonzero(test == i)[0]) for i in indices])
        np.savez_compressed(OUT / "field_examples.npz", coordinates=data.coordinates,
                            triangles=data.triangles, circles=data.circles,
                            indices=indices, k=data.k0[indices], theta=data.theta[indices],
                            reference=data.u_scat[indices], sno=prediction[positions],
                            gms=baseline["gmsfem_n20_type2_M16"][selected])
    result = dict(passed=True, date="2026-11-13", scope="No retraining or new FEM solves",
                  torch=str(torch.__version__), numpy=str(np.__version__),
                  reloaded_test_fields=len(test), frequency_groups=[len(g) for g in groups],
                  disjoint_physical_frequencies=True, source_hash_parity=source_parity,
                  checkpoint_sha256=verification["checkpoint_sha256"], label_hash_verified=True,
                  max_reload_relative_l2=float(reload_error.max()),
                  max_scattered_metric_difference_percentage_points=metric_difference,
                  loss_identity_samples=len(chosen), quadratic_loss=quadratic,
                  direct_consistent_mass_loss=direct, loss_absolute_difference=abs(quadratic-direct),
                  chebyshev_inverse_max_errors=transform_errors,
                  field_examples_sha256=sha(OUT / "field_examples.npz"),
                  baseline_fem_solves="Not rerun; archived verification checked",
                  full_pde_accuracy="Not established by these checks")
    (OUT / "operator_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
