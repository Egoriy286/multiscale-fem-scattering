"""Fit and independently evaluate ordinary grid FNOs on paired FEM queries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu

from neural_operator.data import mass_matrix
from neural_operator.fno.model import FNO2d
from neural_operator.mesh_transfer import P1Transfer

ROOT = Path(__file__).resolve().parents[1]
SIZES = (400, 600, 800, 1000)
DEFAULT = {"width": 16, "modes": 8, "depth": 3, "padding": 4}
LARGER = {"width": 20, "modes": 10, "depth": 3, "padding": 4}
MATERIAL = ("alpha1", "alpha2", "n1", "n2")
RESOLUTION, BATCH, SEED = 32, 32, 20261202


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def csv_write(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def energies(rows, mass):
    return np.einsum("ij,ij->i", rows.conj(), (mass @ rows.T).T).real


def grid_map(coordinates, resolution):
    lower = coordinates.min(axis=0)
    span = np.ptp(coordinates, axis=0)
    scaled = np.clip((coordinates - lower) / span, 0., 1.) * (resolution - 1)
    base = np.minimum(scaled.astype(int), resolution - 2)
    x, y = base.T
    ax, ay = (scaled - base).T
    columns = np.stack((y * resolution + x, y * resolution + x + 1,
                        (y + 1) * resolution + x, (y + 1) * resolution + x + 1), axis=1)
    weights = np.stack(((1 - ax)*(1 - ay), ax*(1 - ay), (1 - ax)*ay, ax*ay), axis=1)
    transfer = sparse.csr_matrix((weights.ravel(), (np.repeat(np.arange(len(x)), 4), columns.ravel())),
                                 shape=(len(x), resolution**2))
    xx, yy = np.meshgrid(*(np.linspace(l, l+s, resolution) for l, s in zip(lower, span)))
    return transfer, np.column_stack((xx.ravel(), yy.ravel()))


def spatial_mask(coordinates, triangles, tags, points):
    # Area-independent fraction of incident cells at each node, interpolated
    # to the FNO grid. The disjoint material groups are also supplied globally.
    locator = P1Transfer(coordinates, triangles, points)
    count = np.bincount(triangles.ravel(), minlength=len(coordinates))
    groups = []
    for region in (np.isin(tags, (2, 3)), tags == 4):
        nodal = np.bincount(triangles.ravel(), weights=np.repeat(region, 3), minlength=len(coordinates)) / count
        groups.append(locator.apply(nodal))
    return np.asarray(groups)


def inputs_for(q, points, spatial_mask, train):
    material = np.stack([q[name] for name in MATERIAL], axis=1)
    scalar = np.column_stack((q["k"], material))
    scalar = (scalar - scalar[train].mean(axis=0)) / scalar[train].std(axis=0)
    angles = np.column_stack((np.cos(q["theta"]), np.sin(q["theta"])))
    phase = (angles @ points.T) * q["k"][:, None]
    medium_fields = []
    for raw in (material[:, :2], material[:, 2:]):
        values = 1 + (raw - 1) @ spatial_mask
        medium_fields.append((values - values[train].mean()) / values[train].std())
    nrows, npoints = phase.shape
    xy = np.broadcast_to((points - np.mean(points, axis=0)) / (np.ptp(points, axis=0)/2), (nrows, npoints, 2))
    global_features = np.broadcast_to(np.column_stack((scalar, angles))[:, None], (nrows, npoints, 7))
    local = np.stack((np.cos(phase), np.sin(phase), *medium_fields), axis=-1)
    return np.concatenate((xy, global_features, local), axis=-1).reshape(nrows, RESOLUTION, RESOLUTION, 13).astype(np.float32)


def prepare(root):
    data = root / "data"
    with (data / "complete.json").open() as stream:
        completion = json.load(stream)
    for name in ("queries.npz", "reference_problem.npz", "fields.npy"):
        if sha256(data / name) != completion["sha256"][name]:
            raise ValueError(f"Completed FEM file has a different checksum: {name}")
    with np.load(data / "queries.npz", allow_pickle=False) as archive:
        q = {key: archive[key] for key in ("k", "theta", "medium", "split", *MATERIAL)}
    if np.any(q["k"] <= 0):
        raise ValueError("FNO field normalization requires positive k")
    train_ids, val, test = (np.flatnonzero(q["split"] == s) for s in (0, 1, 2))
    sets = [set(q["medium"][ids]) for ids in (train_ids, val, test)]
    if [len(ids) for ids in (train_ids, val, test)] != [1000, 120, 120] or [len(s) for s in sets] != [100, 12, 12]:
        raise ValueError("Expected 100/12/12 full media in train/validation/test")
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Medium leaked across splits")
    _, first = np.unique(q["medium"][train_ids], return_index=True)
    media = q["medium"][train_ids[np.sort(first)]]
    nested = {n: train_ids[np.isin(q["medium"][train_ids], media[:n // 10])] for n in SIZES}
    for n, ids in nested.items():
        if len(ids) != n:
            raise ValueError(f"{n} train cases do not form a complete medium prefix")
    with np.load(data / "reference_problem.npz", allow_pickle=False) as geometry:
        coordinates, triangles, tags = (geometry[key] for key in ("coordinates", "triangles", "cell_tags"))
    if set(tags) != {1, 2, 3, 4}:
        raise ValueError("Unknown reference material markers")
    mass = mass_matrix(coordinates, triangles)
    interpolate, points = grid_map(coordinates, RESOLUTION)
    if abs(interpolate @ np.ones(interpolate.shape[1]) - 1).max() > 1e-12:
        raise ArithmeticError("Invalid reference-node interpolation")
    gram = (interpolate.T @ mass @ interpolate).tocsr()
    factor = splu(gram.tocsc())
    labels = np.load(data / "fields.npy", mmap_mode="r", allow_pickle=False)
    if labels.shape != (len(q["k"]), len(coordinates)) or labels.dtype != np.complex128:
        raise ValueError("Unexpected native FEM target shape or dtype")
    targets = np.empty((len(labels), interpolate.shape[1]), dtype=complex)
    ref_energy, floor = np.empty(len(labels)), np.empty(len(labels))
    for start in range(0, len(labels), 24):
        sl = slice(start, start + 24)
        fields = labels[sl] / q["k"][sl, None]
        rhs = interpolate.T @ (mass @ fields.T)
        coef = (factor.solve(rhs.real) + 1j * factor.solve(rhs.imag)).T
        targets[sl] = coef
        ref_energy[sl] = energies(fields, mass)
        floor[sl] = energies(fields - (interpolate @ coef.T).T, mass)
    if not np.isfinite(targets).all() or np.any(ref_energy <= 0) or np.any(floor < 0):
        raise ArithmeticError("Invalid consistent FEM projection")
    mask = spatial_mask(coordinates, triangles, tags, points)
    return dict(q=q, nested=nested, val=val, test=test, labels=labels, mass=mass,
                transfer=interpolate, gram=gram, targets=targets, floor=floor,
                ref_energy=ref_energy, points=points, mask=mask, fem_sha256=completion["sha256"])


def fit_config(grid, n, config, epochs, patience, path, *, stage, extension_epoch=None):
    start = perf_counter()
    train = grid["nested"][n]
    x = torch.from_numpy(inputs_for(grid["q"], grid["points"], grid["mask"], train))
    targets = grid["targets"]
    mean = targets[train].mean(axis=0)
    scale = float(np.sqrt(np.mean(np.abs(targets[train] - mean)**2)))
    if scale <= 0:
        raise ValueError("Training target has no variance")
    desired = (targets - mean) / scale
    y = torch.tensor(np.stack((desired.real, desired.imag), axis=2), dtype=torch.float32)
    coo = grid["gram"].tocoo()
    gram = torch.sparse_coo_tensor(np.stack((coo.row, coo.col)), coo.data.astype(np.float32), coo.shape, check_invariants=True).coalesce()
    floor = torch.tensor(grid["floor"] / scale**2, dtype=torch.float32)
    denom = torch.tensor(grid["ref_energy"] / scale**2, dtype=torch.float32)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = FNO2d(input_channels=x.shape[-1], **config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=9, factor=.5, min_lr=1e-5)
    val = grid["val"]
    best = np.inf
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for ix in np.array_split(rng.permutation(train), (len(train) + BATCH - 1)//BATCH):
            prediction = model(x[ix]).reshape(len(ix), -1, 2)
            delta = prediction - y[ix]
            flat = delta.permute(0, 2, 1).reshape(-1, coo.shape[0])
            error = (flat * torch.sparse.mm(gram, flat.T).T).sum(1).reshape(-1, 2).sum(1)
            loss = torch.sqrt((error.clamp_min(0) + floor[ix]) / denom[ix]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        mapped = predict_grid(model, x, val, mean, scale)
        metrics = relative_grid(grid, mapped, val)
        val_error = float(metrics.mean())
        if not np.isfinite(val_error):
            raise ArithmeticError("Nonfinite validation error")
        scheduler.step(val_error)
        history.append(dict(epoch=epoch, train_pct=100 * np.mean(losses),
                            val_pct=100*val_error, lr=optimizer.param_groups[0]["lr"]))
        if val_error < best:
            best = val_error
            best_epoch = epoch
            payload = dict(state=model.state_dict(), config=config, n=n, epoch=epoch,
                           val_pct=100*best, mean=torch.from_numpy(mean), scale=scale,
                           parameters=sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()))
            temp = path.with_suffix(".tmp")
            torch.save(payload, temp)
            temp.replace(path)
        if epoch % 5 == 0 or epoch == 1:
            print(f"{stage} n={n} epoch={epoch}/{epochs}: validation {100*val_error:.4f}%, best {100*best:.4f}%, {perf_counter()-start:.1f}s", flush=True)
        if epoch == extension_epoch:
            baseline = min(row["val_pct"] for row in history[:epoch-15])
            if 100 * best >= .99 * baseline:
                print(f"{stage} n={n}: validation improved by under 1% relative in the latest 15 epochs", flush=True)
                break
            print(f"{stage} n={n}: continuing after validation improvement over the latest 15 epochs", flush=True)
        if epoch - best_epoch >= patience:
            break
    csv_write(path.with_suffix(".csv"), history)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return dict(checkpoint=path.name, epoch=checkpoint["epoch"], epochs_run=epoch,
                val_pct=checkpoint["val_pct"], parameters=checkpoint["parameters"],
                seconds=perf_counter()-start)


@torch.no_grad()
def predict_grid(model, x, rows, mean, scale):
    model.eval()
    predictions = []
    for j in range(0, len(rows), BATCH):
        idx = rows[j:j+BATCH]
        pred = model(x[idx]).numpy().astype(np.float64).reshape(len(idx), -1, 2)
        predictions.append((pred[..., 0] + 1j * pred[..., 1]) * scale + mean)
    return np.concatenate(predictions, axis=0)


def relative_grid(grid, predictions, rows):
    error = energies(predictions - grid["targets"][rows], grid["gram"]) + grid["floor"][rows]
    return np.sqrt(error / grid["ref_energy"][rows])


def evaluate(root, grid, n, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = FNO2d(input_channels=13, **checkpoint["config"])
    model.load_state_dict(checkpoint["state"])
    train = grid["nested"][n]
    x = torch.from_numpy(inputs_for(grid["q"], grid["points"], grid["mask"], train))
    test = grid["test"]
    pred_grid = predict_grid(model, x, test, checkpoint["mean"].numpy(), checkpoint["scale"])
    val = grid["val"]
    pred_val = predict_grid(model, x, val, checkpoint["mean"].numpy(), checkpoint["scale"])
    q = grid["q"]
    scores = {}
    for scope, rows, mapped in (("test", test, pred_grid), ("val", val, pred_val)):
        errors = []
        for j in range(0, len(rows), BATCH):
            idx = rows[j:j+BATCH]
            decoded = (grid["transfer"] @ mapped[j:j+BATCH].T).T * q["k"][idx, None]
            exact = grid["labels"][idx]
            errors.append(np.sqrt(energies(decoded - exact, grid["mass"]) / energies(exact, grid["mass"])))
        errors = np.concatenate(errors)
        if not np.isfinite(errors).all() or np.max(np.abs(errors - relative_grid(grid, mapped, rows))) > 1e-10:
            raise ArithmeticError("Direct consistent FEM replay differs from grid Gram identity")
        scores[scope] = 100 * errors
    if abs(scores["val"].mean() - checkpoint["val_pct"]) > 1e-10:
        raise ArithmeticError("Saved checkpoint validation mismatch")
    if n == 1000:
        pos = int(np.flatnonzero(q["k"][test] == 4.)[0])
        row = test[pos]
        field = (grid["transfer"] @ pred_grid[pos]) * q["k"][row]
        np.savez(root / "fno" / "example.npz", field=field, row=row, k=q["k"][row], theta=q["theta"][row])
    out = scores["test"]
    summary = dict(n=n, parameters=checkpoint["parameters"], mean_pct=float(out.mean()),
                   median_pct=float(np.median(out)), p95_pct=float(np.quantile(out, .95)),
                   val_pct=float(scores["val"].mean()))
    per_case = [dict(n=n, row=int(i), error_pct=float(e)) for i, e in zip(test, out)]
    return summary, per_case



def evaluate_nearzero(root, grid, output, checkpoint_path):
    path = root / "data" / "nearzero.npz"
    with np.load(path, allow_pickle=False) as archive:
        special = {key: archive[key] for key in ("k", "theta", *MATERIAL)}
        fields = archive["fields"]
    if (fields.shape != (6, grid["mass"].shape[0]) or not np.iscomplexobj(fields)
            or not np.array_equal(np.unique(special["k"]), [.001, .01, .025])):
        raise ValueError("Unexpected near-zero FEM diagnostic input")
    train = grid["nested"][1000]
    combined = {key: np.concatenate((grid["q"][key][train], value))
                for key, value in special.items()}
    x = torch.from_numpy(inputs_for(combined, grid["points"], grid["mask"], np.arange(len(train))))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = FNO2d(input_channels=13, **checkpoint["config"])
    model.load_state_dict(checkpoint["state"])
    native = predict_grid(model, x, np.arange(len(train), len(train)+len(fields)),
                          checkpoint["mean"].numpy(), checkpoint["scale"])
    decoded = (grid["transfer"] @ native.T).T * special["k"][:, None]
    errors = 100 * np.sqrt(energies(decoded - fields, grid["mass"]) / energies(fields, grid["mass"]))
    if not np.isfinite(errors).all():
        raise ArithmeticError("Nonfinite near-zero FEM errors")
    rows = [dict(row=i, k=float(k), theta=float(angle), error_pct=float(e))
            for i, (k, angle, e) in enumerate(zip(special["k"], special["theta"], errors))]
    csv_write(output / "nearzero.csv", rows)
    return sha256(path)


def run(root, epochs, threads, mode):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    print("Preparing consistent FEM projection for the grid", flush=True)
    grid = prepare(root)
    output = root / "fno"
    runs = output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    if mode == "fit":
        trials = [fit_config(grid, 1000, c, 12, 12, runs/f"scout_{j}.pt", stage=f"scout{j}")
                  for j, c in enumerate((DEFAULT, LARGER))]
        choice = min(range(len(trials)), key=lambda j: trials[j]["val_pct"])
        config = (DEFAULT, LARGER)[choice]
        extension_epoch = 50 if epochs > 50 else None
        results = [fit_config(grid, n, config, epochs, 20, runs/f"fno_{n}.pt",
                              stage="main", extension_epoch=extension_epoch) for n in SIZES]
        (runs/"training.json").write_text(json.dumps(dict(scouts=trials, selected=choice, config=config,
                                                          results=results, epoch_limit=epochs, extension_epoch=extension_epoch,
                                                          extension_min_relative_gain_pct=1.), indent=2)+"\n")
    summaries, cases = [], []
    for n in SIZES:
        summary, per_case = evaluate(root, grid, n, runs/f"fno_{n}.pt")
        print(summary, flush=True)
        summaries.append(summary)
        cases.extend(per_case)
    csv_write(output / "summary.csv", summaries)
    csv_write(output / "cases.csv", cases)
    edges = np.linspace(.05, 4., 4)
    band_rows = []
    for n in SIZES:
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            error = np.asarray([row["error_pct"] for row in cases if row["n"] == n
                                and lo <= grid["q"]["k"][row["row"]]
                                and (grid["q"]["k"][row["row"]] < hi
                                     or j == 2 and grid["q"]["k"][row["row"]] == hi)])
            band_rows.append(dict(n=n, k_lo=float(lo), k_hi=float(hi), count=len(error),
                                  mean_pct=float(error.mean()), median_pct=float(np.median(error)),
                                  p95_pct=float(np.quantile(error, .95))))
    csv_write(output / "bands.csv", band_rows)
    diagnostic_hash = evaluate_nearzero(root, grid, output, runs / "fno_1000.pt")
    verification = dict(passed=True, train_sizes=list(SIZES), test_rows=grid["test"].tolist(),
                        observed_mesh_bilinear_floor_pct=dict(
                            val=float(100 * np.mean(np.sqrt(grid["floor"][grid["val"]] / grid["ref_energy"][grid["val"]]))),
                            test=float(100 * np.mean(np.sqrt(grid["floor"][grid["test"]] / grid["ref_energy"][grid["test"]])))),
                        hashes={name: sha256(output/name) for name in ("summary.csv", "cases.csv", "bands.csv", "nearzero.csv", "example.npz")},
                        fem_sha256=grid["fem_sha256"], nearzero_input_sha256=diagnostic_hash,
                        runtime=dict(backend="cpu", torch_version=torch.__version__, threads=threads),
                        checkpoints={str(n):sha256(runs/f"fno_{n}.pt") for n in SIZES},
                        training=json.loads((runs/"training.json").read_text()),
                        validation_and_test="Direct complex nodal P1 consistent mass and independent grid Gram replay")
    (output/"verification.json").write_text(json.dumps(verification, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "score"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.root.resolve(), args.epochs, args.threads, args.mode)
