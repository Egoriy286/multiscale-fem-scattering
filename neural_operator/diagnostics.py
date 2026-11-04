"""Train-only SVD diagnostics and validation reconstruction curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import subspace_angles

from .basis import PODBasis
from .grid import GridTransfer


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def svd_diagnostics(dataset, output_dir, max_rank=64, seed=42, methods=("full", "randomized")):
    """Compare invariant subspaces, not the arbitrary signs of singular vectors."""
    train, val, _ = dataset.split_indices()
    if not len(train) or not len(val):
        raise ValueError("SVD diagnostics require train and validation samples.")
    if not methods or any(method not in ("full", "randomized") for method in methods):
        raise ValueError("methods must contain full, randomized, or both")
    methods = tuple(method for method in ("full", "randomized") if method in methods)
    _, first = np.unique(dataset.medium_id[train], return_index=True)
    unique_train = train[first]
    weights = dataset.lumped_weights
    rows, bases = [], {}
    for name, fields, indices in (("A", dataset.A, unique_train),
                                   ("n", dataset.n, unique_train),
                                   ("real", dataset.u_scat.real, train),
                                   ("imag", dataset.u_scat.imag, train)):
        bases[name] = {}
        for method in methods:
            started = perf_counter()
            basis = PODBasis.fit(fields[indices], rank=max_rank, weights=weights,
                                 method=method, seed=seed)
            elapsed = perf_counter()-started
            bases[name][method] = basis
            gram = basis.modes.T @ (weights[:, None] * basis.modes)
            row = {"field": name, "method": method, "snapshots": len(indices),
                   "algorithm_used": ("constant" if basis.total_energy == 0 else
                                      "full" if method == "full" or min(max_rank, len(indices), fields.shape[1])+10
                                      >= min(len(indices), fields.shape[1]) else "randomized"),
                   "computed_rank": basis.rank, "seconds": elapsed,
                   "orthogonality_fro": float(np.linalg.norm(gram-np.eye(basis.rank))),
                   "retained_centered_energy": float(basis.energy[-1]) if basis.rank else 0.,
                   "snapshot_array_bytes": int(fields[indices].nbytes),
                   "mode_array_bytes": int(basis.modes.nbytes)}
            if method == "randomized" and "full" in bases[name]:
                exact = bases[name]["full"]
                rank = min(exact.rank, basis.rank)
                if rank:
                    a = exact.modes[:, :rank] * np.sqrt(weights)[:, None]
                    b = basis.modes[:, :rank] * np.sqrt(weights)[:, None]
                    row["largest_subspace_angle_degrees"] = float(np.degrees(subspace_angles(a, b)).max())
                    row["singular_error_relative_to_leading"] = float(np.max(np.abs(
                        exact.singular_values[:rank]-basis.singular_values[:rank])) / exact.singular_values[0])
            rows.append(row)
    curves = []
    for method in methods:
        br, bi = bases["real"][method], bases["imag"][method]
        cr, ci = br.encode(dataset.u_scat[val].real), bi.encode(dataset.u_scat[val].imag)
        for rank in range(1, min(br.rank, bi.rank)+1):
            approximation = (br.mean + cr[:, :rank] @ br.modes[:, :rank].T
                             + 1j*(bi.mean + ci[:, :rank] @ bi.modes[:, :rank].T))
            error = dataset.relative_l2(dataset.u_scat[val], approximation)
            curves.append({"method": method, "rank": rank,
                           "mean_fem_l2": float(error.mean()),
                           "p95_fem_l2": float(np.percentile(error, 95))})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir/"svd_summary.csv", rows)
    write_csv(output_dir/"validation_ranks.csv", curves)
    spectra = {f"{name}_{method}": basis.singular_values
               for name, methods in bases.items() for method, basis in methods.items()}
    spectra.update({f"{name}_{method}_energy": basis.energy
                    for name, methods in bases.items() for method, basis in methods.items()})
    np.savez_compressed(output_dir/"spectra.npz", **spectra)
    (output_dir/"protocol.json").write_text(json.dumps({
        "seed": seed, "max_rank": max_rank, "pod_metric": "mass_lumped",
        "methods": list(methods),
        "reconstruction_metric": "consistent_complex_FEM_L2", "split": "validation_only",
        "randomized_oversampling": 10, "power_iterations": 2,
        "small_problem_fallback": "full SVD if requested rank + 10 reaches the smaller matrix dimension",
        "memory_columns": "stored arrays, not peak process/BLAS memory",
    }, indent=2)+"\n")
    return rows, curves, spectra


def transfer_diagnostics(dataset, indices=None, resolution=32):
    """Isolate representation/interpolation error from neural prediction error."""
    if indices is None:
        indices = dataset.split_indices()[1]
    indices = np.asarray(indices)
    rows = []
    for kind in ("uniform", "chebyshev"):
        transfer = GridTransfer(dataset.coordinates, dataset.triangles, resolution, kind)
        for field in ("A", "n", "u_scat"):
            values = getattr(dataset, field)[indices]
            returned = transfer.to_nodes(transfer.to_grid(values))
            errors = dataset.relative_l2(values, returned)
            rows.append({"grid": kind, "resolution": resolution, "field": field,
                         "mean_fem_l2": float(errors.mean()),
                         "p95_fem_l2": float(np.percentile(errors, 95))})
    return rows


def chebyshev_reconstruction(dataset, indices, resolution=32, modes=8):
    """Truncated Lobatto transform, with the same train mean as the model.

    This is an interpolation/truncation diagnostic, not the minimum possible
    consistent FEM error over the polynomial span.
    """
    import torch
    from .models import chebyshev_matrices
    transfer = GridTransfer(dataset.coordinates, dataset.triangles, resolution, "chebyshev")
    train = dataset.split_indices()[0]
    mean = dataset.u_scat[train].mean(axis=0)
    grid = transfer.to_grid(dataset.u_scat[indices]-mean)
    analysis, synthesis = [value.numpy().astype(np.float64)
                           for value in chebyshev_matrices(resolution, dtype=torch.float64)]
    coefficients = np.einsum("ih,bhw,jw->bij", analysis[:modes], grid, analysis[:modes])
    returned = np.einsum("hi,bij,wj->bhw", synthesis[:, :modes], coefficients, synthesis[:, :modes])
    return mean + transfer.to_nodes(returned)
