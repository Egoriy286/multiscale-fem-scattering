"""Train-only SVD diagnostics and validation reconstruction curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import subspace_angles

from .sno.basis import PODBasis
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
    from .chebyshev.model import chebyshev_matrices
    transfer = GridTransfer(dataset.coordinates, dataset.triangles, resolution, "chebyshev")
    train = dataset.split_indices()[0]
    mean = dataset.u_scat[train].mean(axis=0)
    grid = transfer.to_grid(dataset.u_scat[indices]-mean)
    analysis, synthesis = [value.numpy().astype(np.float64)
                           for value in chebyshev_matrices(resolution, dtype=torch.float64)]
    coefficients = np.einsum("ih,bhw,jw->bij", analysis[:modes], grid, analysis[:modes])
    returned = np.einsum("hi,bij,wj->bhw", synthesis[:, :modes], coefficients, synthesis[:, :modes])
    return mean + transfer.to_nodes(returned)


def chebyshev_projection_diagnostics(dataset, *, resolution=32, modes=(8, 12, 16)):
    """Validation-only consistent FEM floors for the actual Chebyshev decoder.

    Unlike DCT truncation, this projects onto the polynomial synthesis followed
    by the model's bilinear grid-to-node transfer. Its span includes the frozen
    train mean and arbitrary real/imag coefficients. The output scales do not
    change that span. It measures representation, without fitting network weights
    or selecting modes from test labels.
    """
    import torch
    from scipy.linalg import cho_factor, cho_solve
    from .chebyshev.model import chebyshev_matrices
    from .training import summarize

    train, val, _ = dataset.split_indices()
    if not len(train) or not len(val):
        raise ValueError("Chebyshev projection requires train and validation samples")
    modes = tuple(modes)
    if not modes or any(not isinstance(m, (int, np.integer)) or m < 1 or m > resolution
                        for m in modes):
        raise ValueError("modes must contain positive integers <= resolution")
    mean = dataset.u_scat[train].mean(axis=0)
    residual = dataset.u_scat[val] - mean
    transfer = GridTransfer(dataset.coordinates, dataset.triangles, resolution, "chebyshev")
    _, synthesis = chebyshev_matrices(resolution, dtype=torch.float64)
    synthesis = synthesis.numpy()
    rows = []
    for count in modes:
        tensor_basis = np.einsum("hi,wj->hwij", synthesis[:, :count], synthesis[:, :count])
        phi = transfer.to_nodes_matrix @ tensor_basis.reshape(resolution ** 2, -1)
        mphi = dataset.mass_matrix @ phi
        gram = phi.T @ mphi
        gram = (gram + gram.T) / 2
        factor = cho_factor(gram)
        coefficients = cho_solve(factor, (residual @ mphi).T).T
        prediction = mean + coefficients @ phi.T
        rows.append({"modes_per_axis": count, "coefficient_dofs_per_component": count ** 2,
                     "resolution": resolution, "validation_samples": len(val),
                     "metric": "consistent_complex_FEM_L2",
                     "projection_floor": summarize(dataset.relative_l2(dataset.u_scat[val], prediction)),
                     "gram_condition_number": float(np.linalg.cond(gram))})
    return rows


def main(argv=None):
    """Reproduce validation-only Chebyshev decoder floors without training."""
    import argparse
    import hashlib
    from .data import load_dataset
    from .training import dataset_fingerprint

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--modes", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset, mmap=True)
    rows = chebyshev_projection_diagnostics(dataset, resolution=args.resolution, modes=args.modes)
    report = {
        "scope": "Validation-only approximation diagnostic; no neural fit and no test inspection",
        "dataset": str(args.dataset), "dataset_fingerprint": dataset_fingerprint(dataset),
        "resolution": args.resolution, "kind": "chebyshev", "mean": "train-only complex FEM field",
        "validation_samples": rows[0]["validation_samples"],
        "method": "Consistent mass least-squares projection into the actual affine decoder span",
        "interpretation": "Lower bound for this fixed decoder; not deployable predictor error or FEM PDE accuracy",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": [{"modes_per_axis": row["modes_per_axis"],
                     "coefficients": row["coefficient_dofs_per_component"],
                     "validation_fem_l2": row["projection_floor"],
                     "gram_condition_number": row["gram_condition_number"]} for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
