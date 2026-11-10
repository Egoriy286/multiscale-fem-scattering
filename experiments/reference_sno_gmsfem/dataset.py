"""Exact GMsFEM-reference labels, with compact materials and resumable NPY storage.

The returned object is the normal neural_operator.data.Dataset. This module
owns its custom config/storage; neural_operator.data.load_dataset is for NPZ
archives and must not be used to open these experiment directories.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from neural_operator.data import Dataset, mass_matrix

REPO = Path(__file__).resolve().parents[2]
REFERENCE_MESH = REPO / "gmsfem/meshes/reference_n20_h0p04.msh"
REFERENCE_SHA256 = "9273efc7c0b500209ff847de4ee36d3790b93d64b09e67f62348318ed6b6cb47"
ANCHOR_FIELDS = REPO / "gmsfem/results/notebook/fields.npz"
BANDS = {"low": (.1, 2.), "high": (2., 4.)}
SCHEMA = 1


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _save_npz(path, **arrays):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def _save_npy(path, values):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def make_queries(band="low", n_frequencies=160, n_directions=8, seed=2026):
    """Plan disjoint frequency groups; all endpoints and low k=1 are test.

    n_frequencies counts the linspace grid. Low-band k=1 is appended if absent,
    making one extra group. It is not appended outside the high band's range.
    Counts floor .67/.17 of the actual group count, with remainder for test.
    """
    if band not in BANDS:
        raise ValueError(f"band must be one of {tuple(BANDS)}")
    for name, value, minimum in (("n_frequencies", n_frequencies, 10),
                                  ("n_directions", n_directions, 1),
                                  ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    frequencies = np.linspace(*BANDS[band], n_frequencies)
    if band == "low" and not np.any(frequencies == 1.):
        frequencies = np.sort(np.r_[frequencies, 1.])
    groups = len(frequencies)
    forced = (frequencies == frequencies[0]) | (frequencies == frequencies[-1])
    if band == "low":
        forced |= frequencies == 1.
    ntrain, nval = np.floor(np.array([.67, .17]) * groups).astype(int)
    remaining = np.flatnonzero(~forced)
    if len(remaining) < ntrain + nval:
        raise ValueError("Too few frequency groups to reserve the benchmark test groups")
    permutation = np.random.default_rng(seed).permutation(remaining)
    group_split = np.full(groups, 2, dtype=np.int8)
    group_split[permutation[:ntrain]] = 0
    group_split[permutation[ntrain:ntrain + nval]] = 1
    frequency_id = np.repeat(np.arange(groups, dtype=np.int32), n_directions)
    theta = np.empty(groups * n_directions)
    for group, k in enumerate(frequencies):
        # theta=0 at k=1 reproduces the existing table's incident direction.
        offset = 0. if band == "low" and k == 1. else np.random.default_rng(
            [seed, group, 2]).uniform(0, 2 * np.pi / n_directions)
        theta[group * n_directions:(group + 1) * n_directions] = (
            offset + 2 * np.pi * np.arange(n_directions) / n_directions)
    return dict(frequencies=frequencies, frequency_id=frequency_id,
                k0=frequencies[frequency_id], theta=theta,
                medium_id=np.zeros(len(theta), dtype=np.int32),
                split=group_split[frequency_id])


def _check_geometry(solver, anchor):
    if len(solver.coordinates) != 53246 or len(solver.triangles) != 105690:
        raise ValueError("Expected the existing 53246-DOF/105690-triangle reference mesh")
    if (solver.coordinates.shape != anchor["reference_coordinates"].shape or
            not np.allclose(solver.coordinates, anchor["reference_coordinates"], rtol=0, atol=1e-13) or
            not np.array_equal(solver.triangles, anchor["reference_triangles"])):
        raise ValueError("Reference mesh DOF order/connectivity differs from stored table fields")


def _contrast_rhs(solver, alpha, n, k, theta):
    """Same element quadrature, omitting exactly zero background contributions."""
    cells = np.flatnonzero((np.asarray(alpha) != 1.) | (np.asarray(n) != 1.))
    direction = np.array([np.cos(theta), np.sin(theta)])
    incident = np.exp(1j * k * (solver.qpoints[cells] @ direction))
    integral = np.einsum("cq,q->c", incident, solver.qweights) * solver.areas[cells]
    mass_integral = np.einsum("cq,qi,q->ci", incident, solver.phi,
                              solver.qweights) * solver.areas[cells, None]
    local = ((1 - alpha[cells])[:, None] *
             (1j * k * (solver.gradients[cells] @ direction)) * integral[:, None] +
             k * k * (n[cells] - 1)[:, None] * mass_integral)
    result = np.zeros(len(solver.coordinates), dtype=np.complex128)
    np.add.at(result, solver.triangles[cells].ravel(), local.ravel())
    return result


def _reference_problem(directory):
    """Prepare the exact materials and independently validate the k=1 anchor."""
    from gmsfem.fem import MATERIALS
    from gmsfem.geometry import inclusion_circles
    from neural_operator.fem import HelmholtzSolver

    mesh = directory / "etal.msh"
    if _sha256(REFERENCE_MESH) != REFERENCE_SHA256:
        raise ValueError("Authoritative reference mesh checksum changed")
    if not mesh.exists():
        shutil.copyfile(REFERENCE_MESH, mesh)
    if _sha256(mesh) != REFERENCE_SHA256:
        raise ValueError("Experiment mesh differs from the authoritative reference mesh")
    class ContrastOnlySolver(HelmholtzSolver):
        rhs = _contrast_rhs

    solver = ContrastOnlySolver(mesh)
    with np.load(ANCHOR_FIELDS, allow_pickle=False) as anchor:
        _check_geometry(solver, anchor)
        stored_anchor = anchor["reference_scattered"]
    tags = solver.cell_tags
    if not set(np.unique(tags)).issubset(MATERIALS["notebook"]) or np.any(tags == 0):
        raise ValueError("Unexpected physical material tags")
    alpha = np.array([MATERIALS["notebook"][int(tag)][0] for tag in tags])
    n = np.array([MATERIALS["notebook"][int(tag)][1] for tag in tags])
    started = perf_counter()
    rhs_checks = []
    for k, theta in ((1., 0.), (4., .173)):
        full = HelmholtzSolver.rhs(solver, alpha, n, k, theta)
        optimized = solver.rhs(alpha, n, k, theta)
        difference = float(np.linalg.norm(full - optimized) / np.linalg.norm(full))
        if not np.isfinite(difference) or difference > 1e-12:
            raise ValueError(f"Contrast-only RHS verification failed at k={k}: {difference:g}")
        rhs_checks.append(dict(k=k, theta=theta, relative_difference=difference))
    computed_anchor = solver.solve(alpha, n, 1., 0.)
    mass = mass_matrix(solver.coordinates, solver.triangles)
    delta = computed_anchor - stored_anchor
    error = float(np.sqrt(np.real(np.vdot(delta, mass @ delta)) /
                          np.real(np.vdot(stored_anchor, mass @ stored_anchor))))
    residual = solver.build_problem(alpha, n, 1., 0.)
    relative_residual = float(np.linalg.norm(residual.operator @ computed_anchor - residual.rhs) /
                              np.linalg.norm(residual.rhs))
    if not np.isfinite(error) or error > 1e-6:
        raise ValueError(f"Solver anchor differs from stored table reference: relative L2={error:g}")
    if not np.isfinite(relative_residual) or relative_residual > 1e-8:
        raise ValueError(f"Solver anchor residual is too large: {relative_residual:g}")
    arrays = dict(coordinates=solver.coordinates, triangles=solver.triangles,
                  cell_tags=tags, circles=inclusion_circles(),
                  A=solver.nodal_coefficients(alpha)[None],
                  n=solver.nodal_coefficients(n)[None],
                  A_cells=alpha[None], n_cells=n[None],
                  anchor_scattered=computed_anchor, anchor_stored_scattered=stored_anchor)
    path = directory / "reference_problem.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[key], value)
                                                   for key, value in arrays.items()
                                                   if not key.startswith("anchor_")):
                raise ValueError("Saved reference geometry/materials differ from the current exact source")
    else:
        _save_npz(path, **arrays)
    provenance = dict(schema_version=SCHEMA, case="notebook",
                      reference_mesh=str(REFERENCE_MESH.relative_to(REPO)),
                      reference_mesh_sha256=REFERENCE_SHA256,
                      anchor_fields=str(ANCHOR_FIELDS.relative_to(REPO)),
                      anchor_fields_sha256=_sha256(ANCHOR_FIELDS),
                      materials={str(k): list(v) for k, v in MATERIALS["notebook"].items()},
                      nodes=len(solver.coordinates), triangles=len(solver.triangles),
                      anchor=dict(k=1., theta=0., mesh_order_verified=True,
                                  consistent_relative_l2=error,
                                  normalized_linear_residual=relative_residual,
                                  validation_seconds=perf_counter() - started),
                      rhs_optimization=dict(method="exact zero-contrast background omission",
                                            active_cells=int(np.count_nonzero((alpha != 1) | (n != 1))),
                                            checks=rhs_checks),
                      generator_sha256=_sha256(Path(__file__)),
                      fem_backend_sha256=_sha256(REPO / "neural_operator/fem.py"))
    _write_json(directory / "provenance.json", provenance)
    return solver, arrays


def generate_dataset(root, band="low", n_frequencies=160, n_directions=8, seed=2026):
    """Create/resume one band directly in root; a completed result opens read-only.

    Completion is committed only after every row of one frequency is flushed.
    A crash can repeat the current frequency, and never exposes partial labels
    as a completed Dataset. Configuration mismatches are always rejected.
    """
    queries = make_queries(band, n_frequencies, n_directions, seed)
    directory = Path(root).expanduser().resolve()
    recipe = dict(schema_version=SCHEMA, storage="native_npy_complex128",
                  band=band, k_range=list(BANDS[band]), n_frequencies=int(n_frequencies),
                  actual_frequency_groups=len(queries["frequencies"]),
                  n_directions=int(n_directions), seed=int(seed), medium_mode="static",
                  case="notebook", split_fractions=[.67, .17, .16],
                  heldout_policy="all endpoint groups; low k=1 group including theta=0",
                  reference_mesh_sha256=REFERENCE_SHA256, h=.02, n_coarse=20)
    config_path = directory / "config.json"
    directory.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        if json.loads(config_path.read_text()) != recipe:
            raise ValueError("Existing experiment dataset has a different configuration")
    else:
        if any(directory.iterdir()):
            raise ValueError("Nonempty dataset directory has no config.json")
        _write_json(config_path, recipe)
    if (directory / "dataset_info.json").exists():
        return load_dataset(directory)
    solver, problem = _reference_problem(directory)
    samples_path = directory / "samples.npz"
    if samples_path.exists():
        with np.load(samples_path, allow_pickle=False) as saved:
            if set(saved.files) != set(queries) or any(not np.array_equal(saved[k], v)
                                                      for k, v in queries.items()):
                raise ValueError("Saved wave parameters/splits differ from the exact recipe")
    else:
        _save_npz(samples_path, **queries)
    labels_path = directory / "u_scat.npy"
    shape = (len(queries["theta"]), len(solver.coordinates))
    if labels_path.exists():
        labels = np.load(labels_path, mmap_mode="r+", allow_pickle=False)
        if labels.shape != shape or labels.dtype != np.complex128:
            raise ValueError("Saved labels have a different shape or dtype")
    else:
        labels = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.complex128, shape=shape)
        labels[:] = np.nan + 1j * np.nan
        labels.flush()
    completed_path = directory / "completed.npy"
    completed = (np.load(completed_path, allow_pickle=False) if completed_path.exists()
                 else np.zeros(len(queries["frequencies"]), dtype=bool))
    if completed.shape != (len(queries["frequencies"]),) or completed.dtype != bool:
        raise ValueError("Invalid frequency completion bitmap")
    started = perf_counter()
    completed_before = int(completed.sum())
    for group, k in enumerate(queries["frequencies"]):
        rows = slice(group * n_directions, (group + 1) * n_directions)
        if completed[group]:
            if not np.all(np.isfinite(labels[rows])):
                raise ValueError(f"Committed frequency {group} contains nonfinite labels")
            continue
        for index in range(rows.start, rows.stop):
            # Copy the independently checked solve into its reserved benchmark row.
            labels[index] = (problem["anchor_scattered"] if k == 1. and queries["theta"][index] == 0.
                             else solver.solve(problem["A_cells"][0], problem["n_cells"][0],
                                               float(k), float(queries["theta"][index])))
        if not np.all(np.isfinite(labels[rows])):
            raise FloatingPointError(f"Nonfinite FEM labels in frequency {group}")
        labels.flush()
        completed[group] = True
        _save_npy(completed_path, completed)
        if completed.sum() == completed_before + 1 or completed.sum() % 10 == 0 or completed.all():
            print(f"{band} reference FEM: {completed.sum()}/{len(completed)} frequencies, "
                  f"{completed.sum() * n_directions}/{len(labels)} fields; "
                  f"current session {perf_counter() - started:.1f}s", flush=True)
    labels.flush()
    # Structural/source hashes are small; labels are hashed as a stream once.
    info = dict(schema_version=SCHEMA, complete=True, samples=shape[0], nodes=shape[1],
                triangles=len(solver.triangles), media=1, split_strategy="frequency",
                frequency_groups=len(completed), raw_label_bytes=int(labels.nbytes),
                current_generation_session_seconds=perf_counter() - started,
                resumed_frequency_groups=completed_before,
                splits=[dict(name=name, samples=int(np.count_nonzero(queries["split"] == j)),
                             groups=len(np.unique(queries["frequency_id"][queries["split"] == j])))
                        for j, name in enumerate(("train", "validation", "test"))],
                sha256={name: _sha256(directory / name) for name in
                        ("etal.msh", "reference_problem.npz", "samples.npz", "u_scat.npy")})
    _write_json(directory / "dataset_info.json", info)
    del labels
    return load_dataset(directory)


def load_dataset(path):
    """Open complete native storage; labels mmap and material broadcasts are read-only."""
    directory = Path(path).expanduser().resolve()
    info = json.loads((directory / "dataset_info.json").read_text())
    if info.get("schema_version") != SCHEMA or info.get("complete") is not True:
        raise ValueError("Dataset is incomplete or uses an unsupported storage schema")
    if _sha256(directory / "etal.msh") != REFERENCE_SHA256:
        raise ValueError("Saved experiment reference mesh checksum differs")
    for name in ("reference_problem.npz", "samples.npz"):
        if _sha256(directory / name) != info["sha256"][name]:
            raise ValueError(f"Saved {name} checksum differs")
    with np.load(directory / "reference_problem.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("coordinates", "triangles", "cell_tags", "circles")}
        materials = {key: archive[key] for key in ("A", "n", "A_cells", "n_cells")}
    with np.load(directory / "samples.npz", allow_pickle=False) as archive:
        data.update({key: archive[key] for key in ("theta", "k0", "medium_id", "split", "frequency_id")})
    data["u_scat"] = np.load(directory / "u_scat.npy", mmap_mode="r", allow_pickle=False)
    if data["u_scat"].shape != (info["samples"], info["nodes"]) or data["u_scat"].dtype != np.complex128:
        raise ValueError("Saved labels have a different shape or dtype")
    completed = np.load(directory / "completed.npy", allow_pickle=False)
    if completed.shape != (info["frequency_groups"],) or completed.dtype != bool or not completed.all():
        raise ValueError("Dataset completion bitmap is not fully committed")
    for name, values in materials.items():
        if values.shape[0] != 1:
            raise ValueError("Static materials must be stored as one row")
        data[name] = np.broadcast_to(values, (info["samples"], values.shape[1]))
    return Dataset(**data, split_strategy="frequency", directory=directory)
