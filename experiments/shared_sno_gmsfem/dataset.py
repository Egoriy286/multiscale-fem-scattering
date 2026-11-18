"""One frequency-grouped dataset on [0.1,4], reusing verified FEM snapshots.

All source splits are preserved. The two sets of angles at k=2 become ONE
held-out frequency group. The resulting native storage uses the usual loader.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from experiments.reference_sno_gmsfem.dataset import (
    REFERENCE_SHA256, SCHEMA, _save_npy, _save_npz, _sha256,
    _write_json, load_dataset,
)

SOURCES = ("low", "high")


def merge_queries(sources):
    """Merge query arrays, rejecting split conflicts and duplicate (k,theta)."""
    arrays = {name: [] for name in ("k0", "theta", "split", "medium_id")}
    bands, positions = [], []
    for band in SOURCES:
        source = sources[band]
        n = len(source["k0"])
        for name in arrays:
            value = np.asarray(source[name])
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {band} query array {name}")
            arrays[name].append(value)
        bands.extend([band] * n)
        positions.extend(range(n))
    arrays = {name: np.concatenate(parts) for name, parts in arrays.items()}
    if (not np.isin(arrays["split"], (0, 1, 2)).all()
            or not np.issubdtype(arrays["split"].dtype, np.integer)
            or np.any(arrays["k0"] <= 0) or np.any(arrays["medium_id"] != 0)):
        raise ValueError("Expected positive frequencies, integer splits and fixed medium 0")
    bands, positions = np.asarray(bands), np.asarray(positions, dtype=np.int64)
    order = np.lexsort((positions, bands, arrays["k0"]))
    queries = {name: value[order] for name, value in arrays.items()}
    queries["source_band"], queries["source_index"] = bands[order], positions[order]
    frequencies, groups = np.unique(queries["k0"], return_inverse=True)
    queries["frequency_id"] = groups.astype(np.int32)
    queries["frequencies"] = frequencies
    pairs = np.rec.fromarrays([queries["k0"], queries["theta"] % (2 * np.pi)])
    if len(np.unique(pairs)) != len(pairs):
        raise ValueError("Duplicate physical (k,theta) queries across sources")
    for group in range(len(frequencies)):
        if len(np.unique(queries["split"][groups == group])) != 1:
            raise ValueError(f"Conflicting source splits at k={frequencies[group]}")
    if any(not np.any(queries["split"] == split) for split in (0, 1, 2)):
        raise ValueError("Merged dataset needs nonempty train, validation and test")
    return queries


def _verify_sources(root):
    datasets, snapshots = {}, {}
    reference = None
    for band in SOURCES:
        directory = root / band
        datasets[band] = load_dataset(directory)
        info = json.loads((directory / "dataset_info.json").read_text())
        # Native loader checks small structural files; verify the large labels
        # explicitly before trusting inherited labels or baseline scores.
        if _sha256(directory / "u_scat.npy") != info["sha256"]["u_scat.npy"]:
            raise ValueError(f"Source {band} label checksum differs")
        config = json.loads((directory / "config.json").read_text())
        provenance = json.loads((directory / "provenance.json").read_text())
        if (config.get("case") != "notebook" or config.get("band") != band
                or config.get("reference_mesh_sha256") != REFERENCE_SHA256
                or provenance.get("reference_mesh_sha256") != REFERENCE_SHA256):
            raise ValueError(f"Source {band} does not describe the reference problem")
        with np.load(directory / "reference_problem.npz", allow_pickle=False) as archive:
            problem = {key: archive[key] for key in
                       ("coordinates", "triangles", "cell_tags", "circles",
                        "A", "n", "A_cells", "n_cells")}
        if reference is not None and any(not np.array_equal(problem[k], reference[k])
                                         for k in reference):
            raise ValueError("Source geometry, DOF ordering or materials differ")
        reference = problem
        snapshots[band] = {
            "sha256": info["sha256"], "samples": info["samples"],
            "config_sha256": _sha256(directory / "config.json"),
            "provenance_sha256": _sha256(directory / "provenance.json"),
            "dataset_info_sha256": _sha256(directory / "dataset_info.json"),
        }
    return datasets, snapshots


def prepare_shared_dataset(output_dir, source_root, *, progress=print):
    """Create/resume output_dir without new PDE solves or split changes."""
    root = Path(source_root).expanduser().resolve()
    datasets, snapshots = _verify_sources(root)
    sources = {band: {name: getattr(data, name) for name in
                     ("k0", "theta", "split", "medium_id")}
               for band, data in datasets.items()}
    queries = merge_queries(sources)
    if queries["frequencies"][0] != .1 or queries["frequencies"][-1] != 4.:
        raise ValueError("Source union must cover exactly [0.1,4.0]")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    recipe = dict(schema_version=SCHEMA, storage="native_npy_complex128",
                  band="shared", k_range=[.1, 4.], medium_mode="static", case="notebook",
                  reference_mesh_sha256=REFERENCE_SHA256, h=.02, n_coarse=20,
                  actual_frequency_groups=len(queries["frequencies"]),
                  heldout_policy="preserve source splits; group identical physical k globally",
                  merge_policy="sorted union; no duplicate queries; no train/test promotion",
                  sources=snapshots)
    config = directory / "config.json"
    if config.exists():
        if json.loads(config.read_text()) != recipe:
            raise ValueError("Existing shared dataset uses different sources/settings")
    else:
        if any(directory.iterdir()):
            raise ValueError("Nonempty shared directory has no config.json")
        _write_json(config, recipe)
    info_path = directory / "dataset_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if _sha256(directory / "u_scat.npy") != info["sha256"]["u_scat.npy"]:
            raise ValueError("Shared label checksum differs")
        return load_dataset(directory)
    for name in ("etal.msh", "reference_problem.npz"):
        target = directory / name
        if target.exists():
            if _sha256(target) != _sha256(root / "low" / name):
                raise ValueError(f"Shared {name} differs from source")
        else:
            shutil.copyfile(root / "low" / name, target)
    samples = directory / "samples.npz"
    if samples.exists():
        with np.load(samples, allow_pickle=False) as old:
            if set(old.files) != set(queries) or any(not np.array_equal(old[k], queries[k])
                                                    for k in queries):
                raise ValueError("Shared query mapping differs")
    else:
        _save_npz(samples, **queries)
    shape = (len(queries["k0"]), len(datasets["low"].coordinates))
    label_path = directory / "u_scat.npy"
    if label_path.exists():
        labels = np.load(label_path, mmap_mode="r+", allow_pickle=False)
        if labels.shape != shape or labels.dtype != np.complex128:
            raise ValueError("Shared label shape or dtype differs")
    else:
        labels = np.lib.format.open_memmap(label_path, mode="w+", shape=shape,
                                          dtype=np.complex128)
    bitmap = directory / "completed.npy"
    completed = (np.load(bitmap, allow_pickle=False) if bitmap.exists()
                 else np.zeros(len(queries["frequencies"]), dtype=bool))
    if completed.shape != queries["frequencies"].shape or completed.dtype != bool:
        raise ValueError("Invalid shared completion bitmap")
    started = perf_counter()
    for group in range(len(completed)):
        rows = np.flatnonzero(queries["frequency_id"] == group)
        if completed[group]:
            if not np.isfinite(labels[rows]).all():
                raise ValueError("Committed shared labels are nonfinite")
            continue
        for band in SOURCES:
            selected = rows[queries["source_band"][rows] == band]
            labels[selected] = datasets[band].u_scat[queries["source_index"][selected]]
        if not np.isfinite(labels[rows]).all():
            raise ValueError("Copied labels are nonfinite")
        labels.flush()
        completed[group] = True
        _save_npy(bitmap, completed)
    labels.flush()
    del labels
    info = dict(schema_version=SCHEMA, complete=True, samples=shape[0], nodes=shape[1],
                triangles=len(datasets["low"].triangles), media=1, split_strategy="frequency",
                frequency_groups=len(completed), raw_label_bytes=int(np.prod(shape) * 16),
                new_fem_solves=0, preparation_seconds=perf_counter() - started,
                splits=[dict(name=name, samples=int(np.count_nonzero(queries["split"] == j)),
                             groups=len(np.unique(queries["frequency_id"][queries["split"] == j])))
                        for j, name in enumerate(("train", "validation", "test"))],
                sha256={name: _sha256(directory / name) for name in
                        ("etal.msh", "reference_problem.npz", "samples.npz", "u_scat.npy")})
    _write_json(directory / "provenance.json", dict(
        reference_mesh_sha256=REFERENCE_SHA256, sources=snapshots,
        new_fem_solves=0, source_split_preservation=True,
        inherited_physics_provenance={band: str(root / band / "provenance.json")
                                     for band in SOURCES},
        generator_sha256=_sha256(Path(__file__))))
    _write_json(info_path, info)
    if progress:
        progress(f"Shared dataset: {shape[0]} fields, {len(completed)} frequency groups; "
                 f"splits {[s['samples'] for s in info['splits']]}; no new FEM solves")
    return load_dataset(directory)
