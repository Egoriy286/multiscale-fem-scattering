"""Native union storage copies exact fields and detects stale/corrupt sources."""
import hashlib
import json

import numpy as np
import pytest

from experiments.reference_sno_gmsfem import dataset as native
from experiments.shared_sno_gmsfem import dataset as shared


@pytest.fixture
def source_root(tmp_path, monkeypatch):
    mesh = b"tiny shared reference mesh"
    digest = hashlib.sha256(mesh).hexdigest()
    monkeypatch.setattr(native, "REFERENCE_SHA256", digest)
    monkeypatch.setattr(shared, "REFERENCE_SHA256", digest)
    root = tmp_path / "sources"
    problem = dict(coordinates=np.array([[0., 0.], [1., 0.], [0., 1.]]),
                   triangles=np.array([[0, 1, 2]], dtype=np.int32),
                   cell_tags=np.array([2]), circles=np.array([[.2, .2, .1, 2.]]),
                   A=np.full((1, 3), .2), n=np.full((1, 3), 10.),
                   A_cells=np.array([[.2]]), n_cells=np.array([[10.]]))
    for band in shared.SOURCES:
        path = root / band
        path.mkdir(parents=True)
        queries = native.make_queries(band, 10, 2)
        (path / "etal.msh").write_bytes(mesh)
        native._save_npz(path / "reference_problem.npz", **problem)
        native._save_npz(path / "samples.npz", **queries)
        labels = (queries["k0"] + 1j * queries["theta"])[:, None] * np.arange(1., 4.)
        native._save_npy(path / "u_scat.npy", labels)
        native._save_npy(path / "completed.npy", np.ones(len(queries["frequencies"]), dtype=bool))
        native._write_json(path / "config.json", dict(case="notebook", band=band,
                                                     reference_mesh_sha256=digest))
        native._write_json(path / "provenance.json", dict(reference_mesh_sha256=digest))
        native._write_json(path / "dataset_info.json", dict(
            schema_version=1, complete=True, samples=len(labels), nodes=3, triangles=1,
            frequency_groups=len(queries["frequencies"]),
            sha256={name: native._sha256(path / name) for name in
                    ("etal.msh", "reference_problem.npz", "samples.npz", "u_scat.npy")}))
    return root


def test_exact_readonly_union_and_reuse(tmp_path, source_root):
    path = tmp_path / "shared"
    dataset = shared.prepare_shared_dataset(path, source_root, progress=None)
    assert dataset.u_scat.flags.writeable is False
    assert dataset.A.flags.writeable is False
    with np.load(path / "samples.npz") as mapping:
        for band in shared.SOURCES:
            source = native.load_dataset(source_root / band)
            selected = np.flatnonzero(mapping["source_band"] == band)
            original = mapping["source_index"][selected]
            assert np.array_equal(dataset.u_scat[selected], source.u_scat[original])
            assert np.array_equal(dataset.split[selected], source.split[original])
    info_before = (path / "dataset_info.json").read_bytes()
    again = shared.prepare_shared_dataset(path, source_root, progress=None)
    assert np.array_equal(dataset.u_scat, again.u_scat)
    assert (path / "dataset_info.json").read_bytes() == info_before
    assert json.loads(info_before)["new_fem_solves"] == 0


def test_corrupted_source_labels_rejected(tmp_path, source_root):
    with (source_root / "high" / "u_scat.npy").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="Source high label checksum"):
        shared.prepare_shared_dataset(tmp_path / "shared", source_root, progress=None)


def test_corrupted_completed_union_rejected(tmp_path, source_root):
    path = tmp_path / "shared"
    shared.prepare_shared_dataset(path, source_root, progress=None)
    with (path / "u_scat.npy").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="Shared label checksum"):
        shared.prepare_shared_dataset(path, source_root, progress=None)
