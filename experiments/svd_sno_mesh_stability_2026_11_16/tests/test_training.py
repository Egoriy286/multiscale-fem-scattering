import json

import numpy as np
import pytest
from scipy import sparse

from experiments.gms_sno_100.pod import ComplexPOD
from experiments.svd_sno_mesh_stability_2026_11_16.training import (
    NativeDataset, _encode, _validate_protocol, load_queries,
)


def test_protocol_accepts_root_aliases_and_freezes_architecture():
    protocol = _validate_protocol({"pod_rank": 192, "width": 8, "depth": 4,
        "scales": [1, 40, 80, 100, 120, 140, 180, 200], "seeds": [11, 29, 47],
        "train_sizes": [1000], "lr": .001, "val_interval": 5, "device": "cpu"})
    assert protocol["output_modes"] == 192
    assert protocol["learning_rate"] == .001
    assert protocol["validation_interval"] == 5
    assert protocol["train_count"] == 1000


@pytest.mark.parametrize("protocol", [
    {"pod_rank": 64}, {"width": 4}, {"seeds": [1, 2, 3]}, {"train_sizes": [1000, 2000]},
])
def test_protocol_rejects_architecture_or_primary_phase_changes(protocol):
    with pytest.raises(ValueError):
        _validate_protocol(protocol)


def test_native_dataset_loads_six_dof_triangles_and_shared_query_rows(tmp_path):
    directory = tmp_path/"data"/"level0"
    directory.mkdir(parents=True)
    coordinates = np.array([[0.,0.],[1.,0.],[0.,1.],[.5,0.],[.5,.5],[0.,.5]])
    np.savez(directory/"geometry.npz", coordinates=coordinates,
             triangles=np.array([[0,1,2,3,4,5]], dtype=np.int32),
             cell_tags=np.array([1], dtype=np.int32))
    sparse.save_npz(directory/"mass.npz", sparse.eye(6, format="csr"))
    fields = np.ones((3,6), dtype=np.complex128)
    np.save(directory/"fields.npy", fields)
    queries = [{"id": i, "k": float(i+1), "theta": 0., "split": i} for i in range(3)]
    (tmp_path/"queries.json").write_text(json.dumps({"queries": queries}))
    rows, digest = load_queries(tmp_path)
    data = NativeDataset(tmp_path, 0, rows)
    assert len(digest) == 64
    assert data.triangles.shape == (1,6)
    assert [ids.tolist() for ids in data.split_indices()] == [[0],[1],[2]]
    assert np.array_equal(data.fields, fields)


def test_mass_projection_encoder_matches_complex_pod():
    rng = np.random.default_rng(4)
    fields = rng.normal(size=(8,6)) + 1j*rng.normal(size=(8,6))
    mass = sparse.diags(np.arange(1.,7.), format="csr")
    pod = ComplexPOD.fit(fields, mass, rank=3, max_rank=3)
    encoded = _encode(pod, fields[:3], mass)
    assert np.allclose(encoded, pod.encode(fields[:3], mass), rtol=1e-10, atol=1e-10)

