"""The quality gate must not hide poor labels or treat missing evidence as good."""
import json
import numpy as np
import pytest

from neural_operator.datasets.curation import audit_dataset, quality_status


def test_quality_gate_requires_refinement_and_preserves_stress_case():
    assert quality_status([]) == 'refinement_pending'
    assert quality_status([], [.0065]) == 'sampled_refinement_pass'
    assert quality_status([], [.0065, .136]) == 'stress_only'
    assert quality_status([], [.136], obsolete=True) == 'obsolete_recipe'
    assert quality_status(['CRC failed'], [.0065], obsolete=True) == 'invalid_integrity'
    for values in ([np.nan], [np.inf], [-.1], [[.1]]):
        with pytest.raises(ValueError):
            quality_status([], values)


def test_streaming_audit_detects_nonfinite_payload_and_split_leakage(tmp_path):
    config = dict(n_media=3, n_directions=2, h=.1, n_coarse=1,
                  k_range=[1., 2.], seed=2026,
                  active_range=[3, 10], a_range=[.6, .9], n_range=[1.5, 3.],
                  split_fractions=[.34, .33, .33])
    (tmp_path / 'config.json').write_text(json.dumps(config))
    np.savez(tmp_path / 'geometry.npz',
             dof_coords=np.array([[0., 0.], [1., 0.], [0., 1.]]),
             triangles=np.array([[0, 1, 2]]))
    values = np.ones((6, 3), dtype=complex)
    values[0, 0] = np.nan
    np.savez(tmp_path / 'fem_dataset.npz', A=np.ones((6, 3)), n=np.ones((6, 3)),
             u_scat=values, k0=np.ones(6), theta=np.zeros(6),
             medium_id=np.repeat(np.arange(3), 2),
             split=np.array([0, 2, 2, 2, 2, 2]))
    record = audit_dataset(tmp_path)
    assert record['status'] == 'invalid_integrity'
    assert 'u_scat: nonfinite values' in record['integrity_errors']
    assert 'Group leakage between splits' in record['integrity_errors']
    assert len(record['files']['fem_dataset.npz']['sha256']) == 64


def test_corrupt_archive_never_passes_as_unmeasured(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'fem_dataset.npz').write_bytes(b'not a zip archive')
    record = audit_dataset(tmp_path)
    assert record['status'] == 'invalid_integrity'
    assert record['integrity_errors']


def test_extra_field_axis_and_unused_geometry_node_are_rejected(tmp_path):
    config = dict(n_media=3, n_directions=2, split_fractions=[.34, .33, .33])
    (tmp_path / 'config.json').write_text(json.dumps(config))
    np.savez(tmp_path / 'geometry.npz',
             dof_coords=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
             triangles=np.array([[0, 1, 2]]))
    np.savez(tmp_path / 'fem_dataset.npz', A=np.ones((6, 3)), n=np.ones((6, 3)),
             u_scat=np.ones((6, 3, 1), dtype=complex), k0=np.ones(6), theta=np.zeros(6),
             medium_id=np.repeat(np.arange(3), 2), split=np.array([0, 0, 2, 2, 2, 2]))
    record = audit_dataset(tmp_path)
    assert record['status'] == 'invalid_integrity'
    assert 'u_scat: expected exactly [samples, nodes]' in record['integrity_errors']
    assert 'Geometry node count/shape differs from field nodes' in record['integrity_errors']
