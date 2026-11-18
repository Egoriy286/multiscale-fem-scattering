import json
import numpy as np
import pytest
from experiments.svd_sno_mesh_stability_2026_11_16 import physics


def test_native_snapshots_preserve_manifest_order_and_resume(tmp_path, monkeypatch):
    class Native:
        def solve(self,k,theta):
            return np.array([k+1j*theta, 2*k+1j*theta]), {'residual': 1e-13, 'seconds': 0.}
    class Backend:
        native=Native()
    out=tmp_path/'data'/'level0'; out.mkdir(parents=True)
    (out/'geometry.npz').write_bytes(b'geometry'); (out/'mass.npz').write_bytes(b'mass')
    monkeypatch.setattr(physics,'build_level',lambda level,root: (Backend(),out,{'dofs':2,'mesh_sha256':'mesh'}))
    queries=[{'k':2.,'theta':0.2},{'k':1.,'theta':0.1}]
    manifest=tmp_path/'queries.json'; manifest.write_text(json.dumps(queries))
    physics.generate(0,manifest,tmp_path)
    fields=np.load(out/'fields.npy')
    assert fields.dtype == np.complex128
    np.testing.assert_allclose(fields[:,0],[2+.2j,1+.1j])
    physics.generate(0,manifest,tmp_path)
    manifest.write_text(json.dumps(queries[::-1]))
    with pytest.raises(ValueError,match='different manifest'): physics.generate(0,manifest,tmp_path)


def test_geometry_audit_detects_changed_interface(tmp_path):
    # Two triangles with one interface; native mesh order may differ.
    for i in range(3):
        d=tmp_path/'data'/f'level{i}';d.mkdir(parents=True)
        xy=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        np.savez(d/'geometry.npz', vertices=xy,vertex_triangles=[[0,1,2],[1,3,2]],cell_tags=[1,2])
    assert physics.audit_geometry(tmp_path)['identical_material_interface_segments_at_1e_12']
    d=tmp_path/'data'/'level2'
    xy[1,0]=1.01
    np.savez(d/'geometry.npz', vertices=xy,vertex_triangles=[[0,1,2],[1,3,2]],cell_tags=[1,2])
    with pytest.raises(ValueError,match='polygons differ'): physics.audit_geometry(tmp_path)
