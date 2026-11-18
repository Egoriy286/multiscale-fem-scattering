"""Independent audit numerical checks; no production fields or models."""
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from experiments.multiscale_sample_study import audit as module


def data_fixture():
    return SimpleNamespace(coordinates=np.array([[0.,0.],[1.,0.],[0.,1.]]),
        triangles=np.array([[0,1,2]]), u_scat=np.array([[1+2j]*3,[2-1j]*3]),
        k0=np.array([.3,2.]),theta=np.array([0.,.4]),
        mass_matrix=sparse.csr_matrix((np.ones((3,3))+np.eye(3))/24))


def test_direct_mass_scores_both_denominators_and_bounded_batches():
    data=data_fixture()
    prediction=data.u_scat*1.1
    scattered,total=module.direct_scores(data,[0,1],prediction,np.array([10.,20.]),batch_size=1)
    np.testing.assert_allclose(scattered,[10.,10.],rtol=1e-14)
    np.testing.assert_allclose(total,100*np.sqrt(np.array([.025,.025])/[10.,20.]),rtol=1e-14)
    with pytest.raises(ValueError,match='Invalid predicted'):
        module.direct_scores(data,[0,1],prediction.real,[10.,20.])


def test_analytic_incident_total_norm_and_cache_tampering():
    data=data_fixture()
    data.u_scat[:]=0
    norms=module.total_denominators(data,[0,1])
    # Analytic plane wave has modulus one throughout the area-1/2 triangle.
    np.testing.assert_allclose(norms,[.5,.5],rtol=1e-14)
    module.compare_norm_cache(norms,norms.copy())
    with pytest.raises(ValueError,match='denominators differ'):
        module.compare_norm_cache(norms,norms+[0.,.01])


def test_per_case_scores_and_identity_are_checked_independently():
    rows=[dict(index=i,k=k,theta=t,scattered_pct=s,total_pct=e) for i,k,t,s,e in
          [(100,.2,0.,3.,1.),(101,.4,0.,4.,2.)]]
    args=([100,101],[.2,.4],[0.,0.],[3.,4.],[1.,2.],'synthetic')
    assert module.compare_rows(rows[::-1],*args)==0
    rows[1]['total_pct']=2.1
    with pytest.raises(ValueError,match='physical score mismatch'):
        module.compare_rows(rows,*args)
    rows[1]['total_pct']=2.
    rows[1]['k']=.5
    with pytest.raises(ValueError,match='query differs'):
        module.compare_rows(rows,*args)
    with pytest.raises(ValueError,match='Duplicate'):
        module.compare_rows([rows[0],rows[0]],*args)


def test_audit_rejects_changed_frozen_source_before_any_writes(tmp_path,monkeypatch):
    protocol=dict(train_sizes=[400,600,800,1000],methods=sorted(module.METHODS),
                  seeds=[11,29,47],source_sha256={'frozen.py':'invalid'},dataset_sha256={})
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'frozen.py').write_text('source')
    (tmp_path/'results').mkdir()
    records=[dict(n_train=n,method=m,seed=s) for n,m,s in itertools.product(protocol['train_sizes'],protocol['methods'],protocol['seeds'])]
    verification=dict(passed=True,runs=records,protocol_sha256=module._sha256(tmp_path/'protocol.json'))
    path=tmp_path/'results/verification.json'
    path.write_text(json.dumps(verification))
    original=path.read_bytes()
    monkeypatch.setattr(module,'REPO',tmp_path)
    with pytest.raises(ValueError,match='training source changed'):
        module.audit(tmp_path)
    assert path.read_bytes()==original
    assert not (tmp_path/'results/physical_audit.json').exists()


def test_interpolation_roundtrip_is_batched_physical_diagnostic():
    data=data_fixture()
    data.u_scat[:,2]+=1
    seen=[]
    class Sampler:
        def apply(self, fields):
            seen.append(len(fields))
            return fields[:,:2]
    back=sparse.csr_matrix([[1.,0.],[0.,1.],[.5,.5]])
    errors=module.interpolation_roundtrip_errors(data,Sampler(),back,batch_size=1)
    reconstruction=(back@data.u_scat[:,:2].T).T
    expected,_=module.direct_scores(data,[0,1],reconstruction,[1.,1.])
    np.testing.assert_allclose(errors,expected,rtol=1e-14)
    assert seen==[1,1] and np.all(errors>0)
    # A full identity decoder has zero round-trip interpolation error.
    identity=SimpleNamespace(apply=lambda fields:fields)
    np.testing.assert_array_equal(module.interpolation_roundtrip_errors(data,identity,sparse.eye(3)),[0.,0.])
