"""Independent physical-loss equivalence and train-only POD checks."""
from types import SimpleNamespace

import numpy as np
from scipy import sparse
import torch

from experiments.multiscale_sample_study.sno import SNOAdapter, CoefficientLoss, _norm2


def fixture_data():
    rng=np.random.default_rng(7)
    a=rng.normal(size=(7,7))
    return SimpleNamespace(u_scat=3+rng.normal(size=(6,7))+1j*rng.normal(size=(6,7)),
        mass_matrix=sparse.csr_matrix(a.T@a+np.eye(7)), k0=np.linspace(.1,4,6),theta=np.zeros(6))


def test_complex_coefficient_loss_equals_direct_mass_and_gradient():
    data=fixture_data()
    adapter=SNOAdapter(data,np.arange(4),rank=2,width=2,depth=1)
    prediction=torch.randn(6,2,2,dtype=torch.float64,requires_grad=True)
    actual=adapter.validation_loss(prediction,np.arange(6))
    pairs=prediction.detach().numpy()*adapter.prep['scale']
    decoded=adapter.pod.decode(pairs[...,0]+1j*pairs[...,1])
    expected=np.sqrt(_norm2(decoded-data.u_scat,data.mass_matrix)/_norm2(data.u_scat,data.mass_matrix))
    np.testing.assert_allclose(actual.detach(),expected,rtol=1e-12,atol=1e-12)
    actual.mean().backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum()>0


def test_holdouts_do_not_affect_pod_or_normalizers_and_cache_roundtrip(tmp_path):
    data=fixture_data()
    a=SNOAdapter(data,np.arange(4),rank=2,width=2,depth=1)
    data.u_scat[4:]*=1000;data.k0[4:]+=1000
    b=SNOAdapter(data,np.arange(4),rank=2,width=2,depth=1)
    for key in ('scale','k_mean','k_std'):
        assert a.prep[key]==b.prep[key]
    np.testing.assert_array_equal(a.pod.modes,b.pod.modes)
    a.save_preprocessing(tmp_path)
    restored=SNOAdapter.load(tmp_path)
    restored.model.load_state_dict(a.model.state_dict())
    np.testing.assert_array_equal(a.predict_fields(np.arange(6)),restored.predict_fields(np.arange(6)))


def test_float32_online_loss_retains_positive_projection_residual():
    data=fixture_data()
    adapter=SNOAdapter(data,np.arange(4),rank=2,width=2,depth=1)
    pairs=torch.tensor(adapter.prep['pairs']/adapter.prep['scale'],dtype=torch.float32)
    online=CoefficientLoss(adapter.prep,'cpu',torch.float32)(pairs,np.arange(6))
    exact=adapter.validation_loss(pairs,np.arange(6))
    np.testing.assert_allclose(online,exact,rtol=2e-6,atol=1e-7)
    assert online.min()>0
