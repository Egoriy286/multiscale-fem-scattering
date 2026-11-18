"""Small deterministic checks using the repository's actual SVD-SNO classes.

Run from the repository root: .venv/bin/python svd_sno_manual/mini_demo.py
These synthetic fields are algebra exercises, not Helmholtz FEM solutions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigvalsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural_operator.data import Dataset, mass_matrix
from neural_operator.sno.basis import PODBasis
from neural_operator.sno.model import SVDSNO
from neural_operator.training import TrainConfig, Predictor, _svd_preprocessing
from neural_operator.losses import PODFEMLoss, relative_root_mean
from experiments.gms_sno_100.pod import ComplexPOD
from experiments.gms_sno_100.model import SpectralSurrogate


def main():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    mass = mass_matrix(coords, triangles)
    weights = np.asarray(mass.sum(axis=1)).ravel()
    eigenvalues=eigvalsh(np.diag(weights),mass.toarray())
    assert eigenvalues.min()>=1-1e-12 and eigenvalues.max()<=4+1e-12
    t = np.linspace(-1., 1., 12)
    real = 2 + t[:, None]*np.array([1., 2., -1., .5])
    real += t[:, None]**2*np.array([.2, -.1, .3, .1])
    imag = -.5 + np.sin(t[:, None])*np.array([.4, -.2, .6, .8])
    imag += t[:, None]**3*np.array([-.2, .1, .4, -.3])
    fields = real + 1j*imag
    split = np.array([0]*8+[1]*2+[2]*2, dtype=np.int8)
    data = Dataset(A=np.ones_like(real), n=np.ones_like(real), u_scat=fields,
                   theta=np.zeros(12), k0=2+t/2, medium_id=np.zeros(12,dtype=int),
                   split=split, coordinates=coords, triangles=triangles,
                   frequency_id=np.arange(12), split_strategy='frequency')
    train, val, test = data.split_indices()

    # A full retained real POD is orthonormal in the lumped metric.
    basis = PODBasis.fit(real[train], 3, weights=weights)
    orth = np.linalg.norm(basis.modes.T@(weights[:,None]*basis.modes)-np.eye(basis.rank))
    recon = np.max(np.abs(basis.decode(basis.encode(real))-real))
    assert orth < 1e-12 and recon < 1e-12
    assert PODBasis.fit(np.ones((8,4)), 3, weights=weights).rank == 0

    # Keep one mode per component deliberately, so the loss has a residual.
    prep = _svd_preprocessing(data, TrainConfig(rank=1), train, val)
    kwargs = dict(input_dim=3, output_modes=prep['rank'], width=4, depth=2,
                  input_ranks=(0,0), architecture_version=2)
    predictor = Predictor('svd', kwargs, prep, coords, triangles)
    features = predictor.encode(data.A, data.n, data.theta, data.k0)
    token_shape = list(predictor.model.coefficient_tokens(torch.tensor(features)).shape)
    loss_fn = PODFEMLoss(predictor, data, fields, target_dtype=np.float32)
    raw = predictor.model(torch.tensor(features)).detach().requires_grad_(True)
    fast = loss_fn(raw, torch.arange(len(fields)))
    decoded = predictor.decode(raw).double()
    truth = torch.tensor(np.stack([real,imag],-1).astype('float32')).double()
    m = torch.tensor(mass.toarray(),dtype=torch.float64)
    delta = decoded-truth
    e2 = torch.einsum('bnc,nm,bmc->b',delta,m,delta)
    n2 = torch.einsum('bnc,nm,bmc->b',truth,m,truth)
    direct = relative_root_mean(e2,n2)
    grad_fast, = torch.autograd.grad(fast, raw, retain_graph=True)
    grad_direct, = torch.autograd.grad(direct, raw)
    # The field decoder adds float32 means; the quadratic evaluates its
    # float32-rounded basis and mean in float64. Allow that rounding level.
    assert torch.allclose(fast,direct,rtol=2e-6,atol=1e-7)
    assert torch.allclose(grad_fast,grad_direct,rtol=2e-5,atol=1e-7)

    # A complex mass POD obeys the Pythagorean error identity.
    cpod = ComplexPOD.fit(fields[train],mass,rank=1,max_rank=1)
    a = cpod.encode(fields,mass)
    residual = fields-cpod.decode(a)
    err = cpod.decode(a+.01*(1+1j))-fields
    norm2 = lambda v: np.real(np.sum(v.conj()*(mass@v.T).T,axis=-1))
    pyth = np.max(np.abs(norm2(err)-(norm2(residual)+.0002)))
    assert pyth < 1e-12
    network = SpectralSurrogate(output_modes=3,width=4,depth=2,
                                scales=(1.,4.),activation='sine')
    formula_count = 2*(4*3*4+2*(2*4**2+4+3**2)+2*4+2)+2*2
    actual_count = sum(p.numel() for p in network.parameters())
    assert formula_count == actual_count
    out = network(torch.tensor([[0.,1.,0.],[1.,0.,1.]]))
    assert tuple(out.shape)==(2,3,2)

    result = dict(description='Synthetic algebra checks; no physical training',
                  real_pod_rank=basis.rank,lumped_orthogonality_error=orth,
                  max_real_reconstruction_error=recon,token_shape=token_shape,
                  quadratic_loss=float(fast.detach()),direct_loss=float(direct.detach()),
                  gradient_max_difference=float((grad_fast-grad_direct).abs().max()),
                  complex_pythagorean_error=pyth,multiscale_parameter_count=actual_count,
                  lumped_to_consistent_generalized_eigenvalues=eigenvalues.tolist(),
                  versions=dict(numpy=np.__version__,torch=str(torch.__version__)))
    target=Path(__file__).parent/'assets'/'demo_results.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
