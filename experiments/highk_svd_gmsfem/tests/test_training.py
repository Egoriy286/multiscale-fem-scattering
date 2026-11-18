import numpy as np
import pytest
import torch
from scipy import sparse
from experiments.gms_sno_100.pod import ComplexPOD
from experiments.highk_svd_gmsfem.training import fit_streaming_pod, TransferredSurrogate


def test_streamed_pod_matches_exact_complex_mass_projection_and_ignores_heldouts():
    rng=np.random.default_rng(9)
    fields=rng.normal(size=(12,9))+1j*rng.normal(size=(12,9))
    mass=sparse.diags(np.linspace(.2,1.4,9),format='csr')
    train=np.arange(2,12)
    pod,diagnostics=fit_streaming_pod(fields,mass,train,9,1e-5,batch_size=3)
    exact=ComplexPOD.fit(fields[train],mass,rank=9,max_rank=9)
    np.testing.assert_allclose(pod.mean,exact.mean,atol=2e-14)
    np.testing.assert_allclose(pod.decode(pod.encode(fields,mass)),exact.decode(exact.encode(fields,mass)),atol=2e-12)
    np.testing.assert_allclose(pod.modes.conj().T@(mass@pod.modes),np.eye(pod.rank),atol=2e-13)
    fields[:2]*=1e6
    again,_=fit_streaming_pod(fields,mass,train,9,1e-5,batch_size=3)
    np.testing.assert_array_equal(again.mean,pod.mean)
    np.testing.assert_array_equal(again.modes,pod.modes)
    assert diagnostics['training_ids']==train.tolist()


def test_streamed_rank_cap_cannot_hide_high_projection_floor():
    fields=np.eye(8,dtype=complex)
    with pytest.raises(ValueError,match='cap 1 insufficient'):
        fit_streaming_pod(fields,sparse.eye(8,format='csr'),np.arange(8),1,.01,3)


def test_transfer_preserves_projected_old_prediction_and_fine_tunes_old_weights():
    torch.manual_seed(11)
    old=dict(output_modes=2,width=3,depth=1,scales=[1,4],activation='sine')
    new=dict(output_modes=5,width=3,depth=1,scales=[1,4],activation='sine')
    rng=np.random.default_rng(1); mapping=rng.normal(size=(2,5))+1j*rng.normal(size=(2,5)); bias=rng.normal(size=5)+1j*rng.normal(size=5)
    model=TransferredSurrogate(old,new,np.stack((mapping.real,mapping.imag),-1),np.stack((bias.real,bias.imag),-1),2.,.7,10.,5.)
    features=torch.tensor([[3.,1.,0.],[12.,1.,0.]])
    with torch.no_grad():
        prior=model.old(torch.tensor([[(3.-2)/.7,1.,0.],[(12.-2)/.7,1.,0.]])).numpy()
        expected=(prior[...,0]+1j*prior[...,1])@mapping+bias
        actual=model(features).numpy()
    np.testing.assert_allclose(actual[...,0]+1j*actual[...,1],expected,rtol=2e-6,atol=2e-6)
    before=model.old.branches[0].lift.weight.detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    model(features).square().mean().backward(); optimizer.step()
    assert not torch.equal(before,model.old.branches[0].lift.weight)
    assert model.residual.output_modes==5


def test_reduced_coefficient_loss_includes_unresolved_physical_field():
    from experiments.multiscale_sample_study.sno import CoefficientLoss, _norm2
    rng=np.random.default_rng(71)
    fields=rng.normal(size=(11,7))+1j*rng.normal(size=(11,7))
    mass=sparse.diags(np.linspace(.2,1.,7),format='csr')
    pod=ComplexPOD.fit(fields[:8],mass,rank=3,max_rank=3)
    c=pod.encode(fields,mass); scale=1.7
    prep=dict(pairs=np.stack((c.real,c.imag),-1),scale=scale,
        residual2=_norm2(fields-pod.decode(c),mass),norm2=_norm2(fields,mass))
    guess=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)); ids=np.arange(8,11)
    pairs=torch.tensor(np.stack((guess.real,guess.imag),-1)/scale,dtype=torch.float64)
    reduced=CoefficientLoss(prep,'cpu',torch.float64)(pairs,ids).numpy()
    physical=np.sqrt(_norm2(pod.decode(guess)-fields[ids],mass)/_norm2(fields[ids],mass))
    np.testing.assert_allclose(reduced,physical,rtol=2e-13,atol=2e-13)


def test_chunked_encoder_matches_reference_without_conjugating_decoder():
    from experiments.highk_svd_gmsfem.training import _encode
    rng=np.random.default_rng(33)
    fields=rng.normal(size=(15,11))+1j*rng.normal(size=(15,11))
    mass=sparse.diags(np.linspace(.2,1.,11),format='csr')
    pod=ComplexPOD.fit(fields[:10],mass,rank=7,max_rank=7)
    np.testing.assert_allclose(_encode(pod,fields,mass),pod.encode(fields,mass),atol=1e-13)
    np.testing.assert_allclose(_encode(pod,fields[0],mass),pod.encode(fields[0],mass),atol=1e-13)


def test_declared_expansion_selects_from_training_candidates_only():
    fields=np.eye(8,dtype=complex); mass=sparse.eye(8,format='csr')
    pod,diagnostics=fit_streaming_pod(fields,mass,np.arange(8),1,.01,3,
        expansion_rank=7,candidate_ranks=[1,4,7])
    assert pod.rank==7
    assert diagnostics['expanded'] and diagnostics['expansion_limit']==7


def test_nested_cache_dispatch_never_passes_larger_training_prefix(monkeypatch,tmp_path):
    from neural_operator.data import Dataset
    from experiments.highk_svd_gmsfem.dataset import make_queries
    from experiments.highk_svd_gmsfem import training
    q=make_queries(); count=len(q['k0']); fields=np.arange(count*3).reshape(count,3).astype(complex)
    data=Dataset(A=np.ones((count,3)),n=np.ones((count,3)),u_scat=fields,
        theta=q['theta'],k0=q['k0'],medium_id=q['medium_id'],split=q['split'],frequency_id=q['frequency_id'],
        coordinates=np.array([[0.,0.],[1.,0.],[0.,1.]]),triangles=np.array([[0,1,2]]),split_strategy='frequency')
    received=[]
    def prepare(root,view,protocol):
        train,val,test=view.split_indices(); received.append(len(train))
        assert len(val)==len(test)==200
        assert len(view.u_scat)==400+len(train)
        assert np.shares_memory(view.u_scat,data.u_scat)
        np.testing.assert_array_equal(view.u_scat[train[-1]],data.u_scat[400+len(train)-1])
        return dict(n_train=len(train))
    monkeypatch.setattr(training,'_prepare_one_cache',prepare)
    sizes=[400,600,800,1000,2000]
    result=training.prepare_cache(tmp_path,data,dict(train_sizes=sizes))
    assert received==sizes and [r['n_train'] for r in result]==sizes
    assert training.run_directory(tmp_path,11,400)!=training.run_directory(tmp_path,11,2000)


def test_training_and_replay_dispatch_cover_all_fifteen_runs(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from experiments.highk_svd_gmsfem import training,dataset
    sizes=[400,600,800,1000,2000]; seeds=[11,29,47]
    def view(size):
        return SimpleNamespace(n_train=size,split_indices=lambda:(np.arange(400,400+size),np.arange(200),np.arange(200,400)))
    data=view(2000)
    monkeypatch.setattr(dataset,'training_view',lambda source,size:view(size))
    monkeypatch.setattr(training,'prepare_cache',lambda root,source,protocol:(root/'cache').mkdir(parents=True,exist_ok=True))
    monkeypatch.setattr(training,'_total_norms',lambda source,ids:np.ones(len(ids)))
    monkeypatch.setattr(training,'train_one',lambda root,source,protocol,seed,norms:dict(n_train=source.n_train,seed=seed))
    monkeypatch.setattr(training,'_verify_one',lambda root,source,protocol:[dict(n_train=source.n_train,seed=seed) for seed in protocol['seeds']])
    protocol=dict(train_sizes=sizes,seeds=seeds,threads=2)
    runs=training.train_all(tmp_path,data,protocol)
    expected=[(size,seed) for size in sizes for seed in seeds]
    assert [(r['n_train'],r['seed']) for r in runs]==expected
    replay=training.verify(tmp_path,data,protocol)
    assert replay['passed'] and [(r['n_train'],r['seed']) for r in replay['runs']]==expected


def test_replay_rejects_unpaired_queries_and_tampered_case_metrics(monkeypatch,tmp_path):
    import json
    from types import SimpleNamespace
    from experiments.highk_svd_gmsfem import training
    root=tmp_path; directory=training.run_directory(root,11,400); directory.mkdir(parents=True)
    (root/'cache').mkdir();np.save(root/'cache/test_total_norms.npy',np.full(2,4.))
    truth=np.ones((403,2),dtype=complex);test=np.arange(2);prediction=truth[:2]+np.array([[.1],[.2]])
    data=SimpleNamespace(u_scat=truth,mass_matrix=sparse.eye(2,format='csr'),k0=np.arange(403)+1.,theta=np.zeros(403),
        split_indices=lambda:(np.arange(3,403),np.array([2]),test))
    model=torch.nn.Linear(1,1)
    torch.save(dict(state_dict=model.state_dict(),best_epoch=1),directory/'model.pt')
    np.save(directory/'predictions.npy',prediction)
    (directory/'history.json').write_text(json.dumps([dict(epoch=1,val_relative_l2=.1)]))
    scattered=np.array([10.,20.]);total=scattered/np.sqrt(2.)
    rows=[dict(index=int(i),k=float(data.k0[i]),theta=0.,scattered_pct=float(s),total_pct=float(t)) for i,s,t in zip(test,scattered,total)]
    (directory/'metrics.json').write_text(json.dumps(rows))
    result=dict(best_epoch=1,scattered_mean_pct=float(scattered.mean()),total_mean_pct=float(total.mean()),old_weights_l2_change=.1,
        checkpoint_sha256=training._sha256(directory/'model.pt'),predictions_sha256=training._sha256(directory/'predictions.npy'))
    (directory/'result.json').write_text(json.dumps(result))
    class Replay:
        def __init__(self,*args,**kwargs):self.model=torch.nn.Linear(1,1)
        def predict_fields(self,ids):yield ids,prediction
    monkeypatch.setattr(training,'Adapter',Replay)
    assert len(training._verify_one(root,data,dict(seeds=[11],device='cpu')))==1
    rows[0]['index']=1;(directory/'metrics.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='Unpaired'):training._verify_one(root,data,dict(seeds=[11],device='cpu'))
    rows[0]['index']=0;rows[0]['scattered_pct']=11.;(directory/'metrics.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='Per-case FEM scattered'):training._verify_one(root,data,dict(seeds=[11],device='cpu'))


def test_compact_prior_gate_preserves_old_domain_and_removes_extrapolation():
    torch.manual_seed(11)
    old=dict(output_modes=2,width=3,depth=1,scales=[1,4],activation='sine')
    new=dict(output_modes=5,width=3,depth=1,scales=[1,4],activation='sine')
    rng=np.random.default_rng(1);mapping=rng.normal(size=(2,5,2));bias=rng.normal(size=(5,2))
    ungated=TransferredSurrogate(old,new,mapping,bias,2.,.7,10.,5.)
    gated=TransferredSurrogate(old,new,mapping,bias,2.,.7,10.,5.,prior_gate=dict(enabled=True,start=4.,stop=5.))
    gated.load_state_dict(ungated.state_dict())
    features=torch.tensor([[1.,1.,0.],[2.,1.,0.],[4.,1.,0.],[4.5,1.,0.],[5.,1.,0.],[20.,1.,0.]])
    with torch.no_grad():
        original=ungated(features);actual=gated(features)
    torch.testing.assert_close(actual[:3],original[:3],rtol=0.,atol=0.)
    torch.testing.assert_close(actual[3],original[3]*.5,rtol=0.,atol=0.)
    torch.testing.assert_close(actual[4:],torch.zeros_like(actual[4:]),rtol=0.,atol=0.)
    # Both endpoints have zero slope, so the piecewise extension is C1.
    k=torch.tensor([4.,4.5,5.,20.],requires_grad=True)
    gated.prior_weight(k).sum().backward()
    torch.testing.assert_close(k.grad[[0,2,3]],torch.zeros(3),rtol=0.,atol=0.)
    gated.zero_grad(set_to_none=True);gated(features[4:]).sum().backward()
    assert all(parameter.grad is not None and not torch.count_nonzero(parameter.grad) for parameter in gated.old.parameters())
    assert all(torch.count_nonzero(branch.project.bias.grad)>0 for branch in gated.residual.branches)
    # New residual still contributes and trains above the old domain.
    with torch.no_grad():
        for branch in gated.residual.branches:branch.project.bias.fill_(1.)
    torch.testing.assert_close(gated(features[4:]),torch.ones_like(actual[4:]))
