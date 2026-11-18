"""Train-only mass POD and genuine expanded-decoder SNO checkpoint fine tuning.

All field operations stream sample blocks. No heldout label enters the mean,
POD spectrum/rank, normalizers, or transferred coefficient map.
"""
from __future__ import annotations
import gc
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import linalg
import torch
from torch import nn
from experiments.gms_sno_100.model import SpectralSurrogate
from experiments.gms_sno_100.pod import ComplexPOD
from experiments.multiscale_sample_study.sno import CoefficientLoss, model_config, _norm2
from experiments.multiscale_sample_study.training import _save, _json_hash, _model_state, parameter_count, _total_norms
from experiments.reference_sno_gmsfem.dataset import _sha256, _write_json
from neural_operator.mesh_transfer import P1Transfer

OLD_ROOT = Path(__file__).resolve().parents[1]/'multiscale_sample_study'
METHOD = 'mscale_sno'


def _blocks(ids, size=16):
    ids = np.asarray(ids, dtype=np.int64)
    for start in range(0, len(ids), size):
        yield ids[start:start+size]


def _encode(pod, fields, mass):
    """Mass projection conjugating sample blocks, never the large decoder."""
    fields=np.asarray(fields,dtype=np.complex128)
    shape=fields.shape[:-1]
    centered=(fields-pod.mean).reshape(-1,pod.mean.size)
    weighted=(mass@centered.T).T
    coefficients=(weighted.conj()@pod.modes).conj()
    return coefficients.reshape(*shape,pod.rank)


def fit_streaming_pod(fields, mass, train_ids, max_rank=128, relative_tolerance=.01, batch_size=16, *, expansion_rank=None, candidate_ranks=None):
    """Exact snapshot POD; smallest allowed rank meeting train relative RMS floor.

    Gram eigenvalues and the rank rule use only supplied training rows. The
    relative residual rule weights each training field equally, like the loss.
    Optional predeclared candidates and expansion ceiling use this same train-only
    spectrum. Exhausting the ceiling causes an error, never silent use of an
    inadequate decoder. Numerical rank uses the same cutoff as ComplexPOD.
    """
    train_ids = np.asarray(train_ids, dtype=np.int64)
    if len(train_ids) < 2 or max_rank < 1 or not 0 < relative_tolerance < 1:
        raise ValueError('At least two training rows, positive cap and tolerance in (0,1) required')
    n, nodes = len(train_ids), fields.shape[1]
    mean = np.zeros(nodes, np.complex128)
    for ids in _blocks(train_ids, batch_size):
        mean += np.asarray(fields[ids]).sum(axis=0)/n
    gram = np.empty((n,n), np.complex128)
    norm2 = np.empty(n)
    for start in range(0,n,batch_size):
        ids = train_ids[start:start+batch_size]
        raw = np.asarray(fields[ids]); centered = raw-mean
        norm2[start:start+len(ids)] = _norm2(raw,mass)
        weighted = (mass @ centered.T).T
        for stop in range(0,start+1,batch_size):
            other = np.asarray(fields[train_ids[stop:stop+batch_size]])-mean
            block = other.conj() @ weighted.T
            gram[stop:stop+len(other),start:start+len(ids)] = block
            gram[start:start+len(ids),stop:stop+len(other)] = block.conj().T
    gram = (gram+gram.conj().T)*.5
    values, vectors = linalg.eigh(gram, check_finite=False)
    order = np.argsort(values)[::-1]; values,vectors=values[order],vectors[:,order]
    cutoff = 32*np.finfo(float).eps*n*np.max(np.abs(values),initial=0.)
    positive = np.flatnonzero(values>cutoff)[:n-1]
    values,vectors=values[positive],vectors[:,positive]
    if not len(values) or np.any(norm2 <= 0):
        raise ValueError('Nonzero snapshots and field norms required')
    ceiling=max_rank if expansion_rank is None else max(max_rank,int(expansion_rank))
    available = min(ceiling,len(values))
    # Each training row's coefficients are sqrt(lambda)*conjugate(V[row]).
    captured = np.cumsum(abs(vectors[:,:available])**2*values[None,:available],axis=1)
    residual = np.maximum(np.real(np.diag(gram))[:,None]-captured,0.)
    curve = np.sqrt(np.mean(residual/norm2[:,None],axis=0))
    reached = np.flatnonzero(curve<=relative_tolerance)
    if not len(reached):
        raise ValueError(f'Train-only POD cap {ceiling} insufficient: RMS floor {curve[-1]:.6g} > {relative_tolerance}; increase pod_max_rank before test evaluation')
    rank = int(reached[0]+1)
    if candidate_ranks is not None:
        candidates=sorted(set(int(r) for r in candidate_ranks if 0<int(r)<=available))
        if len(values)<=ceiling and len(values) not in candidates: candidates.append(len(values))
        admissible=[r for r in candidates if r>=rank]
        if not admissible: raise ValueError('Declared POD rank candidates cannot achieve train-only tolerance')
        rank=min(admissible)
    modes = np.zeros((nodes,rank),np.complex128)
    for start in range(0,n,batch_size):
        ids = train_ids[start:start+batch_size]
        centered=np.asarray(fields[ids])-mean
        for column in range(0,rank,batch_size):
            modes[:,column:column+batch_size] += centered.T@vectors[start:start+len(ids),column:column+batch_size]
    modes /= np.sqrt(values[:rank])[None,:]
    mode_gram = np.empty((rank,rank),np.complex128)
    # Conjugate only one narrow weighted block, never the full large decoder.
    for start in range(0,rank,batch_size):
        weighted_modes=mass@modes[:,start:start+batch_size]
        mode_gram[:,start:start+batch_size]=(modes.T@weighted_modes.conj()).conj()
    upper = linalg.cholesky((mode_gram+mode_gram.conj().T)*.5,lower=False)
    for row in range(0,nodes,4096):
        modes[row:row+4096]=linalg.solve_triangular(upper.T,modes[row:row+4096].T,lower=True,check_finite=False).T
    return ComplexPOD(mean,modes,values), dict(rank=rank, numerical_rank=len(values),
        max_rank=max_rank, expansion_limit=ceiling, expanded=bool(rank>max_rank),candidate_ranks=None if candidate_ranks is None else list(candidate_ranks), relative_tolerance=relative_tolerance,
        rank_rule='smallest declared candidate rank with train RMS relative FEM projection residual <= tolerance' if candidate_ranks is not None else 'smallest positive rank with train RMS relative FEM projection residual <= tolerance',
        training_rms_relative_floor_by_rank=curve.tolist(), training_ids=train_ids.tolist())


class TransferredSurrogate(nn.Module):
    """Old learned network + zero-initialized new full-rank multiscale residual.

    Old decoder projection is fixed; every old network parameter is trainable.
    Features are physical k/cos/sin, normalized separately for each network.
    Output is normalized new POD complex coefficient pairs.
    """
    def __init__(self, old_config, new_config, transfer, bias, old_mean, old_std, new_mean, new_std, prior_gate=None):
        super().__init__()
        gate=dict(prior_gate or dict(enabled=False,start=4.,stop=5.))
        self.prior_gate=dict(enabled=bool(gate['enabled']),start=float(gate['start']),stop=float(gate['stop']))
        if not (np.isfinite(self.prior_gate['start']) and np.isfinite(self.prior_gate['stop']) and self.prior_gate['stop']>self.prior_gate['start']):
            raise ValueError('Finite increasing physical-k prior gate bounds required')
        self.old = SpectralSurrogate(**old_config)
        self.residual = SpectralSurrogate(**new_config)
        for branch in self.residual.branches:
            nn.init.zeros_(branch.project.weight); nn.init.zeros_(branch.project.bias)
        for name,value in dict(transfer=transfer,bias=bias,old_mean=old_mean,old_std=old_std,new_mean=new_mean,new_std=new_std).items():
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))

    def prior_weight(self,k):
        """C1 compact gate: one through k=4, smooth fade, zero from k=5.

        Bounds are explicit protocol constants; no new labels determine them.
        The full projected old prediction, including its decoder-mean offset,
        is gated. The new full-rank residual remains ungated at every k.
        """
        if not self.prior_gate['enabled']:return torch.ones_like(k)
        start,stop=self.prior_gate['start'],self.prior_gate['stop']
        t=((k-start)/(stop-start)).clamp(0.,1.)
        return 1.-3.*t.square()+2.*t.pow(3)

    def forward(self, physical):
        old_input=torch.cat(((physical[:,:1]-self.old_mean)/self.old_std,physical[:,1:]),dim=1)
        new_input=torch.cat(((physical[:,:1]-self.new_mean)/self.new_std,physical[:,1:]),dim=1)
        old = self.old(old_input)
        # transfer[j,r] is complex; multiplying real pairs avoids complex autograd.
        real=old[...,0]@self.transfer[...,0]-old[...,1]@self.transfer[...,1]
        imag=old[...,0]@self.transfer[...,1]+old[...,1]@self.transfer[...,0]
        prior=torch.stack((real,imag),dim=-1)+self.bias
        return prior*self.prior_weight(physical[:,0])[:,None,None]+self.residual(new_input)


def _prepare_one_cache(root,data,protocol):
    n_train=len(data.split_indices()[0])
    root=Path(root); directory=root/'cache'/f'n{n_train}'/'sno'; directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'cache.json'
    spec=dict(protocol_sha256=_json_hash(protocol),n_train=n_train)
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['specification']!=spec: raise ValueError('POD cache protocol differs')
        for name,digest in saved['files'].items():
            if _sha256(directory/name)!=digest: raise ValueError('POD cache hash mismatch')
        for name,digest in saved.get('transfer_sources',{}).items():
            if _sha256(Path(name))!=digest: raise ValueError('Historical transfer source changed')
        return saved
    started=perf_counter(); train=data.split_indices()[0]
    pod,diagnostics=fit_streaming_pod(data.u_scat,data.mass_matrix,train,
        protocol.get('pod_max_rank',128),protocol.get('pod_relative_tolerance',.01),protocol.get('pod_batch_size',16),
        expansion_rank=protocol.get('pod_expansion_rank',512),
        candidate_ranks=protocol.get('pod_rank_candidates',[32,64,96,128,192,256,384,512]))
    coefficients=[]; residual2=[]; norm2=[]
    for ids in _blocks(np.arange(len(data.k0)),16):
        fields=np.asarray(data.u_scat[ids]); c=_encode(pod,fields,data.mass_matrix)
        coefficients.append(c); residual2.extend(_norm2(fields-pod.decode(c),data.mass_matrix)); norm2.extend(_norm2(fields,data.mass_matrix))
    coefficients=np.concatenate(coefficients); scale=float(np.sqrt(np.mean(abs(coefficients[train])**2)))
    prep=dict(inputs=np.column_stack((data.k0,np.cos(data.theta),np.sin(data.theta))).astype('float32'),
        pairs=np.stack((coefficients.real,coefficients.imag),-1),scale=scale,
        k_mean=float(data.k0[train].mean()),k_std=float(data.k0[train].std()),
        residual2=np.array(residual2),norm2=np.array(norm2),train_ids=train,
        floor_relative_l2=np.sqrt(np.array(residual2)/norm2))
    # Historical decoder is evaluated as P1 fields on the new fine mesh.
    old_pod=ComplexPOD.load(OLD_ROOT/'cache/n1000/sno/pod.npz')
    with np.load(OLD_ROOT/'data/reference_problem.npz',allow_pickle=False) as mesh:
        transfer=P1Transfer(mesh['coordinates'],mesh['triangles'],data.coordinates)
    mean=transfer.apply(old_pod.mean); modes=transfer.matrix@old_pod.modes
    with np.load(OLD_ROOT/'cache/n1000/sno/sno_preprocessing.npz',allow_pickle=False) as old:
        weighted_old=data.mass_matrix@modes
        mapping=(weighted_old.conj().T@pod.modes).conj()*float(old['scale'])/scale
        bias=_encode(pod,mean,data.mass_matrix)/scale
        prep.update(transfer=np.stack((mapping.real,mapping.imag),-1),bias=np.stack((bias.real,bias.imag),-1),
            old_mean=float(old['k_mean']),old_std=float(old['k_std']))
    pod.save(directory/'pod.npz'); np.savez(directory/'preprocessing.npz',**prep)
    old_config=json.loads((OLD_ROOT/'cache/n1000/sno/sno_config.json').read_text())
    config=dict(old=old_config,new=model_config(pod.rank,METHOD,protocol.get('sno_width',8),protocol.get('depth',4)),
        old_prior_gate=protocol.get('old_prior_gate',dict(enabled=False,start=4.,stop=5.)))
    _write_json(directory/'model_config.json',config)
    subsets=dict(zip(('train','validation','test'),data.split_indices()))
    measured_train_rms=float(np.sqrt(np.mean(prep['floor_relative_l2'][train]**2)))
    if measured_train_rms>protocol.get('pod_relative_tolerance',.01)*(1+1e-7)+1e-10:
        raise ValueError('Direct reconstructed training POD residual does not satisfy the declared rank criterion')
    diagnostics.update(preprocessing_seconds=perf_counter()-started,
        projection_floor_mean_pct={name:float(100*prep['floor_relative_l2'][ids].mean()) for name,ids in subsets.items()},
        projection_floor_rms_pct={name:float(100*np.sqrt(np.mean(prep['floor_relative_l2'][ids]**2))) for name,ids in subsets.items()},
        projection_floor_max_pct={name:float(100*prep['floor_relative_l2'][ids].max()) for name,ids in subsets.items()})
    transfer_sources={str(p):_sha256(p) for p in (OLD_ROOT/'cache/n1000/sno/pod.npz',OLD_ROOT/'cache/n1000/sno/sno_preprocessing.npz',OLD_ROOT/'cache/n1000/sno/sno_config.json',OLD_ROOT/'data/reference_problem.npz')}
    result=dict(specification=spec,diagnostics=diagnostics,transfer_sources=transfer_sources,files={p.name:_sha256(p) for p in directory.iterdir() if p.is_file() and p.name!='cache.json'})
    _write_json(marker,result); return result


def prepare_cache(root,data,protocol,n_train=None):
    """Fit isolated train-prefix representations, never the union decoder."""
    from .dataset import training_view
    sizes=protocol.get('train_sizes',[len(data.split_indices()[0])]) if n_train is None else [n_train]
    results=[]
    for size in sizes:
        view=training_view(data,int(size))
        results.append(_prepare_one_cache(root,view,protocol))
        gc.collect()
    return results if n_train is None else results[0]


class Adapter:
    def __init__(self,root,seed,device='cpu',initialize_old=True,n_train=1000):
        directory=Path(root)/'cache'/f'n{int(n_train)}'/'sno'; self.device=torch.device(device)
        self.pod=ComplexPOD.load(directory/'pod.npz')
        with np.load(directory/'preprocessing.npz',allow_pickle=False) as z: self.prep={k:z[k] for k in z.files}
        self.config=json.loads((directory/'model_config.json').read_text())
        torch.manual_seed(seed)
        p=self.prep
        self.model=TransferredSurrogate(self.config['old'],self.config['new'],p['transfer'],p['bias'],p['old_mean'],p['old_std'],p['k_mean'],p['k_std'],prior_gate=self.config.get('old_prior_gate'))
        self.old_checkpoint=OLD_ROOT/f'runs/n1000/mscale_sno/seed_{seed}/model.pt'
        if initialize_old:
            checkpoint=torch.load(self.old_checkpoint,map_location='cpu',weights_only=False)
            self.model.old.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.inputs=torch.as_tensor(p['inputs'],device=self.device)
        self.loss=CoefficientLoss(p,device,torch.float32)
        self.validation_loss=CoefficientLoss(p,'cpu',torch.float64)

    @torch.no_grad()
    def values(self,ids,batch_size=50):
        self.model.eval()
        return np.concatenate([self.validation_loss(self.model(self.inputs[b]),b).cpu().numpy() for b in _blocks(ids,batch_size)])

    @torch.no_grad()
    def predict_fields(self,ids,batch_size=16):
        self.model.eval()
        for b in _blocks(ids,batch_size):
            pairs=self.model(self.inputs[b]).cpu().numpy().astype(float)*float(self.prep['scale'])
            yield b,self.pod.decode(pairs[...,0]+1j*pairs[...,1])


def run_directory(root,seed,n_train=1000): return Path(root)/'runs'/f'n{int(n_train)}'/METHOD/f'seed_{seed}'


def train_one(root,data,protocol,seed,total_norms):
    n_train=len(data.split_indices()[0])
    directory=run_directory(root,seed,n_train); directory.mkdir(parents=True,exist_ok=True)
    spec=dict(protocol_sha256=_json_hash(protocol),seed=seed,method=METHOD,n_train=len(data.split_indices()[0]))
    if (directory/'result.json').exists():
        if json.loads((directory/'config.json').read_text())['specification']!=spec: raise ValueError('Run protocol differs')
        return json.loads((directory/'result.json').read_text())
    adapter=Adapter(root,seed,protocol.get('device','cpu'),n_train=n_train)
    train,val,test=data.split_indices(); batches=int(np.ceil(len(train)/protocol['batch_size']))
    representation=dict(r_complex=int(adapter.pod.rank),pod_complex_rank=int(adapter.pod.rank),
        pod_real_unknowns=2*int(adapter.pod.rank),support_dofs=len(data.coordinates),
        support_real_unknowns=2*len(data.coordinates),model_real_parameters=parameter_count(adapter.model),
        old_model_real_parameters=parameter_count(adapter.model.old),old_prior_gate=adapter.model.prior_gate)
    config=dict(specification=spec,model=adapter.config,parameters=parameter_count(adapter.model),**representation,
        transfer='old seed checkpoint network jointly fine tuned; P1 old decoder projected into new train-only POD; explicit physical-k C1 prior gate; zero-output ungated full-rank residual SNO',
        old_checkpoint=str(adapter.old_checkpoint),old_checkpoint_sha256=_sha256(adapter.old_checkpoint))
    if (directory/'config.json').exists() and json.loads((directory/'config.json').read_text())!=config:
        raise ValueError('Existing training configuration or transferred checkpoint differs')
    _write_json(directory/'config.json',config)
    optimizer=torch.optim.Adam(adapter.model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol.get('weight_decay',0.),foreach=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=protocol['epochs']*batches)
    rng=np.random.default_rng(seed); first=1; history=[]; best=float('inf'); best_epoch=0; bad=0; steps=0; elapsed=0.; best_state=None
    initial_values=adapter.values(val,protocol['batch_size'])
    _write_json(directory/'initial_validation.json',dict(mean_relative_l2=float(initial_values.mean()),old_checkpoint_sha256=config['old_checkpoint_sha256']))
    if (directory/'resume.pt').exists():
        state=torch.load(directory/'resume.pt',map_location='cpu',weights_only=False)
        if state['specification']!=spec: raise ValueError('Resume protocol differs')
        adapter.model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        rng.bit_generator.state=state['numpy_rng']; torch.set_rng_state(state['torch_rng'])
        first=state['epoch']+1; history=state['history']; best=state['best']; best_epoch=state['best_epoch']; bad=state['bad']; steps=state['steps']; elapsed=state['elapsed']; best_state=state['best_state']
    started=perf_counter()
    for epoch in range(first,protocol['epochs']+1):
        if bad>=protocol['patience_checks']: break
        adapter.model.train()
        for ids in np.array_split(rng.permutation(train),batches):
            optimizer.zero_grad(set_to_none=True); loss=adapter.loss(adapter.model(adapter.inputs[ids]),ids).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); optimizer.step(); scheduler.step(); steps+=1
        if epoch!=1 and epoch%protocol['validation_interval'] and epoch!=protocol['epochs']: continue
        # Both train/validation curves are actual errors of this epoch's model.
        training=adapter.values(train,protocol['batch_size']); validation=adapter.values(val,protocol['batch_size'])
        if not np.isfinite(validation).all(): raise FloatingPointError('Nonfinite validation')
        score=float(validation.mean())
        history.append(dict(epoch=epoch,optimizer_steps=steps,train_relative_l2=float(training.mean()),val_relative_l2=score,val_p95=float(np.percentile(validation,95)),elapsed_seconds=elapsed+perf_counter()-started,learning_rate=optimizer.param_groups[0]['lr']))
        if score<best:
            best,best_epoch,bad=score,epoch,0; best_state=_model_state(adapter.model)
            _save(directory/'model.pt',dict(state_dict=best_state,best_epoch=best_epoch,config=config))
        else: bad+=1
        _save(directory/'resume.pt',dict(specification=spec,model=adapter.model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),epoch=epoch,steps=steps,history=history,best=best,best_epoch=best_epoch,bad=bad,elapsed=history[-1]['elapsed_seconds'],best_state=best_state))
        _write_json(directory/'history.json',history)
        print(f'high-k seed={seed} epoch={epoch}/{protocol["epochs"]} val={100*score:.5f}% best={100*best:.5f}% elapsed={history[-1]["elapsed_seconds"]:.1f}s',flush=True)
    adapter.model.load_state_dict(best_state)
    # Never materialize all test fields: persist and score each native FEM batch.
    prediction=np.lib.format.open_memmap(directory/'predictions.npy',mode='w+',dtype=np.complex128,shape=(len(test),len(data.coordinates)))
    rows=[]; offset=0
    for ids,fields in adapter.predict_fields(test):
        prediction[offset:offset+len(ids)]=fields
        error=_norm2(fields-np.asarray(data.u_scat[ids]),data.mass_matrix)
        scattered=100*np.sqrt(error/adapter.prep['norm2'][ids]); total=100*np.sqrt(error/total_norms[offset:offset+len(ids)])
        rows.extend(dict(n_train=len(train),method=METHOD,seed=seed,index=int(i),k=float(data.k0[i]),theta=float(data.theta[i]),scattered_pct=float(s),total_pct=float(t)) for i,s,t in zip(ids,scattered,total)); offset+=len(ids)
    prediction.flush(); del prediction
    _write_json(directory/'metrics.json',rows)
    scattered=np.array([r['scattered_pct'] for r in rows]); total=np.array([r['total_pct'] for r in rows])
    state=adapter.model.state_dict(); old=torch.load(adapter.old_checkpoint,map_location='cpu',weights_only=False)['state_dict']
    delta=float(sum((state['old.'+key].cpu()-value).double().square().sum().item() for key,value in old.items())**.5)
    result=dict(n_train=len(train),method=METHOD,variant=METHOD,**representation,directory=str(directory.resolve()),seed=seed,parameters=config['parameters'],epochs_completed=history[-1]['epoch'],best_epoch=best_epoch,best_validation_relative_l2=best,training_seconds=history[-1]['elapsed_seconds'],optimizer_steps=steps,stop_reason='validation_patience' if bad>=protocol['patience_checks'] else 'epochs_completed',scattered_mean_pct=float(scattered.mean()),scattered_p95_pct=float(np.percentile(scattered,95)),total_mean_pct=float(total.mean()),total_p95_pct=float(np.percentile(total,95)),old_weights_l2_change=delta,checkpoint_sha256=_sha256(directory/'model.pt'),predictions_sha256=_sha256(directory/'predictions.npy'))
    if delta<=0: raise ValueError('Fine tuning failed to change old checkpoint weights')
    _write_json(directory/'result.json',result); del adapter,optimizer,scheduler; gc.collect(); return result


def train_all(root,data,protocol):
    import fcntl
    root=Path(root); (root/'runs').mkdir(parents=True,exist_ok=True); (root/'results').mkdir(parents=True,exist_ok=True)
    with (root/'runs/.training.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another high-k trainer is running') from None
        from .dataset import training_view
        torch.set_num_threads(protocol.get('threads',2)); prepare_cache(root,data,protocol)
        norms_path=root/'cache/test_total_norms.npy'
        if not norms_path.exists(): np.save(norms_path,_total_norms(data,data.split_indices()[2]),allow_pickle=False)
        norms=np.load(norms_path,allow_pickle=False); runs=[]
        for size in protocol.get('train_sizes',[len(data.split_indices()[0])]):
            view=training_view(data,int(size))
            for seed in protocol['seeds']:
                runs.append(train_one(root,view,protocol,seed,norms))
                _write_json(root/'results/training_summary.json',dict(runs=runs))
        return runs


def _verify_one(root,data,protocol):
    root=Path(root); records=[]; test=data.split_indices()[2]; n_train=len(data.split_indices()[0])
    for seed in protocol['seeds']:
        directory=run_directory(root,seed,n_train); result=json.loads((directory/'result.json').read_text())
        adapter=Adapter(root,seed,protocol.get('device','cpu'),initialize_old=False,n_train=n_train)
        checkpoint=torch.load(directory/'model.pt',map_location='cpu',weights_only=False); adapter.model.load_state_dict(checkpoint['state_dict'])
        history=json.loads((directory/'history.json').read_text())
        selected=min(history,key=lambda r:r['val_relative_l2'])['epoch']
        if selected!=checkpoint['best_epoch'] or selected!=result['best_epoch']: raise ValueError('Not validation-selected')
        saved=np.load(directory/'predictions.npy',mmap_mode='r',allow_pickle=False); scores=[]; totals=[]; offset=0
        rows=json.loads((directory/'metrics.json').read_text())
        if [row['index'] for row in rows]!=list(test): raise ValueError('Unpaired saved physical queries')
        for row,i in zip(rows,test):
            if row['k']!=float(data.k0[i]) or row['theta']!=float(data.theta[i]): raise ValueError('Saved physical query changed')
        total_norms=np.load(root/'cache/test_total_norms.npy',allow_pickle=False)
        for ids,fields in adapter.predict_fields(test):
            if not np.allclose(fields,saved[offset:offset+len(ids)],rtol=3e-5,atol=2e-6): raise ValueError('Checkpoint replay differs')
            error2=_norm2(fields-np.asarray(data.u_scat[ids]),data.mass_matrix)
            scores.extend(100*np.sqrt(error2/_norm2(np.asarray(data.u_scat[ids]),data.mass_matrix)))
            totals.extend(100*np.sqrt(error2/total_norms[offset:offset+len(ids)])); offset+=len(ids)
        if not np.isclose(np.mean(scores),result['scattered_mean_pct'],rtol=1e-6,atol=1e-6): raise ValueError('Independent FEM metric differs')
        if not np.allclose(scores,[row['scattered_pct'] for row in rows],rtol=1e-6,atol=1e-6): raise ValueError('Per-case FEM scattered scores differ')
        if not np.allclose(totals,[row['total_pct'] for row in rows],rtol=1e-6,atol=1e-6): raise ValueError('Per-case FEM total scores differ')
        if not np.isclose(np.mean(totals),result['total_mean_pct'],rtol=1e-6,atol=1e-6): raise ValueError('Total FEM metric differs')
        for name,key in [('model.pt','checkpoint_sha256'),('predictions.npy','predictions_sha256')]:
            if _sha256(directory/name)!=result[key]: raise ValueError('Run artifact changed')
        records.append(dict(n_train=n_train,seed=seed,checkpoint_sha256=result['checkpoint_sha256'],predictions_sha256=result['predictions_sha256'],old_weights_l2_change=result['old_weights_l2_change'])); del adapter,saved; gc.collect()
    return records


def verify(root,data,protocol):
    from .dataset import training_view
    records=[]
    for size in protocol.get('train_sizes',[len(data.split_indices()[0])]):
        records.extend(_verify_one(root,training_view(data,int(size)),protocol))
    result=dict(passed=True,runs=records,protocol_sha256=_json_hash(protocol))
    _write_json(Path(root)/'results/verification.json',result)
    return result


def timing_pilot(root, *, rank=128, batch_size=50, threads=2, measured_steps=8, device='cpu'):
    """Bounded synthetic CPU/MPS pilot, without data access or model selection.

    Times the actual transferred architecture, Adam update, and coefficient
    objective. Extrapolation includes complete 1000-row train/100-row validation
    evaluation every five epochs; disk checkpoint I/O and POD setup are excluded.
    """
    if min(rank,batch_size,threads,measured_steps)<1 or measured_steps>30:
        raise ValueError('Positive pilot dimensions and at most 30 measured steps required')
    if device not in ('cpu','mps'): raise ValueError('CPU or MPS pilot required')
    if device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    torch.set_num_threads(threads); torch.manual_seed(11)
    rng=np.random.default_rng(11)
    old_config=model_config(20,METHOD,8,4); new_config=model_config(rank,METHOD,8,4)
    mapping=rng.normal(scale=.05,size=(20,rank,2)); bias=np.zeros((rank,2))
    model=TransferredSurrogate(old_config,new_config,mapping,bias,2.,1.,10.,5.)
    # Match steady-state nonzero residual heads; pristine heads initially block
    # deeper residual gradients, so measuring only initialization is optimistic.
    for branch in model.residual.branches:
        nn.init.normal_(branch.project.weight,std=.02)
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,foreach=False)
    inputs=torch.randn(batch_size,3); inputs[:,0]=inputs[:,0]*5+10; inputs=inputs.to(device)
    prep=dict(pairs=rng.normal(size=(batch_size,rank,2)),scale=1.,
        residual2=np.full(batch_size,.01),norm2=np.full(batch_size,2.*rank))
    loss=CoefficientLoss(prep,device,torch.float32); ids=np.arange(batch_size)
    synchronize=lambda: torch.mps.synchronize() if device=='mps' else None
    def update():
        optimizer.zero_grad(set_to_none=True)
        loss(model(inputs),ids).mean().backward(); optimizer.step()
    for _ in range(2): update()
    times=[]
    for _ in range(measured_steps):
        synchronize(); started=perf_counter(); update(); synchronize(); times.append(perf_counter()-started)
    evaluation=[]
    with torch.no_grad():
        for _ in range(measured_steps):
            synchronize(); started=perf_counter(); model(inputs); synchronize(); evaluation.append(perf_counter()-started)
    step=float(np.median(times)); forward=float(np.median(evaluation))
    batches=int(np.ceil(1000/batch_size)); validation_batches=int(np.ceil(100/batch_size))
    seconds_per_seed=1000*batches*step+201*(batches+validation_batches)*forward
    result=dict(synthetic=True,data_access=False,selection_use=False,device=device,threads=threads,
        rank=rank,batch_size=batch_size,seed=11,warmup_updates=2,measured_updates=measured_steps,
        parameters=parameter_count(model),update_seconds=times,median_update_seconds=step,
        median_forward_seconds=forward,projected_1000_epoch_seconds_per_seed=seconds_per_seed,
        projected_1000_epoch_three_seed_hours=seconds_per_seed*3/3600,
        assumptions='1000 training rows,100 validation rows,1000 epochs,validation every5 plus epoch1; includes full train evaluation; excludes POD, disk I/O and test scoring',
        torch_version=torch.__version__,mps_available=bool(torch.backends.mps.is_available()))
    directory=Path(root)/'results'; directory.mkdir(parents=True,exist_ok=True)
    _write_json(directory/f'training_timing_pilot_{device}_rank{rank}_batch{batch_size}.json',result)
    return result


def device_consistency_pilot(root, *, rank=128, batch_size=50, threads=2):
    """Compare identical CPU/MPS weights, inputs, coefficient loss and gradients."""
    import copy
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    torch.set_num_threads(threads); torch.manual_seed(11); rng=np.random.default_rng(11)
    model=TransferredSurrogate(model_config(20,METHOD,8,4),model_config(rank,METHOD,8,4),
        rng.normal(scale=.05,size=(20,rank,2)),np.zeros((rank,2)),2.,1.,10.,5.)
    for branch in model.residual.branches: nn.init.normal_(branch.project.weight,std=.02)
    accelerated=copy.deepcopy(model).to('mps')
    inputs=torch.randn(batch_size,3); inputs[:,0]=inputs[:,0]*5+10
    prep=dict(pairs=rng.normal(size=(batch_size,rank,2)),scale=1.,residual2=np.full(batch_size,.01),norm2=np.full(batch_size,2.*rank))
    ids=np.arange(batch_size); cpu_prediction=model(inputs); mps_prediction=accelerated(inputs.to('mps'))
    cpu_loss=CoefficientLoss(prep,'cpu',torch.float32)(cpu_prediction,ids).mean()
    mps_loss=CoefficientLoss(prep,'mps',torch.float32)(mps_prediction,ids).mean()
    cpu_loss.backward(); mps_loss.backward(); torch.mps.synchronize()
    gradient_error2=0.; gradient_norm2=0.; details=[]
    for (name,parameter),(other_name,other) in zip(model.named_parameters(),accelerated.named_parameters()):
        assert name==other_name
        reference=parameter.grad.detach().double(); candidate=other.grad.detach().cpu().double()
        error2=float((candidate-reference).square().sum()); norm2=float(reference.square().sum())
        gradient_error2+=error2; gradient_norm2+=norm2
        details.append(dict(parameter=name,relative_l2=float(np.sqrt(error2/max(norm2,1e-30)))))
    error=float(np.sqrt(gradient_error2/max(gradient_norm2,1e-30)))
    cpu_value=float(cpu_loss.detach()); mps_value=float(mps_loss.detach().cpu())
    reference=cpu_prediction.detach().double(); candidate=mps_prediction.detach().cpu().double()
    prediction_error=float(torch.linalg.vector_norm(candidate-reference)/torch.linalg.vector_norm(reference))
    result=dict(synthetic=True,data_access=False,rank=rank,batch_size=batch_size,seed=11,threads=threads,
        cpu_loss=cpu_value,mps_loss=mps_value,loss_relative_error=abs(cpu_value-mps_value)/abs(cpu_value),
        prediction_relative_l2=prediction_error,gradient_relative_l2=error,
        finite=bool(np.isfinite(error) and np.isfinite(mps_value)),gradient_details=details)
    directory=Path(root)/'results'; directory.mkdir(parents=True,exist_ok=True)
    _write_json(directory/f'device_consistency_rank{rank}_batch{batch_size}.json',result)
    return result
