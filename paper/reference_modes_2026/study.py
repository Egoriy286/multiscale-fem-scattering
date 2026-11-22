"""Weighted snapshot SVD and controlled training runs for the reference-grid paper."""
from copy import deepcopy
import csv
import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.linalg import eigh
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from neural_operator.data import mass_matrix
from neural_operator.sno.model import SVDSNO

DATA = HERE / 'data'
RUNS = HERE / 'runs'
SIZES = (400, 600, 800, 1000)
RANKS = (12, 24, 48)
WIDTHS = (24, 48)
DEPTH, SEED, EPOCHS, BATCH = 4, 2026, 180, 64
CONFIGS = ([(n, 24, 48) for n in SIZES[:-1]] +
           [(1000, r, w) for r in RANKS for w in WIDTHS])

def write_csv(path, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)

def squared_norm(x, mass):
    return np.sum(x.conj() * (mass @ x.T).T, axis=1).real

def prepare(n, fields, mass, train):
    """Solve two snapshot eigenproblems; retain direct projections of all rows."""
    path = RUNS / f'pod_{n}.npz'
    if path.exists():
        return
    rank = max(r for count,r,w in CONFIGS if count == n)
    norms = np.empty(len(fields))
    for i in range(0, len(fields), 16):
        norms[i:i+16] = squared_norm(fields[i:i+16], mass)
    out = dict(norm2=norms, train=train)
    for part in ('real','imag'):
        samples = np.ascontiguousarray(getattr(fields[train], part))
        avg = samples.mean(axis=0)
        samples -= avg
        # Tiling keeps the dense mass-weighted snapshot matrix out of memory.
        gram = np.empty((n,n))
        for i in range(0,n,64):
            gram[:,i:i+64] = samples @ (mass @ samples[i:i+64].T)
        eig, vectors = eigh((gram+gram.T)*.5, subset_by_index=(n-rank,n-1))
        assert eig[0] > eig[-1]*1e-12
        basis = samples.T @ (vectors[:, ::-1] / np.sqrt(eig[::-1]))
        # Orient each mode by its largest nodal value for reproducible replay.
        basis *= np.sign(basis[np.abs(basis).argmax(axis=0), np.arange(rank)])
        assert np.max(np.abs(basis.T @ (mass @ basis)-np.eye(rank))) < 1e-8
        coeff, centered_norm = np.empty((len(fields),rank)), np.empty(len(fields))
        for i in range(0,len(fields),24):
            centered = getattr(fields[i:i+24],part) - avg
            weighted = (mass @ centered.T).T
            coeff[i:i+24] = weighted @ basis
            centered_norm[i:i+24] = np.sum(weighted*centered, axis=1)
        out.update({part+'_mean':avg, part+'_basis':basis, part+'_c':coeff,
                    part+'_norm2':centered_norm})
    tmp = RUNS / f'pod_{n}.tmp.npz'
    np.savez(tmp, **out)
    tmp.replace(path)

def components(pod, r):
    c = np.stack([pod[p+'_c'][:,:r] for p in ('real','imag')], axis=-1)
    remainder = pod['real_norm2'] + pod['imag_norm2'] - np.sum(c*c,axis=(1,2))
    assert np.min(remainder / pod['norm2']) >= -1e-10
    return c, np.maximum(remainder,0)

def features(q, tr):
    k = q['k']
    x = np.column_stack((np.cos(q['theta']),np.sin(q['theta']),
                         (k-k[tr].mean())/k[tr].std()))
    return torch.tensor(x,dtype=torch.float32)

def train_run(n,r,w,q,pod):
    path = RUNS / f'n{n}_r{r}_w{w}.pt'
    if path.exists():
        return
    torch.manual_seed(SEED)
    tr, va = pod['train'], np.flatnonzero(q['split']==1)
    net = SVDSNO(3,r,w,DEPTH)
    opt = torch.optim.Adam(net.parameters(),lr=.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.00015)
    c, remainder = components(pod,r)
    std = torch.tensor(c[tr].std(axis=0),dtype=torch.float32)
    target = torch.tensor(c,dtype=torch.float32)
    rem = torch.tensor(remainder,dtype=torch.float32)
    norm = torch.tensor(pod['norm2'],dtype=torch.float32)
    x = features(q,tr)
    def loss(ids):
        delta = net(x[ids])*std-target[ids]
        return torch.sqrt((delta.square().sum((1,2))+rem[ids])/norm[ids]).mean()
    best, history = float('inf'), []
    for epoch in range(1,EPOCHS+1):
        for ids in torch.as_tensor(tr)[torch.randperm(n)].split(BATCH):
            opt.zero_grad()
            loss(ids).backward()
            opt.step()
        scheduler.step()
        with torch.no_grad():
            val = float(loss(va))
        history.append(val)
        if val < best:
            best, best_epoch, weights = val, epoch, deepcopy(net.state_dict())
    temp = path.with_suffix('.tmp')
    torch.save(dict(weights=weights, scale=std, epoch=best_epoch, val=best,
                    history=history, n=n, r=r, w=w, seed=SEED),temp)
    temp.replace(path)
    print(f'N={n} R={r} w={w}: val {100*best:.3f}% epoch {best_epoch}', flush=True)

def evaluate(q, fields, mass):
    summary, cases = [], []
    test = np.flatnonzero(q['split']==2)
    # This example is fixed by its row number, independent of prediction errors.
    sample = test[len(test)//2]
    for n,r,w in CONFIGS:
        with np.load(RUNS / f'pod_{n}.npz') as pod:
            tr = pod['train']
            cp = torch.load(RUNS / f'n{n}_r{r}_w{w}.pt',map_location='cpu')
            net = SVDSNO(3,r,w,DEPTH)
            net.load_state_dict(cp['weights'])
            with torch.no_grad():
                pred = (net(features(q,tr)[test])*cp['scale']).numpy().astype('float64')
            c, rem = components(pod,r)
            relative = np.sqrt((np.sum((pred-c[test])**2,axis=(1,2))+rem[test])/pod['norm2'][test])
            floors = np.sqrt(rem[test]/pod['norm2'][test])
            model = dict(n=n,r=r,w=w,parameters=sum(p.numel() for p in net.parameters()))
            assert model['parameters'] == DEPTH*(2*w*w+w+r*r)+8*w+2
            # Independent nodal reconstruction and mass norm of every prediction.
            def decode(coefficients):
                out = []
                for j, part in enumerate(('real','imag')):
                    out.append(coefficients[:,:,j] @ pod[part+'_basis'][:,:r].T + pod[part+'_mean'])
                return out[0]+1j*out[1]
            for i in range(0,len(test),16):
                sl = slice(i,i+16)
                ids = test[sl]
                for coeff,expected in ((pred[sl],relative[sl]),(c[ids],floors[sl])):
                    error = fields[ids]-decode(coeff)
                    actual = np.sqrt(squared_norm(error,mass)/squared_norm(fields[ids],mass))
                    np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-9)
            for idx,err,floor in zip(test,100*relative,100*floors):
                cases.append(dict(**model,row=idx,k=q['k'][idx],theta=q['theta'][idx],
                                  error_pct=err,projection_pct=floor))
            summary.append(dict(**model,epoch=cp['epoch'],val_pct=100*cp['val'],
                   mean_pct=100*relative.mean(),median_pct=100*np.median(relative),
                   p95_pct=100*np.quantile(relative,.95),projection_pct=100*floors.mean()))
            if (n,r,w) == (1000,24,48):
                loc = np.flatnonzero(test==sample)
                np.savez(RUNS/'example.npz',row=sample,k=q['k'][sample],theta=q['theta'][sample],
                         reference=fields[sample],prediction=decode(pred[loc])[0])
    write_csv(HERE/'sno_cases.csv',cases)
    write_csv(HERE/'sno_summary.csv',summary)

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def run_protocol():
    info = json.loads((DATA/'complete.json').read_text())
    for name, expected in info['sha256'].items():
        if digest(DATA/name) != expected:
            raise ValueError(f'FEM inputs changed: {name}')
    code = [HERE/'study.py', HERE.parents[1]/'neural_operator/sno/model.py',
            HERE.parents[1]/'neural_operator/_layers.py', HERE.parents[1]/'neural_operator/data.py']
    return dict(data=info, code={p.name:digest(p) for p in code},
                numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                depth=DEPTH, seed=SEED, epochs=EPOCHS, batch=BATCH, runs=CONFIGS)

def check_run_cache():
    path = RUNS/'protocol.json'
    protocol = json.dumps(run_protocol(), indent=2) + '\n'
    if path.exists():
        if path.read_text() != protocol:
            raise ValueError('Input data, runtime or model configuration changed. Use a new run directory.')
    elif any(RUNS.iterdir()):
        raise ValueError('Existing runs have no matching protocol.')
    else:
        path.write_text(protocol)

def main():
    assert (DATA/'complete.json').exists(), 'Complete FEM generation before training'
    RUNS.mkdir(exist_ok=True)
    check_run_cache()
    torch.set_num_threads(2)
    with np.load(DATA/'queries.npz') as q, np.load(DATA/'reference_problem.npz') as geom:
        fields = np.load(DATA/'fields.npy',mmap_mode='r')
        mass = mass_matrix(geom['coordinates'],geom['triangles'])
        groups = [set(q['k'][q['split']==s]) for s in (0,1,2)]
        assert not (groups[0]&groups[1] or groups[1]&groups[2] or groups[0]&groups[2])
        for n in SIZES:
            tr = np.flatnonzero(q['split']==0)[:n]
            assert len(np.unique(q['k'][tr]))*8 == len(tr) == n
            prepare(n,fields,mass,tr)
            with np.load(RUNS / f'pod_{n}.npz') as pod:
                for count,r,w in CONFIGS:
                    if count == n:
                        train_run(n,r,w,q,pod)
        evaluate(q,fields,mass)

if __name__ == '__main__':
    main()
