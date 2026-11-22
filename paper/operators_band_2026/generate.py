"""New reference FEM data with four variable material parameters."""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.sparse.linalg import splu
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.reference_sno_gmsfem.dataset import _reference_problem, _save_npz, _sha256, _write_json

def queries():
    points = qmc.Sobol(7, scramble=True, seed=20261124).random_base2(7)[:124]
    media = qmc.scale(points[:,:4],[.4,.4,2,2],[.9,.9,6,6])
    edges = np.linspace(.05,4,4)
    interior = edges[:-1] + points[:,4:]*np.diff(edges)
    bands = np.column_stack((np.full(124,.05),interior,np.full(124,4.)))
    group_split = np.repeat([0,1,2],[100,12,12])
    return dict(medium=np.repeat(np.arange(len(media)),10),
                split=np.repeat(group_split,10), k=np.repeat(bands.ravel(),2),
                theta=np.tile([np.pi/8,5*np.pi/8],len(media)*5),
                **{name:np.repeat(media[:,j],10) for j,name in
                   enumerate(('alpha1','alpha2','n1','n2'))})

def material_fields(tags, q, i):
    region = np.where(tags==1,0,np.where(tags==4,2,1))
    assert set(np.unique(tags))=={1,2,3,4}
    return [np.array([1.,q[f'{key}1'][i],q[f'{key}2'][i]])[region]
            for key in ('alpha','n')]

def main():
    d=HERE/'data'
    d.mkdir(exist_ok=True)
    q=queries()
    manifest=d/'queries.npz'
    if manifest.exists():
        with np.load(manifest) as saved:
            assert all(np.array_equal(saved[k],v) for k,v in q.items())
    else:
        _save_npz(manifest,**q)
    if (d/'complete.json').exists():
        info=json.loads((d/'complete.json').read_text())
        assert all(_sha256(d/name)==value for name,value in info['sha256'].items())
        return
    solver,_=_reference_problem(d)
    path=d/'fields.npy'
    shape=(len(q['medium']),len(solver.coordinates))
    fields=(np.load(path,mmap_mode='r+') if path.exists() else
            np.lib.format.open_memmap(path,'w+',np.complex128,shape))
    assert fields.shape==shape and fields.dtype==np.complex128
    state=d/'residuals.npz'
    errors=np.load(state)['values'] if state.exists() else np.full(shape[0],np.nan)
    for medium in np.unique(q['medium']):
        ids=np.flatnonzero(q['medium']==medium)
        if np.all(np.isfinite(errors[ids])):
            assert np.isfinite(fields[ids]).all()
            continue
        i=ids[0]
        a,n=material_fields(solver.cell_tags,q,i)
        p=solver.build_problem(a,n,1.,q['theta'][i])
        if medium==0:
            rhs0=solver.rhs(a,n,0.,q['theta'][i])
            state0=dict(rhs_norm=float(np.linalg.norm(rhs0)),
                        constant_kernel_residual=float(np.linalg.norm(p.K@np.ones(p.K.shape[0]))))
            assert state0['rhs_norm']==0. and state0['constant_kernel_residual']<1.e-9
            _write_json(d/'static.json',state0)
        solutions=[]
        residual=[]
        for start in range(0,10,2):
            pair=ids[start:start+2]
            k=q['k'][pair[0]]
            mat=(p.K-k*k*p.Mn-1j*k*p.B).tocsc()
            rhs=np.column_stack([solver.rhs(a,n,k,t) for t in q['theta'][pair]])
            pair_sol=splu(mat).solve(rhs)
            residual.extend(np.linalg.norm(mat@pair_sol-rhs,axis=0)/np.linalg.norm(rhs,axis=0))
            solutions.append(pair_sol)
        sol=np.column_stack(solutions)
        rel=np.array(residual)
        if not np.isfinite(rel).all() or rel.max()>1e-8:
            raise ArithmeticError(f'Bad residual for medium {medium}: {rel}')
        fields[ids]=sol.T
        fields.flush()
        errors[ids]=rel
        _save_npz(state,values=errors)
        if medium%20==0: print(f'{medium+1}/124 media',flush=True)
    names=['queries.npz','fields.npy','etal.msh','reference_problem.npz','residuals.npz','static.json']
    _write_json(d/'complete.json',dict(samples=shape[0],media=124,residual_max=float(errors.max()),
               sha256={name:_sha256(d/name) for name in names}))
if __name__=='__main__':
    main()
