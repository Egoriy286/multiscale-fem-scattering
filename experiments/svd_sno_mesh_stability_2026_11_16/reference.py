"""Independent native CG3 test reference on the middle, identical-polygon mesh."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.highk_svd_gmsfem.highorder import HighOrderSolver
from .physics import ROOT, digest


def generate(root=ROOT):
    root=Path(root).resolve(); manifest=root/'queries.json'; mesh=root/'meshes/level1.msh'
    raw=json.loads(manifest.read_text()); queries=raw['queries'] if isinstance(raw,dict) else raw
    indices=[i for i,q in enumerate(queries) if q['split'] in (2,'test')]
    if len(indices)!=200: raise ValueError(f'Expected 200 test queries, got {len(indices)}')
    out=root/'data/reference'; out.mkdir(parents=True,exist_ok=True)
    solver=HighOrderSolver(mesh,degree=3,iterative=True)
    shape=(len(indices),solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs)
    if shape[1]!=377530: raise ValueError(f'Unexpected reference DOFs: {shape[1]}')
    repo=ROOT.parents[1]
    sources=[Path(__file__),repo/'experiments/highk_svd_gmsfem/highorder.py',repo/'gmsfem/fem.py']
    identity=dict(manifest_sha256=digest(manifest),mesh_sha256=digest(mesh),degree=3,shape=list(shape),dtype='complex128',
                  query_indices=indices,sources_sha256={str(p.relative_to(repo)):digest(p) for p in sources})
    path=out/'fields.npy'; progress=out/'progress.json'
    if progress.exists():
        state=json.loads(progress.read_text())
        if state['identity']!=identity: raise ValueError('Reference identity changed; refusing unsafe resume')
        fields=np.lib.format.open_memmap(path,mode='r+')
        if fields.shape!=shape or fields.dtype!=np.complex128: raise ValueError('Reference array mismatch')
    else:
        fields=np.lib.format.open_memmap(path,mode='w+',dtype=np.complex128,shape=shape)
        state=dict(identity=identity,completed=[],stats={}); progress.write_text(json.dumps(state,indent=2)+'\n')
    completed=set(state['completed'])
    for j in sorted(range(len(indices)),key=lambda j:float(queries[indices[j]]['k'])):
        if j in completed: continue
        q=queries[indices[j]]; field,stat=solver.solve(float(q['k']),float(q['theta']))
        fields[j]=field; fields.flush(); state['completed'].append(j); state['stats'][str(j)]=stat
        tmp=progress.with_suffix('.tmp'); tmp.write_text(json.dumps(state,indent=2)+'\n'); tmp.replace(progress)
        print(f'CG3 reference {len(state["completed"])}/{len(indices)}: residual {stat["residual"]:.2e}',flush=True)
    if any(not np.isfinite(x['residual']) or x['residual']>=1e-10 for x in state['stats'].values()): raise ValueError('Unacceptable reference residual')
    result=dict(identity=identity,completed_count=len(completed | set(state['completed'])),
                maximum_original_equation_residual=max(x['residual'] for x in state['stats'].values()),
                file_sha256={p.name:digest(p) for p in (path,progress)})
    (out/'dataset_info.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=ROOT)
    print(json.dumps(generate(parser.parse_args().root),indent=2))
