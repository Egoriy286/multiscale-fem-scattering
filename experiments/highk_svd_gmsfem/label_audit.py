"""Independent direct-LU replay of selected newly generated native FEM labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .dataset import HERE, verify_dataset
from .highorder import selected_reference
from experiments.reference_sno_gmsfem.dataset import _sha256, _write_json


def run(root=HERE):
    root=Path(root).resolve();data=verify_dataset(root/'data')
    info=json.loads((root/'data/dataset_info.json').read_text())
    residuals=np.load(root/'data/linear_residuals.npy')
    if not np.isfinite(residuals).all() or np.max(residuals)>=1e-10:
        raise ValueError('Native original-equation residual exceeds declared tolerance')
    reference=selected_reference(root);solver=reference['solver']
    solver.native.iterative=False
    # Index-only plan, independent of trained errors or labels.
    train,val,test=data.split_indices()
    indices=np.unique(np.r_[val[[0,len(val)//2,-1]],test[[0,len(test)//2,-1]],
                              train[np.linspace(0,len(train)-1,6,dtype=int)]]).astype(int)
    signature=dict(dataset_sha256=info['sha256'],reference=reference['metadata'],
                   indices=indices.tolist(),auditor_sha256=_sha256(Path(__file__)))
    path=root/'results/fem_label_audit.json';rows=[]
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['specification']!=signature:raise ValueError('Label audit signature changed')
        rows=saved['rows']
    done={r['index'] for r in rows}
    for i in indices:
        if int(i) in done:continue
        start=perf_counter();result=solver.solve_on_support(float(data.k0[i]),float(data.theta[i]))
        delta=result['field']-data.u_scat[i];mass=data.mass_matrix
        relative=float(np.sqrt(np.real(np.vdot(delta,mass@delta))/
                               np.real(np.vdot(data.u_scat[i],mass@data.u_scat[i]))))
        if not np.isfinite(relative) or relative>1e-7:
            raise ValueError(f'Direct replay differs at row {i}: {relative}')
        rows.append(dict(index=int(i),k=float(data.k0[i]),theta=float(data.theta[i]),
                         direct_vs_saved_relative_mass_l2=relative,
                         direct_native_residual=result['relative_residual'],seconds=perf_counter()-start))
        _write_json(path,dict(passed=False,specification=signature,rows=rows))
        print(f'Direct native FEM label replay {len(rows)}/{len(indices)}; k={data.k0[i]:.7g}; '
              f'difference={relative:.3g}',flush=True)
    # k=0 is verified algebraically and remains outside all relative-error sets.
    from dolfinx.fem import petsc
    from petsc4py import PETSc
    native=solver.native;native.k.value=0.
    vector=petsc.assemble_vector(native.load)
    vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)
    zero_rhs=float(np.linalg.norm(vector.array));vector.destroy()
    constant=np.ones(native.K.shape[0])
    from scipy.sparse.linalg import norm
    kernel=float(np.linalg.norm(native.K@constant)/(norm(native.K)*np.linalg.norm(constant)))
    if zero_rhs!=0 or kernel>1e-12:raise ValueError('Unexpected static-limit formulation')
    value=dict(passed=True,specification=signature,rows=rows,
               all_native_original_equation_residuals=dict(cases=len(residuals),maximum=float(residuals.max())),
               direct_replay_max_relative_mass_l2=max(r['direct_vs_saved_relative_mass_l2'] for r in rows),
               zero_limit=dict(k=0.,rhs_norm=zero_rhs,relative_constant_kernel_defect=kernel,
                               interpretation='nonunique static constant; relative scattered error undefined; excluded from datasets'))
    _write_json(path,value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=HERE)
    args=parser.parse_args();run(args.root)
