"""Cross-mesh check on the same new test materials and directions."""
import sys
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import splu
from generate import HERE, material_fields
from neural_operator.fem import HelmholtzSolver
from neural_operator.mesh_transfer import P1Transfer
from neural_operator.data import mass_matrix
from experiments.reference_sno_gmsfem.baselines import FINE_MESH,FINE_SHA256,_checked_mesh
from experiments.reference_sno_gmsfem.dataset import _save_npz

def solve():
    d=HERE/'data'
    q=np.load(d/'queries.npz')
    geom=np.load(d/'reference_problem.npz')
    solver=HelmholtzSolver(_checked_mesh(FINE_MESH,FINE_SHA256))
    transfer=P1Transfer(solver.coordinates,solver.triangles,geom['coordinates'])
    rows=np.flatnonzero(q['split']==2)
    out=[]
    for group in np.unique(q['medium'][rows]):
        ids=rows[q['medium'][rows]==group]
        a,n=material_fields(solver.cell_tags,q,ids[0])
        p=solver.build_problem(a,n,1.,0.)
        for pair in ids.reshape(-1,2):
            k=q['k'][pair[0]]
            assert q['k'][pair[1]]==k
            mat=(p.K-k*k*p.Mn-1j*k*p.B).tocsc()
            rhs=np.stack([solver.rhs(a,n,k,t) for t in q['theta'][pair]],axis=1)
            out.extend(transfer.apply(splu(mat).solve(rhs).T))
    _save_npz(d/'mesh_check.npz',rows=rows,values=np.array(out))

if __name__=='__main__':
    solve()
