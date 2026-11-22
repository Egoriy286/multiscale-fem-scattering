"""Additional positive-k extrapolation set isolated from all training and validation."""
import numpy as np
from scipy.sparse.linalg import splu
from generate import HERE,material_fields,_save_npz
from neural_operator.fem import HelmholtzSolver
from experiments.reference_sno_gmsfem.dataset import _contrast_rhs

class Solver(HelmholtzSolver):
    rhs=_contrast_rhs

def main():
    d=HERE/'data'
    original=np.load(d/'queries.npz')
    first=np.flatnonzero(original['split']==2)[0]
    q={key:np.repeat(original[key][first],6) for key in ('alpha1','alpha2','n1','n2','medium','split')}
    q.update(k=np.repeat([.001,.01,.025],2),theta=np.tile(original['theta'][first:first+2],3))
    solver=Solver(d/'etal.msh')
    a,n=material_fields(solver.cell_tags,q,0)
    basis=solver.build_problem(a,n,1.,0.)
    solutions=[];rel=[]
    for k in q['k'][::2]:
        A=(basis.K-k*k*basis.Mn-1j*k*basis.B).tocsc()
        F=np.stack([solver.rhs(a,n,k,t) for t in q['theta'][:2]],axis=1)
        v=splu(A).solve(F)
        rel.extend(np.linalg.norm(A@v-F,axis=0)/np.linalg.norm(F,axis=0))
        solutions.extend(v.T)
    assert np.isfinite(solutions).all() and max(rel)<1e-8
    _save_npz(d/'nearzero.npz',**q,fields=np.array(solutions),residual=rel)
if __name__=='__main__':main()
