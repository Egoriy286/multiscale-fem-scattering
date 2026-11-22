"""Paired Type-I GMsFEM on the twelve blind media, two incident angles, five frequencies."""
import csv
import numpy as np
from generate import HERE,material_fields
from check_mesh import HelmholtzSolver,P1Transfer,FINE_MESH,FINE_SHA256,_checked_mesh,mass_matrix
from gmsfem.offline import build_offline_space
from experiments.reference_sno_gmsfem.baselines import _ReducedSystem

def main():
    q=np.load(HERE/'data/queries.npz')
    ref=np.load(HERE/'data/reference_problem.npz')
    fields=np.load(HERE/'data/fields.npy',mmap_mode='r')
    src=HelmholtzSolver(_checked_mesh(FINE_MESH,FINE_SHA256))
    transfer=P1Transfer(src.coordinates,src.triangles,ref['coordinates'])
    mass=mass_matrix(ref['coordinates'],ref['triangles'])
    with (HERE/'gms_cases.csv').open('w') as file:
        out=csv.writer(file);out.writerow(('row','modes','dofs','error_pct','residual'))
        for medium in np.unique(q['medium'][q['split']==2]):
            ids=np.flatnonzero(q['medium']==medium)
            a,n=material_fields(src.cell_tags,q,ids[0])
            space=None
            for pair in ids.reshape(-1,2):
                k=q['k'][pair[0]]
                p=src.build_problem(a,n,k,q['theta'][pair[0]])
                if space is None:
                    space=build_offline_space(p,n_coarse=20,max_modes=8,basis_type=1,progress=None)
                rhs=np.column_stack([p.rhs,src.rhs(a,n,k,q['theta'][pair[1]])])
                for modes in (4,8):
                    system=_ReducedSystem(p,space,modes)
                    b=system.PH@rhs
                    u=system.P@system.factor.solve(b)
                    res=np.linalg.norm(system.PH@(p.operator@u-rhs),axis=0)/np.linalg.norm(b,axis=0)
                    assert max(res)<1e-7
                    for row,v,r in zip(pair,transfer.apply(u.T),res):
                        z=v-fields[row];t=fields[row]
                        e=100*np.sqrt(np.vdot(z,mass@z).real/np.vdot(t,mass@t).real)
                        out.writerow((row,modes,system.P.shape[1],e,r))
            file.flush()
            print(medium,flush=True)
if __name__=='__main__': main()
