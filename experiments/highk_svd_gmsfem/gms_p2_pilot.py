"""Native CG2 GMsFEM cost audit at a non-dataset pilot frequency."""
import argparse,gc,json,resource
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import splu
from .gms_highorder import P2Backend,build_offline_space
from .reduced import ReducedSystem as _ReducedSystem
from experiments.reference_sno_gmsfem.dataset import _sha256,_write_json


def pilot(mesh,output,*,k=18.12345,modes=(16,32),n_coarse=20):
    start=perf_counter();mesh=Path(mesh);output=Path(output);backend=P2Backend(mesh)
    problem=backend.problem(k,0.)
    result=dict(mesh=str(mesh),mesh_sha256=_sha256(mesh),k=k,theta=0.,n_coarse=n_coarse,
                fine_degree=2,dofs_fine=problem.ndofs,provenance=backend.provenance,complete=False,methods=[])
    _write_json(output,result)
    space=build_offline_space(problem,n_coarse=n_coarse,max_modes=max(modes),basis_type=2)
    result.update(offline_seconds=space.build_seconds,diagnostics=space.diagnostics,
                  eigenvalue_min=float(space.eigenvalues.min()),eigenvalue_max=float(space.eigenvalues.max()))
    _write_json(output,result)
    t=perf_counter();factor=splu(problem.operator.tocsc());native=factor.solve(problem.rhs)
    result['native_fine_seconds']=perf_counter()-t
    result['native_fine_residual']=float(np.linalg.norm(problem.operator@native-problem.rhs)/np.linalg.norm(problem.rhs))
    del factor;gc.collect()
    den=np.real(np.vdot(native,problem.M0@native))
    for m in modes:
        t=perf_counter();system=_ReducedSystem(problem,space,m);setup=perf_counter()-t
        field,timing=system.solve(problem);diff=field-native
        error=100*float(np.sqrt(max(np.real(np.vdot(diff,problem.M0@diff)),0)/den))
        result['methods'].append(dict(modes=m,system_setup_seconds=setup,error_vs_native_cg2_percent=error,**timing))
        del system,field;gc.collect();_write_json(output,result)
    result.update(complete=True,wall_seconds=perf_counter()-start,
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                  scope='One non-dataset pilot query; accuracy against native CG2 only, not against converged CG3 truth')
    _write_json(output,result);print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh',required=True);p.add_argument('--output',required=True)
    p.add_argument('--modes',type=int,nargs='+',default=[16,32]);p.add_argument('--k',type=float,default=18.12345)
    a=p.parse_args();pilot(a.mesh,a.output,k=a.k,modes=a.modes)
