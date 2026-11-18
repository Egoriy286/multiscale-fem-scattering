"""Compare sparse orderings for the exact same stored reduced matrix."""
from pathlib import Path
from time import perf_counter
import gc,json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .reduced import nested_dissection
from .gms_highorder import P2Backend,build_offline_space
from experiments.reference_sno_gmsfem.dataset import _write_json,_sha256


def main():
    root=Path(__file__).resolve().parent;out=root/'results/ordering_pilot';out.mkdir(exist_ok=True)
    mesh=root/'meshes/graded_geometry256.msh';apath=out/'matrix.npz';bpath=out/'rhs.npy'
    metadata_path=out/'matrix_metadata.json'
    if not apath.exists():
        backend=P2Backend(mesh);problem=backend.problem(18.12345,0.)
        space=build_offline_space(problem,n_coarse=20,max_modes=32,basis_type=2)
        P=space.prolongation(32);t=perf_counter();A=(P.T@(problem.operator@P)).tocsc();projection=perf_counter()-t
        rhs=P.T@problem.rhs;sparse.save_npz(apath,A);np.save(bpath,rhs)
        _write_json(metadata_path,dict(mesh_sha256=_sha256(mesh),k=18.12345,n_coarse=20,modes=32,
            matrix_projection_seconds=projection,offline_seconds=space.build_seconds,matrix_nnz=A.nnz,dofs=A.shape[0]))
        del backend,problem,space,P;gc.collect()
    else:A=sparse.load_npz(apath);rhs=np.load(bpath)
    result=dict(matrix_sha256=_sha256(apath),rhs_sha256=_sha256(bpath),metadata=json.loads(metadata_path.read_text()),rows=[],complete=False)
    output=root/'results/gms_ordering_pilot.json';reference=None
    for ordering in ('MMD_AT_PLUS_A','ND','COLAMD'):
        perm=nested_dissection(20,32) if ordering=='ND' else np.arange(A.shape[0])
        matrix=A[perm][:,perm].tocsc();t=perf_counter()
        factor=splu(matrix,permc_spec='NATURAL' if ordering=='ND' else ordering)
        seconds=perf_counter()-t;solution=np.empty_like(rhs);solution[perm]=factor.solve(rhs[perm])
        if reference is None:reference=solution.copy()
        result['rows'].append(dict(ordering=ordering,factor_seconds=seconds,factor_nnz=int(factor.L.nnz+factor.U.nnz),
            residual=float(np.linalg.norm(A@solution-rhs)/np.linalg.norm(rhs)),
            relative_solution_difference=float(np.linalg.norm(solution-reference)/np.linalg.norm(reference))))
        _write_json(output,result);print(result['rows'][-1],flush=True);del factor,matrix;gc.collect()
    result['complete']=True;_write_json(output,result)
if __name__=='__main__':main()
