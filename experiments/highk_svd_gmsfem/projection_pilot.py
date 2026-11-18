"""Verify real/imaginary split projection against the saved original complex matrix."""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from scipy import sparse
from .gms_highorder import P2Backend,build_offline_space
from .reduced import project_matrix,apply_real
from experiments.reference_sno_gmsfem.dataset import _write_json


def main():
    root=Path(__file__).resolve().parent
    reference=sparse.load_npz(root/'results/ordering_pilot/matrix.npz')
    reference_rhs=np.load(root/'results/ordering_pilot/rhs.npy')
    backend=P2Backend(root/'meshes/graded_geometry256.msh');problem=backend.problem(18.12345,0.)
    space=build_offline_space(problem,n_coarse=20,max_modes=32,basis_type=2);P=space.prolongation(32)
    started=perf_counter();observed=project_matrix(problem.operator,P);seconds=perf_counter()-started
    defect=float(sparse.linalg.norm(observed-reference)/sparse.linalg.norm(reference))
    rhs=apply_real(P.T,problem.rhs);rhs_defect=float(np.linalg.norm(rhs-reference_rhs)/np.linalg.norm(reference_rhs))
    rng=np.random.default_rng(3);c=rng.normal(size=P.shape[1])+1j*rng.normal(size=P.shape[1])
    full=P@c;split=apply_real(P,c);reconstruction_defect=float(np.linalg.norm(split-full)/np.linalg.norm(full))
    result=dict(passed=max(defect,rhs_defect,reconstruction_defect)<1e-12,relative_matrix_difference=defect,
        rhs_relative_difference=rhs_defect,reconstruction_relative_difference=reconstruction_defect,
        split_projection_seconds=seconds,original_projection_seconds=json.loads((root/'results/ordering_pilot/matrix_metadata.json').read_text())['matrix_projection_seconds'])
    _write_json(root/'results/gms_projection_pilot.json',result)
    if not result['passed']:raise RuntimeError(result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
