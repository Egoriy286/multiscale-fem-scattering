"""Bounded k=20 full-snapshot GMsFEM timing/accuracy pilot (not test tuning)."""
from pathlib import Path
from time import perf_counter
import gc
import json
import resource
import numpy as np
from experiments.reference_sno_gmsfem.baselines import _materials, _ReducedSystem
from experiments.reference_sno_gmsfem.dataset import _contrast_rhs, _sha256, _write_json
from neural_operator.fem import HelmholtzSolver
from gmsfem.offline import build_offline_space


def main():
    root=Path(__file__).resolve().parent
    out=root/'results/baseline_timing_pilot.json'
    mesh=root/'meshes/graded_1.msh'
    wall_start=perf_counter()
    started=perf_counter(); solver=HelmholtzSolver(mesh)
    setup=perf_counter()-started
    solver.rhs=lambda a,n,k,theta:_contrast_rhs(solver,a,n,k,theta)
    a,n=_materials(solver.cell_tags)
    started=perf_counter(); problem=solver.build_problem(a,n,20.,0.)
    assembly=perf_counter()-started
    result=dict(k=20.,theta=0.,n_coarse=10,max_modes=32,dofs_fine=problem.ndofs,
                mesh_sha256=_sha256(mesh),setup_seconds=setup,assembly_seconds=assembly,
                complete=False,methods=[])
    _write_json(out,result)
    started=perf_counter();space=build_offline_space(problem,n_coarse=10,max_modes=32,basis_type=2)
    result.update(offline_seconds=perf_counter()-started,diagnostics=space.diagnostics,
                  eigenvalue_min=float(space.eigenvalues.min()),eigenvalue_max=float(space.eigenvalues.max()),
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    _write_json(out,result)
    refpath=root/'results/highorder/level0_cg3_on_p1level1.npy'
    true=np.load(refpath,mmap_mode='r')[-1] if refpath.exists() else None
    nativepath=root/'results/mesh_convergence/level_1_fields.npz'
    with np.load(nativepath) as z:
        if not np.array_equal(z['coordinates'],problem.coordinates) or not np.array_equal(z['triangles'],problem.triangles):
            raise ValueError('Native archive ordering differs')
        j=int(np.flatnonzero(z['frequencies']==20.)[0]);native=z['fields'][j]
    def err(pred,ref):
        def norm(x): return np.real(np.vdot(x,problem.M0@x))
        return float(np.sqrt(max(norm(pred-ref),0)/norm(ref)))
    if true is not None:
        result['reference']=dict(path=str(refpath.relative_to(root)),sha256=_sha256(refpath),
             kind='CG3 level0 sampled on graded1 P1 nodes; scored with consistent P1 mass',
             native_p1_error_percent=100*err(native,true))
    for m in (16,32):
        started=perf_counter();system=_ReducedSystem(problem,space,m);construction=perf_counter()-started
        prediction,timing=system.solve(problem)
        row=dict(modes=m,system_seconds=construction,**timing,error_vs_native_p1_percent=100*err(prediction,native))
        if true is not None: row['error_vs_projected_cg3_percent']=100*err(prediction,true)
        result['methods'].append(row); _write_json(out,result)
        del system,prediction;gc.collect()
    result.update(complete=True,component_seconds_total=setup+assembly+result['offline_seconds']+sum(r['system_seconds']+r['solve_seconds']+r['reconstruction_seconds'] for r in result['methods']),
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,wall_seconds=perf_counter()-wall_start)
    _write_json(out,result); print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
