"""Actual native CG2 GMsFEM fine-grid refinement against CG3 at common quadrature.

This audit does not replace the paired primary metric on the observation support.
All native fields are evaluated directly; no interpolation through support nodes.
"""
from pathlib import Path
from time import perf_counter
import argparse,gc,json
import numpy as np
from scipy.sparse.linalg import splu
from scipy import sparse
from gmsfem.metrics import BARYCENTRIC,WEIGHTS
from .gms_highorder import P2Backend,build_offline_space
from .highorder import HighOrderSolver
from .reduced import ReducedSystem as _ReducedSystem
from experiments.reference_sno_gmsfem.dataset import _sha256,_write_json


def audit(meshes,reference_mesh,quadrature_archive,output,*,frequencies=(10.,20.),modes=(16,32),n_coarse=20):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parent;repo=here.parents[1]
    signature=dict(mesh_sha256=[_sha256(p) for p in meshes],reference_sha256=_sha256(reference_mesh),
        quadrature_archive_sha256=_sha256(quadrature_archive),frequencies=list(frequencies),modes=list(modes),n_coarse=n_coarse,
        source_sha256={str(p.relative_to(repo)):_sha256(p) for p in
            [Path(__file__),here/'gms_highorder.py',here/'reduced.py',here/'highorder.py',repo/'gmsfem/offline.py']})
    if output.exists():
        previous=json.loads(output.read_text())
        if previous.get('complete') and previous.get('signature')==signature:return previous
    with np.load(quadrature_archive) as z:
        coordinates=z['coordinates'];triangles=z['triangles']
    xy=coordinates[triangles];edges=xy[:,1:]-xy[:,:1]
    areas=abs(edges[:,0,0]*edges[:,1,1]-edges[:,0,1]*edges[:,1,0])/2
    weights=(areas[:,None]*WEIGHTS).ravel()
    points=np.einsum('qj,cjd->cqd',BARYCENTRIC,xy).reshape(-1,2)
    def evaluate(backend,field):
        result=np.empty(len(points),complex)
        for i in range(0,len(points),100000):result[i:i+100000]=backend.evaluate(field,points[i:i+100000])
        return result
    truth=HighOrderSolver(reference_mesh,degree=3)
    refs={};truth_info=[]
    for k in frequencies:
        field,timing=truth.solve(k,0.);refs[k]=evaluate(truth,field)
        truth_info.append(dict(k=k,**timing))
    del truth,field;gc.collect()
    result=dict(signature=signature,meshes=[str(p) for p in meshes],mesh_sha256=[_sha256(p) for p in meshes],
                reference_mesh=str(reference_mesh),reference_sha256=_sha256(reference_mesh),
                reference_degree=3,truth_solves=truth_info,n_coarse=n_coarse,modes=list(modes),frequencies=list(frequencies),
                quadrature_archive_sha256=_sha256(quadrature_archive),quadrature_points=len(points),
                metric='Direct native CG2 and CG3 evaluation at common degree-five quadrature; individual complex L2 ratio',rows=[],differences=[],complete=False)
    _write_json(output,result)
    cache={}
    for level,mesh in enumerate(meshes):
        backend=P2Backend(mesh)
        for k in frequencies:
            started=perf_counter();problem=backend.problem(k,0.)
            factor=splu(problem.operator.tocsc());native=factor.solve(problem.rhs);del factor;gc.collect()
            native_residual=float(np.linalg.norm(problem.operator@native-problem.rhs)/max(np.linalg.norm(problem.rhs),1e-30))
            if not np.isfinite(native_residual) or native_residual>1e-10:
                raise RuntimeError(f'Native CG2 refinement residual {native_residual}')
            fields={'native_cg2':evaluate(backend,native)}
            native_seconds=perf_counter()-started
            space=build_offline_space(problem,n_coarse=n_coarse,max_modes=max(modes),basis_type=2) if modes else None
            timings={}
            for m in modes:
                system=_ReducedSystem(problem,space,m);field,timing=system.solve(problem)
                fields[f'gms_M{m}']=evaluate(backend,field);timings[f'gms_M{m}']=timing
                if level==0 and k==frequencies[0] and m==modes[0]:
                    original=(system.P.T@(problem.operator@system.P)).tocsc()
                    defect=float(sparse.linalg.norm(system.matrix-original)/max(sparse.linalg.norm(original),1e-30))
                    result['projection_validation']=dict(relative_matrix_difference=defect,passed=defect<1e-12,
                        original_equation_galerkin_residual=timing['galerkin_residual'])
                    if defect>=1e-12:raise RuntimeError('Real/imaginary split projection differs from original matrix')
                    del original
                del system,field;gc.collect()
            denominator=float(np.sum(weights*abs(refs[k])**2))
            for method,values in fields.items():
                error=float(np.sqrt(np.sum(weights*abs(values-refs[k])**2)/denominator))
                result['rows'].append(dict(level=level,mesh=str(mesh),k=k,method=method,fine_dofs=problem.ndofs,
                    reduced_dofs=problem.ndofs if method=='native_cg2' else (n_coarse+1)**2*int(method[5:]),
                    error_percent=100*error,offline_seconds=space.build_seconds if space else 0.,native_solve_evaluation_seconds=native_seconds,
                    diagnostics=space.diagnostics if space else {},native_linear_residual=native_residual,**timings.get(method,{})))
                if level:
                    previous=cache[k,method]
                    result['differences'].append(dict(coarse_level=level-1,fine_level=level,k=k,method=method,
                        difference_relative_l2=float(np.sqrt(np.sum(weights*abs(values-previous)**2)/denominator))))
                cache[k,method]=values
            _write_json(output,result);del space,problem,native;gc.collect()
        del backend;gc.collect()
    result['complete']=True;_write_json(output,result)
    return result

def setup(root=None):
    """Recreate fixed CG2 meshes, audit full native/reduced fields, freeze selection.

    The pass criterion measures native fine-discretization adequacy, never the
    size of a reduced-space error. Final main modes remain the declared 16/32.
    """
    from .mesh import generate_graded_mesh
    root=Path(root or Path(__file__).resolve().parent).resolve()
    reference_manifest=root/'results/highorder/selected_reference.json'
    reference=json.loads(reference_manifest.read_text())
    if not reference.get('passed'):raise ValueError('Selected native CG3 reference checks have not passed')
    for name,digest in reference['verified_files'].items():
        if _sha256(root/name)!=digest:raise ValueError(f'Frozen reference artifact differs: {name}')
    base=generate_graded_mesh(root/'meshes/graded_geometry256.msh',h_background=.04,h_inclusion=.005,n_coarse=20,circle_points=256)
    refined=generate_graded_mesh(root/'meshes/gms_geometry256_h02.msh',h_background=.02,h_inclusion=.005,n_coarse=20,circle_points=256)
    audit_path=root/'results/gms_refinement.json'
    measured=audit([base,refined],root/reference['native_mesh'],root/reference['support_archive'],audit_path,
                   frequencies=(10.,20.),modes=(16,32),n_coarse=20)
    native=[r for r in measured['rows'] if r['method']=='native_cg2']
    primary=[r['error_percent'] for r in native if r['level']==0]
    successive=[100*r['difference_relative_l2'] for r in measured['differences'] if r['method']=='native_cg2']
    adequacy=max(primary)<1. and max(successive)<1.
    algebra=all(np.isfinite(r['error_percent']) and r['native_linear_residual']<1e-10 and
                (r['method']=='native_cg2' or r['galerkin_residual']<1e-7) for r in measured['rows'])
    configuration=dict(basis_type=2,n_coarse=20,modes=[16,32],fine_degree=2,
        fine_mesh=str(base.relative_to(root)),fine_mesh_sha256=_sha256(base),rebuild_each_frequency=True,sparse_ordering='COLAMD')
    selection=dict(schema_version=1,passed=bool(adequacy and algebra and measured['projection_validation']['passed']),
        configuration=configuration,criterion=dict(native_cg2_vs_cg3_max_percent=1.,native_successive_difference_max_percent=1.,
            scope='Fine-discretization adequacy only; no reduced-accuracy selection threshold'),
        measured=dict(base_native_max_error_percent=max(primary),native_successive_max_difference_percent=max(successive)),
        audit_refs={str(audit_path.relative_to(root)):_sha256(audit_path),str(reference_manifest.relative_to(root)):_sha256(reference_manifest)},
        limitation='Background-only CG2 refinement .04→.02, fixed inclusion size .005 and geometry256; independent native CG3 spatial, degree, geometry and representation checks remain separate. The declared mode counts are finite and can yield large reduction errors; this does not rank the entire GMsFEM family.',
        pilot_exposure='The historical n10 P1 pilot exposed k20 before final protocol; CG2 cost/order pilots used k18.12345. Main M16/32 were retained; the M64 trial was stopped for resource limits before observing field accuracy.')
    for name in ('gms_native_refinement.json','gms_ordering_pilot.json','gms_projection_pilot.json','gms_p2_pilot_n20_m64.json','baseline_timing_pilot.json'):
        path=root/'results'/name
        if path.exists():selection['audit_refs'][str(path.relative_to(root))]=_sha256(path)
    _write_json(root/'results/gms_mesh_selection.json',selection)
    if not selection['passed']:raise RuntimeError('CG2 fine-discretization audit failed; selection was written with passed=false')
    print(json.dumps(selection,indent=2),flush=True)
    return selection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,help='Run fixed two-mesh setup/audit and write gms_mesh_selection.json')
    p.add_argument('--meshes',nargs='+');p.add_argument('--reference-mesh')
    p.add_argument('--quadrature-archive');p.add_argument('--output')
    p.add_argument('--frequencies',type=float,nargs='+',default=[10.,20.]);p.add_argument('--modes',type=int,nargs='+',default=[16,32])
    p.add_argument('--native-only',action='store_true')
    a=p.parse_args()
    if a.root is not None or a.meshes is None:return setup(a.root)
    if not all((a.reference_mesh,a.quadrature_archive,a.output)):
        p.error('Explicit mesh audit requires --reference-mesh, --quadrature-archive, and --output')
    return audit(a.meshes,a.reference_mesh,a.quadrature_archive,a.output,
                 frequencies=a.frequencies,modes=() if a.native_only else a.modes)


if __name__=='__main__':main()
