"""Independent native CG2/CG3 truth and measured P1 observation projection.

Affine geometry retains the graded mesh's polygonal circles. Native functions
are evaluated by DOLFINx, preserving high-order element maps and orientation.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import splu,gmres,LinearOperator
from .mesh import HERE,FREQUENCIES


class HighOrderSolver:
    def __init__(self,mesh_path,degree=3,*,iterative=False):
        from mpi4py import MPI
        from petsc4py import PETSc
        from dolfinx import fem
        from dolfinx.fem import petsc
        from dolfinx.io import gmsh as gmshio
        from matplotlib.tri import Triangulation
        from gmsfem.fem import MATERIALS,_petsc_to_csr
        import ufl
        self.mesh_path=Path(mesh_path);self.degree=int(degree);self.iterative=bool(iterative)
        self.data=gmshio.read_from_msh(str(mesh_path),MPI.COMM_SELF,gdim=2)
        domain=self.data.mesh;self.V=fem.functionspace(domain,('Lagrange',self.degree))
        low=fem.functionspace(domain,('Lagrange',1));ncells=domain.topology.index_map(2).size_local
        self.coordinates=low.tabulate_dof_coordinates()[:,:2].copy()
        self.triangles=np.array([low.dofmap.cell_dofs(c) for c in range(ncells)],dtype=np.int32)
        tri=self.triangles.copy();xy=self.coordinates[tri];e=xy[:,1:]-xy[:,:1]
        neg=e[:,0,0]*e[:,1,1]-e[:,0,1]*e[:,1,0]<0;tri[neg]=tri[neg][:,[0,2,1]]
        self.locator=Triangulation(self.coordinates[:,0],self.coordinates[:,1],tri).get_trifinder()
        self.function=fem.Function(self.V)
        W=fem.functionspace(domain,('DG',0));a=fem.Function(W);n=fem.Function(W)
        tags=np.zeros(ncells,dtype=int);tags[self.data.cell_tags.indices]=self.data.cell_tags.values
        for c,tag in enumerate(tags):
            a.x.array[W.dofmap.cell_dofs(c)]=MATERIALS['notebook'][tag][0]
            n.x.array[W.dofmap.cell_dofs(c)]=MATERIALS['notebook'][tag][1]
        u,v=ufl.TrialFunction(self.V),ufl.TestFunction(self.V)
        dx=ufl.Measure('dx',domain=domain,metadata={'quadrature_degree':max(12,2*self.degree+4)})
        ds=ufl.Measure('ds',domain=domain,subdomain_data=self.data.facet_tags)
        forms=[ufl.inner(a*ufl.grad(u),ufl.grad(v))*dx,ufl.inner(n*u,v)*dx,
               sum(ufl.inner(u,v)*ds(tag) for tag in (11,12,13,14))]
        matrices=[]
        for form in forms:
            matrix=petsc.assemble_matrix(fem.form(form));matrix.assemble()
            matrices.append(_petsc_to_csr(matrix));matrix.destroy()
        self.K,self.Mn,self.B=matrices
        self.k=fem.Constant(domain,PETSc.ScalarType(1.));self.d=fem.Constant(domain,np.array([1.,0.],dtype=PETSc.ScalarType))
        x=ufl.SpatialCoordinate(domain);ui=ufl.exp(1j*self.k*(self.d[0]*x[0]+self.d[1]*x[1]))
        self.load=fem.form((ufl.inner((1-a)*ufl.grad(ui),ufl.grad(v))+self.k**2*ufl.inner((n-1)*ui,v))*dx)
        self._key=None

    def solve(self,k,theta=0.):
        from dolfinx.fem import petsc
        from petsc4py import PETSc
        self.k.value=k;self.d.value=np.array([np.cos(theta),np.sin(theta)],dtype=PETSc.ScalarType)
        t=perf_counter()
        self.operator=(self.K-k*k*self.Mn-1j*k*self.B).tocsc()
        vector=petsc.assemble_vector(self.load)
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)
        rhs=vector.array.copy();vector.destroy()
        iterations=0;used_iterative=False;anchor=self._key;gmres_info=None
        if self.iterative and self._key is not None and abs(k-self._key)<=.5:
            preconditioner=LinearOperator(self.operator.shape,matvec=self.factor.solve,dtype=np.complex128)
            counts=[]
            u,info=gmres(self.operator,rhs,M=preconditioner,x0=getattr(self,'_previous_solution',None),rtol=2e-12,atol=0.,
                         restart=50,maxiter=1,callback=lambda value:counts.append(float(value)),callback_type='pr_norm')
            iterations=len(counts);gmres_info=int(info)
            residual=float(np.linalg.norm(self.operator@u-rhs)/np.linalg.norm(rhs))
            # GMRES info concerns its stricter internal stopping rule. Acceptance
            # is defined by the finite solution and original, unpreconditioned PDE.
            used_iterative=bool(np.isfinite(u).all() and np.isfinite(residual) and residual<1e-10)
        if not used_iterative:
            # Release a stale factor before allocating its replacement.
            if 'preconditioner' in locals():del preconditioner
            self.factor=None
            self.factor=splu(self.operator);self._key=float(k)
            u=self.factor.solve(rhs)
            residual=float(np.linalg.norm(self.operator@u-rhs)/np.linalg.norm(rhs))
        if not np.isfinite(u).all() or residual>=1e-10:
            raise RuntimeError(f'Native high-order solve failed explicit residual {residual:g}')
        self._previous_solution=u.copy()
        return u,dict(seconds=perf_counter()-t,residual=residual,dofs=len(u),
                      reused_lu=used_iterative,gmres_iterations=iterations,gmres_info=gmres_info,
                      original_equation_residual=residual,
                      anchor_k=anchor if used_iterative else float(k))

    def evaluate(self,field,points):
        points=np.asarray(points)
        # Locate on an inward-rounded square boundary; evaluate the original points.
        query=np.clip(points,-2.+1e-12,2.-1e-12)
        cells=self.locator(query[:,0],query[:,1]).astype(np.int32)
        if np.any(cells<0):raise ValueError('High-order evaluation point outside native mesh')
        self.function.x.array[:]=field
        points3=np.column_stack((points,np.zeros(len(points))))
        return self.function.eval(points3,cells).reshape(-1)


def solve_highorder(mesh,degree,archive,frequencies=FREQUENCIES,theta=0.):
    """Solve native CGp, store coefficients plus exact source mesh hash."""
    archive=Path(archive)
    signature=dict(mesh_sha256=hashlib.sha256(Path(mesh).read_bytes()).hexdigest(),degree=degree,
                   frequencies=list(frequencies),theta=theta)
    if archive.exists():
        with np.load(archive,allow_pickle=False) as z:
            if json.loads(str(z['signature']))==signature:return archive
    t=perf_counter();solver=HighOrderSolver(mesh,degree);fields=[];stats=[]
    for k in frequencies:
        field,stat=solver.solve(k,theta);fields.append(field);stats.append(stat)
        print(f'CG{degree} {Path(mesh).name} k={k:g}: {stat}',flush=True)
    archive.parent.mkdir(parents=True,exist_ok=True)
    np.savez(archive,fields=fields,frequencies=frequencies,theta=theta,signature=json.dumps(signature),
             stats=json.dumps(stats),total_seconds=perf_counter()-t)
    del solver;gc.collect();return archive


def compare_native_to_p1(mesh,degree,archive,target_archive,*,chunk_cells=20000):
    """Compare high-order truth to existing P1 solves and its P1 node sampling.

    All three fields use the same finer mesh quadrature. Projection error tests
    loss caused by storing native truth as nodal values on the observation mesh.
    """
    from gmsfem.metrics import BARYCENTRIC,WEIGHTS
    with np.load(archive,allow_pickle=False) as z:hi={k:z[k] for k in z.files}
    with np.load(target_archive,allow_pickle=False) as z:lo={k:z[k] for k in z.files}
    solver=HighOrderSolver(mesh,degree);n=len(hi['frequencies'])
    projected=np.stack([solver.evaluate(u,lo['coordinates']) for u in hi['fields']])
    numerator=np.zeros((2,n));denominator=np.zeros(n)
    for start in range(0,len(lo['triangles']),chunk_cells):
        tri=lo['triangles'][start:start+chunk_cells];xy=lo['coordinates'][tri];e=xy[:,1:]-xy[:,:1]
        areas=np.abs(e[:,0,0]*e[:,1,1]-e[:,0,1]*e[:,1,0])/2
        points=np.einsum('qj,cjd->cqd',BARYCENTRIC,xy);w=areas[:,None]*WEIGHTS
        native=np.stack([solver.evaluate(u,points.reshape(-1,2)).reshape(len(tri),-1) for u in hi['fields']])
        p1=np.einsum('fcj,qj->fcq',lo['fields'][:,tri],BARYCENTRIC)
        sampled=np.einsum('fcj,qj->fcq',projected[:,tri],BARYCENTRIC)
        numerator[0]+=np.sum(abs(p1-native)**2*w,axis=(1,2))
        numerator[1]+=np.sum(abs(sampled-native)**2*w,axis=(1,2))
        denominator+=np.sum(abs(native)**2*w,axis=(1,2))
    errors=np.sqrt(numerator/denominator)
    rows=[dict(k=float(k),degree=degree,native_dofs=hi['fields'].shape[1],
               p1_nodes=len(lo['coordinates']),p1_fem_relative_l2=float(errors[0,j]),
               p1_representation_relative_l2=float(errors[1,j])) for j,k in enumerate(hi['frequencies'])]
    del solver;gc.collect();return rows,projected


def compare_highorder(mesh_a,degree_a,archive_a,mesh_b,degree_b,archive_b,quadrature_archive,*,chunk_cells=20000):
    """Compare two native high-order functions directly on common quadrature."""
    from gmsfem.metrics import BARYCENTRIC,WEIGHTS
    with np.load(archive_a,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(archive_b,allow_pickle=False) as z:b={k:z[k] for k in z.files}
    with np.load(quadrature_archive,allow_pickle=False) as z:q={k:z[k] for k in ('coordinates','triangles')}
    if not np.array_equal(a['frequencies'],b['frequencies']):raise ValueError('Different physical frequencies')
    sa,sb=HighOrderSolver(mesh_a,degree_a),HighOrderSolver(mesh_b,degree_b)
    num=np.zeros(len(a['frequencies']));den=np.zeros_like(num)
    for start in range(0,len(q['triangles']),chunk_cells):
        tri=q['triangles'][start:start+chunk_cells];xy=q['coordinates'][tri];e=xy[:,1:]-xy[:,:1]
        areas=np.abs(e[:,0,0]*e[:,1,1]-e[:,0,1]*e[:,1,0])/2
        points=np.einsum('qj,cjd->cqd',BARYCENTRIC,xy).reshape(-1,2);w=areas[:,None]*WEIGHTS
        for j,(ua,ub) in enumerate(zip(a['fields'],b['fields'])):
            aq=sa.evaluate(ua,points).reshape(len(tri),-1);bq=sb.evaluate(ub,points).reshape(len(tri),-1)
            num[j]+=np.sum(abs(aq-bq)**2*w);den[j]+=np.sum(abs(bq)**2*w)
    return [dict(k=float(k),relative_scattered_l2=float(np.sqrt(x/y)),degree_a=degree_a,degree_b=degree_b)
            for k,x,y in zip(a['frequencies'],num,den)]


class SupportGeometry:
    """Only the P1 ordering/material geometry needed by dataset generation."""
    def __init__(self,archive):
        with np.load(archive,allow_pickle=False) as z:
            for key in ('coordinates','triangles','cell_tags'):setattr(self,key,z[key])
    def nodal_coefficients(self,values):
        return np.bincount(self.triangles.ravel(),weights=np.repeat(values,3),minlength=len(self.coordinates))/np.bincount(self.triangles.ravel(),minlength=len(self.coordinates))


class ReferenceOnSupport:
    def __init__(self,mesh,degree,support):
        self.native=HighOrderSolver(mesh,degree,iterative=True);self.support=support
    def solve_on_support(self,k,theta=0.):
        u,stats=self.native.solve(k,theta)
        return dict(field=self.native.evaluate(u,self.support.coordinates),
                    relative_residual=stats['residual'],native_solve=stats)


def selected_reference(root=HERE):
    """Load a frozen, passed native-refinement and representation selection."""
    root=Path(root).resolve();manifest=root/'results/highorder/selected_reference.json'
    selection=json.loads(manifest.read_text())
    if not selection.get('passed'):raise ValueError('High-order reference audits have not passed')
    for relative,digest in selection['verified_files'].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Changed selected reference source: {relative}')
    support=SupportGeometry(root/selection['support_archive'])
    solver=ReferenceOnSupport(root/selection['native_mesh'],selection['native_degree'],support)
    return dict(solver=solver,support_solver=support,support_mesh=root/selection['support_mesh'],metadata=selection)


def pilot(root=HERE,frequencies=FREQUENCIES):
    root=Path(root);out=root/'results/highorder';out.mkdir(parents=True,exist_ok=True);results=[]
    for level,degree in [(0,2),(0,3),(1,2)]:
        mesh=root/f'meshes/graded_{level}.msh';archive=out/f'level{level}_cg{degree}.npz'
        solve_highorder(mesh,degree,archive,frequencies)
        for target in [1]:
            rows,projected=compare_native_to_p1(mesh,degree,archive,root/f'results/mesh_convergence/level_{target}_fields.npz')
            for row in rows:row.update(native_mesh_level=level,p1_mesh_level=target)
            results.extend(rows)
            np.save(out/f'level{level}_cg{degree}_on_p1level{target}.npy',projected)
            (out/'pilot.json').write_text(json.dumps(results,indent=2)+'\n')
            print('HIGHORDER',json.dumps(rows),flush=True)
    return results


def validate_iterative_solver(mesh,output):
    """Replay warm-start LU reuse against direct solves on the selected mesh."""
    solver=HighOrderSolver(mesh,3,iterative=True);rows=[]
    for anchor,k in [(.05,.06),(.06,.1),(8.,8.05),(19.95,20.)]:
        _,a=solver.solve(anchor);u,it=solver.solve(k)
        solver.iterative=False;direct,d=solver.solve(k);solver.iterative=True
        difference=float(np.linalg.norm(u-direct)/np.linalg.norm(direct))
        rows.append(dict(anchor_k=anchor,k=k,anchor=a,iterative=it,direct=d,
                         relative_coefficient_difference=difference))
        print('Iterative validation:',json.dumps(rows[-1]),flush=True)
    _,low=solver.solve(.05)
    result=dict(passed=all(r['relative_coefficient_difference']<1e-7 and r['iterative']['residual']<1e-10 and r['iterative']['reused_lu'] for r in rows),
                native_mesh=str(Path(mesh).name),warm_start=True,rows=rows,low_k_check=low)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    del solver;gc.collect();return result


def run_reference_study(root=HERE,*,force_audits=False,tolerance=.02):
    """Fully regenerate the mesh, degree, spatial and representation audits.

    Cached exact mesh/solution recipes are resumable. --force-audits recomputes
    the common-quadrature comparisons and independent iterative/direct replay.
    No failed P1 convergence result is relabeled as a continuum certificate.
    """
    from .mesh import generate_graded_mesh,run_convergence
    root=Path(root).resolve();out=root/'results/highorder';out.mkdir(parents=True,exist_ok=True)
    # Includes all three material-aware P1 refinements and their measured failure.
    run_convergence(root)
    geometry=generate_graded_mesh(root/'meshes/graded_geometry256.msh',h_background=.04,
                                 h_inclusion=.005,circle_points=256)
    selected=out/'geometry256_cg3.npz';lower=out/'geometry256_cg2.npz';refined=out/'level1_cg2.npz'
    for mesh,degree,archive in [(geometry,3,selected),(geometry,2,lower),(root/'meshes/graded_1.msh',2,refined)]:
        solve_highorder(mesh,degree,archive)
    support=root/'results/mesh_convergence/level_1_fields.npz'
    def cached(name,compute):
        path=out/name
        if path.exists() and not force_audits:return json.loads(path.read_text())
        value=compute();path.write_text(json.dumps(value,indent=2)+'\n');return value
    degree=cached('geometry256_degree_convergence.json',lambda:compare_highorder(geometry,2,lower,geometry,3,selected,support))
    spatial=cached('geometry256_spatial_convergence.json',lambda:compare_highorder(geometry,3,selected,root/'meshes/graded_1.msh',2,refined,support))
    representation=cached('geometry256_representation.json',lambda:compare_native_to_p1(geometry,3,selected,support)[0])
    iterative=cached('selected_iterative_validation.json',lambda:validate_iterative_solver(geometry,out/'selected_iterative_validation.json'))
    # Native mesh ordering and actual edge sizes come from the same DOLFINx reader.
    native=HighOrderSolver(geometry,3);support_geometry=SupportGeometry(support)
    def geometry_stats(coords,tri,tags):
        xy=coords[tri];lengths=np.linalg.norm(xy-np.roll(xy,1,axis=1),axis=2)
        result=dict(vertices=len(coords),triangles=len(tri),edge_min=float(lengths.min()),edge_max=float(lengths.max()),
                    inclusion_edge_max=float(lengths[np.asarray(tags)>1].max()))
        for n in [10,20]:
            defect=0
            for axis in [0,1]:
                lo=xy[:,:,axis].min(axis=1);hi=xy[:,:,axis].max(axis=1)
                for line in np.linspace(-2,2,n+1)[1:-1]:defect+=int(np.count_nonzero((lo<line-1e-10)&(hi>line+1e-10)))
            result[f'crossings_coarse_grid_{n}']=defect
        return result
    native_tags=np.zeros(len(native.triangles),dtype=int)
    native_tags[native.data.cell_tags.indices]=native.data.cell_tags.values
    statistics=dict(native=geometry_stats(native.coordinates,native.triangles,native_tags),
                    support=geometry_stats(support_geometry.coordinates,support_geometry.triangles,support_geometry.cell_tags))
    with np.load(selected,allow_pickle=False) as z:native_dofs=int(z['fields'].shape[1])
    del native;gc.collect()
    maxima=dict(native_degree=max(r['relative_scattered_l2'] for r in degree),
                native_spatial=max(r['relative_scattered_l2'] for r in spatial),
                p1_representation=max(r['p1_representation_relative_l2'] for r in representation))
    checked=[geometry,root/'meshes/graded_1.msh',support,selected,lower,refined,
             out/'geometry256_degree_convergence.json',out/'geometry256_spatial_convergence.json',
             out/'geometry256_representation.json',out/'selected_iterative_validation.json',Path(__file__),root/'mesh.py']
    selection=dict(schema_version=1,passed=bool(max(maxima.values())<=tolerance and iterative['passed']),
                   native_mesh=str(geometry.relative_to(root)),native_degree=3,native_dofs=native_dofs,
                   support_mesh='meshes/graded_1.msh',support_archive=str(support.relative_to(root)),
                   support_dofs=len(support_geometry.coordinates),validation_frequencies=list(FREQUENCIES),
                   separate_relative_l2_tolerance=tolerance,observed_relative_l2_maxima=maxima,
                   reference_definition='Native affine-geometry CG3 scattering solution sampled onto declared P1 support nodes',
                   geometry=statistics,
                   native_linear_solver='Finite original-equation residual <1e-10 determines acceptance independently of GMRES internal info; previous-solution warm-start, nearby-frequency LU preconditioning, max50iterations; direct LU fallback',
                   interpretation='Empirical degree/spatial/representation diagnostics at five anchors, not a rigorous uniform continuum error certificate. Polygonal material geometry and fixed first-order absorbing boundary are declared model approximations. P1 support is an observation representation, not P1 FEM truth.',
                   verified_files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in checked})
    (out/'selected_reference.json').write_text(json.dumps(selection,indent=2)+'\n')
    if not selection['passed']:raise RuntimeError('Reference adequacy audit did not pass its separate tolerances')
    return selection


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=HERE)
    parser.add_argument('--stage',choices=('reference','pilot'),default='reference')
    parser.add_argument('--force-audits',action='store_true')
    args=parser.parse_args()
    result=run_reference_study(args.root,force_audits=args.force_audits) if args.stage=='reference' else pilot(args.root)
    print(json.dumps(result,indent=2))
