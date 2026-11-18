"""Native CG2 fine operators for the existing Type-2 GMsFEM algorithm.

No fine solution or RHS enters offline basis construction. Element matrices are
assembled over the actual patch elements (never global principal submatrices).
Only ``offline.local_matrices`` is temporarily dispatched to its CG2 analogue;
this wrapper is for single-threaded offline construction in one process.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass,field
from pathlib import Path
from threading import RLock
from time import perf_counter
import numpy as np
from scipy import sparse
from gmsfem.fem import FineProblem,LocalMatrices,MATERIALS,_petsc_to_csr
from .highorder import HighOrderSolver

_LOCK=RLock()


@dataclass
class P2FineProblem(FineProblem):
    degree:int=2
    edge_vertices:tuple=((1,2),(0,2),(0,1))
    edge_closure_dofs:tuple=((1,2,3),(0,2,4),(0,1,5))
    assembly_diagnostics:dict=field(default_factory=dict)


def assemble_elements(connectivity,values,ndofs):
    connectivity=np.asarray(connectivity,dtype=np.int32);values=np.asarray(values)
    width=connectivity.shape[1]
    if values.shape!=(len(connectivity),width,width):raise ValueError('Element tensor/connectivity shape differs')
    return sparse.coo_matrix((values.ravel(),(np.repeat(connectivity,width,axis=1).ravel(),
        np.tile(connectivity,(1,width)).ravel())),shape=(ndofs,ndofs)).tocsr()


def element_matrices(coordinates,connectivity,alpha):
    """Affine P2 stiffness and unit mass, in Basix/DOLFINx local DOF order."""
    import basix
    element=basix.create_element(basix.ElementFamily.P,basix.CellType.triangle,2,basix.LagrangeVariant.equispaced)
    points,weights=basix.make_quadrature(basix.CellType.triangle,4)
    table=element.tabulate(1,points)
    values=table[0,:,:,0];gradients=np.stack((table[1,:,:,0],table[2,:,:,0]),axis=-1)
    vertices=np.asarray(coordinates)[np.asarray(connectivity)[:,:3]]
    jacobian=np.swapaxes(vertices[:,1:]-vertices[:,:1],1,2)
    determinant=np.linalg.det(jacobian)
    if np.any(abs(determinant)<1e-20):raise ValueError('Degenerate affine CG2 cell')
    mapped=np.einsum('qir,crd->cqid',gradients,np.linalg.inv(jacobian))
    stiffness=np.einsum('cqid,cqjd,q,c->cij',mapped,mapped,weights,np.abs(determinant)*alpha,optimize=True)
    reference_mass=np.einsum('qi,qj,q->ij',values,values,weights,optimize=True)
    mass=np.abs(determinant)[:,None,None]*reference_mass
    return stiffness,mass,abs(determinant)/2,element


def local_matrices(problem,cells):
    """True CG2 patch Neumann matrices and all vertex/edge trace DOFs."""
    if not isinstance(problem,P2FineProblem):raise TypeError('CG2 local assembly requires P2FineProblem')
    cells=np.asarray(cells,dtype=np.int32)
    if not len(cells):raise ValueError('Empty local CG2 domain')
    global_cells=problem.triangles[cells];dofs=np.unique(global_cells)
    connectivity=np.searchsorted(dofs,global_cells)
    edges=np.concatenate([np.sort(connectivity[:,vertices],axis=1) for vertices in problem.edge_vertices])
    _,inverse,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    if np.any(counts>2):raise ValueError('Nonmanifold CG2 patch')
    boundary=[];ncells=len(cells)
    for edge,closure in enumerate(problem.edge_closure_dofs):
        owners=np.flatnonzero(counts[inverse[edge*ncells:(edge+1)*ncells]]==1)
        boundary.append(connectivity[owners][:,closure].ravel())
    trace=np.unique(np.concatenate(boundary))
    return LocalMatrices(dofs,trace,
        assemble_elements(connectivity,problem.element_K[cells],len(dofs)),
        assemble_elements(connectivity,problem.n[cells,None,None]*problem.element_M0[cells],len(dofs)),
        assemble_elements(connectivity,problem.alpha[cells,None,None]*problem.element_M0[cells],len(dofs)))


@contextmanager
def _local_dispatch():
    """Scoped in-memory dispatch; always restore the unchanged P1 implementation."""
    import gmsfem.offline as offline
    with _LOCK:
        original=offline.local_matrices
        def dispatch(problem,cells):
            return local_matrices(problem,cells) if isinstance(problem,P2FineProblem) else original(problem,cells)
        offline.local_matrices=dispatch
        try:yield
        finally:offline.local_matrices=original


def build_offline_space(problem,**kwargs):
    import gmsfem.offline as offline
    if not isinstance(problem,P2FineProblem):raise TypeError('CG2 wrapper requires P2FineProblem')
    with _local_dispatch():result=offline.build_offline_space(problem,**kwargs)
    result.diagnostics.update(fine_degree=2,local_assembly='Basix affine P2 quadrature degree4',
        scoped_dispatch=True,global_assembly_errors=problem.assembly_diagnostics)
    return result


class P2Backend:
    """Reusable native CG2 geometry, operators, local tensors and loads."""
    def __init__(self,mesh,*,assembly_tolerance=1e-10):
        from dolfinx import fem
        from dolfinx.fem import petsc
        import basix
        import ufl
        started=perf_counter();self.native=HighOrderSolver(mesh,degree=2)
        native=self.native;self.mesh_path=Path(mesh)
        domain=native.data.mesh;self.V=native.V
        self.coordinates=self.V.tabulate_dof_coordinates()[:,:2].copy()
        ncells=domain.topology.index_map(2).size_local
        self.triangles=np.array([self.V.dofmap.cell_dofs(c) for c in range(ncells)],dtype=np.int32)
        self.tags=np.zeros(ncells,dtype=np.int32);self.tags[native.data.cell_tags.indices]=native.data.cell_tags.values
        self.alpha=np.array([MATERIALS['notebook'][int(t)][0] for t in self.tags])
        self.n=np.array([MATERIALS['notebook'][int(t)][1] for t in self.tags])
        self.element_K,self.element_M0,self.areas,element=element_matrices(self.coordinates,self.triangles,self.alpha)
        self.edge_vertices=tuple(tuple(edge) for edge in basix.topology(basix.CellType.triangle)[1])
        self.edge_closure_dofs=tuple(tuple(edge) for edge in element.entity_closure_dofs[1])
        ndofs=len(self.coordinates)
        self.M0=assemble_elements(self.triangles,self.element_M0,ndofs)
        self.Salpha=assemble_elements(self.triangles,self.alpha[:,None,None]*self.element_M0,ndofs)
        # Independently assemble every local tensor's global analogue in UFL.
        u,v=ufl.TrialFunction(self.V),ufl.TestFunction(self.V)
        dx=ufl.Measure('dx',domain=domain,metadata={'quadrature_degree':12})
        W=fem.functionspace(domain,('DG',0));alpha_f=fem.Function(W)
        for c in range(ncells):alpha_f.x.array[W.dofmap.cell_dofs(c)]=self.alpha[c]
        independent=[]
        for expression in [ufl.inner(u,v)*dx,ufl.inner(alpha_f*u,v)*dx]:
            matrix=petsc.assemble_matrix(fem.form(expression));matrix.assemble()
            independent.append(_petsc_to_csr(matrix));matrix.destroy()
        pairs=dict(K=(assemble_elements(self.triangles,self.element_K,ndofs),native.K),
            Mn=(assemble_elements(self.triangles,self.n[:,None,None]*self.element_M0,ndofs),native.Mn),
            M0=(self.M0,independent[0]),Salpha=(self.Salpha,independent[1]))
        self.assembly_errors={name:float(sparse.linalg.norm(a-b)/max(sparse.linalg.norm(b),1e-30)) for name,(a,b) in pairs.items()}
        if max(self.assembly_errors.values())>=assembly_tolerance:
            raise RuntimeError(f'CG2 element/global DOLFINx assembly mismatch: {self.assembly_errors}')
        self.setup_seconds=perf_counter()-started
        self.provenance=dict(degree=2,quadrature_degree=4,basix_version=basix.__version__,
            global_assembly_errors=self.assembly_errors,setup_seconds=self.setup_seconds,
            ndofs=ndofs,cells=ncells,local_dofs=6)

    def validate_local_assembly(self,cells,*,tolerance=1e-10):
        """Independent UFL integral over only selected cells, including interfaces."""
        from dolfinx import fem
        from dolfinx.fem import petsc
        import ufl
        cells=np.asarray(cells,dtype=np.int32)
        problem=self.problem(1.);local=local_matrices(problem,cells)
        domain=self.native.data.mesh;W=fem.functionspace(domain,('DG',0))
        indicator=fem.Function(W);alpha=fem.Function(W);n=fem.Function(W)
        selected=np.zeros(len(self.triangles));selected[cells]=1.
        for cell in range(len(self.triangles)):
            dof=W.dofmap.cell_dofs(cell)
            indicator.x.array[dof]=selected[cell]
            alpha.x.array[dof]=self.alpha[cell];n.x.array[dof]=self.n[cell]
        u,v=ufl.TrialFunction(self.V),ufl.TestFunction(self.V)
        dx=ufl.Measure('dx',domain=domain,metadata={'quadrature_degree':12})
        expressions=dict(K=ufl.inner(indicator*alpha*ufl.grad(u),ufl.grad(v))*dx,
            Mn=ufl.inner(indicator*n*u,v)*dx,Salpha=ufl.inner(indicator*alpha*u,v)*dx)
        errors={}
        for name,expression in expressions.items():
            matrix=petsc.assemble_matrix(fem.form(expression));matrix.assemble()
            independent=_petsc_to_csr(matrix)[local.dofs][:,local.dofs];matrix.destroy()
            reference=getattr(local,name)
            errors[name]=float(sparse.linalg.norm(reference-independent)/max(sparse.linalg.norm(independent),1e-30))
        if max(errors.values())>=tolerance:raise RuntimeError(f'CG2 local patch/UFL mismatch: {errors}')
        return dict(cells=len(cells),dofs=len(local.dofs),boundary_dofs=len(local.boundary),
            material_tags=np.unique(self.tags[cells]).tolist(),relative_assembly_errors=errors)

    def problem(self,k,theta=0.):
        from petsc4py import PETSc
        from dolfinx.fem import petsc
        if not np.isfinite(k) or k<=0:raise ValueError('Positive finite physical k required')
        started=perf_counter();native=self.native;direction=np.array([np.cos(theta),np.sin(theta)])
        native.k.value=k;native.d.value=direction.astype(PETSc.ScalarType)
        vector=petsc.assemble_vector(native.load)
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)
        rhs=vector.array.copy();vector.destroy()
        return P2FineProblem(mesh=native.data.mesh,V=native.V,cell_tags=native.data.cell_tags,
            facet_tags=native.data.facet_tags,alpha_function=None,n_function=None,
            coordinates=self.coordinates,triangles=self.triangles,material_tags=self.tags,
            alpha=self.alpha,n=self.n,areas=self.areas,element_K=self.element_K,element_M0=self.element_M0,
            K=native.K,Mn=native.Mn,M0=self.M0,Salpha=self.Salpha,B=native.B,
            operator=(native.K-k*k*native.Mn-1j*k*native.B).tocsr(),rhs=rhs,
            incident=np.exp(1j*k*(self.coordinates@direction)),k=float(k),direction=tuple(direction),
            case='notebook',assembly_seconds=perf_counter()-started,
            element_assembly_error=max(self.assembly_errors.values()),edge_vertices=self.edge_vertices,
            edge_closure_dofs=self.edge_closure_dofs,assembly_diagnostics=self.assembly_errors)

    def evaluate(self,field,points):return self.native.evaluate(field,points)
    def solve_fine(self,k,theta=0.):return self.native.solve(k,theta)
