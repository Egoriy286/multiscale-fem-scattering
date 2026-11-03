"""Local conforming FEM generation and exact material reuse by GMsFEM."""
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .data import Dataset, group_split, load_dataset


@dataclass(frozen=True)
class DatasetConfig:
    n_media: int = 48
    n_directions: int = 4
    h: float = .10
    n_coarse: int = 4
    k_range: tuple = (1., 2.)
    seed: int = 2026
    active_range: tuple = (3, 10)
    a_range: tuple = (.6, .9)
    n_range: tuple = (1.5, 3.)
    split_fractions: tuple = (.67, .17, .16)


def generate_mesh(path, *, h=.10, n_coarse=4):
    """Same CAD as GMsFEM; each individual circle gets its own physical tag."""
    import gmsh
    from gmsfem.geometry import inclusion_circles, _interior_point, _GEO_DIR
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    previous = gmsh.model.getCurrent()
    try:
        gmsh.option.setNumber('General.Verbosity', 1)
        gmsh.onelab.setNumber('GMSFEM/n_coarse', [int(n_coarse)])
        gmsh.onelab.setNumber('GMSFEM/h', [float(h)])
        gmsh.open(str(_GEO_DIR / 'scattering.geo'))
        circles = inclusion_circles()
        groups = {tag: [] for tag in range(1, len(circles)+2)}
        for _, surface in gmsh.model.getEntities(2):
            xy = _interior_point(gmsh, surface)[:2]
            inside = np.flatnonzero(np.sum((circles[:, :2]-xy)**2, axis=1) < circles[:, 2]**2)
            groups[int(inside[0])+2 if len(inside) else 1].append(surface)
        for tag, surfaces in groups.items():
            gmsh.model.addPhysicalGroup(2, surfaces, tag)
        # Keep curvature refinement near the small circles rather than propagating
        # its tiny edge sizes throughout the 4×4 background.
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        if own:
            gmsh.finalize()
        else:
            gmsh.model.remove()
            if previous:
                gmsh.model.setCurrent(previous)
    return Path(path)


class HelmholtzSolver:
    """P1 sparse operator; 8×8 Duffy quadrature of the analytic incident wave.

    Geometry, quadrature and boundary mass are cached. One LU is reused for
    all directions having identical DG0 coefficients and wavenumber.
    """
    def __init__(self, mesh_path):
        from mpi4py import MPI
        from dolfinx import fem
        from dolfinx.io import gmsh as gmshio
        from gmsfem.fem import assemble_elements
        self.mesh_path = Path(mesh_path)
        data = gmshio.read_from_msh(str(mesh_path), MPI.COMM_SELF, gdim=2)
        self.mesh, self.ct, self.ft = data.mesh, data.cell_tags, data.facet_tags
        self.V = fem.functionspace(self.mesh, ('Lagrange', 1))
        self.coordinates = self.V.tabulate_dof_coordinates()[:, :2].copy()
        ncells = self.mesh.topology.index_map(2).size_local
        self.triangles = np.array([self.V.dofmap.cell_dofs(c) for c in range(ncells)], dtype=np.int32)
        self.cell_tags = np.zeros(ncells, dtype=np.int32)
        self.cell_tags[self.ct.indices] = self.ct.values
        xy = self.coordinates[self.triangles]
        mapping = np.concatenate((np.ones((ncells, 3, 1)), xy), axis=2)
        self.areas = np.abs(np.linalg.det(mapping))/2
        self.gradients = np.linalg.inv(mapping)[:, 1:, :].transpose(0, 2, 1)
        self.element_K0 = self.areas[:, None, None] * np.einsum('cik,cjk->cij', self.gradients, self.gradients)
        self.element_M0 = self.areas[:, None, None]/12 * (np.ones((3,3))+np.eye(3))[None]
        self.M0 = assemble_elements(self.triangles, self.element_M0, len(self.coordinates))
        edges = np.sort(np.concatenate([self.triangles[:, [0,1]], self.triangles[:, [1,2]], self.triangles[:, [2,0]]]), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        edges = edges[counts == 1]
        lengths = np.linalg.norm(self.coordinates[edges[:, 0]]-self.coordinates[edges[:, 1]], axis=1)
        vals = lengths[:,None,None]/6 * np.array([[2.,1.],[1.,2.]])[None]
        self.B = sparse.coo_matrix((vals.ravel(), (np.repeat(edges,2,axis=1).ravel(), np.tile(edges,(1,2)).ravel())), shape=self.M0.shape).tocsr()
        q,w = np.polynomial.legendre.leggauss(8)
        q,w = (q+1)/2,w/2
        r,s = np.meshgrid(q,q,indexing='ij')
        self.phi = np.stack((1-r-(1-r)*s,r,(1-r)*s),axis=-1).reshape(-1,3)
        self.qweights = ((w[:,None]*w[None,:])*(1-r)).ravel()*2
        self.qpoints = np.einsum('qi,cid->cqd', self.phi,xy)
        self._key = None

    def nodal_coefficients(self, values):
        return np.bincount(self.triangles.ravel(), weights=np.repeat(values,3), minlength=len(self.coordinates))/np.bincount(self.triangles.ravel(), minlength=len(self.coordinates))

    def rhs(self, alpha, n, k, theta):
        d = np.array([np.cos(theta), np.sin(theta)])
        ui = np.exp(1j*k*(self.qpoints@d))
        integral = np.einsum('cq,q->c',ui,self.qweights)*self.areas
        mass_integral = np.einsum('cq,qi,q->ci',ui,self.phi,self.qweights)*self.areas[:,None]
        local = ((1-alpha)[:,None] * (1j*k*(self.gradients@d))*integral[:,None] + k*k*(n-1)[:,None]*mass_integral)
        result = np.zeros(len(self.coordinates),dtype=complex)
        np.add.at(result,self.triangles.ravel(),local.ravel())
        return result

    def build_problem(self, alpha, n, k, theta):
        from gmsfem.fem import FineProblem, assemble_elements
        started = perf_counter()
        alpha,n = np.asarray(alpha),np.asarray(n)
        K = assemble_elements(self.triangles,alpha[:,None,None]*self.element_K0,len(self.coordinates))
        Mn = assemble_elements(self.triangles,n[:,None,None]*self.element_M0,len(self.coordinates))
        Salpha = assemble_elements(self.triangles,alpha[:,None,None]*self.element_M0,len(self.coordinates))
        rhs = self.rhs(alpha,n,k,theta)
        d = (float(np.cos(theta)),float(np.sin(theta)))
        return FineProblem(self.mesh,self.V,self.ct,self.ft,None,None,self.coordinates,self.triangles,self.cell_tags,alpha.copy(),n.copy(),self.areas,alpha[:,None,None]*self.element_K0,self.element_M0,K,Mn,self.M0,Salpha,self.B,(K-k*k*Mn-1j*k*self.B).tocsr(),rhs,np.exp(1j*k*(self.coordinates@d)),float(k),d,'dataset',perf_counter()-started,0.)

    def solve(self, A_cells, n_cells, k0, theta):
        key = (np.asarray(A_cells).tobytes(), np.asarray(n_cells).tobytes(),float(k0))
        if key != self._key:
            problem = self.build_problem(A_cells,n_cells,k0,theta)
            self._lu = splu(problem.operator.tocsc())
            self._key = key
        return self._lu.solve(self.rhs(A_cells,n_cells,k0,theta))


def build_fine_problem(dataset, index=0):
    if dataset.directory is None or dataset.A_cells is None or dataset.n_cells is None:
        raise ValueError('Classical comparisons require saved mesh and exact DG0 coefficients')
    solver = HelmholtzSolver(dataset.directory / 'etal.msh')
    if not np.array_equal(solver.triangles,dataset.triangles) or not np.allclose(solver.coordinates,dataset.coordinates):
        raise ValueError('Mesh DOF ordering differs from stored labels')
    return solver.build_problem(dataset.A_cells[index],dataset.n_cells[index],dataset.k0[index],dataset.theta[index])


def generate_dataset(output_dir, config=DatasetConfig(), progress=print):
    from gmsfem.geometry import inclusion_circles
    if config.n_media < 1 or config.n_directions < 1 or config.n_coarse < 1 or config.h <= 0:
        raise ValueError('Positive sample counts, coarse resolution and h required')
    if not 0 <= config.active_range[0] <= config.active_range[1] <= 24:
        raise ValueError('active_range must lie in [0,24]')
    for interval in (config.k_range,config.a_range,config.n_range):
        if not 0 < interval[0] <= interval[1]:
            raise ValueError('Positive ordered parameter ranges required')
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True,exist_ok=True)
    cfg = json.loads(json.dumps(asdict(config)))
    config_path = directory/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Output directory contains a different configuration')
    if (directory/'fem_dataset.npz').exists():
        if not config_path.exists():
            raise ValueError('Existing dataset has no configuration')
        return load_dataset(directory)
    config_path.write_text(json.dumps(cfg,indent=2)+'\n')
    generate_mesh(directory/'etal.msh',h=config.h,n_coarse=config.n_coarse)
    solver = HelmholtzSolver(directory/'etal.msh')
    circles = inclusion_circles().copy()
    circles[:,3] = np.arange(2,26)
    np.savez_compressed(directory/'geometry.npz',dof_coords=solver.coordinates,triangles=solver.triangles,cell_tags=solver.cell_tags,etal_circles=circles)
    ns = config.n_media*config.n_directions
    nn, nc = len(solver.coordinates),len(solver.triangles)
    arrays = {key: np.empty((ns,nn),dtype=np.float64) for key in ('A','n')}
    arrays.update({key: np.empty((ns,nc),dtype=np.float64) for key in ('A_cells','n_cells')})
    arrays['u_scat'] = np.empty((ns,nn),dtype=np.complex128)
    arrays.update({key: np.empty(ns,dtype=np.float64) for key in ('theta','k0')})
    arrays['medium_id'] = np.empty(ns,dtype=np.int32)
    def report(completed):
        if progress is not None:
            message = f'FEM dataset: {completed}/{config.n_media} media, {completed*config.n_directions}/{ns} fields'
            if progress is print:
                print(message,flush=True)
            else:
                progress(message)
    report(0)
    for mid in range(config.n_media):
        rng = np.random.default_rng([config.seed,mid])
        k = rng.uniform(*config.k_range)
        alpha,n = np.ones(len(solver.triangles)),np.ones(len(solver.triangles))
        chosen = rng.choice(24,size=rng.integers(config.active_range[0],config.active_range[1]+1),replace=False)
        for cid in chosen:
            cells = solver.cell_tags == cid+2
            alpha[cells] = np.exp(rng.uniform(*np.log(config.a_range)))
            n[cells] = np.exp(rng.uniform(*np.log(config.n_range)))
        offset = rng.uniform(0,2*np.pi/config.n_directions)
        first = mid*config.n_directions
        rows = slice(first,first+config.n_directions)
        # Nodal averages are medium properties, computed once for all directions.
        arrays['A'][rows] = solver.nodal_coefficients(alpha)
        arrays['n'][rows] = solver.nodal_coefficients(n)
        arrays['A_cells'][rows] = alpha
        arrays['n_cells'][rows] = n
        arrays['k0'][rows] = k
        arrays['medium_id'][rows] = mid
        for direction,theta in enumerate(offset+2*np.pi*np.arange(config.n_directions)/config.n_directions):
            sample = first+direction
            arrays['theta'][sample] = theta
            arrays['u_scat'][sample] = solver.solve(alpha,n,k,theta)
        if mid == 0 or (mid+1)%25 == 0 or mid+1 == config.n_media:
            report(mid+1)
    arrays['split'] = group_split(arrays['medium_id'],config.split_fractions,config.seed)
    # Passing an open file avoids NumPy adding a suffix; replace is atomic on
    # this filesystem and an interrupted save never becomes a valid cache.
    temporary = directory/'fem_dataset.tmp.npz'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    temporary.replace(directory/'fem_dataset.npz')
    return Dataset(**arrays,coordinates=solver.coordinates,triangles=solver.triangles,
                   cell_tags=solver.cell_tags,circles=circles,directory=directory)
