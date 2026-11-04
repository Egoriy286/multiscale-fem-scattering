"""Classical Galerkin comparisons on the exact saved neural-operator sample."""
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .fem import build_fine_problem


def mesh_resolution(dataset, index=0):
    xy = dataset.coordinates[dataset.triangles]
    lengths = np.linalg.norm(xy-xy[:, [1,2,0]],axis=-1)
    h_cell = lengths.max(axis=1)
    if dataset.A_cells is not None and dataset.n_cells is not None:
        k_cell = dataset.k0[index]*np.sqrt(dataset.n_cells[index]/dataset.A_cells[index])
        k_local = float(k_cell.max())
        kh_max = float(np.max(k_cell*h_cell))
        estimate = 'cellwise DG0'
    else:
        k_local = float(dataset.k0[index]*np.sqrt(np.max(dataset.n[index]/dataset.A[index])))
        kh_max = float(k_local*h_cell.max())
        estimate = 'conservative global bound from nodal coefficients'
    return dict(h_max=float(lengths.max()),h_median=float(np.median(lengths)),k_local_max=k_local,kh_max=kh_max,points_per_wavelength_min=float(2*np.pi/kh_max),resolution_estimate=estimate)



def coarse_prolongation(coordinates, n_coarse):
    """Nodal Q1 traces interpolated into the common conforming fine P1 space."""
    scaled = (coordinates+2)*n_coarse/4
    ij = np.clip(np.floor(scaled).astype(int),0,n_coarse-1)
    local = scaled-ij
    rows,cols,values = [],[],[]
    for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
        rows.append(np.arange(len(coordinates)))
        cols.append((ij[:,1]+dy)*(n_coarse+1)+ij[:,0]+dx)
        values.append((local[:,0] if dx else 1-local[:,0])*(local[:,1] if dy else 1-local[:,1]))
    return sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(len(coordinates),(n_coarse+1)**2)).tocsc()


def compare_classical(dataset, index=0, n_coarse=4, modes=(1,2,4), basis_types=(1,2)):
    """Offline timings describe one space built at max(modes), shared by all M.

    reconstruction_seconds includes reconstruction and residual validation overhead.
    """
    from gmsfem.offline import build_offline_space, solve_reduced
    started = perf_counter()
    problem = build_fine_problem(dataset,index)
    assembly = perf_counter()-started
    reference = dataset.u_scat[index]
    started = perf_counter()
    solution = splu(problem.operator.tocsc()).solve(problem.rhs)
    online = perf_counter()-started
    resolution = mesh_resolution(dataset,index)
    def row(method,u,dofs,offline,online,**extra):
        return dict(method=method,error_fem_l2=float(dataset.relative_l2(reference,u)),assembly_seconds=assembly,offline_seconds=offline,online_seconds=online,dofs=int(dofs),sample_index=int(index),medium_id=int(dataset.medium_id[index]),split_group_id=int(dataset.split_group_ids[index]),split_strategy=dataset.split_strategy,**resolution,**extra)
    rows = [row('Fine FEM reference',solution,problem.ndofs,0.,online,role='reference')]
    started = perf_counter()
    P = coarse_prolongation(problem.coordinates,n_coarse)
    offline = perf_counter()-started
    started = perf_counter()
    u = P @ splu((P.T@problem.operator@P).tocsc()).solve(np.asarray(P.T@problem.rhs))
    rows.append(row('Coarse Q1 traces in fine P1',u,P.shape[1],offline,perf_counter()-started))
    for basis_type in basis_types:
        space = build_offline_space(problem,n_coarse=n_coarse,max_modes=max(modes),basis_type=basis_type,progress=None)
        for m in modes:
            started = perf_counter()
            u,info = solve_reduced(problem,space,m)
            online = perf_counter()-started
            reconstruction = max(0.,online-info['projection_seconds']-info['solve_seconds'])
            rows.append(row(f'GMsFEM Type {basis_type}, M={m}',u,info['dofs_coarse'],space.build_seconds,online,built_max_modes=int(space.max_modes),reconstruction_seconds=reconstruction,basis_type=int(basis_type),modes=int(m),projection_seconds=info['projection_seconds'],solve_seconds=info['solve_seconds']))
    return rows


def compare_refined_fem(dataset, index=0, *, h=None, n_coarse=4):
    """Mesh-convergence evidence: compare coarse P1 field to refined FEM.

    Both solves use the same circle materials and analytic incident wave.
    Error integrates their difference with the refined consistent mass matrix;
    curved interfaces are independently polygonalized on the two meshes.
    """
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import json
    from dolfinx import geometry
    from .fem import generate_mesh, HelmholtzSolver
    from .data import mass_matrix
    if dataset.A_cells is None or dataset.n_cells is None or dataset.cell_tags is None:
        raise ValueError('Refinement requires exact DG0 coefficients and circle tags')
    if h is None:
        config = json.loads((Path(dataset.directory)/'config.json').read_text())
        h = config['h']/2
        n_coarse = config['n_coarse']
    original = build_fine_problem(dataset,index)
    with TemporaryDirectory() as directory:
        path = generate_mesh(Path(directory)/'refined.msh',h=h,n_coarse=n_coarse)
        solver = HelmholtzSolver(path)
        alpha,n = np.ones(len(solver.triangles)),np.ones(len(solver.triangles))
        for tag in np.unique(dataset.cell_tags):
            source = dataset.cell_tags == tag
            # Each circle (including inactive circles) is constant DG0.
            if np.ptp(dataset.A_cells[index,source]) > 1e-12 or np.ptp(dataset.n_cells[index,source]) > 1e-12:
                raise ValueError('Material must be constant within each physical surface tag')
            alpha[solver.cell_tags == tag] = dataset.A_cells[index,source][0]
            n[solver.cell_tags == tag] = dataset.n_cells[index,source][0]
        started = perf_counter()
        refined = solver.solve(alpha,n,dataset.k0[index],dataset.theta[index])
        solve_seconds = perf_counter()-started
        function = original.as_function(dataset.u_scat[index])
        points = np.column_stack((solver.coordinates,np.zeros(len(solver.coordinates))))
        tree = geometry.bb_tree(original.mesh,2)
        candidates = geometry.compute_collisions_points(tree,points)
        collisions = geometry.compute_colliding_cells(original.mesh,candidates,points)
        cells = np.array([collisions.links(j)[0] if len(collisions.links(j)) else -1 for j in range(len(points))],dtype=np.int32)
        if np.any(cells < 0):
            raise RuntimeError('Refinement evaluation point outside original square mesh')
        coarse = function.eval(points,cells).reshape(-1)
        M = mass_matrix(solver.coordinates,solver.triangles)
        error = coarse-refined
        num,den = np.real(np.vdot(error,M@error)),np.real(np.vdot(refined,M@refined))
        relative = np.sqrt(num/den) if den > 0 else (0. if num == 0 else np.inf)
        return dict(method='Fine FEM vs refined FEM',error_fem_l2=float(relative),dofs_original=original.ndofs,dofs_refined=len(refined),refined_h=float(h),refined_solve_seconds=solve_seconds)
