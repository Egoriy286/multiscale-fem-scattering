import numpy as np
import pytest
import importlib.util
from neural_operator.data import Dataset, mass_matrix, group_split


def test_consistent_mass_and_complex_norm():
    xy = np.array([[0.,0.],[1.,0.],[0.,1.]])
    tri = np.array([[0,1,2]])
    M = mass_matrix(xy,tri)
    assert np.isclose(np.ones(3)@M@np.ones(3),.5)
    ds = Dataset(np.ones((1,3)),np.ones((1,3)),np.ones((1,3),complex),np.zeros(1),np.ones(1),np.zeros(1,int),np.zeros(1,int),xy,tri)
    assert ds.relative_l2(np.ones(3)*1j,np.ones(3)*2j) == pytest.approx(1)
    assert ds.relative_l2(np.zeros(3),np.zeros(3)) == 0
    assert np.isinf(ds.relative_l2(np.zeros(3),np.ones(3)))


def test_group_split_has_no_leakage():
    ids = np.repeat(np.arange(20),4)
    split = group_split(ids)
    assert np.array_equal(split,group_split(ids))
    assert set(split) == {0,1,2}
    assert all(len(np.unique(split[ids == mid])) == 1 for mid in np.unique(ids))


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    if importlib.util.find_spec('dolfinx') is None:
        pytest.skip('DOLFINx is needed for the PDE tests')
    from neural_operator.fem import DatasetConfig,generate_dataset
    return generate_dataset(tmp_path_factory.mktemp('fem'),DatasetConfig(n_media=3,n_directions=2,h=.3,n_coarse=2,active_range=(1,2)))


def test_homogeneous_and_material_reuse(generated):
    from neural_operator.fem import HelmholtzSolver,build_fine_problem
    solver = HelmholtzSolver(generated.directory/'etal.msh')
    ones = np.ones(len(solver.triangles))
    assert np.max(np.abs(solver.solve(ones,ones,1.,.3))) < 1e-13
    problem = build_fine_problem(generated,0)
    assert np.array_equal(problem.alpha,generated.A_cells[0])
    assert np.array_equal(problem.n,generated.n_cells[0])
    u = solver.solve(generated.A_cells[0],generated.n_cells[0],generated.k0[0],generated.theta[0])
    assert generated.relative_l2(generated.u_scat[0],u) < 1e-12


def test_rhs_independent_ufl(generated):
    """Compare analytic quadrature RHS to independently assembled UFL weak form."""
    from dolfinx import fem
    from dolfinx.fem import petsc
    import ufl
    from petsc4py import PETSc
    from neural_operator.fem import HelmholtzSolver
    solver = HelmholtzSolver(generated.directory/'etal.msh')
    W = fem.functionspace(solver.mesh,('DG',0))
    a,n = fem.Function(W),fem.Function(W)
    for cell in range(len(solver.triangles)):
        a.x.array[W.dofmap.cell_dofs(cell)] = generated.A_cells[0,cell]
        n.x.array[W.dofmap.cell_dofs(cell)] = generated.n_cells[0,cell]
    k,theta = generated.k0[0],generated.theta[0]
    x = ufl.SpatialCoordinate(solver.mesh)
    ui = ufl.exp(1j*k*(np.cos(theta)*x[0]+np.sin(theta)*x[1]))
    v = ufl.TestFunction(solver.V)
    form = (ufl.inner((1-a)*ufl.grad(ui),ufl.grad(v))+k*k*ufl.inner((n-1)*ui,v))*ufl.dx(metadata={'quadrature_degree':12})
    vector = petsc.assemble_vector(fem.form(form))
    vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)
    expected = vector.array.copy()
    vector.destroy()
    actual = solver.rhs(generated.A_cells[0],generated.n_cells[0],k,theta)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 1e-11


def test_sparse_matrices_independent_ufl(generated):
    """Independent UFL assembly checks variable DG0 stiffness, mass and Robin edges."""
    from dolfinx import fem
    from dolfinx.fem import petsc
    from scipy import sparse
    import ufl
    from neural_operator.fem import HelmholtzSolver
    solver = HelmholtzSolver(generated.directory/'etal.msh')
    W = fem.functionspace(solver.mesh,('DG',0))
    a,n = fem.Function(W),fem.Function(W)
    # Use every circle with distinct contrast, beyond the sparse random sample.
    alpha = .5+.03*solver.cell_tags
    refractive = 1.+.1*solver.cell_tags
    for cell in range(len(solver.triangles)):
        a.x.array[W.dofmap.cell_dofs(cell)] = alpha[cell]
        n.x.array[W.dofmap.cell_dofs(cell)] = refractive[cell]
    problem = solver.build_problem(alpha,refractive,1.7,.3)
    u,v = ufl.TrialFunction(solver.V),ufl.TestFunction(solver.V)
    dx = ufl.Measure('dx',domain=solver.mesh)
    ds = ufl.Measure('ds',domain=solver.mesh,subdomain_data=solver.ft)
    expressions = (
        (ufl.inner(a*ufl.grad(u),ufl.grad(v))*dx,problem.K),
        (ufl.inner(n*u,v)*dx,problem.Mn),
        (ufl.inner(u,v)*dx,problem.M0),
        (sum(ufl.inner(u,v)*ds(tag) for tag in (11,12,13,14)),problem.B),
    )
    for expression,actual in expressions:
        matrix = petsc.assemble_matrix(fem.form(expression))
        matrix.assemble()
        indptr,indices,values = matrix.getValuesCSR()
        expected = sparse.csr_matrix((values.copy(),indices.copy(),indptr.copy()),shape=matrix.getSize())
        matrix.destroy()
        assert sparse.linalg.norm(actual-expected)/sparse.linalg.norm(expected) < 1e-11


def test_cellwise_resolution():
    from types import SimpleNamespace
    from neural_operator.benchmark import mesh_resolution
    # High contrast occurs in the small element: a product of global maxima is wrong.
    ds = SimpleNamespace(coordinates=np.array([[0.,0.],[1.,0.],[0.,1.],[2.,0.],[2.1,0.],[2.,.1]]),triangles=np.array([[0,1,2],[3,4,5]]),A_cells=np.ones((1,2)),n_cells=np.array([[1.,100.]]),k0=np.ones(1))
    info = mesh_resolution(ds)
    assert info['kh_max'] == pytest.approx(np.sqrt(2))
    assert info['k_local_max'] == 10
    assert info['resolution_estimate'] == 'cellwise DG0'


@pytest.mark.parametrize('name,value',[('A',0.),('n',-1.),('k0',0.),('medium_id',.5),('split',.5),('A_cells',0.),('n_cells',-1.)])
def test_dataset_rejects_invalid_physics_and_ids(generated,name,value):
    from dataclasses import replace
    arr = getattr(generated,name).astype(float).copy()
    arr.flat[0] = value
    with pytest.raises(ValueError):
        replace(generated,**{name:arr})


def test_generation_preallocation_roundtrip_cache_and_progress(tmp_path,monkeypatch):
    pytest.importorskip('dolfinx')
    import neural_operator.fem as module
    from neural_operator.data import load_dataset
    config = module.DatasetConfig(n_media=3,n_directions=2,h=.3,n_coarse=2)
    messages = []
    # Fresh generation must return its existing arrays without a second load.
    def unexpected_load(path):
        raise AssertionError('Fresh generation must not reload the archive')
    monkeypatch.setattr(module,'load_dataset',unexpected_load)
    fresh = module.generate_dataset(tmp_path,config,progress=messages.append)
    loaded = load_dataset(tmp_path)
    for key in ('A','n','A_cells','n_cells','u_scat','theta','k0','medium_id','split','coordinates','triangles'):
        assert np.array_equal(getattr(fresh,key),getattr(loaded,key))
    assert fresh.A.dtype == np.float64
    assert fresh.A_cells.dtype == np.float64
    assert fresh.u_scat.dtype == np.complex128
    assert messages[0].endswith('0/6 fields')
    assert messages[-1].endswith('3/3 media, 6/6 fields')
    assert not (tmp_path/'fem_dataset.tmp.npz').exists()
    monkeypatch.setattr(module,'load_dataset',load_dataset)
    def unexpected_mesh(*args,**kwargs):
        raise AssertionError('Cached dataset must not rebuild its mesh')
    monkeypatch.setattr(module,'generate_mesh',unexpected_mesh)
    cached_messages = []
    cached = module.generate_dataset(tmp_path,config,progress=cached_messages.append)
    assert np.array_equal(cached.u_scat,fresh.u_scat)
    assert cached_messages == []
