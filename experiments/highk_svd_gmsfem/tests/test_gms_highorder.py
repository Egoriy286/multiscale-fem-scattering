"""Native Docker tests: exact CG2 assembly, traces, local solve, scoped dispatch."""
import numpy as np
import pytest
pytest.importorskip('dolfinx')
from scipy import sparse
from gmsfem.fem import solve_dirichlet
from experiments.highk_svd_gmsfem.gms_highorder import P2Backend,local_matrices,build_offline_space


@pytest.fixture(scope='module')
def backend(tmp_path_factory):
    import gmsh
    path=tmp_path_factory.mktemp('cg2')/'square.msh'
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0)
        points=[gmsh.model.geo.addPoint(x,y,0,.6) for x,y in [(-2,-2),(2,-2),(2,2),(-2,2)]]
        lines=[gmsh.model.geo.addLine(points[i],points[(i+1)%4]) for i in range(4)]
        loop=gmsh.model.geo.addCurveLoop(lines);surface=gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize();gmsh.model.addPhysicalGroup(2,[surface],1)
        for tag,line in zip((11,12,13,14),lines):gmsh.model.addPhysicalGroup(1,[line],tag)
        gmsh.model.mesh.generate(2);gmsh.write(str(path))
    finally:gmsh.finalize()
    return P2Backend(path)


def test_cg2_element_global_equivalence_and_midpoint_boundary(backend):
    assert max(backend.assembly_errors.values())<1e-10
    problem=backend.problem(2.)
    assert problem.triangles.shape[1]==6
    local=local_matrices(problem,np.arange(len(problem.triangles)))
    xy=problem.coordinates[local.dofs]
    expected=np.flatnonzero(np.any(np.isclose(abs(xy),2.),axis=1))
    np.testing.assert_array_equal(local.boundary,expected)
    vertices=np.unique(problem.triangles[:,:3])
    assert np.count_nonzero(~np.isin(local.dofs[local.boundary],vertices))>0
    assert abs(problem.M0.sum()-16)<1e-11
    assert sparse.linalg.norm(local.K-problem.K)<1e-10


def test_local_cg2_affine_harmonic_extension_uses_only_patch_elements(backend):
    problem=backend.problem(2.)
    cells=np.flatnonzero(problem.coordinates[problem.triangles[:,:3]].mean(axis=1)[:,0]<0)
    local=local_matrices(problem,cells)
    audit=backend.validate_local_assembly(cells)
    assert max(audit['relative_assembly_errors'].values())<1e-10
    xy=problem.coordinates[local.dofs];expected=1+xy[:,0]+2*xy[:,1]
    solution,residual=solve_dirichlet(local.K,local.boundary,expected[local.boundary,None])
    np.testing.assert_allclose(solution[:,0],expected,rtol=2e-12,atol=2e-12)
    assert residual<1e-12
    assert sparse.linalg.norm(local.K-problem.K[local.dofs][:,local.dofs])>1e-3


def test_native_p2_evaluation_and_offline_dispatch_restoration(backend,monkeypatch):
    import gmsfem.offline as offline
    problem=backend.problem(2.)
    coordinates=backend.coordinates;field=coordinates[:,0]**2+1j*coordinates[:,1]
    points=np.array([[-2.,-2.],[2.,2.],[.123,.456]])
    np.testing.assert_allclose(backend.evaluate(field,points),points[:,0]**2+1j*points[:,1],atol=3e-12)
    original=offline.local_matrices
    space=build_offline_space(problem,n_coarse=1,max_modes=2,basis_type=1,progress=None)
    assert space.P.shape==(problem.ndofs,8)
    assert offline.local_matrices is original
    assert space.diagnostics['fine_degree']==2
    def fail(*args,**kwargs):raise RuntimeError('deliberate failure')
    monkeypatch.setattr(offline,'build_offline_space',fail)
    with pytest.raises(RuntimeError,match='deliberate'):
        build_offline_space(problem,n_coarse=1,max_modes=2,basis_type=2,progress=None)
    assert offline.local_matrices is original
