import numpy as np
import pytest
from experiments.svd_sno_mesh_stability_2026_11_16.report import NativeEvaluator, physical_errors, quadrature


def evaluator(degree):
    vertices=np.array([[0.,0.],[1.,0.],[0.,1.]])
    indices=np.array([(a,b,degree-a-b) for a in range(degree+1) for b in range(degree+1-a)])
    nodes=indices@vertices/degree
    return NativeEvaluator(vertices,np.array([[0,1,2]]),nodes,np.array([np.arange(len(nodes))]),degree),vertices

@pytest.mark.parametrize('degree',[2,3])
def test_exact_native_polynomial(degree):
    e,vertices=evaluator(degree)
    def f(x): return 1+2*x[:,0]+3j*x[:,1]+x[:,0]*x[:,1]
    points=np.array([[.1,.2],[.4,.2],[.01,.93]])
    np.testing.assert_allclose(e.evaluate_prepared(f(e.coordinates),e.prepare(points)),f(points),atol=1e-13)
    err=physical_errors({'a':e,'b':e},{'a':2*f(e.coordinates),'b':f(e.coordinates)}, {'x':('a','b')},vertices,np.array([[0,1,2]]),chunk_cells=1)
    assert err['x']==pytest.approx(1.)

@pytest.mark.parametrize('degree',[5,7])
def test_quadrature_polynomial_exactness(degree):
    b,w=quadrature(degree)
    from math import factorial
    for a in range(degree+1):
        for c in range(degree+1-a):
            exact=2*factorial(a)*factorial(c)/factorial(a+c+2)
            assert np.dot(w,b[:,1]**a*b[:,2]**c)==pytest.approx(exact,abs=2e-14)


def test_dolfinx_native_cg2_matches_evaluator():
    pytest.importorskip('dolfinx')
    from mpi4py import MPI
    from dolfinx import fem, mesh
    domain=mesh.create_unit_square(MPI.COMM_SELF,3,3)
    p1=fem.functionspace(domain,('Lagrange',1)); p2=fem.functionspace(domain,('Lagrange',2))
    n=domain.topology.index_map(2).size_local
    vt=np.array([p1.dofmap.cell_dofs(c) for c in range(n)])
    ht=np.array([p2.dofmap.cell_dofs(c) for c in range(n)])
    xy=p1.tabulate_dof_coordinates()[:,:2]
    e=NativeEvaluator(xy,vt,p2.tabulate_dof_coordinates()[:,:2],ht,2)
    f=fem.Function(p2); rng=np.random.default_rng(14)
    f.x.array[:]=rng.standard_normal(len(f.x.array))
    points=np.einsum('j,cjd->cd',[.2,.3,.5],xy[vt])
    expected=f.eval(np.column_stack((points,np.zeros(len(points)))),np.arange(n,dtype=np.int32)).reshape(-1)
    np.testing.assert_allclose(e.evaluate_prepared(f.x.array,e.prepare(points)),expected,atol=1e-12)
