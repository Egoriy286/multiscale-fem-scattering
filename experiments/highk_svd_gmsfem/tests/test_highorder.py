"""Verify native CGp evaluation preserves a polynomial across cells."""
import numpy as np
import pytest
pytest.importorskip('dolfinx')
from experiments.highk_svd_gmsfem.highorder import HighOrderSolver
from experiments.highk_svd_gmsfem.mesh import HERE


def test_native_quadratic_evaluation_and_boundary():
    mesh=HERE/'meshes/graded_0.msh'
    if not mesh.exists():pytest.skip('Pilot mesh not generated')
    solver=HighOrderSolver(mesh,degree=2)
    dofs=solver.V.tabulate_dof_coordinates()[:,:2]
    field=1+dofs[:,0]**2+2j*dofs[:,1]
    points=np.array([[-2.,-2.],[2.,2.],[.125,.375],[-1.2,.8]])
    assert np.allclose(solver.evaluate(field,points),1+points[:,0]**2+2j*points[:,1],rtol=1e-12,atol=1e-12)
