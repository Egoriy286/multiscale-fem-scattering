from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from experiments.highk_svd_gmsfem.reduced import ReducedSystem,nested_dissection,project_matrix,apply_real


@pytest.mark.parametrize('n',[1,2,5,20])
@pytest.mark.parametrize('width',[1,2])
def test_nested_dissection_is_complete(n,width):
    permutation=nested_dissection(n,3,width)
    assert np.array_equal(np.sort(permutation),np.arange((n+1)**2*3))


@pytest.mark.parametrize('ordering',['COLAMD','ND','ND2','ND_NOPIVOT'])
def test_exact_galerkin_equations_and_real_complex_parity(ordering):
    rng=np.random.default_rng(22);P=sparse.csc_matrix(rng.normal(size=(31,18)))
    stiffness=rng.normal(size=(31,31));stiffness=(stiffness+stiffness.T)/2
    H=sparse.csr_matrix(stiffness-2j*np.eye(31));rhs=rng.normal(size=31)+1j*rng.normal(size=31)
    reference=(P.T@(H@P)).toarray()
    assert np.allclose(project_matrix(H,P).toarray(),reference,rtol=1e-13,atol=1e-13)
    problem=SimpleNamespace(operator=H,rhs=rhs)
    space=SimpleNamespace(n_coarse=2,prolongation=lambda m:P)
    system=ReducedSystem(problem,space,2,ordering=ordering)
    prediction,timing=system.solve(problem)
    expected=P@np.linalg.solve(reference,P.T@rhs)
    assert np.allclose(prediction,expected,rtol=1e-10,atol=1e-10)
    assert timing['galerkin_residual']<1e-10
    assert np.allclose(apply_real(P,1j*np.ones(18)),P@(1j*np.ones(18)))
