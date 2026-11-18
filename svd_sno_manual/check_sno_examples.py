"""Check the SNO chapter's algebra examples; no author-network training."""
from pathlib import Path
import json
import numpy as np


def main():
    U=np.array([[1.,0.],[0.,2.]])
    A=np.array([[1.,-1.],[1.,1.]])
    B=np.array([[1.,1.],[1.,-1.]])
    bias=np.array([[0.,0.],[1.,0.]])
    UA=U@A; BUA=B@UA; Z=BUA+bias
    activated=np.logaddexp(0,Z)
    np.testing.assert_allclose(UA,[[1,-1],[2,2]])
    np.testing.assert_allclose(BUA,[[3,1],[-1,-3]])
    np.testing.assert_allclose(Z,[[3,1],[0,-3]])
    np.testing.assert_allclose(np.kron(A.T,B)@U.ravel(order='F'),BUA.ravel(order='F'))
    params=sum(a.size for a in (A,B,bias))
    assert params==12
    x=np.linspace(-1,1,101)
    cheb=1.5+x+.5*(2*x*x-1)
    cheb_error=float(abs(cheb-(1+x+x*x)).max())
    assert cheb_error<1e-14
    frequencies=np.arange(-2,3)
    d=np.array([.2+.1j,-.3+.2j,1.,.4-.1j,-.2+.3j])
    nodes=-1+2*np.arange(5)/5
    vals=np.exp(1j*np.pi*np.outer(nodes,frequencies))@d
    recovered=vals@np.exp(-1j*np.pi*np.outer(nodes,frequencies))/5
    fourier_error=float(abs(recovered-d).max())
    assert fourier_error<1e-14
    result=dict(description='Manually chosen SNO algebra examples; not trained paper results',
                input=U.tolist(),channel=A.tolist(),mode=B.tolist(),bias=bias.tolist(),
                UA=UA.tolist(),BUA=BUA.tolist(),affine_output=Z.tolist(),
                activated_internal_output=activated.tolist(),real_layer_parameters=params,
                dense_real_parameters=20,chebyshev_max_error=cheb_error,
                fourier_coefficient_max_error=fourier_error,
                softplus_linear_branch_exact_error_bound=float(np.exp(-35)))
    target=Path(__file__).parent/'assets/original_sno_example.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
