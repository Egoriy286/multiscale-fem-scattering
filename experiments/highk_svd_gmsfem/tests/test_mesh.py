"""Convergence diagnostics must compare native fields with a shared integral."""
import numpy as np
from experiments.highk_svd_gmsfem.mesh import compare_meshes


def test_common_quadrature_exact_affine_field(tmp_path):
    coarse=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    fine=np.vstack([coarse,[[.5,.5]]])
    tri0=np.array([[0,1,2],[0,2,3]])
    tri1=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
    def write(name,coords,tri,offset):
        value=(1+coords[:,0]+2j*coords[:,1]+offset)[None]
        path=tmp_path/name
        np.savez(path,coordinates=coords,triangles=tri,fields=value,frequencies=[4.],theta=0.)
        return path
    a=write('a.npz',coarse,tri0,0)
    b=write('b.npz',fine,tri1,0)
    result=compare_meshes(a,b,chunk_cells=1)[0]
    assert result['scattered_relative_l2']<1e-14
    shifted=write('c.npz',coarse,tri0,1.)
    result=compare_meshes(shifted,b,chunk_cells=2)[0]
    # Integral |1+x+2iy|² dxdy = 7/3 + 4/3 = 11/3.
    assert np.isclose(result['scattered_relative_l2'],np.sqrt(3/11))
    assert result['coarse_nodes']==4 and result['fine_nodes']==5
