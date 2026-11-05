"""Numerical P1 transfer checks without a FEM runtime."""
import unittest

import numpy as np
from neural_operator.mesh_transfer import P1Transfer


def irregular_square():
    coordinates = np.array([[-2.,-2.], [2.,-2.], [2.,2.], [-2.,2.], [.3,-.4]])
    triangles = np.array([[0,1,4], [1,2,4], [2,3,4], [3,0,4]])
    return coordinates, triangles


class P1TransferTests(unittest.TestCase):
    def test_affine_fields_boundary_and_sparse_partition(self):
        coordinates, triangles = irregular_square()
        points = np.array([[-2.,-2.], [2.,2.], [-2.,.7], [.2,2.],
                           [1.2,-.3], [-.8,.1], [.3,-.4]])
        transfer = P1Transfer(coordinates, triangles, points)
        fields = np.stack((np.ones(5), 2*coordinates[:,0]-3*coordinates[:,1]+.9))
        expected = np.stack((np.ones(len(points)), 2*points[:,0]-3*points[:,1]+.9))
        np.testing.assert_allclose(transfer.apply(fields), expected, atol=1e-13)
        np.testing.assert_allclose(np.asarray(transfer.matrix.sum(axis=1)).ravel(), 1., atol=1e-14)
        self.assertTrue(np.all(np.diff(transfer.matrix.indptr) <= 3))

    def test_piecewise_linear_hat_and_complex_batch(self):
        coordinates, triangles = irregular_square()
        barycentric = np.array([[.2,.3,.5], [.6,.15,.25], [.1,.8,.1], [.3,.4,.3]])
        points = np.einsum("ij,ijk->ik", barycentric, coordinates[triangles])
        transfer = P1Transfer(coordinates, triangles, points)
        # Interior-node hat is piecewise affine, with a different gradient in
        # each actual source triangle. Its value is the third barycentric weight.
        hat = np.array([0.,0.,0.,0.,1.])
        np.testing.assert_allclose(transfer.apply(hat), barycentric[:,2], atol=1e-14)
        complex_field = hat + 1j*(2*coordinates[:,0]+coordinates[:,1])
        batch = np.stack((complex_field, 2*complex_field+3)).reshape(1,2,5)
        result = transfer.apply(batch)
        expected = barycentric[:,2] + 1j*(2*points[:,0]+points[:,1])
        np.testing.assert_allclose(result[0,0], expected, atol=1e-14)
        np.testing.assert_allclose(result[0,1], 2*expected+3, atol=1e-14)

    def test_identity_and_permuted_source_order(self):
        coordinates, triangles = irregular_square()
        transfer = P1Transfer(coordinates, triangles[:,::-1], coordinates)
        np.testing.assert_allclose(transfer.matrix.toarray(), np.eye(5), atol=1e-12)
        permutation = np.array([4,2,0,3,1])
        inverse = np.argsort(permutation)
        permuted = P1Transfer(coordinates[permutation], inverse[triangles], coordinates)
        field = np.array([.2,1.,-3.,7.,2.])
        np.testing.assert_allclose(permuted.apply(field[permutation]), field, atol=1e-12)

    def test_coarser_round_trip_loses_hat(self):
        fine, fine_triangles = irregular_square()
        coarse = fine[:4]
        coarse_triangles = np.array([[0,1,2], [0,2,3]])
        down = P1Transfer(fine, fine_triangles, coarse)
        up = P1Transfer(coarse, coarse_triangles, fine)
        hat = np.array([0.,0.,0.,0.,1.])
        recovered = up.apply(down.apply(hat))
        np.testing.assert_allclose(recovered, 0., atol=1e-14)
        self.assertEqual(float(np.max(np.abs(recovered-hat))), 1.)

    def test_outside_and_hole_are_rejected(self):
        coordinates, triangles = irregular_square()
        with self.assertRaises(ValueError):
            P1Transfer(coordinates, triangles, [[2.01,0.]])
        # Ring triangulation around an open central square.
        outer = np.array([[-2.,-2.], [2.,-2.], [2.,2.], [-2.,2.]])
        inner = outer*.3
        ring = np.concatenate((outer,inner))
        ring_triangles = []
        for j in range(4):
            k = (j+1)%4
            ring_triangles.extend(((j,k,k+4), (j,k+4,j+4)))
        with self.assertRaises(ValueError):
            P1Transfer(ring, ring_triangles, [[0.,0.]])
        accepted = P1Transfer(ring, ring_triangles, [[1.,0.], [.6,0.]])
        np.testing.assert_allclose(accepted.apply(np.ones(8)), 1.)

    def test_boundary_segment_roundoff_inside_bounding_box(self):
        # Top edge has a rounding-sized indentation, while its endpoints keep
        # the bbox at y=2. Clipping to that bbox leaves these queries outside.
        coordinates = np.array([[-2.,-2.], [2.,-2.], [2.,2.],
                                [0.,2.-1e-13], [-2.,2.], [.3,-.4]])
        triangles = np.array([[j,(j+1)%5,5] for j in range(5)])
        points = np.array([[0.,2.], [.7,2.], [-1.1,2.]])
        transfer = P1Transfer(coordinates,triangles,points)
        field = 2*coordinates[:,0]-3*coordinates[:,1]+.4
        np.testing.assert_allclose(transfer.apply(field),2*points[:,0]-3*points[:,1]+.4,
                                   atol=1e-13)
        np.testing.assert_allclose(np.asarray(transfer.matrix.sum(axis=1)).ravel(),1.,atol=1e-14)
        # A physically different indentation must not be filled by the fallback.
        coordinates[3,1] = 2.-1e-6
        with self.assertRaises(ValueError):
            P1Transfer(coordinates,triangles,points)

    def test_roundoff_bbox_and_invalid_inputs(self):
        coordinates, triangles = irregular_square()
        points = np.array([[2.+1e-13, .2], [-2.-1e-13,-.5]])
        transferred = P1Transfer(coordinates, triangles, points).apply(coordinates[:,0])
        # Original points are evaluated, rather than clipped field values.
        np.testing.assert_allclose(transferred, points[:,0], atol=1e-14)
        for bad_points in ([[np.nan,0.]], [0.,1.], [[1.,2.,3.]]):
            with self.assertRaises(ValueError):
                P1Transfer(coordinates, triangles, bad_points)
        for bad_triangles in ([[0,0,1]], [[0,1,9]], [[0.,1.,4.]]):
            with self.assertRaises(ValueError):
                P1Transfer(coordinates, bad_triangles, [[0.,0.]])


if __name__ == "__main__":
    unittest.main()
