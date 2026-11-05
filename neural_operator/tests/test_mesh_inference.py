"""Frozen-model inference across physical meshes, without training or FEM."""
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial import Delaunay
import torch

from neural_operator.basis import PODBasis
from neural_operator.mesh_transfer import P1Transfer
from neural_operator.training import Predictor


def geometry():
    source = np.array([[-2.,-2.], [2.,-2.], [2.,2.], [-2.,2.], [.3,-.4]])
    triangles = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
    target = np.array([[-2.,-2.], [2.,-2.], [2.,2.], [-2.,2.],
                       [-.7,.1], [.8,.5], [-.3,1.1]])
    return source, triangles, target, Delaunay(target).simplices.copy()


def predictor_fixture(static=True, method="svd"):
    torch.manual_seed(31)
    rng = np.random.default_rng(91)
    source, triangles, _, _ = geometry()
    x, y = source.T
    if method != "svd":
        prep = {"resolution":8, "A_stats":{"mean":1.5,"std":.3},
                "n_stats":{"mean":1.2,"std":.2}, "k_stats":{"mean":1.5,"std":.4},
                "output_mean":rng.normal(size=(5,2)).tolist(), "output_scale":[.3,.8]}
        kwargs = {"input_channels":7,"modes":3,"width":4,"depth":1}
        if method == "fno":
            kwargs["padding"] = 2
        return Predictor(method, kwargs, prep, source, triangles)
    a = np.broadcast_to(1.5+.1*x, (15,5)).copy()
    n = np.broadcast_to(1.3+.07*y, (15,5)).copy()
    if not static:
        a += rng.normal(size=(15,1))*.08*x + rng.normal(size=(15,1))*.06*y
        n += rng.normal(size=(15,1))*.04*y
    bases = {"A":PODBasis.fit(a,3), "n":PODBasis.fit(n,3),
             "real":PODBasis.fit(rng.normal(size=(15,5)),3),
             "imag":PODBasis.fit(rng.normal(size=(15,5)),2)}
    inputs = bases["A"].rank+bases["n"].rank
    prep = {"bases":{key:b.to_dict() for key,b in bases.items()}, "rank":3,
            "input_stats":{"mean":np.linspace(-.1,.2,inputs).tolist(),
                           "std":np.linspace(.3,1.2,inputs).tolist()},
            "output_stats":{"mean":[[.3,-.7],[.8,.1],[.2,0.]],
                            "std":[[.4,1.2],[.6,.8],[1.3,1.]]},
            "k_stats":{"mean":1.5,"std":.4}}
    return Predictor("svd", {"input_dim":inputs+3,"output_modes":3,"width":8,"depth":2},
                     prep, source, triangles)


def affine_inputs(points):
    x, y = points.T
    amplitudes = np.array([-.3,.4,1.])[:,None]
    return (1.6+.1*x+amplitudes*.07*y,
            1.25+.05*y+amplitudes*.03*x)


class MeshInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_static_scalar_and_batch_equal_transferred_source_and_frozen_weights(self):
        source = predictor_fixture()
        before = {key:value.clone() for key,value in source.model.state_dict().items()}
        _, _, target, _ = geometry()
        mapped = source.on_mesh(target)  # Static inference needs no target connectivity.
        theta, k = np.array([.2,1.7,4.]), 1.9
        A = np.broadcast_to(source.bases["A"].mean, (3,5))
        n = np.broadcast_to(source.bases["n"].mean, (3,5))
        expected = P1Transfer(source.coordinates,source.triangles,target).apply(
            source.predict(A,n,theta,k,batch_size=2))
        actual = mapped.predict_static(theta,k,batch_size=2)
        np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=2e-6)
        single = mapped.predict_static(theta[0],k)
        self.assertEqual(single.shape, (len(target),))
        np.testing.assert_allclose(single, actual[0], atol=3e-7, rtol=2e-6)
        for key,value in source.model.state_dict().items():
            torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_permutation_matches_source_node_order(self):
        source = predictor_fixture(static=False)
        permutation = np.array([4,2,0,3,1])
        inverse = np.argsort(permutation)
        mapped = source.on_mesh(source.coordinates[permutation],inverse[source.triangles])
        A,n = affine_inputs(source.coordinates)
        theta,k = np.array([.2,1.7,4.]), np.array([1.1,1.9,2.3])
        expected = source.predict(A,n,theta,k)
        actual = mapped.predict(A[:,permutation],n[:,permutation],theta,k)
        np.testing.assert_allclose(actual,expected[:,permutation],atol=1e-7,rtol=1e-7)

    def test_variable_affine_inputs_keep_original_coefficients(self):
        source = predictor_fixture(static=False)
        self.assertGreater(source.bases["A"].rank,0)
        self.assertGreater(source.bases["n"].rank,0)
        _,_,target,triangles = geometry()
        mapped = source.on_mesh(target,triangles)
        A,n = affine_inputs(target)
        expected_A,expected_n = affine_inputs(source.coordinates)
        np.testing.assert_allclose(mapped.to_source.apply(A),expected_A,atol=1e-14)
        np.testing.assert_allclose(mapped.to_source.apply(n),expected_n,atol=1e-14)
        theta,k = np.array([.2,1.7,4.]), np.array([1.1,1.9,2.3])
        encoded = source.encode(mapped.to_source.apply(A),mapped.to_source.apply(n),theta,k)
        np.testing.assert_allclose(encoded,source.encode(expected_A,expected_n,theta,k),atol=1e-7)
        expected = mapped.to_target.apply(source.predict(expected_A,expected_n,theta,k))
        actual = mapped.predict(A,n,theta,k)
        np.testing.assert_allclose(actual,expected,atol=3e-7,rtol=2e-6)
        single = mapped.predict(A[0],n[0],theta[0],k[0])
        np.testing.assert_allclose(single,actual[0],atol=3e-7,rtol=2e-6)

    def test_dataset_requires_explicit_mesh_transfer_and_matches_wrapper(self):
        source = predictor_fixture(static=False)
        _,_,target,triangles = geometry()
        A,n = affine_inputs(target)
        dataset = SimpleNamespace(A=A,n=n,theta=np.array([.2,1.7,4.]),
                                  k0=np.array([1.1,1.9,2.3]),coordinates=target,triangles=triangles)
        with self.assertRaises(ValueError):
            source.predict_dataset(dataset)
        mapped = source.on_mesh(target,triangles)
        indices = np.array([2,0])
        expected = mapped.predict_dataset(dataset,indices,batch_size=1)
        actual = source.predict_dataset(dataset,indices,batch_size=1,allow_mesh_transfer=True)
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-7)
        dataset.coordinates = target.copy()
        dataset.coordinates[-1,0] += .01
        with self.assertRaises(ValueError):
            mapped.predict_dataset(dataset)

    def test_invalid_target_or_static_use_is_rejected(self):
        source = predictor_fixture(static=False)
        _,_,target,triangles = geometry()
        with self.assertRaises(ValueError):
            source.on_mesh([[2.1,0.]])
        with self.assertRaises(ValueError):
            source.on_mesh(target,[[0,1,100]])
        with self.assertRaises(ValueError):
            source.on_mesh(target,triangles).predict_static(.3,1.7)
        A,n = affine_inputs(target)
        with self.assertRaises(ValueError):
            source.on_mesh(target).predict(A,n,[.2,1.7,4.],1.9)
        with self.assertRaises(ValueError):
            source.on_mesh(target,triangles).predict(A[:,:-1],n[:,:-1],[.2,1.7,4.],1.9)

    def test_grid_operator_fallback_transfers_decoded_source_field(self):
        _,_,target,triangles = geometry()
        A,n = affine_inputs(target)
        theta,k = np.array([.2,1.7,4.]), np.array([1.1,1.9,2.3])
        for method in ("fno","chebyshev"):
            with self.subTest(method=method):
                source = predictor_fixture(method=method)
                mapped = source.on_mesh(target,triangles)
                source_A,source_n = affine_inputs(source.coordinates)
                expected = mapped.to_target.apply(source.predict(source_A,source_n,theta,k,batch_size=2))
                actual = mapped.predict(A,n,theta,k,batch_size=2)
                np.testing.assert_allclose(actual,expected,atol=5e-7,rtol=2e-6)
                with self.assertRaises(ValueError):
                    mapped.predict_static(.2,1.9)


if __name__ == "__main__":
    unittest.main()
