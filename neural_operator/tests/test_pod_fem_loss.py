"""Compare the compressed physical objective with full FEM reconstruction."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import Delaunay
import torch

from neural_operator.basis import PODBasis
from neural_operator.data import mass_matrix
from neural_operator.losses import PODFEMLoss
from neural_operator.training import FEMLoss, Predictor


def fixture(constant_imaginary=False):
    rng = np.random.default_rng(71)
    coordinates = np.array([[-2.,-2.], [2.,-2.], [2.,2.], [-2.,2.],
                            [-.9,-.4], [.7,.3], [-.6,1.1]])
    triangles = Delaunay(coordinates).simplices.copy()
    mass = mass_matrix(coordinates, triangles)
    lumped = np.asarray(mass.sum(axis=1)).ravel()
    bases = {
        "real": PODBasis.fit(rng.normal(size=(25, 7))+.3, 2, weights=lumped),
        "imag": PODBasis.fit(np.full((25, 7), 1.75) if constant_imaginary else
                              rng.normal(size=(25, 7))-.4, 1, weights=lumped),
        "A": PODBasis.fit(np.ones((2, 7)), 0),
        "n": PODBasis.fit(np.ones((2, 7)), 0),
    }
    preprocessing = {"bases": {k:b.to_dict() for k,b in bases.items()}, "rank": 2,
                     "output_stats": {"mean": [[1.2,-.8],[.7,10.]],
                                      "std": [[.3,2.1],[3.4,.05]]}}
    predictor = Predictor("svd", {"input_dim":3,"output_modes":2,"width":4,"depth":1},
                          preprocessing, coordinates, triangles)
    coefficients = torch.tensor(rng.normal(size=(19,2,2)), dtype=torch.float32)
    targets = predictor.decode(coefficients) + torch.tensor(
        .07*rng.normal(size=(19,7,2)), dtype=torch.float32)
    if constant_imaginary:
        targets[..., 1] = 1.75
    dataset = SimpleNamespace(mass_matrix=mass, coordinates=coordinates, triangles=triangles)
    return predictor, dataset, targets


class PODFEMLossTests(unittest.TestCase):
    def test_values_gradients_nonorthogonal_mass_and_subset(self):
        torch.manual_seed(8)
        predictor, dataset, target = fixture()
        compressed = PODFEMLoss(predictor, dataset, target)
        self.assertFalse(torch.allclose(compressed.gram[:, :, 0], torch.eye(2, dtype=torch.float64)))
        self.assertTrue(torch.all(compressed.floor_squared > 0))
        indices = torch.tensor([15,2,8,2,0])
        coefficients = torch.randn(5,2,2, requires_grad=True)
        direct_coefficients = coefficients.detach().clone().requires_grad_()
        actual = compressed(coefficients, indices)
        expected = FEMLoss(dataset.coordinates, dataset.triangles)(
            predictor.decode(direct_coefficients), target[indices])
        actual.backward()
        expected.backward()
        torch.testing.assert_close(actual.float(), expected, atol=2e-7, rtol=1e-5)
        torch.testing.assert_close(coefficients.grad, direct_coefficients.grad, atol=2e-7, rtol=1e-5)
        self.assertEqual(compressed.projected_coefficients.shape, (19,2,2))
        self.assertTrue(all(b.device.type == "cpu" for b in compressed.buffers()))

    def test_constant_component_zero_rank_and_numpy_targets(self):
        predictor, dataset, target = fixture(constant_imaginary=True)
        compressed = PODFEMLoss(predictor, dataset, target.numpy())
        self.assertEqual(predictor.bases["imag"].rank, 0)
        self.assertEqual(torch.count_nonzero(compressed.gram[...,1]), 0)
        self.assertEqual(torch.count_nonzero(compressed.projected_coefficients[...,1]), 0)
        coefficients = torch.randn(3,2,2, requires_grad=True)
        indices = torch.tensor([1,7,13])
        actual = compressed(coefficients, indices)
        expected = FEMLoss(dataset.coordinates, dataset.triangles)(predictor.decode(coefficients), target[indices])
        torch.testing.assert_close(actual.float(), expected, atol=2e-7, rtol=1e-5)
        actual.backward()
        torch.testing.assert_close(coefficients.grad[...,1], torch.zeros_like(coefficients.grad[...,1]))

    def test_complex_arrays_and_memmaps_match_rounded_targets(self):
        predictor, dataset, target = fixture()
        components = target.numpy().astype(np.float64) + 1e-9
        complex_targets = np.tile(components[...,0] + 1j*components[...,1], (29,1))
        indices = torch.tensor([550,256,0,513,256])
        expected_targets = torch.tensor(np.stack((complex_targets.real, complex_targets.imag), axis=-1), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.npy"
            np.save(path, complex_targets)
            mapped = np.load(path, mmap_mode="r")
            self.assertIsInstance(mapped, np.memmap)
            for source in (complex_targets, mapped):
                with self.subTest(memmap=isinstance(source, np.memmap)):
                    loss = PODFEMLoss(predictor, dataset, source, target_dtype=np.float32)
                    coefficients = torch.randn(len(indices),2,2, requires_grad=True)
                    direct_coefficients = coefficients.detach().clone().requires_grad_()
                    actual = loss(coefficients, indices)
                    expected = FEMLoss(dataset.coordinates, dataset.triangles)(
                        predictor.decode(direct_coefficients), expected_targets[indices])
                    actual.backward()
                    expected.backward()
                    torch.testing.assert_close(actual.float(), expected, atol=2e-7, rtol=1e-5)
                    torch.testing.assert_close(coefficients.grad, direct_coefficients.grad, atol=2e-7, rtol=1e-5)
                    self.assertEqual(loss.projected_coefficients.shape, (551,2,2))
                    self.assertFalse(any(tuple(buffer.shape) == source.shape for buffer in loss.buffers()))

    def test_tiny_residual_energy_is_integrated_without_cancellation(self):
        predictor, dataset, _ = fixture()
        rng = np.random.default_rng(5)
        normalized = torch.tensor(rng.normal(size=(4,2,2)), dtype=torch.float32)
        stats = predictor.preprocessing["output_stats"]
        coefficients = normalized * torch.tensor(stats["std"]) + torch.tensor(stats["mean"])
        # A double-precision oracle uses precisely the rounded decoder buffers,
        # isolating cancellation in residual energies from float32 field sums.
        phi = predictor.output_modes.double()
        mean = predictor.output_mean.double()
        target = mean + torch.einsum("brc,nrc->bnc", coefficients.double(), phi)
        target += torch.tensor(1e-10*rng.normal(size=target.shape), dtype=torch.float64)
        compressed = PODFEMLoss(predictor, dataset, target)
        self.assertTrue(torch.all(compressed.floor_squared >= 0))
        self.assertTrue(torch.all(compressed.floor_squared < 1e-16))
        # At a small parameter displacement, the accurate quadratic form
        # agrees with sparse full-field integration down to near-zero error.
        displaced = (normalized + 1e-5).requires_grad_()
        actual = compressed(displaced, torch.arange(4))
        denorm = displaced * compressed.std + compressed.mean
        error = mean + torch.einsum("brc,nrc->bnc", denorm.double(), phi) - target
        matrix = torch.tensor(dataset.mass_matrix.toarray(), dtype=torch.float64)
        error2 = torch.einsum("bnc,nm,bmc->b", error, matrix, error)
        norm2 = torch.einsum("bnc,nm,bmc->b", target, matrix, target)
        expected = torch.sqrt(error2.clamp_min(1e-24)/norm2.clamp_min(1e-24)).mean()
        torch.testing.assert_close(actual, expected, atol=2e-15, rtol=1e-8)
        actual.backward()
        self.assertTrue(torch.isfinite(displaced.grad).all())

    def test_zero_target_clamping_matches_direct_loss(self):
        predictor, dataset, target = fixture()
        predictor.output_mean.zero_()
        target.zero_()
        predictor.preprocessing["output_stats"]["mean"] = [[0.,0.],[0.,0.]]
        compressed = PODFEMLoss(predictor, dataset, target)
        coefficients = torch.zeros(2,2,2, requires_grad=True)
        indices = torch.tensor([0,1])
        actual = compressed(coefficients, indices)
        expected = FEMLoss(dataset.coordinates, dataset.triangles)(predictor.decode(coefficients), target[indices])
        torch.testing.assert_close(actual.float(), expected)
        self.assertEqual(float(actual.detach()), 0.)
        actual.backward()
        self.assertTrue(torch.isfinite(coefficients.grad).all())


if __name__ == "__main__":
    unittest.main()
