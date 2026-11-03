"""Numerical operator tests, runnable with python -m unittest discover."""
import io
import json
import unittest
import numpy as np
import torch
from neural_operator.basis import PODBasis
from neural_operator.grid import GridTransfer
from neural_operator.models import SVDSNO, ChebyshevSNO, FNO2d, SpectralConv2d, chebyshev_matrices


class PODTests(unittest.TestCase):
    def test_weighted_reconstruction_and_serialization(self):
        rng = np.random.default_rng(4)
        fields = rng.normal(size=(12, 19))
        weights = rng.uniform(.2, 3, 19)
        basis = PODBasis.fit(fields, 19, weights=weights)
        np.testing.assert_allclose(basis.modes.T @ (weights[:, None] * basis.modes), np.eye(basis.rank), atol=1e-12)
        np.testing.assert_allclose(basis.decode(basis.encode(fields)), fields, atol=1e-12)
        self.assertAlmostEqual(basis.energy[-1], 1)
        restored = PODBasis.from_dict(json.loads(json.dumps(basis.to_dict())))
        np.testing.assert_allclose(restored.decode(restored.encode(fields)), fields, atol=1e-12)

    def test_roundoff_noise_does_not_become_pod_modes(self):
        rng = np.random.default_rng(33)
        noisy_constant = 2 + 1e-15 * rng.standard_normal((40, 70))
        self.assertEqual(PODBasis.fit(noisy_constant, 20).rank, 0)
        # Scalar material fields varying across samples have one physical mode.
        scalar_fields = rng.uniform(.5, 2, (40, 1)) * np.ones((1, 70))
        scalar_fields += 1e-15 * rng.standard_normal(scalar_fields.shape)
        self.assertEqual(PODBasis.fit(scalar_fields, 20).rank, 1)
        # A small field with zero mean still retains its relative variations.
        small_fields = 1e-18 * rng.standard_normal((40, 70))
        self.assertEqual(PODBasis.fit(small_fields, 20).rank, 20)

    def test_constant_and_randomized_total_energy(self):
        constant = np.full((10, 8), 7.3)
        basis = PODBasis.fit(constant, 6)
        self.assertEqual(basis.rank, 0)
        np.testing.assert_allclose(PODBasis.from_dict(basis.to_dict()).decode(np.zeros((2, 0))), constant[:2])
        rng = np.random.default_rng(5)
        data = rng.normal(size=(60, 80))
        basis = PODBasis.fit(data, 6, method="randomized")
        self.assertAlmostEqual(basis.total_energy, np.sum((data-data.mean(0))**2), places=10)
        self.assertLess(basis.energy[-1], 1)
        np.testing.assert_allclose(basis.modes.T @ basis.modes, np.eye(6), atol=1e-12)


class TransferTests(unittest.TestCase):
    def test_linear_exact_actual_triangles(self):
        nodes = np.array([[-2,-2], [2,-2], [2,2], [-2,2], [.3,.1]])
        triangles = [[0,1,4], [1,2,4], [2,3,4], [3,0,4]]
        values = 2*nodes[:,0]-3*nodes[:,1]+.7
        for kind in ("uniform", "chebyshev"):
            transfer = GridTransfer(nodes, triangles, (13,17), kind)
            result = transfer.to_grid(values[None])
            expected = (2*transfer.points[:,0]-3*transfer.points[:,1]+.7).reshape(1,13,17)
            np.testing.assert_allclose(result, expected, atol=1e-12)
            np.testing.assert_allclose(transfer.to_nodes(result), values[None], atol=1e-12)
            self.assertEqual(transfer.to_nodes_matrix.shape, (5, 13*17))

    def test_mesh_holes_are_not_filled(self):
        with self.assertRaises(ValueError):
            GridTransfer([[-2,-2], [2,-2], [-2,2]], [[0,1,2]], 5)


class ModelTests(unittest.TestCase):
    def test_chebyshev_transform_polynomials(self):
        for n in (2, 9, 16):
            analysis, synthesis = chebyshev_matrices(n, dtype=torch.float64)
            torch.testing.assert_close(analysis @ synthesis, torch.eye(n, dtype=torch.float64), atol=1e-12, rtol=1e-12)
            coefficients = analysis @ torch.ones(n, dtype=torch.float64)
            torch.testing.assert_close(coefficients[0], torch.tensor(1., dtype=torch.float64))
            torch.testing.assert_close(coefficients[1:], torch.zeros(n-1, dtype=torch.float64), atol=1e-12, rtol=1e-12)

    def test_chebyshev_polynomial_coefficients(self):
        n = 11
        nodes = -torch.cos(torch.linspace(0, torch.pi, n, dtype=torch.float64))
        analysis, synthesis = chebyshev_matrices(n, dtype=torch.float64)
        # 3 + 2*x + 4*x^2 - x^3 = 5 + 1.25*T1 + 2*T2 - .25*T3.
        values = 3 + 2*nodes + 4*nodes**2 - nodes**3
        expected = torch.zeros(n, dtype=torch.float64)
        expected[:4] = torch.tensor([5., 1.25, 2., -.25], dtype=torch.float64)
        torch.testing.assert_close(analysis @ values, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(synthesis[:, :4] @ expected[:4], values, atol=1e-12, rtol=1e-12)

    def test_shapes_gradients_and_state_round_trip(self):
        torch.manual_seed(0)
        cases = [(SVDSNO, dict(input_dim=11,output_modes=5,width=8,depth=2), (2,11), (2,5,2)),
                 (ChebyshevSNO, dict(modes=(4,5),width=8,depth=2), (2,9,12,7), (2,9,12,2)),
                 (FNO2d, dict(modes=(4,5),width=8,depth=2,padding=3), (2,9,12,7), (2,9,12,2))]
        for constructor, config, shape, output_shape in cases:
            model = constructor(**config)
            x = torch.randn(shape, requires_grad=True)
            output = model(x)
            self.assertEqual(tuple(output.shape), output_shape)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            output = model(x.detach())
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            restored = constructor(**config)
            restored.load_state_dict(torch.load(stream, weights_only=True))
            torch.testing.assert_close(restored(x.detach()), output.detach())

    def test_fourier_signed_modes_and_no_overlap(self):
        for h,w in ((7,9), (8,10)):
            conv = SpectralConv2d(1, 1, (20,20))
            with torch.no_grad():
                conv.weight_positive.fill_(1)
                conv.weight_negative.fill_(1)
            x = torch.randn(2,1,h,w)
            torch.testing.assert_close(conv(x), x, atol=1e-6, rtol=1e-5)
            conv.zero_grad()
            conv(x).square().mean().backward()
            self.assertGreater(conv.weight_positive.grad.abs().sum(), 0)
            self.assertGreater(conv.weight_negative.grad.abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
