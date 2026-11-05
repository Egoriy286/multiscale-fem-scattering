"""Static-medium regression: parameter groups, zero-rank inputs, portable SVD."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from neural_operator.data import Dataset
from neural_operator.training import TrainConfig, load_predictor, train_model


def static_dataset():
    """Algebraic P1 fields keep this regression independent of a PDE solver."""
    x, y = np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-2, 2, 7))
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    triangles = []
    for j in range(6):
        for i in range(7):
            a = j*8+i
            triangles.extend(((a, a+1, a+9), (a, a+9, a+8)))
    triangles = np.asarray(triangles, dtype=np.int64)
    frequency_id = np.repeat(np.arange(20), 3)
    k0 = np.repeat(np.linspace(.6, 3.2, 20), 3)
    theta = np.tile(np.array([0., 1.2, 3.8]), 20)
    split = np.repeat(np.r_[np.zeros(14), np.ones(3), np.full(3, 2)], 3).astype(np.int8)
    xx, yy = coordinates.T / 2
    spatial = np.stack((np.ones_like(xx), xx, yy, xx*yy, xx**2, yy**2), axis=0)
    real_coefficients = np.stack((np.sin(k0), np.cos(theta), k0*np.sin(theta),
                                  np.cos(2*k0), np.sin(2*theta+k0), k0**2), axis=1)
    imag_coefficients = np.stack((np.cos(k0), np.sin(theta), k0*np.cos(theta),
                                  np.sin(2*k0), np.cos(2*theta-k0), k0**3), axis=1)
    values = real_coefficients @ spatial + 1j*(imag_coefficients @ spatial)
    A = np.broadcast_to(1.5 + .1*xx + .05*yy**2, values.shape).copy()
    n = np.broadcast_to(1.2 + .2*np.cos(xx)*np.cos(yy), values.shape).copy()
    return Dataset(A=A, n=n, u_scat=values, theta=theta, k0=k0,
                   medium_id=np.zeros(60, dtype=np.int64), frequency_id=frequency_id,
                   split_strategy="frequency", split=split,
                   coordinates=coordinates, triangles=triangles)


class StaticDatasetTests(unittest.TestCase):
    def test_one_physical_medium_has_disjoint_frequency_splits(self):
        dataset = static_dataset()
        self.assertEqual(len(np.unique(dataset.medium_id)), 1)
        np.testing.assert_array_equal(dataset.split_group_ids, dataset.frequency_id)
        train, val, test = dataset.split_indices()
        self.assertEqual(tuple(map(len, (train, val, test))), (42, 9, 9))
        self.assertFalse(set(dataset.frequency_id[train]) & set(dataset.frequency_id[val]))
        self.assertFalse(set(dataset.frequency_id[train]) & set(dataset.frequency_id[test]))
        self.assertIs(dataset.validate(), dataset)

    def test_frequency_leakage_is_rejected(self):
        dataset = static_dataset()
        dataset.split[1] = 1  # Same frequency as sample zero, now across splits.
        with self.assertRaises(ValueError):
            dataset.validate()

    def test_randomized_svd_static_inputs_train_and_portable_checkpoint(self):
        dataset = static_dataset()
        config = TrainConfig(method="svd", pod_method="randomized", rank=2,
                             input_rank=5, epochs=2, batch_size=14, width=8,
                             depth=1, threads=1, patience=0, min_epochs=1,
                             target_projection_error=0.)
        with tempfile.TemporaryDirectory() as temporary:
            # 42 train samples and requested rank=2 force a 12-vector random
            # range; a silently substituted full SVD would be 42 by 56.
            with patch("numpy.linalg.svd", wraps=np.linalg.svd) as svd:
                predictor, metrics, history = train_model(dataset, temporary, config, verbose=False)
            shapes = [call.args[0].shape for call in svd.call_args_list]
            self.assertIn((12, 56), shapes)
            self.assertNotIn((42, 56), shapes)
            self.assertEqual(metrics["ranks"]["A"], 0)
            self.assertEqual(metrics["ranks"]["n"], 0)
            self.assertEqual(predictor.model_kwargs["input_dim"], 3)
            self.assertEqual(len(history["train"]), 2)
            self.assertTrue(np.isfinite(history["train"]).all())
            self.assertTrue(np.isfinite(history["val"]).all())
            self.assertTrue(np.isfinite(metrics["test_fem_l2"]["mean"]))
            train, _, test = dataset.split_indices()
            features = predictor.encode(dataset.A, dataset.n, dataset.theta, dataset.k0)
            self.assertEqual(features.shape, (60, 3))
            np.testing.assert_allclose(features[:, 0], np.cos(dataset.theta), atol=1e-7)
            np.testing.assert_allclose(features[:, 1], np.sin(dataset.theta), atol=1e-7)
            np.testing.assert_allclose(features[train, 2].mean(), 0., atol=1e-7)
            self.assertGreater(float(features[:, 2].std()), 0)
            expected = predictor.predict_dataset(dataset, test)
            loaded = load_predictor(Path(temporary) / "model.pt")
            self.assertEqual(loaded.bases["A"].rank, 0)
            self.assertEqual(loaded.bases["n"].rank, 0)
            actual = loaded.predict_dataset(dataset, test)
            self.assertEqual(actual.shape, (9, 56))
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            self.assertTrue(np.isfinite(dataset.relative_l2(dataset.u_scat[test], actual)).all())


if __name__ == "__main__":
    unittest.main()
