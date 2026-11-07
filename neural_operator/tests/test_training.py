"""Integration checks for the physical loss and portable inference bundles."""

import numpy as np
import pytest

from neural_operator.data import Dataset
from neural_operator.training import FEMLoss, TrainConfig, load_predictor, train_model
import torch


@pytest.fixture
def tiny_dataset():
    xy = np.array([[-2., -2.], [2., -2.], [2., 2.], [-2., 2.], [.1, -.3]])
    triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    rng = np.random.default_rng(4)
    A = rng.uniform(.7, 1.2, (12, 5))
    n = rng.uniform(1., 2., (12, 5))
    theta = np.linspace(0., 5., 12)
    k = np.linspace(1., 2., 12)
    # An algebraic fixture, used only for loss/serialization verification.
    u = (A-.5)*np.exp(1j*(theta[:, None]+xy[:, 0]))
    return Dataset(A, n, u, theta, k, np.arange(12), np.repeat([0, 1, 2], [8, 2, 2]), xy, triangles)


def test_loss_matches_consistent_complex_mass(tiny_dataset):
    ds = tiny_dataset
    pred = .7*ds.u_scat + .1j
    stacked = lambda x: torch.tensor(np.stack([x.real, x.imag], axis=-1), dtype=torch.float32)
    loss = FEMLoss(ds.coordinates, ds.triangles)(stacked(pred), stacked(ds.u_scat))
    assert float(loss) == pytest.approx(ds.relative_l2(ds.u_scat, pred).mean(), rel=2e-6)


@pytest.mark.parametrize("method", ["svd", "chebyshev", "fno"])
def test_training_bundle_roundtrip_and_batch_parity(tiny_dataset, tmp_path, method):
    config = TrainConfig(method=method, epochs=2, width=4, depth=1, rank=4,
                         resolution=8, modes=3, padding=2, batch_size=4)
    original, metrics, _ = train_model(tiny_dataset, tmp_path/method, config, verbose=False)
    saved = load_predictor(tmp_path/method/"model.pt")
    indices = tiny_dataset.split_indices()[2]
    expected = original.predict_dataset(tiny_dataset, indices)
    actual = saved.predict_dataset(tiny_dataset, indices)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    single = saved.predict(tiny_dataset.A[indices[0]], tiny_dataset.n[indices[0]],
                           tiny_dataset.theta[indices[0]], tiny_dataset.k0[indices[0]])
    np.testing.assert_allclose(single, actual[0], atol=2e-6)
    components = saved.predict_components(tiny_dataset.A[indices[0]], tiny_dataset.n[indices[0]],
                                          tiny_dataset.theta[indices[0]], tiny_dataset.k0[indices[0]])
    batch_components = saved.predict_dataset_components(tiny_dataset, indices)
    for name, expected_component in (("real", actual.real), ("imag", actual.imag),
                                     ("abs", np.abs(actual))):
        assert components[name].shape == (len(tiny_dataset.coordinates),)
        np.testing.assert_allclose(components[name], expected_component[0], atol=2e-6)
        np.testing.assert_allclose(batch_components[name], expected_component, atol=1e-7)
    with np.load(tmp_path/method/"test_predictions.npz") as archive:
        for name in ("real", "imag", "abs"):
            np.testing.assert_allclose(archive[f"prediction_{name}"], batch_components[name], atol=1e-7)
        np.testing.assert_allclose(archive["prediction"], actual, atol=1e-7)
    assert metrics["best_epoch"] in (1, 2)
    assert np.isfinite(metrics["test_fem_l2"]["mean"])


def test_svd_preserves_constant_imaginary_component(tiny_dataset, tmp_path):
    tiny_dataset.u_scat = tiny_dataset.u_scat.real + 2j
    predictor, metrics, _ = train_model(tiny_dataset, tmp_path, TrainConfig(
        epochs=1, width=4, depth=1, rank=4, input_rank=3), verbose=False)
    assert metrics["ranks"]["imag"] == 0
    assert metrics["ranks"]["real"] > 0
    prediction = load_predictor(tmp_path/"model.pt").predict_dataset(tiny_dataset, [10])
    np.testing.assert_allclose(prediction.imag, 2., atol=1e-7)
