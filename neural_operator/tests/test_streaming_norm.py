"""Large/broadcast complex fields keep the same consistent FEM norm."""

import numpy as np

from neural_operator.data import Dataset


def test_relative_norm_batches_leading_axes_and_zero_reference():
    xy = np.array([[0., 0.], [1., 0.], [0., 1.]])
    ds = Dataset(np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 3), complex),
                 np.zeros(1), np.ones(1), np.zeros(1, int), np.zeros(1, int),
                 xy, np.array([[0, 1, 2]]))
    rng = np.random.default_rng(17)
    true = rng.normal(size=(2, 300, 3)) + 1j*rng.normal(size=(2, 300, 3))
    predicted = rng.normal(size=(1, 3)) + 1j*rng.normal(size=(1, 3))
    error = predicted-true
    mass = ds.mass_matrix.toarray()
    numerator = np.real(np.einsum('...i,ij,...j->...', error.conj(), mass, error))
    denominator = np.real(np.einsum('...i,ij,...j->...', true.conj(), mass, true))
    np.testing.assert_allclose(ds.relative_l2(true, predicted), np.sqrt(numerator/denominator), rtol=2e-14)
    true.flat[:6] = 0
    approximation = np.broadcast_to(predicted, true.shape).copy()
    approximation[0, 0] = 0
    actual = ds.relative_l2(true, approximation)
    assert actual[0, 0] == 0
    assert np.isinf(actual[0, 1])
