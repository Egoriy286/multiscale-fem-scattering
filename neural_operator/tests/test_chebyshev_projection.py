"""Check the physical decoder span, rather than DCT truncation error."""
import numpy as np

from neural_operator.diagnostics import chebyshev_projection_diagnostics
from neural_operator.tests.test_static_training import static_dataset


def test_projection_span_is_nested_and_respects_affine_train_mean():
    dataset = static_dataset()
    x, y = dataset.coordinates.T
    # The mean can be outside the polynomial span: it is added independently.
    fixed = np.sin(2 * x + y)
    parameter = dataset.k0[:, None] - 1.7
    dataset.u_scat = fixed[None] + parameter * x + 1j * (fixed[None] + parameter * y)
    rows = chebyshev_projection_diagnostics(dataset, resolution=9, modes=(1, 2, 3))
    errors = [row["projection_floor"]["mean"] for row in rows]
    assert errors[0] > .01
    # Linear polynomials are exactly preserved by both tensor synthesis and
    # bilinear interpolation onto the actual FEM nodes.
    assert errors[1] < 1e-12
    assert errors[2] < 1e-12
    assert all(row["validation_samples"] == 9 for row in rows)
