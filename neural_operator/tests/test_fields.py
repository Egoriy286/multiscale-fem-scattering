"""Physical output components must preserve phase and compare like quantities."""

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_operator.fields import scattering_components
from neural_operator.plotting import plot_prediction


def test_components_preserve_complex_field_and_amplitude_for_batches():
    field = np.array([[3+4j, -3-4j, 0], [1j, -1j, -2]])
    result = scattering_components(field)
    assert set(result) == {"real", "imag", "abs"}
    np.testing.assert_array_equal(result["real"] + 1j*result["imag"], field)
    np.testing.assert_array_equal(result["abs"], [[5, 5, 0], [1, 1, 2]])
    assert all(values.shape == field.shape for values in result.values())


def test_plot_shows_imaginary_phase_error_and_zero_amplitude_error():
    dataset = SimpleNamespace(coordinates=np.array([[0., 0.], [1., 0.], [0., 1.]]),
                              triangles=np.array([[0, 1, 2]]),
                              u_scat=np.full((1, 3), 1j), k0=[1.], medium_id=[0])
    figure = plot_prediction(dataset, 0, np.full(3, -1j))
    try:
        # Colorbar axes have no titles; the nine field panels are row-major.
        panels = [ax for ax in figure.axes if ax.get_title()]
        assert len(panels) == 9
        assert all("Im(" in ax.get_title() for ax in panels[3:6])
        assert all("Abs(" in ax.get_title() for ax in panels[6:9])
        np.testing.assert_array_equal(panels[5].collections[0].get_array(), [2., 2., 2.])
        np.testing.assert_array_equal(panels[8].collections[0].get_array(), [0., 0., 0.])
        for row in (0, 3, 6):
            assert panels[row].collections[0].get_clim() == panels[row+1].collections[0].get_clim()
        figure.canvas.draw()
    finally:
        plt.close(figure)
