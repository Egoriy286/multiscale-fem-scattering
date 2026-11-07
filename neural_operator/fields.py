"""Physical components of the decoded complex scattered field."""

import numpy as np


def scattering_components(field):
    """Return Re, Im and Abs of u_scattering, preserving all field dimensions.

    Apply this to the reconstructed physical field, after denormalization and
    any mesh transfer. The amplitude is derived from the same complex field;
    it is not a third independent network output or the modulus of POD modes.
    """
    field = np.asarray(field)
    return {"real": field.real, "imag": field.imag, "abs": np.abs(field)}
