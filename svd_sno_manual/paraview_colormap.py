"""ParaView's Blue to Red Rainbow preset for nodal fields and matrix entries."""
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# Kitware/ParaView v5.13.3, Remoting/Views/ColorMaps.json.
# Unwrapped HSV from blue to red crosses these five equally spaced RGB knots.
RAINBOW = LinearSegmentedColormap.from_list(
    "Blue to Red Rainbow",
    ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000"],
    N=1025,
).with_extremes(bad="#7f7f7f")


def plot_nodal_field(ax, triangulation, values, norm):
    """Interpolate scalar values on the supplied triangles before mapping to RGB."""
    xmin, xmax = triangulation.x.min(), triangulation.x.max()
    ymin, ymax = triangulation.y.min(), triangulation.y.max()
    step = max(xmax - xmin, ymax - ymin) / 800
    nx, ny = (max(1, round(length / step)) for length in (xmax - xmin, ymax - ymin))
    x = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    y = ymin + (np.arange(ny) + 0.5) * (ymax - ymin) / ny
    field = mtri.LinearTriInterpolator(triangulation, values)(*np.meshgrid(x, y))
    return ax.imshow(field, origin="lower", extent=(xmin, xmax, ymin, ymax),
                     cmap=RAINBOW, norm=norm, interpolation="nearest", aspect="equal")
