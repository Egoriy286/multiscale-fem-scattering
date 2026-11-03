"""Sparse transfers using the supplied P1 triangles and tensor-grid interpolation."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix


class GridTransfer:
    def __init__(self, coordinates, triangles, resolution=64, kind="uniform"):
        coordinates = np.asarray(coordinates, dtype=float)[:, :2]
        triangles = np.asarray(triangles, dtype=int)
        if kind not in ("uniform", "chebyshev"):
            raise ValueError("kind must be uniform or chebyshev")
        ny, nx = (resolution, resolution) if np.isscalar(resolution) else resolution
        if min(nx, ny) < 2:
            raise ValueError("resolution must be at least two in each direction")
        self.kind = kind
        self.x = np.linspace(-2, 2, nx) if kind == "uniform" else -2 * np.cos(np.linspace(0, np.pi, nx))
        self.y = np.linspace(-2, 2, ny) if kind == "uniform" else -2 * np.cos(np.linspace(0, np.pi, ny))
        self.shape = (ny, nx)
        xx, yy = np.meshgrid(self.x, self.y)
        self.points = np.column_stack((xx.ravel(), yy.ravel()))
        assigned = np.zeros(nx * ny, dtype=bool)
        rows, columns, values = [], [], []
        # Bounding-box candidates avoid replacing the actual mesh connectivity
        # with a Delaunay triangulation (which would bridge holes).
        for indices in triangles:
            vertices = coordinates[indices]
            lower, upper = vertices.min(axis=0), vertices.max(axis=0)
            ix = np.flatnonzero((self.x >= lower[0] - 1e-12) & (self.x <= upper[0] + 1e-12))
            iy = np.flatnonzero((self.y >= lower[1] - 1e-12) & (self.y <= upper[1] + 1e-12))
            candidates = (iy[:, None] * nx + ix).ravel()
            candidates = candidates[~assigned[candidates]]
            if not candidates.size:
                continue
            matrix = (vertices[1:] - vertices[0]).T
            if abs(np.linalg.det(matrix)) < 1e-20:
                continue
            uv = np.linalg.solve(matrix, (self.points[candidates] - vertices[0]).T).T
            bary = np.column_stack((1 - uv.sum(axis=1), uv))
            inside = np.all(bary >= -1e-10, axis=1)
            selected, bary = candidates[inside], bary[inside]
            assigned[selected] = True
            rows.extend(np.repeat(selected, 3))
            columns.extend(np.tile(indices, selected.size))
            values.extend(bary.ravel())
        if not assigned.all():
            raise ValueError(f"{np.count_nonzero(~assigned)} grid points lie outside the supplied mesh")
        self.fem_to_grid = csr_matrix((values, (rows, columns)), shape=(nx * ny, len(coordinates)))
        if np.any(coordinates < -2 - 1e-10) or np.any(coordinates > 2 + 1e-10):
            raise ValueError("mesh nodes must lie in [-2,2]^2")
        ix = np.clip(np.searchsorted(self.x, coordinates[:, 0], side="right") - 1, 0, nx - 2)
        iy = np.clip(np.searchsorted(self.y, coordinates[:, 1], side="right") - 1, 0, ny - 2)
        tx = (coordinates[:, 0] - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        ty = (coordinates[:, 1] - self.y[iy]) / (self.y[iy + 1] - self.y[iy])
        cols = np.column_stack((iy * nx + ix, iy * nx + ix + 1, (iy + 1) * nx + ix, (iy + 1) * nx + ix + 1))
        vals = np.column_stack(((1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty))
        self.grid_to_fem = csr_matrix((vals.ravel(), (np.repeat(np.arange(len(coordinates)), 4), cols.ravel())), shape=(len(coordinates), nx * ny))

    @property
    def to_grid_matrix(self):
        return self.fem_to_grid

    @property
    def to_nodes_matrix(self):
        return self.grid_to_fem

    def to_grid(self, fields):
        fields = np.asarray(fields)
        return (self.fem_to_grid @ fields.reshape(-1, fields.shape[-1]).T).T.reshape(*fields.shape[:-1], *self.shape)

    def to_nodes(self, grid):
        grid = np.asarray(grid)
        return (self.grid_to_fem @ grid.reshape(-1, np.prod(self.shape)).T).T.reshape(*grid.shape[:-2], -1)
