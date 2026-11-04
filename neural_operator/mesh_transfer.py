"""Sparse P1 evaluation on a mesh's actual triangular connectivity."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from matplotlib.tri import Triangulation


class P1Transfer:
    """Evaluate source FEM fields at target points by barycentric interpolation.

    Point location respects the supplied triangles, including holes. Only
    roundoff-sized excursions from actual boundary segments are tolerated
    for point location; weights always evaluate the original target points.
    This is an evaluation operator, so a round trip through a coarser mesh
    generally loses information.
    """

    def __init__(self, source_coordinates, source_triangles, target_points):
        coordinates = np.asarray(source_coordinates, dtype=np.float64)
        triangles = np.asarray(source_triangles)
        points = np.asarray(target_points, dtype=np.float64)
        if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or
                len(coordinates) < 3 or not np.isfinite(coordinates).all()):
            raise ValueError("source_coordinates must be finite [nodes,2]")
        if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles) or
                not np.issubdtype(triangles.dtype, np.integer) or
                triangles.min() < 0 or triangles.max() >= len(coordinates)):
            raise ValueError("source_triangles must contain valid integer node triples")
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("target_points must be finite [points,2]")
        vertices = coordinates[triangles]
        edges = vertices[:, 1:] - vertices[:, :1]
        determinants = edges[:, 0, 0]*edges[:, 1, 1] - edges[:, 0, 1]*edges[:, 1, 0]
        edge_scale = np.max(np.sum(edges**2, axis=2), axis=1)
        if np.any(np.abs(determinants) <= 4*np.finfo(np.float64).eps*edge_scale):
            raise ValueError("source mesh contains degenerate triangles")
        # Matplotlib's locator expects anticlockwise triangles. Reorienting
        # their node order preserves every original edge and element.
        triangles = triangles.copy()
        reverse = determinants < 0
        triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
        lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
        tolerance = 1e-12 * max(1., float(np.max(upper-lower)))
        if np.any(points < lower-tolerance) or np.any(points > upper+tolerance):
            raise ValueError("target points lie outside the source mesh")
        if not len(points):
            self.matrix = csr_matrix((0, len(coordinates)), dtype=np.float64)
            return
        query = np.clip(points, lower, upper)
        try:
            triangulation = Triangulation(coordinates[:, 0], coordinates[:, 1], triangles)
            located = triangulation.get_trifinder()(query[:, 0], query[:, 1])
        except (RuntimeError, ValueError) as error:
            raise ValueError("source mesh is not a valid nonoverlapping triangulation") from error
        missing = np.flatnonzero(located < 0)
        if missing.size:
            # A nominally straight boundary can contain coordinates differing
            # by a few ulps. Bounding-box clipping cannot recover points lying
            # just beyond one such segment. Search actual boundary edges only,
            # keeping memory linear in boundary size, not points × triangles.
            all_edges = np.sort(np.concatenate((triangles[:, [0,1]],
                                                triangles[:, [1,2]],
                                                triangles[:, [2,0]])), axis=1)
            owners = np.tile(np.arange(len(triangles)), 3)
            unique, first, counts = np.unique(all_edges, axis=0, return_index=True,
                                              return_counts=True)
            boundary = counts == 1
            boundary_edges = unique[boundary]
            boundary_owners = owners[first[boundary]]
            starts = coordinates[boundary_edges[:, 0]]
            directions = coordinates[boundary_edges[:, 1]] - starts
            lengths_squared = np.einsum("ij,ij->i", directions, directions)
            for index in missing:
                displacement = points[index] - starts
                parameter = np.clip(np.einsum("ij,ij->i", displacement, directions) /
                                    lengths_squared, 0., 1.)
                distances_squared = np.sum((displacement - parameter[:, None]*directions)**2,
                                           axis=1)
                closest = int(np.argmin(distances_squared))
                if distances_squared[closest] > tolerance**2:
                    raise ValueError("target points lie outside source triangles or inside a mesh hole")
                located[index] = boundary_owners[closest]
        containing = triangles[located]
        vertices = coordinates[containing]
        edge_matrix = np.swapaxes(vertices[:, 1:] - vertices[:, :1], 1, 2)
        uv = np.linalg.solve(edge_matrix, (points-vertices[:, 0])[..., None])[..., 0]
        weights = np.column_stack((1-uv.sum(axis=1), uv))
        self.matrix = csr_matrix((weights.ravel(),
                                  (np.repeat(np.arange(len(points)), 3), containing.ravel())),
                                 shape=(len(points), len(coordinates)))
        self.matrix.eliminate_zeros()

    def apply(self, fields):
        """Transfer real or complex fields with source nodes in the last axis."""
        fields = np.asarray(fields)
        if fields.ndim < 1 or fields.shape[-1] != self.matrix.shape[1]:
            raise ValueError("fields must have source nodes in their last axis")
        flat = fields.reshape(-1, fields.shape[-1])
        return (self.matrix @ flat.T).T.reshape(*fields.shape[:-1], self.matrix.shape[0])
