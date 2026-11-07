"""Frozen trained operators evaluated in another mesh's physical coordinates."""

from __future__ import annotations

import numpy as np
import torch

from .mesh_transfer import P1Transfer
from .fields import scattering_components


class MeshPredictor:
    """Re-evaluate the source P1 representation without fitting new bases.

    SVD means and modes are transferred once. Other operator families transfer
    their decoded source field. This preserves the learned source space; a
    finer target mesh does not add fine-scale physics to that space.
    """

    def __init__(self, source, coordinates, triangles=None):
        self.source = source
        self.to_target = P1Transfer(source.coordinates, source.triangles, coordinates)
        self.coordinates = np.asarray(coordinates, dtype=np.float64).copy()
        self.triangles = None if triangles is None else np.asarray(triangles).copy()
        if self.triangles is not None:
            t = self.triangles
            if (t.ndim != 2 or t.shape[1] != 3 or not len(t)
                    or not np.issubdtype(t.dtype, np.integer)
                    or t.min() < 0 or t.max() >= len(self.coordinates)):
                raise ValueError("target triangles must contain valid node triples")
        self._to_source = None
        if source.method == "svd":
            mean = self.to_target.matrix @ source.output_mean.detach().cpu().numpy()
            modes = source.output_modes.detach().cpu().numpy()
            mapped = self.to_target.matrix @ modes.reshape(len(source.coordinates), -1)
            mapped = mapped.reshape(len(self.coordinates), *modes.shape[1:])
            self.output_mean = torch.as_tensor(mean, dtype=torch.float32, device=source.device)
            self.output_modes = torch.as_tensor(mapped, dtype=torch.float32, device=source.device)

    @property
    def to_source(self):
        if self._to_source is None:
            if self.triangles is None:
                raise ValueError("target triangles are required to transfer input fields")
            self._to_source = P1Transfer(self.coordinates, self.triangles, self.source.coordinates)
        return self._to_source

    def _from_source_inputs(self, A, n, theta, k0, batch_size):
        if self.source.method != "svd":
            return self.to_target.apply(self.source.predict(A, n, theta, k0, batch_size=batch_size))
        features = self.source.encode(A, n, theta, k0)
        stats = self.source.preprocessing["output_stats"]
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=self.source.device)
        std = torch.as_tensor(stats["std"], dtype=torch.float32, device=self.source.device)
        self.source.model.eval()
        outputs = []
        with torch.inference_mode():
            for first in range(0, len(features), batch_size):
                x = torch.as_tensor(features[first:first+batch_size], device=self.source.device)
                coefficients = self.source.model(x)*std+mean
                field = self.output_mean + torch.einsum("brc,nrc->bnc", coefficients, self.output_modes)
                values = field.cpu().numpy()
                outputs.append(values[..., 0]+1j*values[..., 1])
        return np.concatenate(outputs)

    def predict(self, A, n, theta, k0, batch_size=32):
        """Input A,n use target-node order; input coefficients keep source bases."""
        single = np.ndim(A) == 1
        A, n = np.atleast_2d(A), np.atleast_2d(n)
        if A.shape != n.shape or A.shape[1] != len(self.coordinates):
            raise ValueError("A and n must use the target FEM-node ordering")
        result = self._from_source_inputs(self.to_source.apply(A), self.to_source.apply(n),
                                          theta, k0, batch_size)
        return result[0] if single else result

    def predict_static(self, theta, k0, batch_size=32):
        """For the fixed-medium SVD models only, supply wave parameters alone."""
        if (self.source.method != "svd" or self.source.bases["A"].rank
                or self.source.bases["n"].rank):
            raise ValueError("predict_static requires zero input ranks for both A and n")
        theta, k0 = np.broadcast_arrays(theta, k0)
        single = theta.ndim == 0
        if theta.ndim > 1:
            raise ValueError("theta and k0 must be scalars or one-dimensional batches")
        theta, k0 = np.atleast_1d(theta), np.atleast_1d(k0)
        count = len(theta)
        A = np.broadcast_to(self.source.bases["A"].mean, (count, len(self.source.coordinates)))
        n = np.broadcast_to(self.source.bases["n"].mean, A.shape)
        result = self._from_source_inputs(A, n, theta, k0, batch_size)
        return result[0] if single else result

    def predict_components(self, A, n, theta, k0, batch_size=32):
        """Return Re, Im and Abs after reconstructing on the target mesh."""
        return scattering_components(self.predict(A, n, theta, k0, batch_size))

    def predict_static_components(self, theta, k0, batch_size=32):
        """Return Re, Im and Abs for a fixed medium on the target mesh."""
        return scattering_components(self.predict_static(theta, k0, batch_size))

    def predict_dataset(self, dataset, indices=None, batch_size=32):
        if (self.triangles is None or not np.array_equal(dataset.coordinates, self.coordinates)
                or not np.array_equal(dataset.triangles, self.triangles)):
            raise ValueError("Dataset geometry differs from the target mesh")
        indices = np.arange(len(dataset.A)) if indices is None else np.asarray(indices)
        return self.predict(dataset.A[indices], dataset.n[indices],
                            dataset.theta[indices], dataset.k0[indices], batch_size)

    def predict_dataset_components(self, dataset, indices=None, batch_size=32):
        """Return physical components for selected target-mesh samples."""
        return scattering_components(self.predict_dataset(dataset, indices, batch_size))
