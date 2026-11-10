"""Train-fitted POD with an optional diagonal (mass-lumped) inner product."""
from __future__ import annotations
import numpy as np


class PODBasis:
    def __init__(self, mean, modes, singular_values, total_energy, weights=None):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.modes = np.asarray(modes, dtype=np.float64)
        self.singular_values = np.asarray(singular_values, dtype=np.float64)
        self.total_energy = float(total_energy)
        self.weights = (np.ones(self.mean.size) if weights is None else
                        np.asarray(weights, dtype=np.float64))

    @classmethod
    def fit(cls, fields, rank, weights=None, method="full", seed=42):
        if np.iscomplexobj(fields):
            raise ValueError("PODBasis is real-valued; fit real and imaginary components separately")
        fields = np.asarray(fields, dtype=np.float64)
        if fields.ndim != 2 or not all(fields.shape) or not np.isfinite(fields).all():
            raise ValueError("fields must be finite [samples, dofs] with nonempty axes")
        if rank < 0 or int(rank) != rank:
            raise ValueError("rank must be a nonnegative integer")
        weights = np.ones(fields.shape[1]) if weights is None else np.asarray(weights, dtype=np.float64)
        if weights.shape != (fields.shape[1],) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("weights must contain one positive finite value per dof")
        if method not in ("full", "randomized"):
            raise ValueError("method must be 'full' or 'randomized'")
        weighted = np.array(fields, dtype=np.float64, order="C", copy=True)
        weighted -= fields[0]
        average_offset = weighted.mean(axis=0)
        mean = fields[0] + average_offset
        weighted -= average_offset
        weighted *= np.sqrt(weights)
        total_energy = float(np.einsum("ij,ij->", weighted, weighted))
        k = min(int(rank), *weighted.shape)
        if k == 0 or total_energy == 0:
            return cls(mean, np.empty((fields.shape[1], 0)), np.empty(0), total_energy, weights)
        if method == "full" or k + 10 >= min(weighted.shape):
            _, s, vh = np.linalg.svd(weighted, full_matrices=False)
        else:
            rng = np.random.default_rng(seed)
            q, _ = np.linalg.qr(weighted @ rng.standard_normal((weighted.shape[1], k + 10)), mode="reduced")
            for _ in range(2):
                z, _ = np.linalg.qr(weighted.T @ q, mode="reduced")
                q, _ = np.linalg.qr(weighted @ z, mode="reduced")
            _, s, vh = np.linalg.svd(q.T @ weighted, full_matrices=False)
        # Centering removes the mean but does not remove its rounding scale.
        # A nearly constant, order-one field must not produce noise-only modes.
        field_scale = float(np.sqrt(np.einsum("ij,ij,j->", fields, fields, weights)))
        tolerance = np.finfo(np.float64).eps * max(weighted.shape) * max(s[0], field_scale)
        k = min(k, int(np.count_nonzero(s > tolerance)))
        return cls(mean, vh[:k].T / np.sqrt(weights)[:, None], s[:k], total_energy, weights)

    @property
    def rank(self):
        return self.modes.shape[1]

    @property
    def energy(self):
        """Cumulative retained fractions of centered training energy."""
        return np.cumsum(self.singular_values ** 2) / self.total_energy if self.total_energy > 0 else np.zeros(self.rank)

    def encode(self, fields):
        if self.rank == 0:
            return np.empty((*np.shape(fields)[:-1], 0), dtype=np.float64)
        return ((np.asarray(fields) - self.mean) * self.weights) @ self.modes

    def decode(self, coefficients):
        return np.asarray(coefficients) @ self.modes.T + self.mean

    def to_dict(self):
        return dict(mean=self.mean.tolist(), modes=self.modes.tolist(),
                    singular_values=self.singular_values.tolist(),
                    total_energy=self.total_energy, weights=self.weights.tolist())

    @classmethod
    def from_dict(cls, state):
        state = dict(state)
        state["modes"] = np.asarray(state["modes"]).reshape(len(state["mean"]), -1)
        return cls(**state)
