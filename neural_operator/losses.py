"""Consistent FEM objectives evaluated in POD coordinates."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import torch
from torch import nn


def relative_root_mean(squared_error, squared_reference, epsilon=1e-24):
    """Relative norms with a finite zero gradient and exact zero at zero error.

    For errors >= epsilon this is the usual square root. Below that scale,
    e / sqrt(epsilon) bounds its derivative. Flooring both numerator
    and denominator instead makes a perfect zero field have loss one.
    """
    error = squared_error.clamp_min(0)
    norms = error / torch.sqrt(error.clamp_min(epsilon))
    return (norms / torch.sqrt(squared_reference.clamp_min(epsilon))).mean()


class PODFEMLoss(nn.Module):
    """Exact consistent-mass relative L2 loss for a fixed POD decoder.

    The FEM projection and its irreducible residual are precomputed in float64
    from the *rounded decoder buffers*. Online work depends on rank, not mesh
    size. Residual energies are integrated directly, avoiding cancellation from
    subtracting a projected energy from a nearly equal total energy.

    ``indices`` address rows of the target_fields array supplied at construction.
    Targets may be real [samples,nodes,2] or complex [samples,nodes],
    including memory-mapped arrays. They are processed on CPU in batches of
    256; no FEM fields are retained. ``target_dtype`` optionally rounds each
    component to the label precision before float64 integration.
    Move the resulting module to the training device with ``.to(device)``.
    """

    def __init__(self, predictor, dataset, target_fields, target_dtype=None):
        super().__init__()
        if torch.is_tensor(target_fields):
            targets = target_fields.detach().cpu().numpy()
        else:
            targets = np.asarray(target_fields)
        modes = predictor.output_modes.detach().cpu().numpy().astype(np.float64)
        field_mean = predictor.output_mean.detach().cpu().numpy().astype(np.float64)
        complex_fields = targets.ndim == 2 and np.iscomplexobj(targets)
        valid_shape = (targets.shape[1:] == field_mean.shape[:1] if complex_fields else
                       targets.ndim == 3 and targets.shape[1:] == field_mean.shape)
        if not valid_shape:
            raise ValueError("target_fields must be complex [samples,nodes] or real [samples,nodes,2]")
        mass = dataset.mass_matrix.astype(np.float64)
        count = targets.shape[0]
        rank = modes.shape[1]
        gram = np.zeros((rank, rank, 2), dtype=np.float64)
        projected = np.zeros((count, rank, 2), dtype=np.float64)
        floor = np.zeros(count, dtype=np.float64)
        norm = np.zeros(count, dtype=np.float64)
        for component, name in enumerate(("real", "imag")):
            actual_rank = predictor.bases[name].rank
            phi = modes[:, :actual_rank, component]
            if actual_rank:
                mphi = mass @ phi
                g = phi.T @ mphi
                g = (g + g.T) * .5
                factor = cho_factor(g, lower=True)
                gram[:actual_rank, :actual_rank, component] = g
            for start in range(0, count, 256):
                stop = min(start + 256, count)
                if complex_fields:
                    block = targets[start:stop].real if component == 0 else targets[start:stop].imag
                else:
                    block = targets[start:stop, :, component]
                if target_dtype is not None:
                    block = np.asarray(block, dtype=target_dtype)
                true = np.asarray(block, dtype=np.float64)
                residual = true - field_mean[:, component]
                if actual_rank:
                    coefficients = cho_solve(factor, (residual @ mphi).T).T
                    projected[start:stop, :actual_rank, component] = coefficients
                    residual = residual - coefficients @ phi.T
                floor[start:stop] += np.einsum("bn,bn->b", residual, (mass @ residual.T).T)
                norm[start:stop] += np.einsum("bn,bn->b", true, (mass @ true.T).T)
        self.register_buffer("gram", torch.from_numpy(gram))
        self.register_buffer("projected_coefficients", torch.from_numpy(projected))
        self.register_buffer("floor_squared", torch.from_numpy(np.maximum(floor, 0)))
        self.register_buffer("norm_squared", torch.from_numpy(np.maximum(norm, 0)))
        stats = predictor.preprocessing["output_stats"]
        # Match the decoder's float32 denormalization before switching to the
        # float64 quadratic form. This also preserves its autograd chain.
        self.register_buffer("mean", torch.as_tensor(stats["mean"], dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(stats["std"], dtype=torch.float32))

    def forward(self, normalized_coefficients, indices):
        coefficients = normalized_coefficients * self.std + self.mean
        difference = coefficients.to(self.gram.dtype) - self.projected_coefficients[indices]
        error = torch.einsum("brc,rsc,bsc->b", difference, self.gram, difference)
        error = error + self.floor_squared[indices]
        return relative_root_mean(error, self.norm_squared[indices])
