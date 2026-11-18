"""FNO adapters trained in the exact, consistent FEM mass norm.

The grid-to-node map is fixed bilinear interpolation T. All projections and
irreducible residuals are computed in float64; no mass lumping or grid
quadrature enters either training or validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.sparse.linalg import splu

from experiments.mscale_fno_100.model import FNO2d, MscaleFNO2d
from experiments.mscale_fno_100.training import _bilinear, _input_fields
from neural_operator.mesh_transfer import P1Transfer

SCALES = (1, 40, 80, 100, 120, 140, 180, 200)


def make_model(config):
    """Build an ordinary or independent eight-branch FNO from explicit config."""
    config = dict(config)
    variant = config.pop('variant', 'fno')
    if variant in ('mscale', 'mscale_fno'):
        return MscaleFNO2d(**config)
    if variant not in ('fno', 'fno_gelu', 'single_gelu', 'single_sine'):
        raise ValueError(f'Unknown FNO variant: {variant}')
    if variant == 'single_sine':
        config.setdefault('activation', 'sine')
    return FNO2d(**config)


def model_config(variant='fno', *, width=8, modes=8, depth=4,
                 use_mlp=True, padding=0, scales=SCALES):
    config = dict(variant=variant, input_channels=7, output_channels=2,
                  width=width, modes=modes, depth=depth, use_mlp=use_mlp,
                  padding=padding)
    if variant in ('mscale', 'mscale_fno'):
        config['scales'] = list(scales)
    else:
        config['activation'] = 'sine' if variant == 'single_sine' else 'gelu'
    return config


def project_targets(transfer, mass, fields):
    """Return G, least-mass-error grid fields, direct residual norms and uMu.

    The direct residual avoids subtracting nearly equal projected/total
    energies. A singular interpolation space is rejected, never regularized.
    """
    transfer = csr_matrix(transfer, dtype=np.float64)
    mass = csr_matrix(mass, dtype=np.float64)
    fields = np.asarray(fields, dtype=np.complex128)
    gram = (transfer.T @ mass @ transfer).tocsc()
    factor = splu(gram)
    rhs = np.asarray(transfer.T @ (mass @ fields.T))
    coefficients = (factor.solve(rhs.real) + 1j * factor.solve(rhs.imag)).T
    residual = fields - (transfer @ coefficients.T).T
    residual2 = np.real(np.sum(residual.conj() * (mass @ residual.T).T, axis=1))
    norm2 = np.real(np.sum(fields.conj() * (mass @ fields.T).T, axis=1))
    if np.any(norm2 <= 0) or not np.isfinite(norm2).all():
        raise ValueError('FEM target norm must be finite and positive')
    return gram.tocsr(), coefficients, np.maximum(residual2, 0), norm2


def prepare_grid(data, train_ids, resolution=64):
    """Prepare train-only statistics and exact FEM projection for all queries."""
    train_ids = np.asarray(train_ids, dtype=np.int64)
    if resolution < 4 or not len(train_ids):
        raise ValueError('resolution >= 4 and a nonempty training subset required')
    coordinates = np.asarray(data.coordinates)
    x = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), resolution)
    y = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    transfer = _bilinear(coordinates, x, y)
    gram, coefficients, residual2, norm2 = project_targets(
        transfer, data.mass_matrix, data.u_scat)
    physical = np.stack((coefficients.real, coefficients.imag), axis=1)
    mean = physical[train_ids].mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean((physical[train_ids] - mean) ** 2)))
    k_mean = float(np.asarray(data.k0)[train_ids].mean())
    k_std = float(np.asarray(data.k0)[train_ids].std())
    if scale <= 0 or k_std <= 0:
        raise ValueError('Training fields and frequencies need positive variance')
    sampler = P1Transfer(coordinates, data.triangles, points)
    A, n = sampler.apply(data.A[0]), sampler.apply(data.n[0])
    inputs = _input_fields(points, A, n, np.asarray(data.k0), np.asarray(data.theta),
                           k_mean, k_std, (resolution, resolution))
    return dict(inputs=inputs, coefficients=physical, mean=mean, scale=scale,
                residual2=residual2, norm2=norm2, gram=gram,
                grid_to_fem=transfer, points=points, A=A, n=n,
                k_mean=k_mean, k_std=k_std, shape=(resolution, resolution),
                train_ids=train_ids, floor_relative_l2=np.sqrt(residual2 / norm2))


class FEMGridLoss:
    """Differentiable exact mass loss using sparse G on CPU or CUDA.

    MPS lacks float64 and sparse support; its predictions are copied to CPU
    differentiably. Network gradients then return through the copy operation.
    """

    online_dtype = 'float64'

    def __init__(self, prep, device='cpu'):
        self.device = torch.device('cpu' if torch.device(device).type == 'mps' else device)
        coo = prep['gram'].tocoo()
        indices = torch.as_tensor(np.stack((coo.row, coo.col)), dtype=torch.int64, device=self.device)
        values = torch.as_tensor(coo.data, dtype=torch.float64, device=self.device)
        self.gram = torch.sparse_coo_tensor(indices, values, coo.shape, device=self.device,
                                            check_invariants=True).coalesce()
        self.coefficients = torch.as_tensor(prep['coefficients'], dtype=torch.float64, device=self.device)
        self.mean = torch.as_tensor(prep['mean'], dtype=torch.float64, device=self.device)
        self.scale = float(prep['scale'])
        self.residual2 = torch.as_tensor(prep['residual2'], dtype=torch.float64, device=self.device)
        self.norm2 = torch.as_tensor(prep['norm2'], dtype=torch.float64, device=self.device)

    def __call__(self, prediction, ids):
        ids = torch.as_tensor(ids, dtype=torch.int64, device=self.device)
        # Copy first: MPS attempts an unsupported Double cast if the destination
        # dtype and device are supplied in one to() operation.
        physical = prediction.to(self.device).double().flatten(2) * self.scale + self.mean
        delta = physical - self.coefficients[ids]
        flat = delta.reshape(-1, delta.shape[-1])
        energy = (flat * torch.sparse.mm(self.gram, flat.T).T).sum(1).reshape(-1, 2).sum(1)
        # Positive projection floors are retained, including during training.
        ratio = (energy.clamp_min(0) + self.residual2[ids]) / self.norm2[ids]
        return ratio.clamp_min(1e-30).sqrt()


class _StencilQuadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta, columns, weights):
        action = (delta[..., columns] * weights).sum(-1)
        ctx.save_for_backward(action)
        return (delta * action).sum((1, 2))

    @staticmethod
    def backward(ctx, gradient):
        # G=T.T M T is symmetric. Its analytic derivative avoids the generic
        # indexed-gather backward's large, contended GPU scatter operation.
        action, = ctx.saved_tensors
        return 2 * gradient[:, None, None] * action, None, None


class MPSStencilLoss:
    """Float32 evaluation of the exact sparse-G algebra on MPS.

    Projection and residual preparation remain float64. This changes only
    floating-point evaluation, never the mass matrix or interpolation space.
    Validation/reporting should use FEMGridLoss or decoded float64 FEM fields.
    """

    online_dtype = 'float32'

    def __init__(self, prep, device='mps'):
        self.device = torch.device(device)
        gram = prep['gram'].tocsr()
        counts = np.diff(gram.indptr)
        width = int(counts.max())
        columns = np.zeros((gram.shape[0], width), dtype=np.int64)
        weights = np.zeros((gram.shape[0], width), dtype=np.float32)
        for row in range(gram.shape[0]):
            start, end = gram.indptr[row:row + 2]
            columns[row, :end-start] = gram.indices[start:end]
            weights[row, :end-start] = gram.data[start:end]
        self.columns = torch.as_tensor(columns, device=self.device)
        self.weights = torch.as_tensor(weights, device=self.device)
        self.coefficients = torch.as_tensor(prep['coefficients'], dtype=torch.float32, device=self.device)
        self.mean = torch.as_tensor(prep['mean'], dtype=torch.float32, device=self.device)
        self.scale = float(prep['scale'])
        # Split normalized targets to retain their low bits near the projection
        # floor. Subtracting physical float32 fields would lose small errors to
        # cancellation against the (potentially large) train mean.
        normalized = (prep['coefficients'] - prep['mean']) / self.scale
        high = normalized.astype(np.float32)
        low = (normalized - high.astype(np.float64)).astype(np.float32)
        self.target_high = torch.as_tensor(high, device=self.device)
        self.target_low = torch.as_tensor(low, device=self.device)
        self.residual2 = torch.as_tensor(prep['residual2'], dtype=torch.float32, device=self.device)
        self.norm2 = torch.as_tensor(prep['norm2'], dtype=torch.float32, device=self.device)

    def __call__(self, prediction, ids):
        ids = torch.as_tensor(ids, dtype=torch.int64, device=self.device)
        normalized = prediction.to(self.device).float().flatten(2)
        delta = ((normalized - self.target_high[ids]) - self.target_low[ids]) * self.scale
        energy = _StencilQuadratic.apply(delta, self.columns, self.weights)
        ratio = (energy.clamp_min(0) + self.residual2[ids]) / self.norm2[ids]
        return ratio.clamp_min(1e-30).sqrt()


class FNOAdapter:
    def __init__(self, data, train_ids, variant='fno', config=None, resolution=64, device='cpu'):
        self.config = model_config(variant) if config is None else dict(config)
        self.prep = prepare_grid(data, train_ids, resolution)
        self.device = torch.device(device)
        self.model = make_model(self.config).to(self.device)
        self.inputs = torch.as_tensor(self.prep['inputs'], device=self.device)
        self.loss = FEMGridLoss(self.prep, self.device)

    @torch.no_grad()
    def predict_fields(self, ids, batch_size=10):
        self.model.eval()
        ids = np.asarray(ids, dtype=np.int64)
        result = []
        for start in range(0, len(ids), batch_size):
            normalized = self.model(self.inputs[ids[start:start + batch_size]])
            result.append(self.decode(normalized))
        return np.concatenate(result)

    def decode(self, prediction):
        """Map normalized network output to the full complex FEM nodal field."""
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        prediction = np.asarray(prediction, dtype=np.float64)
        physical = prediction.reshape(len(prediction), 2, -1) * self.prep['scale'] + self.prep['mean']
        grid = physical[:, 0] + 1j * physical[:, 1]
        return (self.prep['grid_to_fem'] @ grid.T).T

    def save_preprocessing(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'fno_config.json').write_text(json.dumps(self.config, indent=2) + '\n')
        arrays = {k: v for k, v in self.prep.items() if k not in ('gram', 'grid_to_fem')}
        np.savez(directory / 'fno_preprocessing.npz', **arrays)
        save_npz(directory / 'fno_gram.npz', self.prep['gram'])
        save_npz(directory / 'fno_transfer.npz', self.prep['grid_to_fem'])

    @classmethod
    def load(cls, directory, device='cpu'):
        """Load preprocessing; caller restores model state from its checkpoint."""
        directory = Path(directory)
        obj = cls.__new__(cls)
        obj.config = json.loads((directory / 'fno_config.json').read_text())
        with np.load(directory / 'fno_preprocessing.npz', allow_pickle=False) as archive:
            obj.prep = {key: archive[key] for key in archive.files}
        obj.prep.update(gram=load_npz(directory / 'fno_gram.npz'),
                        grid_to_fem=load_npz(directory / 'fno_transfer.npz'))
        obj.device = torch.device(device)
        obj.model = make_model(obj.config).to(obj.device)
        obj.inputs = torch.as_tensor(obj.prep['inputs'], device=obj.device)
        obj.loss = FEMGridLoss(obj.prep, obj.device)
        return obj
