"""Train-only complex POD adapter for the common FEM-norm training loop."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.gms_sno_100.model import SpectralSurrogate
from experiments.gms_sno_100.pod import ComplexPOD

SCALES = (1, 40, 80, 100, 120, 140, 180, 200)


def _norm2(fields, mass):
    return np.maximum(np.real(np.sum(fields.conj() * (mass @ fields.T).T, axis=1)), 0.)


def model_config(rank, method, width=8, depth=4):
    multi = dict(output_modes=rank, width=width, depth=depth,
                 scales=list(SCALES), activation='sine')
    if method == 'mscale_sno':
        return multi
    if method not in ('sno_gelu', 'sno_sine'):
        raise ValueError('Unknown SNO method')
    target = sum(p.numel() for p in SpectralSurrogate(**multi).parameters())
    def count(w):
        # Avoid consuming RNG while choosing a deterministic parameter budget.
        return 4*rank*w + depth*(2*w*w+w+rank*rank) + 2*w+4
    w = 1
    while count(w) < target:
        w += 1
    w = min((max(1, w-1), w), key=lambda x: abs(count(x)-target))
    return dict(output_modes=rank, width=w, depth=depth, scales=[1],
                activation='sine' if method == 'sno_sine' else 'gelu')


class CoefficientLoss:
    def __init__(self, prep, device='cpu', dtype=torch.float64):
        self.device, self.dtype = torch.device(device), dtype
        self.scale = float(prep['scale'])
        target = prep['pairs'] / self.scale
        high = target.astype(np.float32 if dtype == torch.float32 else np.float64)
        self.high = torch.as_tensor(high, dtype=dtype, device=self.device)
        self.low = torch.as_tensor(target-high.astype(np.float64), dtype=dtype, device=self.device)
        self.residual2 = torch.as_tensor(prep['residual2'], dtype=dtype, device=self.device)
        self.norm2 = torch.as_tensor(prep['norm2'], dtype=dtype, device=self.device)

    def __call__(self, prediction, ids):
        ids = torch.as_tensor(ids, dtype=torch.int64, device=self.device)
        # MPS cannot combine a device copy and conversion to double in one op.
        prediction = prediction.to(self.device).to(self.dtype)
        delta = ((prediction-self.high[ids])-self.low[ids]) * self.scale
        return ((delta.square().sum((-1, -2)) + self.residual2[ids]) /
                self.norm2[ids]).clamp_min(1e-30).sqrt()


class SNOAdapter:
    def __init__(self, data, train_ids, method='mscale_sno', *, rank=20,
                 width=8, depth=4, device='cpu'):
        self.device = torch.device(device)
        train_ids = np.asarray(train_ids)
        self.pod = ComplexPOD.fit(data.u_scat[train_ids], data.mass_matrix,
                                  rank=rank, max_rank=rank)
        if self.pod.rank < 1:
            raise ValueError('Positive POD rank required')
        coefficients, residual2, norm2 = [], [], []
        for start in range(0, len(data.k0), 32):
            fields = np.asarray(data.u_scat[start:start+32])
            c = self.pod.encode(fields, data.mass_matrix)
            coefficients.append(c)
            residual2.extend(_norm2(fields-self.pod.decode(c), data.mass_matrix))
            norm2.extend(_norm2(fields, data.mass_matrix))
        coefficients = np.concatenate(coefficients)
        k_mean, k_std = float(data.k0[train_ids].mean()), float(data.k0[train_ids].std())
        scale = float(np.sqrt(np.mean(np.abs(coefficients[train_ids])**2)))
        if min(k_std, scale) <= 0 or np.any(np.asarray(norm2) <= 0):
            raise ValueError('Nonzero training variance and field norms required')
        self.prep = dict(inputs=np.column_stack(((data.k0-k_mean)/k_std,
                            np.cos(data.theta), np.sin(data.theta))).astype('float32'),
                         pairs=np.stack((coefficients.real, coefficients.imag), axis=-1),
                         scale=scale, k_mean=k_mean, k_std=k_std,
                         residual2=np.asarray(residual2), norm2=np.asarray(norm2),
                         train_ids=train_ids, floor_relative_l2=np.sqrt(np.asarray(residual2)/norm2))
        self.config = model_config(self.pod.rank, method, width, depth)
        self.model = SpectralSurrogate(**self.config).to(self.device)
        self.inputs = torch.as_tensor(self.prep['inputs'], device=self.device)
        self.loss = CoefficientLoss(self.prep, device, torch.float32)
        self.validation_loss = CoefficientLoss(self.prep, 'cpu', torch.float64)

    def decode(self, prediction):
        pairs = prediction.detach().cpu().numpy().astype(np.float64)*float(self.prep['scale'])
        return self.pod.decode(pairs[..., 0]+1j*pairs[..., 1])

    @torch.no_grad()
    def predict_fields(self, ids, batch_size=50):
        self.model.eval()
        return np.concatenate([self.decode(self.model(self.inputs[ids[start:start+batch_size]]))
                               for start in range(0, len(ids), batch_size)])

    def save_preprocessing(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.pod.save(directory/'pod.npz')
        np.savez(directory/'sno_preprocessing.npz', **self.prep)
        (directory/'sno_config.json').write_text(json.dumps(self.config, indent=2)+'\n')

    @classmethod
    def load(cls, directory, device='cpu'):
        directory = Path(directory)
        obj = cls.__new__(cls)
        obj.device = torch.device(device)
        obj.pod = ComplexPOD.load(directory/'pod.npz')
        with np.load(directory/'sno_preprocessing.npz', allow_pickle=False) as archive:
            obj.prep = {k: archive[k] for k in archive.files}
        obj.config = json.loads((directory/'sno_config.json').read_text())
        obj.model = SpectralSurrogate(**obj.config).to(obj.device)
        obj.inputs = torch.as_tensor(obj.prep['inputs'], device=obj.device)
        obj.loss = CoefficientLoss(obj.prep, device, torch.float32)
        obj.validation_loss = CoefficientLoss(obj.prep, 'cpu', torch.float64)
        return obj
