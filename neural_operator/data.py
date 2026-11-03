"""Shared FEM samples and consistent P1 integration, without FEM runtime imports."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from scipy import sparse


def mass_matrix(coordinates, triangles):
    xy = np.asarray(coordinates)[np.asarray(triangles)]
    areas = np.abs(np.linalg.det(np.concatenate((np.ones((*xy.shape[:2], 1)), xy), axis=2))) / 2
    if np.any(areas <= 0):
        raise ValueError('Degenerate triangles')
    values = areas[:, None, None] * (np.ones((3, 3)) + np.eye(3))[None] / 12
    t = np.asarray(triangles)
    return sparse.coo_matrix((values.ravel(), (np.repeat(t, 3, axis=1).ravel(), np.tile(t, (1, 3)).ravel())), shape=(len(coordinates),)*2).tocsr()


def group_split(medium_id, fractions=(.67, .17, .16), seed=2026):
    fractions = np.asarray(fractions)
    if fractions.shape != (3,) or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError('Three nonnegative fractions must sum to one')
    ids = np.unique(medium_id)
    permutation = np.random.default_rng(seed).permutation(ids)
    ntrain, nval = np.floor(fractions[:2] * len(ids)).astype(int)
    mapping = {int(mid): (0 if j < ntrain else 1 if j < ntrain+nval else 2) for j, mid in enumerate(permutation)}
    return np.array([mapping[int(mid)] for mid in medium_id], dtype=np.int8)


@dataclass
class Dataset:
    A: np.ndarray
    n: np.ndarray
    u_scat: np.ndarray
    theta: np.ndarray
    k0: np.ndarray
    medium_id: np.ndarray
    split: np.ndarray
    coordinates: np.ndarray
    triangles: np.ndarray
    A_cells: np.ndarray | None = None
    n_cells: np.ndarray | None = None
    cell_tags: np.ndarray | None = None
    circles: np.ndarray | None = None
    directory: Path | None = None
    _mass: object = field(default=None, repr=False)

    def __post_init__(self):
        self.validate()

    def validate(self):
        ns, nn = self.u_scat.shape
        if self.A.shape != (ns, nn) or self.n.shape != (ns, nn):
            raise ValueError('A, n and u_scat must have shape [samples,nodes]')
        if self.coordinates.shape != (nn, 2) or self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError('Expected planar coordinates and triangular connectivity')
        if not np.issubdtype(self.triangles.dtype, np.integer) or self.triangles.min() < 0 or self.triangles.max() >= nn:
            raise ValueError('Invalid triangle indices')
        for name in ('theta', 'k0', 'medium_id', 'split'):
            if getattr(self, name).shape != (ns,):
                raise ValueError(f'{name} must have shape [samples]')
        for name in ('A', 'n', 'u_scat', 'theta', 'k0', 'coordinates'):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f'Nonfinite {name}')
        for name in ('A', 'n', 'k0'):
            if np.any(getattr(self, name) <= 0):
                raise ValueError(f'{name} must be positive')
        for name in ('medium_id', 'split'):
            if not np.issubdtype(getattr(self, name).dtype, np.integer):
                raise ValueError(f'{name} must use an integer dtype')
        if not np.all(np.isin(self.split, (0, 1, 2))):
            raise ValueError('split uses 0=train, 1=val, 2=test')
        for mid in np.unique(self.medium_id):
            if len(np.unique(self.split[self.medium_id == mid])) != 1:
                raise ValueError('Medium leakage between splits')
        for name in ('A_cells', 'n_cells'):
            arr = getattr(self, name)
            if arr is not None and (arr.shape != (ns, len(self.triangles)) or not np.all(np.isfinite(arr)) or np.any(arr <= 0)):
                raise ValueError(f'Invalid {name}')
        return self

    @property
    def mass_matrix(self):
        if self._mass is None:
            self._mass = mass_matrix(self.coordinates, self.triangles)
        return self._mass

    @property
    def lumped_weights(self):
        return np.asarray(self.mass_matrix.sum(axis=1)).ravel()

    def split_indices(self):
        return tuple(np.flatnonzero(self.split == j) for j in range(3))

    def relative_l2(self, true, pred):
        """Consistent complex FEM norm; zero/zero=0, nonzero/zero=inf."""
        true, pred = np.broadcast_arrays(true, pred)
        if true.shape[-1] != len(self.coordinates):
            raise ValueError('Last axis must contain all FEM nodes')
        def norm2(x):
            flat = x.reshape(-1, x.shape[-1])
            return np.maximum(np.real(np.sum(flat.conj() * (self.mass_matrix @ flat.T).T, axis=1)), 0).reshape(x.shape[:-1])
        num, den = norm2(pred-true), norm2(true)
        result = np.sqrt(np.divide(num, den, out=np.full_like(num, np.inf, dtype=float), where=den > 0))
        return np.where((den == 0) & (num == 0), 0., result)


def load_dataset(path):
    path = Path(path).expanduser().resolve()
    directory = path if path.is_dir() else path.parent
    path = directory / 'fem_dataset.npz' if path.is_dir() else path
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ('A','n','u_scat','theta','k0','medium_id','split')}
        data.update({key: archive[key] for key in ('A_cells','n_cells') if key in archive})
    with np.load(directory / 'geometry.npz', allow_pickle=False) as geometry:
        data.update(coordinates=geometry['dof_coords'][:, :2], triangles=geometry['triangles'])
        if 'cell_tags' in geometry:
            data['cell_tags'] = geometry['cell_tags']
        if 'etal_circles' in geometry:
            data['circles'] = geometry['etal_circles']
    return Dataset(**data, directory=directory)
