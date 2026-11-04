"""Shared FEM samples and consistent P1 integration, without FEM runtime imports."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import zipfile
import numpy as np
from scipy import sparse


def _array_batches(values, batch_rows=256):
    """Bound temporaries by rows, including for broadcast and strided views."""
    if not isinstance(batch_rows, (int, np.integer)) or batch_rows < 1:
        raise ValueError('batch_rows must be a positive integer')
    values = np.asanyarray(values)
    if values.ndim == 0:
        yield values
    else:
        for start in range(0, len(values), batch_rows):
            yield values[start:start+batch_rows]


def update_array_digest(digest, values, batch_rows=256):
    """Update a digest with the exact C-order bytes, using bounded row copies."""
    for batch in _array_batches(values, batch_rows):
        contiguous = np.ascontiguousarray(batch)
        if contiguous.size:
            digest.update(memoryview(contiguous).cast('B'))
    return digest


def _all_finite(values):
    return all(np.all(np.isfinite(batch)) for batch in _array_batches(values))


def _all_positive(values):
    return all(np.all(batch > 0) for batch in _array_batches(values))


def _mapped_scattered_field(path):
    """Stream the compressed NPY member once, then reuse a read-only mmap."""
    cache_dir = path.parent/'.array_cache'/path.stem
    cache_path = cache_dir/'u_scat.npy'
    metadata_path = cache_dir/'u_scat.cache.json'
    source = path.stat()
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo('u_scat.npy')
        signature = dict(source_mtime_ns=source.st_mtime_ns, source_size=source.st_size,
                         member_crc=member.CRC, member_file_size=member.file_size)
        try:
            valid = (json.loads(metadata_path.read_text()) == signature
                     and cache_path.stat().st_size == member.file_size)
        except (OSError, ValueError):
            valid = False
        if not valid:
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary_paths = []
            try:
                with tempfile.NamedTemporaryFile(dir=cache_dir, prefix='u_scat.', suffix='.tmp', delete=False) as stream:
                    temporary = Path(stream.name)
                    temporary_paths.append(temporary)
                    with archive.open(member) as compressed:
                        shutil.copyfileobj(compressed, stream, length=1024*1024)
                temporary.replace(cache_path)
                with tempfile.NamedTemporaryFile(mode='w', dir=cache_dir, prefix='u_scat.cache.', suffix='.tmp', delete=False) as stream:
                    temporary = Path(stream.name)
                    temporary_paths.append(temporary)
                    json.dump(signature, stream, sort_keys=True)
                temporary.replace(metadata_path)
            finally:
                for temporary in temporary_paths:
                    temporary.unlink(missing_ok=True)
    return np.load(cache_path, mmap_mode='r', allow_pickle=False)


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
    frequency_id: np.ndarray | None = None
    split_strategy: str = "medium"
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
            if not _all_finite(getattr(self, name)):
                raise ValueError(f'Nonfinite {name}')
        for name in ('A', 'n', 'k0'):
            if not _all_positive(getattr(self, name)):
                raise ValueError(f'{name} must be positive')
        for name in ('medium_id', 'split'):
            if not np.issubdtype(getattr(self, name).dtype, np.integer):
                raise ValueError(f'{name} must use an integer dtype')
        if not np.all(np.isin(self.split, (0, 1, 2))):
            raise ValueError('split uses 0=train, 1=val, 2=test')
        if self.split_strategy not in ('medium', 'frequency'):
            raise ValueError('split_strategy must be medium or frequency')
        if self.split_strategy == 'frequency':
            if (self.frequency_id is None or self.frequency_id.shape != (ns,)
                    or not np.issubdtype(self.frequency_id.dtype, np.integer)):
                raise ValueError('frequency split requires integer frequency_id per sample')
            if len(np.unique(self.medium_id)) != 1:
                raise ValueError('frequency split describes one fixed physical medium')
        for group in np.unique(self.split_group_ids):
            indices = self.split_group_ids == group
            if len(np.unique(self.split[indices])) != 1:
                raise ValueError(f'{self.split_strategy.capitalize()} leakage between splits')
            if self.split_strategy == 'frequency' and np.ptp(self.k0[indices]) != 0:
                raise ValueError('Each frequency_id must have one k0 value')
        for name in ('A_cells', 'n_cells'):
            arr = getattr(self, name)
            if arr is not None and (arr.shape != (ns, len(self.triangles)) or not _all_finite(arr) or not _all_positive(arr)):
                raise ValueError(f'Invalid {name}')
        if self.split_strategy == 'frequency':
            for name in ('A', 'n', 'A_cells', 'n_cells'):
                values = getattr(self, name)
                if values is not None and any(np.any(batch != values[:1]) for batch in _array_batches(values)):
                    raise ValueError(f'{name} must stay fixed for frequency split')
        return self

    @property
    def split_group_ids(self):
        """Physical media for variable coefficients, frequencies for a fixed medium."""
        return self.frequency_id if self.split_strategy == 'frequency' else self.medium_id

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
        shape = true.shape[:-1]
        reference = true.reshape(-1, true.shape[-1])
        approximation = pred.reshape(-1, pred.shape[-1])
        def norm2(x):
            return np.maximum(np.real(np.sum(x.conj() * (self.mass_matrix @ x.T).T, axis=1)), 0)
        num, den = np.empty(len(reference)), np.empty(len(reference))
        for first in range(0, len(reference), 256):
            block = slice(first, first+256)
            num[block] = norm2(approximation[block]-reference[block])
            den[block] = norm2(reference[block])
        result = np.sqrt(np.divide(num, den, out=np.full_like(num, np.inf, dtype=float), where=den > 0))
        return np.where((den == 0) & (num == 0), 0., result).reshape(shape)


def _dataset_path(path):
    path = Path(path).expanduser().resolve()
    return path/'fem_dataset.npz' if path.is_dir() else path


def inspect_dataset(path):
    """Read archive headers and small sample vectors; never unpack field/cache data."""
    path = _dataset_path(path)
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not member.filename.endswith('.npy'):
                continue
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError(f'Unsupported NPY header version {version}')
                shape, _, dtype = reader(stream)
            arrays[member.filename[:-4]] = {'shape': list(shape), 'dtype': str(dtype)}
    with np.load(path, allow_pickle=False) as archive:
        split, medium, k = (archive[key] for key in ('split', 'medium_id', 'k0'))
        strategy = str(archive['split_strategy'].item()) if 'split_strategy' in archive else 'medium'
        groups = archive['frequency_id'] if strategy == 'frequency' else medium
    return {'directory': str(path.parent), 'samples': arrays['u_scat']['shape'][0],
            'nodes': arrays['u_scat']['shape'][1], 'media': len(np.unique(medium)),
            'split_strategy': strategy, 'split_groups': len(np.unique(groups)),
            'k_range_observed': [float(k.min()), float(k.max())],
            'splits': [{'name': name, 'samples': int(np.count_nonzero(split == j)),
                        'groups': len(np.unique(groups[split == j]))}
                       for j, name in enumerate(('train', 'validation', 'test'))],
            'stored_arrays': arrays, 'archive_bytes': path.stat().st_size}


def load_dataset(path, mmap=False):
    """Load shared samples, optionally streaming u_scat into a reusable mmap cache."""
    path = _dataset_path(path)
    directory = path.parent
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ('A','n','theta','k0','medium_id','split')}
        data['u_scat'] = _mapped_scattered_field(path) if mmap else archive['u_scat']
        data.update({key: archive[key] for key in ('A_cells','n_cells') if key in archive})
        data['split_strategy'] = str(archive['split_strategy'].item()) if 'split_strategy' in archive else 'medium'
        if 'frequency_id' in archive:
            data['frequency_id'] = archive['frequency_id']
        if data['split_strategy'] == 'frequency':
            samples = len(data['u_scat'])
            for key in ('A', 'n', 'A_cells', 'n_cells'):
                if key in data and data[key].shape[0] == 1:
                    data[key] = np.broadcast_to(data[key], (samples, data[key].shape[1]))
    with np.load(directory / 'geometry.npz', allow_pickle=False) as geometry:
        data.update(coordinates=geometry['dof_coords'][:, :2], triangles=geometry['triangles'])
        if 'cell_tags' in geometry:
            data['cell_tags'] = geometry['cell_tags']
        if 'etal_circles' in geometry:
            data['circles'] = geometry['etal_circles']
    return Dataset(**data, directory=directory)


def write_dataset_info(dataset):
    """Persist sizes/split counts and array hashes without copying large fields."""
    splits = []
    for name, indices in zip(("train", "validation", "test"), dataset.split_indices()):
        splits.append({"name": name, "samples": len(indices),
                       "media": len(np.unique(dataset.medium_id[indices])),
                       "groups": len(np.unique(dataset.split_group_ids[indices]))})
    hashes, arrays = {}, {}
    for name in ("A", "n", "u_scat", "A_cells", "n_cells", "theta", "k0", "medium_id",
                 "split", "frequency_id", "coordinates", "triangles", "cell_tags", "circles"):
        value = getattr(dataset, name)
        if value is None:
            continue
        hashes[name] = update_array_digest(hashlib.sha256(), value).hexdigest()
        arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes}
    info = {"samples": len(dataset.u_scat), "media": len(np.unique(dataset.medium_id)),
            "split_strategy": dataset.split_strategy,
            "split_groups": len(np.unique(dataset.split_group_ids)),
            "nodes": len(dataset.coordinates), "triangles": len(dataset.triangles),
            "k_range_observed": [float(dataset.k0.min()), float(dataset.k0.max())],
            "splits": splits, "arrays": arrays, "field_sha256": hashes}
    directory = Path(dataset.directory)
    info["files"] = {name: (directory/name).stat().st_size
                     for name in ("fem_dataset.npz", "geometry.npz", "etal.msh")}
    (directory/"dataset_info.json").write_text(json.dumps(info, indent=2)+"\n")
    return info
