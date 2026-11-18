"""New fixed-medium FEM queries for nested sample-size ablations.

One sample is one physical frequency and its complex fine-FEM field. This is
an adaptation of the article's sample counts, not its random-medium mapping.
Heldouts precede training rows so every training view is a zero-copy mmap slice.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from neural_operator.data import Dataset
from experiments.reference_sno_gmsfem.dataset import (
    REFERENCE_SHA256, SCHEMA, _reference_problem, _save_npy, _save_npz,
    _sha256, _write_json, load_dataset,
)

DEFAULT_TRAIN_SIZES = (400, 600, 800, 1000)


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def query_hash(queries):
    """Content digest independent of NPZ ZIP metadata."""
    digest = hashlib.sha256()
    for name in sorted(queries):
        array = np.ascontiguousarray(queries[name])
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_queries(train_sizes=DEFAULT_TRAIN_SIZES, n_validation=100, n_test=100, seed=20261113):
    """Plan nested, near-uniform training prefixes and independent heldouts.

    Begin with an endpoint-inclusive uniform training grid. At every larger
    requested size, bisect the widest gaps, distributing ties across the whole
    interval. New points are jittered within the middle tenth of their gap.
    Heldouts use independent stratified jitter on (0.1,4); they never coincide
    with a training frequency or one another. No FEM labels enter this plan.
    """
    sizes = tuple(_integer(n, 'train size', 2) for n in train_sizes)
    if not sizes or tuple(sorted(set(sizes))) != sizes:
        raise ValueError('train_sizes must be nonempty and strictly increasing')
    nv, nt = _integer(n_validation, 'n_validation'), _integer(n_test, 'n_test')
    seed = _integer(seed, 'seed', 0)
    rng = np.random.default_rng(seed)
    train = list(np.linspace(.1, 4., sizes[0]))
    for size in sizes[1:]:
        remaining = size-len(train)
        while remaining:
            ordered = np.sort(train)
            gaps = np.diff(ordered)
            # Widest bins first. An independent random key avoids left/right bias.
            priority = np.lexsort((rng.random(len(gaps)), -np.round(gaps, 13)))
            count = min(remaining, len(gaps))
            chosen = priority[:count]
            points = ordered[chosen]+gaps[chosen]*rng.uniform(.45, .55, count)
            # Order new rows throughout the domain rather than in gap-size order.
            train.extend(points[np.argsort(points)].tolist())
            remaining -= count
    used = set(train)
    heldout = []
    for count in (nv, nt):
        values = .1+3.9*(np.arange(count)+rng.uniform(.05, .95, count))/count
        for index, value in enumerate(values):
            while float(value) in used:
                value = np.nextafter(value, 4.)
            values[index] = value
            used.add(float(value))
        heldout.append(values)
    k = np.r_[heldout[0], heldout[1], train]
    return dict(frequencies=k.copy(), frequency_id=np.arange(len(k), dtype=np.int32),
                k0=k, theta=np.zeros(len(k)), medium_id=np.zeros(len(k), dtype=np.int32),
                split=np.r_[np.ones(nv, dtype=np.int8), np.full(nt, 2, dtype=np.int8),
                            np.zeros(sizes[-1], dtype=np.int8)])


def training_view(data, n_train):
    """Return a native Dataset sharing label storage, retaining original row IDs.

    The source must have this module's heldout-first layout. Only declared
    training sizes may be requested from a generated dataset.
    """
    n_train = _integer(n_train, 'n_train', 2)
    n_heldout = int(np.count_nonzero(data.split != 0))
    if not np.all(data.split[:n_heldout] != 0) or not np.all(data.split[n_heldout:] == 0):
        raise ValueError('Expected heldout-first nested training layout')
    if n_train > len(data.split)-n_heldout:
        raise ValueError('Requested more training samples than available')
    if data.directory is not None and (Path(data.directory)/'config.json').exists():
        recipe = json.loads((Path(data.directory)/'config.json').read_text())
        if n_train not in recipe['train_sizes']:
            raise ValueError('Training size was not declared in the frozen query plan')
    stop = n_heldout+n_train
    fields = ('A', 'n', 'u_scat', 'theta', 'k0', 'medium_id', 'split', 'A_cells', 'n_cells', 'frequency_id')
    selected = {name: (None if getattr(data, name) is None else getattr(data, name)[:stop]) for name in fields}
    selected.update({name: getattr(data, name) for name in ('coordinates', 'triangles', 'cell_tags', 'circles', 'directory')})
    view = Dataset(**selected, split_strategy='frequency')
    view._mass = data._mass
    view.original_indices = np.arange(stop, dtype=np.int64)
    view.n_train = n_train
    return view


def generate_dataset(root, train_sizes=DEFAULT_TRAIN_SIZES, n_validation=100, n_test=100, seed=20261113):
    """Create/resume independently solved labels; verify all files on reopening."""
    queries = make_queries(train_sizes, n_validation, n_test, seed)
    directory = Path(root).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    total = len(queries['k0'])
    recipe = dict(schema_version=SCHEMA, experiment='multiscale_sample_study',
                  storage='native_npy_complex128', train_sizes=list(map(int, train_sizes)),
                  n_validation=int(n_validation), n_test=int(n_test), seed=int(seed),
                  samples=total, new_generation=True, k_range=[.1, 4.], theta=0.,
                  sample_unit='one independent physical frequency and complex fine FEM solution',
                  adaptation='fixed notebook medium; not article random-medium mapping',
                  split_policy='fixed stratified-jitter heldouts; nested gap-refined training prefixes; endpoints train',
                  row_order='validation,test,nested_training', medium_mode='static',
                  case='notebook', h=.02, n_coarse=20,
                  reference_mesh_sha256=REFERENCE_SHA256,
                  generator_sha256=_sha256(Path(__file__)), query_sha256=query_hash(queries))
    config = directory/'config.json'
    if config.exists():
        if json.loads(config.read_text()) != recipe:
            raise ValueError('Frozen dataset recipe differs; use a new directory')
    else:
        if any(directory.iterdir()):
            raise ValueError('Nonempty dataset directory without recipe')
        _write_json(config, recipe)
    samples = directory/'samples.npz'
    if samples.exists():
        with np.load(samples, allow_pickle=False) as old:
            if set(old.files) != set(queries) or any(not np.array_equal(old[k], v) for k, v in queries.items()):
                raise ValueError('Saved query manifest differs')
    else:
        _save_npz(samples, **queries)
    if (directory/'dataset_info.json').exists():
        info = json.loads((directory/'dataset_info.json').read_text())
        for name, expected in info['sha256'].items():
            if _sha256(directory/name) != expected:
                raise ValueError(f'Saved {name} checksum differs')
        return load_dataset(directory)
    started = perf_counter()
    solver, problem = _reference_problem(directory)
    setup_seconds = perf_counter()-started
    shape = (total, len(solver.coordinates))
    path = directory/'u_scat.npy'
    if path.exists():
        fields = np.load(path, mmap_mode='r+', allow_pickle=False)
        if fields.shape != shape or fields.dtype != np.complex128:
            raise ValueError('Unexpected label array')
    else:
        fields = np.lib.format.open_memmap(path, mode='w+', dtype=np.complex128, shape=shape)
        fields[:] = np.nan+1j*np.nan
        fields.flush()
    bitmap, timings_path = directory/'completed.npy', directory/'generation_times.npy'
    completed = np.load(bitmap, allow_pickle=False) if bitmap.exists() else np.zeros(total, dtype=bool)
    times = np.load(timings_path, allow_pickle=False) if timings_path.exists() else np.zeros(total)
    if completed.shape != (total,) or completed.dtype != bool:
        raise ValueError('Invalid completion bitmap')
    if times.shape != (total,) or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError('Invalid generation timings')
    before = int(completed.sum())
    for index, k in enumerate(queries['k0']):
        if completed[index]:
            if not np.isfinite(fields[index]).all():
                raise ValueError('Committed field is nonfinite')
            continue
        step = perf_counter()
        # Always solve the new query; stored historical labels are never reused.
        fields[index] = solver.solve(problem['A_cells'][0], problem['n_cells'][0], float(k), 0.)
        times[index] = perf_counter()-step
        if not np.isfinite(fields[index]).all():
            raise FloatingPointError('Nonfinite FEM solution')
        fields.flush()
        _save_npy(timings_path, times)
        completed[index] = True
        _save_npy(bitmap, completed)
        if completed.sum() == before+1 or completed.sum() % 10 == 0 or completed.all():
            print(f'New scaling-study FEM labels: {completed.sum()}/{total}; {perf_counter()-started:.1f}s', flush=True)
    del fields
    names = ('config.json', 'etal.msh', 'reference_problem.npz', 'samples.npz', 'u_scat.npy', 'completed.npy', 'generation_times.npy')
    info = dict(schema_version=SCHEMA, complete=True, samples=total,
                nodes=shape[1], triangles=len(solver.triangles), media=1,
                split_strategy='frequency', frequency_groups=total,
                new_fem_labels=total, inherited_fem_labels=0,
                query_sha256=recipe['query_sha256'], setup_and_anchor_seconds=setup_seconds,
                label_solve_seconds=float(times.sum()), resumed_frequency_groups=before,
                current_session_seconds=perf_counter()-started,
                splits=[dict(name=name, samples=int(np.sum(queries['split'] == j)),
                             groups=int(np.sum(queries['split'] == j)))
                        for j, name in enumerate(('train', 'validation', 'test'))],
                sha256={name: _sha256(directory/name) for name in names})
    _write_json(directory/'dataset_info.json', info)
    return load_dataset(directory)
