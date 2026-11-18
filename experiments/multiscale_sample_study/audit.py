"""Independent direct-field physical score audit, without neural model loading."""
from __future__ import annotations

import argparse
import csv
import itertools
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.reference_sno_gmsfem.dataset import _sha256, _write_json, load_dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METHODS = {'sno_gelu', 'mscale_sno', 'fno_gelu', 'mscale_fno'}


def total_denominators(data, indices):
    """Integrate |P1(u_scat)+analytic incident|² on reference quadrature."""
    from gmsfem.metrics import BARYCENTRIC, WEIGHTS
    vertices = data.coordinates[data.triangles]
    edges = vertices[:, 1:]-vertices[:, :1]
    areas = abs(edges[:, 0, 0]*edges[:, 1, 1]-edges[:, 0, 1]*edges[:, 1, 0])/2
    points = np.einsum('qj,cjd->cqd', BARYCENTRIC, vertices)
    weights = areas[:, None]*WEIGHTS[None, :]
    norms = []
    for index in indices:
        direction = np.array([np.cos(data.theta[index]), np.sin(data.theta[index])])
        incident = np.exp(1j*data.k0[index]*(points@direction))
        total = data.u_scat[index][data.triangles]@BARYCENTRIC.T+incident
        norms.append(float(np.sum(abs(total)**2*weights)))
    return np.asarray(norms)


def direct_scores(data, indices, prediction, total_norms, batch_size=8):
    """Compute both physical percentage errors from bounded field batches."""
    indices = np.asarray(indices)
    prediction = np.asarray(prediction)
    total_norms = np.asarray(total_norms)
    if (prediction.shape != (len(indices), len(data.coordinates)) or
            not np.iscomplexobj(prediction) or not np.isfinite(prediction).all() or
            total_norms.shape != (len(indices),) or not np.isfinite(total_norms).all() or
            np.any(total_norms <= 0)):
        raise ValueError('Invalid predicted fields or total-field denominators')
    scattered, total = [], []
    for start in range(0, len(indices), batch_size):
        stop = min(start+batch_size, len(indices))
        true = np.asarray(data.u_scat[indices[start:stop]])
        difference = prediction[start:stop]-true
        def norm2(fields):
            return np.real(np.sum(fields.conj()*(data.mass_matrix@fields.T).T, axis=1))
        error2, true2 = norm2(difference), norm2(true)
        if not np.isfinite(error2).all() or np.any(error2 < 0) or np.any(true2 <= 0):
            raise ValueError('Invalid direct FEM mass quadratic form')
        scattered.extend(100*np.sqrt(error2/true2))
        total.extend(100*np.sqrt(error2/total_norms[start:stop]))
    return np.asarray(scattered), np.asarray(total)


def interpolation_roundtrip_errors(data, sampler, grid_to_fem, batch_size=8):
    """FEM P1 evaluation on grid, then grid reconstruction; never a lower bound."""
    errors = np.empty(len(data.k0), dtype=np.float64)
    for start in range(0, len(errors), batch_size):
        stop = min(start+batch_size, len(errors))
        truth = np.asarray(data.u_scat[start:stop])
        grid = sampler.apply(truth)
        reconstruction = (grid_to_fem@grid.T).T
        indices = np.arange(start, stop)
        # The total denominator is unused here; direct_scores supplies the same
        # independently integrated consistent-mass scattered-field norm.
        errors[start:stop], _ = direct_scores(data, indices, reconstruction,
                                               np.ones(stop-start), batch_size)
    return errors


def _statistics(values):
    values = np.asarray(values)
    return dict(cases=len(values), mean_pct=float(values.mean()),
                p95_pct=float(np.percentile(values,95)), max_pct=float(values.max()))


def representation_diagnostics(root, data, protocol, manifests):
    """Distinguish strict fixed-decoder floors from ordinary interpolation loss."""
    from scipy.sparse import load_npz
    from neural_operator.mesh_transfer import P1Transfer
    directory = Path(root)/'cache/n1000/fno'
    with np.load(directory/'fno_preprocessing.npz', allow_pickle=False) as archive:
        points = archive['points']
    sampler = P1Transfer(data.coordinates, data.triangles, points)
    errors = interpolation_roundtrip_errors(data, sampler,
                    load_npz(directory/'fno_transfer.npz'), batch_size=8)
    train, validation, test = data.split_indices()
    subsets = {f'train_{n}':_statistics(errors[train[:n]]) for n in protocol['train_sizes']}
    subsets.update(validation=_statistics(errors[validation]), test=_statistics(errors[test]),
                   all_queries=_statistics(errors))
    floors = []
    for (n_train, family), manifest in sorted(manifests.items()):
        diagnostics = manifest['diagnostics']
        floors.append(dict(n_train=n_train, family=family, rank=diagnostics.get('rank'),
            quantity='fixed_decoder_projection_floor',
            interpretation=('Strict complex affine POD decoder lower bound in the consistent FEM mass norm; train-only basis.'
                            if family=='sno' else
                            'Strict optimal mass-projection lower bound for the fixed bilinear grid decoder; G=T.T M T.'),
            mean_pct=diagnostics['projection_floor_mean_pct'],
            max_pct=diagnostics['projection_floor_max_pct']))
    return dict(passed=True, cached_fixed_decoder_projection_floors=floors,
        exact_label_interpolation_roundtrip=dict(
            quantity='FEM P1 -> regular grid -> bilinear FEM reconstruction',
            interpretation='Interpolation diagnostic only; not a lower bound on learned-model error.',
            batch_size=8, cases=len(errors), subsets=subsets,
            original_indices=list(range(len(errors))), errors_pct=errors.tolist()))


def compare_norm_cache(expected, stored):
    expected, stored = np.asarray(expected), np.asarray(stored)
    if expected.shape != stored.shape or not np.allclose(expected, stored, rtol=2e-12, atol=1e-12):
        raise ValueError('Cached total-field denominators differ from independent quadrature')


def _read(path):
    return json.loads(Path(path).read_text())


def _compare(actual, expected, label, rtol=2e-7, atol=1e-7):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(f'Independent physical score mismatch: {label}')
    return float(np.max(abs(actual-expected), initial=0.))


def compare_rows(rows, indices, k, theta, scattered, total, label):
    """Require exact identity pairing, then compare both per-case percentages."""
    if len(rows) != len(indices):
        raise ValueError(f'Wrong number of test metrics: {label}')
    lookup = {}
    for row in rows:
        index = int(row['index'])
        if index in lookup:
            raise ValueError(f'Duplicate test metric: {label}')
        lookup[index] = row
    if set(lookup) != set(map(int, indices)):
        raise ValueError(f'Physical test identities differ: {label}')
    maximum = 0.
    for index, kval, angle, s, t in zip(indices, k, theta, scattered, total, strict=True):
        row = lookup[int(index)]
        if float(row['k']) != float(kval) or float(row['theta']) != float(angle):
            raise ValueError(f'Physical test query differs: {label}')
        maximum = max(maximum, _compare([row['scattered_pct'], row['total_pct']], [s, t], label))
    return maximum


def audit(root=HERE, *, include_representation_diagnostics=True):
    """Audit the completed 48-run study, then bind its result to verification."""
    root = Path(root).expanduser().resolve()
    protocol = _read(root/'protocol.json')
    if (set(protocol['train_sizes']) != {400,600,800,1000} or
            set(protocol['methods']) != METHODS or set(protocol['seeds']) != {11,29,47} or
            any(len(protocol[key]) != len(set(protocol[key])) for key in ('train_sizes','methods','seeds'))):
        raise ValueError('Expected the frozen four-size, four-method, three-seed study')
    expected = set(itertools.product(protocol['train_sizes'], protocol['methods'], protocol['seeds']))
    verification = _read(root/'results/verification.json')
    verified = verification.get('runs', [])
    def key(row):
        return int(row['n_train']), row['method'], int(row['seed'])
    if verification.get('passed') is not True or len(verified) != 48 or {key(row) for row in verified} != expected:
        raise ValueError('Main checkpoint replay verification must pass for all 48 runs first')
    # Hash each referenced file at most once throughout this audit.
    hashes = {}
    def hashed(path):
        path = Path(path).resolve()
        if path not in hashes:
            hashes[path] = _sha256(path)
        return hashes[path]
    protocol_hash = hashed(root/'protocol.json')
    run_protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if verification.get('protocol_sha256') != protocol_hash:
        raise ValueError('Replay verification protocol hash differs')
    for name, digest in protocol['source_sha256'].items():
        if hashed(REPO/name) != digest:
            raise ValueError('Frozen training source changed')
    for name, digest in protocol['dataset_sha256'].items():
        if hashed(root/'data'/name) != digest:
            raise ValueError('Frozen physical dataset changed')
    cache_hashes, cache_manifests = {}, {}
    for n_train, family in itertools.product(protocol['train_sizes'], ('sno','fno')):
        directory = root/'cache'/f'n{n_train}'/family
        marker = _read(directory/'cache.json')
        spec = marker['specification']
        cache_manifests[n_train, family] = marker
        if (spec['n_train'] != n_train or spec['family'] != family or
                spec['dataset_sha256'] != protocol['dataset_sha256'] or
                spec['source_sha256'] != protocol['source_sha256'] or
                spec['pod_rank'] != protocol['pod_rank'] or spec['resolution'] != protocol['resolution']):
            raise ValueError('Preprocessing cache specification changed')
        cache_hashes[str((directory/'cache.json').relative_to(root))] = hashed(directory/'cache.json')
        for name, digest in marker['files'].items():
            actual = hashed(directory/name)
            if actual != digest:
                raise ValueError('Preprocessing cache contents changed')
            cache_hashes[str((directory/name).relative_to(root))] = actual
    data = load_dataset(root/'data')
    indices = data.split_indices()[2]
    if (len(indices) != 100 or len(np.unique(data.k0)) != 1200 or len(data.k0) != 1200 or
            not np.array_equal(np.bincount(data.split, minlength=3), [1000,100,100]) or
            len(np.unique(data.frequency_id)) != 1200):
        raise ValueError('Expected 1200 distinct physical queries and 100 test cases')
    norms = total_denominators(data, indices)
    norm_path = root/'cache/test_total_norms.npy'
    compare_norm_cache(norms, np.load(norm_path, allow_pickle=False))
    cache_hashes[str(norm_path.relative_to(root))] = hashed(norm_path)
    with (root/'results/metrics.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    grouped = {}
    for row in csv_rows:
        if key(row) not in expected:
            raise ValueError('Unexpected run in aggregate metrics')
        grouped.setdefault(key(row), []).append(row)
    if set(grouped) != expected:
        raise ValueError('Aggregate metrics omit a run')
    records, max_difference = [], 0.
    verified = {key(row):row for row in verified}
    for n_train, method, seed in sorted(expected):
        ident = (n_train, method, seed)
        directory = root/'runs'/f'n{n_train}'/method/f'seed_{seed}'
        result = _read(directory/'result.json')
        if key(result) != ident:
            raise ValueError('Result identity differs')
        config = _read(directory/'config.json')
        if config['specification'] != dict(protocol_sha256=run_protocol_hash, n_train=n_train, method=method, seed=seed):
            raise ValueError('Run configuration differs from frozen protocol')
        history = _read(directory/'history.json')
        best = min(history, key=lambda row:row['val_relative_l2'])
        if best['epoch'] != result['best_epoch'] or not np.isclose(best['val_relative_l2'], result['best_validation_relative_l2'], rtol=1e-12):
            raise ValueError('Recorded checkpoint is not the validation minimum')
        for name, field in [('model.pt','checkpoint_sha256'), ('predictions.npz','predictions_sha256')]:
            digest = hashed(directory/name)
            if digest != result[field] or digest != verified[ident][field]:
                raise ValueError('Verified checkpoint or prediction hash changed')
        with np.load(directory/'predictions.npz', allow_pickle=False) as saved:
            if (not np.array_equal(saved['indices'], indices) or
                    not np.array_equal(saved['k'], data.k0[indices]) or
                    not np.array_equal(saved['theta'], data.theta[indices])):
                raise ValueError('Saved predictions differ from frozen physical heldout queries')
            scattered, total = direct_scores(data, indices, saved['prediction'], norms)
        rows = _read(directory/'metrics.json')
        if any(key(row) != ident for row in rows):
            raise ValueError('Per-run metric identity differs')
        for label, values in [('metrics.json',rows), ('metrics.csv',grouped[ident])]:
            max_difference = max(max_difference, compare_rows(values,indices,data.k0[indices],data.theta[indices],scattered,total,str(directory/label)))
        for prefix, values in [('scattered',scattered), ('total',total)]:
            max_difference = max(max_difference, _compare(
                [result[prefix+'_mean_pct'],result[prefix+'_p95_pct']],
                [values.mean(),np.percentile(values,95)], str(directory/'result.json')))
        records.append(dict(n_train=n_train,method=method,seed=seed,cases=100,
                            best_epoch=int(best['epoch']),checkpoint_sha256=hashed(directory/'model.pt'),
                            predictions_sha256=hashed(directory/'predictions.npz')))
    value = dict(passed=True, runs=48, count_cases=4800, unique_test_cases=100,
                 max_metric_difference_pct=max_difference, cache_hashes=cache_hashes,
                 protocol_sha256=protocol_hash, auditor_source_sha256=hashed(Path(__file__)),
                 dtype_note='Frozen generic online_dtype field is imprecise: CPU SNO loss is float32; validation and direct physical scoring are float64.',
                 records=records)
    if include_representation_diagnostics:
        diagnostics = representation_diagnostics(root, data, protocol, cache_manifests)
        diagnostics['protocol_sha256'] = protocol_hash
        diagnostics['auditor_source_sha256'] = value['auditor_source_sha256']
        diagnostic_path = root/'results/representation_diagnostics.json'
        _write_json(diagnostic_path, diagnostics)
        value['representation_diagnostics'] = dict(path=diagnostic_path.name, sha256=hashed(diagnostic_path))
    path = root/'results/physical_audit.json'
    _write_json(path,value)
    verification['physical_audit'] = dict(passed=True, path='physical_audit.json',sha256=hashed(path),auditor_source_sha256=value['auditor_source_sha256'])
    _write_json(root/'results/verification.json', verification)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=HERE)
    value = audit(parser.parse_args().root)
    print(f"Physical score audit passed: {value['runs']} runs, {value['count_cases']} cases")


if __name__ == '__main__':
    main()
