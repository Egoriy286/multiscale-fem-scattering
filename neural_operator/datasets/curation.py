"""Read-only, bounded-memory archive audit and explicit dataset quality gate.

This module never removes samples, rewrites a split, or deletes a dataset.
Refinement differences assess sampled discrete labels, not continuous PDE error.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from ..data import inspect_dataset
from .config import load_config, plan_dataset


def quality_status(integrity_errors, refinement_errors=(), *, obsolete=False,
                   threshold=.02):
    """Classify whole recipes without consulting a neural model's test error."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold must be positive and finite')
    values = np.asarray(refinement_errors, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('refinement errors must be finite nonnegative values')
    if integrity_errors:
        return 'invalid_integrity'
    if obsolete:
        return 'obsolete_recipe'
    if not len(values):
        return 'refinement_pending'
    return 'stress_only' if values.max() > threshold else 'sampled_refinement_pass'


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_archive(path):
    """Read each payload to EOF (CRC checked by zipfile), without field copies."""
    arrays, errors = {}, []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not member.filename.endswith('.npy'):
                continue
            name = member.filename[:-4]
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError(f'Unsupported NPY header {version}')
                shape, fortran, dtype = reader(stream)
                if dtype.hasobject:
                    raise ValueError(f'Object array rejected: {name}')
                digest, count, finite, positive = hashlib.sha256(), 0, True, True
                block_bytes = max(1, (1024 * 1024) // dtype.itemsize) * dtype.itemsize
                while chunk := stream.read(block_bytes):
                    count += len(chunk)
                    digest.update(chunk)
                    if dtype.kind in 'biufc':
                        values = np.frombuffer(chunk, dtype=dtype)
                        finite = finite and bool(np.all(np.isfinite(values)))
                        if name in ('A', 'n', 'A_cells', 'n_cells', 'k0'):
                            positive = positive and bool(np.all(values > 0))
                if count != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
                    errors.append(f'{name}: payload size does not match shape')
                if not finite:
                    errors.append(f'{name}: nonfinite values')
                if not positive:
                    errors.append(f'{name}: nonpositive material or frequency')
                arrays[name] = dict(shape=list(shape), dtype=str(dtype),
                                    order='F' if fortran else 'C',
                                    payload_sha256=digest.hexdigest())
    return arrays, errors


def audit_dataset(directory, evidence=None, *, threshold=.02):
    """Audit one dataset; supplied evidence names explicit refinement samples."""
    directory = Path(directory)
    evidence = evidence or {}
    record = dict(dataset=directory.name, quality_threshold=threshold,
                  refinement=evidence.get('refinement', []),
                  obsolete=bool(evidence.get('obsolete', False)),
                  reason=evidence.get('reason', ''), files={}, integrity_errors=[])
    try:
        record['config'] = json.loads((directory / 'config.json').read_text())
        plan = plan_dataset(load_config(directory))
        record['inspection'] = inspect_dataset(directory)
        for filename in ('fem_dataset.npz', 'geometry.npz'):
            path = directory / filename
            arrays, errors = _check_archive(path)
            record['files'][filename] = dict(bytes=path.stat().st_size,
                                             sha256=_sha256(path), arrays=arrays)
            record['integrity_errors'].extend(errors)
        inspection = record['inspection']
        if inspection['samples'] != plan['samples']:
            record['integrity_errors'].append('Sample count differs from recipe')
        if inspection['split_strategy'] != plan['split_strategy']:
            record['integrity_errors'].append('Split strategy differs from recipe')
        if inspection['splits'] != plan['splits']:
            record['integrity_errors'].append('Split counts differ from recipe')
        stored = record['files']['fem_dataset.npz']['arrays']
        ns, nn = inspection['samples'], inspection['nodes']
        if stored['u_scat']['shape'] != [ns, nn]:
            record['integrity_errors'].append('u_scat: expected exactly [samples, nodes]')
        for name in ('A', 'n', 'theta', 'k0'):
            if np.dtype(stored[name]['dtype']).kind not in 'iuf':
                record['integrity_errors'].append(f'{name}: expected real numeric dtype')
        for name in ('medium_id', 'split'):
            if np.dtype(stored[name]['dtype']).kind not in 'iu':
                record['integrity_errors'].append(f'{name}: expected integer dtype')
        if np.dtype(stored['u_scat']['dtype']).kind not in 'iufc':
            record['integrity_errors'].append('u_scat: expected numeric dtype')
        for name in ('A', 'n'):
            allowed = ([ns, nn], [1, nn]) if plan['medium_mode'] == 'static' else ([ns, nn],)
            if stored[name]['shape'] not in allowed:
                record['integrity_errors'].append(f'{name}: wrong sample/node shape')
        for name in ('theta', 'k0', 'medium_id', 'split'):
            if stored[name]['shape'] != [ns]:
                record['integrity_errors'].append(f'{name}: wrong sample shape')
        manifest_path = directory / 'dataset_info.json'
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            matched = []
            for name, metadata in manifest.get('arrays', {}).items():
                # Static manifests hash broadcast materials; compare only exact stored shapes.
                if (name in stored and stored[name]['shape'] == metadata['shape']
                        and name in manifest.get('field_sha256', {})):
                    if stored[name]['payload_sha256'] != manifest['field_sha256'][name]:
                        record['integrity_errors'].append(f'{name}: manifest SHA256 mismatch')
                    matched.append(name)
            record['verified_manifest_arrays'] = matched
        with np.load(directory / 'fem_dataset.npz', allow_pickle=False) as archive:
            split, k0 = archive['split'], archive['k0']
            group_key = 'frequency_id' if inspection['split_strategy'] == 'frequency' else 'medium_id'
            groups = archive[group_key]
            if not np.all(np.isin(split, (0, 1, 2))):
                record['integrity_errors'].append('Unknown split code')
            for group in np.unique(groups):
                mask = groups == group
                if len(np.unique(split[mask])) != 1:
                    record['integrity_errors'].append('Group leakage between splits')
                    break
                if inspection['split_strategy'] == 'frequency' and np.ptp(k0[mask]) != 0:
                    record['integrity_errors'].append('A frequency group has multiple k0')
                    break
        with np.load(directory / 'geometry.npz', allow_pickle=False) as geometry:
            triangles, coords = geometry['triangles'], geometry['dof_coords']
            if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) != nn:
                record['integrity_errors'].append('Geometry node count/shape differs from field nodes')
            if (triangles.ndim != 2 or triangles.shape[1] != 3
                    or triangles.dtype.kind not in 'iu' or not triangles.size
                    or triangles.min() < 0 or triangles.max() >= len(coords)):
                record['integrity_errors'].append('Invalid triangular connectivity')
            else:
                xy = coords[triangles, :2]
                cross = ((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                         - (xy[:, 1, 1] - xy[:, 0, 1]) * (xy[:, 2, 0] - xy[:, 0, 0]))
                if np.any(cross == 0):
                    record['integrity_errors'].append('Degenerate triangles')
        for filename in ('config.json', 'dataset_info.json', 'etal.msh'):
            path = directory / filename
            if path.is_file():
                record['files'][filename] = dict(bytes=path.stat().st_size,
                                                 sha256=_sha256(path))
    except (OSError, ValueError, TypeError, IndexError, KeyError, EOFError, zipfile.BadZipFile) as error:
        record['integrity_errors'].append(f'{type(error).__name__}: {error}')
    record['status'] = quality_status(
        record['integrity_errors'],
        [row['error_fem_l2'] for row in record['refinement']],
        obsolete=record['obsolete'], threshold=threshold)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('neural_operator/data'))
    parser.add_argument('--evidence', type=Path,
                        help='JSON mapping dataset names to refinement/obsolete evidence')
    parser.add_argument('--threshold', type=float, default=.02)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text()) if args.evidence else {}
    records = [audit_dataset(path, evidence.get(path.name), threshold=args.threshold)
               for path in sorted(args.root.iterdir())
               if path.is_dir() and (path / 'fem_dataset.npz').is_file()]
    if not records:
        parser.error('No datasets found; refusing an empty audit')
    report = dict(schema_version=1, threshold=args.threshold,
                  policy='whole-recipe quality gate; no model-test-error filtering',
                  datasets=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    for record in records:
        print(f"{record['dataset']}: {record['status']}")
    return int(any(record['integrity_errors'] for record in records))


if __name__ == '__main__':
    raise SystemExit(main())
