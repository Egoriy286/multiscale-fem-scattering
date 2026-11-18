"""Fresh full-snapshot Type-2 GMsFEM, paired on the frozen high-k mesh.

One maximum-M offline space per frequency supplies the nested, predeclared M
values. No labels enter the offline construction or reduced Galerkin system.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .reduced import ReducedSystem as _ReducedSystem
from experiments.reference_sno_gmsfem.baselines import _materials
from experiments.reference_sno_gmsfem.dataset import _sha256, _write_json, _contrast_rhs
from .dataset import load_dataset


def _csv(path, rows):
    if not rows:
        return
    temporary = Path(str(path) + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate(data_dir, output_dir, *, n_coarse=20, modes=(16, 32), indices=None, fine_mesh=None, sparse_ordering='COLAMD'):
    """Resume complete frequency shards only after checking physical/source hashes.

    Timings on case rows are repeated setup costs; metadata totals count setup
    once per frequency and shared offline construction once for all M values.
    """
    from scipy.sparse.linalg import splu
    from .gms_highorder import P2Backend, build_offline_space

    data_dir, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    data = load_dataset(data_dir)
    if fine_mesh is None:
        raise ValueError('Declare the independent CG2 GMsFEM fine mesh explicitly')
    fine_mesh = Path(fine_mesh).resolve()
    reference_info = json.loads((data_dir/'dataset_info.json').read_text()).get('reference', {})
    selected = data.split_indices()[2] if indices is None else np.asarray(indices)
    if (selected.ndim != 1 or not len(selected) or selected.dtype.kind not in 'iu'
            or len(np.unique(selected)) != len(selected)
            or np.any(selected < 0) or np.any(selected >= len(data.k0))):
        raise ValueError('Expected nonempty distinct valid integer sample indices')
    if np.any(data.split[selected] != 2):
        raise ValueError('Only held-out test cases may be evaluated')
    if (not modes or any(isinstance(m, bool) or int(m) != m or m < 1 for m in modes)
            or isinstance(n_coarse, bool) or int(n_coarse) != n_coarse or n_coarse < 1):
        raise ValueError('Modes and n_coarse must be positive integers')
    expected_a, expected_n = _materials(data.cell_tags)
    for name,expected in (('A_cells',expected_a),('n_cells',expected_n)):
        values=getattr(data,name)
        if values is None or any(not np.array_equal(values[i],expected) for i in selected):
            raise ValueError('CG2 GMsFEM requires the same frozen notebook material table as the labels')
    modes = tuple(sorted(set(map(int, modes))))
    source = Path(__file__).resolve().parents[2]
    spec = dict(schema=2, fine_degree=2, fine_mesh=str(fine_mesh), fine_mesh_sha256=_sha256(fine_mesh), sparse_ordering=sparse_ordering, basis_type=2, n_coarse=int(n_coarse), modes=list(modes),
                indices=selected.tolist(), metric='individual scattered complex consistent P1 FEM L2 ratio',
                dataset_sha256={name: _sha256(data_dir/name) for name in
                                ('etal.msh', 'reference_problem.npz', 'samples.npz', 'u_scat.npy')},
                source_sha256={name: _sha256(source/name) for name in
                               ('gmsfem/offline.py', 'neural_operator/fem.py',
                                'experiments/reference_sno_gmsfem/baselines.py',
                                'experiments/highk_svd_gmsfem/baselines.py',
                                'experiments/highk_svd_gmsfem/dataset.py',
                                'experiments/highk_svd_gmsfem/gms_highorder.py',
                                'experiments/highk_svd_gmsfem/reduced.py',
                                'experiments/highk_svd_gmsfem/gms_refinement.py',
                                'experiments/highk_svd_gmsfem/baseline_parallel.py',
                                'experiments/highk_svd_gmsfem/highorder.py')})
    output.mkdir(parents=True, exist_ok=True)
    (output/'frequency_shards').mkdir(exist_ok=True)
    spec_path = output/'baseline_protocol.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError('Resume protocol differs; use a new output directory')
    _write_json(spec_path, spec)
    solver = None
    rows, records, fine_rows = [], [], []
    frequencies = np.unique(data.k0[selected])
    for position, k in enumerate(frequencies):
        chosen = selected[data.k0[selected] == k]
        key = hashlib.sha256(np.float64(k).tobytes()).hexdigest()[:16]
        shard_path = output/'frequency_shards'/f'k_{key}.json'
        if shard_path.exists():
            shard = json.loads(shard_path.read_text())
            if (shard['spec'] != spec or shard['k'] != float(k)
                    or sorted((r['index'], r['modes']) for r in shard['rows']) !=
                    sorted((int(i), m) for i in chosen for m in modes)):
                raise ValueError(f'Invalid frequency resume shard: {shard_path}')
            for row in shard['rows']:
                if (row['theta'] != float(data.theta[row['index']])
                        or not np.isfinite(row['error_fem_l2'])
                        or not 0 <= row['galerkin_residual'] <= 1e-7):
                    raise ValueError('Invalid resumed metric/query/residual')
        else:
            if solver is None:
                solver = P2Backend(fine_mesh)
            first = int(chosen[0])
            alpha, refractive = data.A_cells[first], data.n_cells[first]
            if any(not np.array_equal(getattr(data, name)[i], getattr(data, name)[first])
                   for name in ('A_cells', 'n_cells') for i in chosen):
                raise ValueError('Frequency reuse requires identical DG0 materials')
            started = perf_counter()
            problem = solver.problem(float(k), float(data.theta[first]))
            assembly_seconds = perf_counter()-started
            space = build_offline_space(problem, n_coarse=n_coarse, max_modes=max(modes), basis_type=2,
                                        progress=lambda message: print(f'k={k:.9g}: {message}', flush=True))
            record = dict(k=float(k), offline_seconds=space.build_seconds,
                          assembly_seconds=assembly_seconds, diagnostics=space.diagnostics,
                          max_modes=max(modes), eigenvalue_min=float(space.eigenvalues.min()),
                          eigenvalue_max=float(space.eigenvalues.max()), systems=[])
            started = perf_counter()
            fine_factor = splu(problem.operator.tocsc())
            record['fine_native_factorization_seconds'] = perf_counter()-started
            fine_cases = []
            for raw_index in chosen:
                index = int(raw_index)
                theta = float(data.theta[index])
                rhs = solver.problem(float(k), theta).rhs
                started = perf_counter()
                native_fine = fine_factor.solve(rhs)
                fine_solve_seconds = perf_counter()-started
                residual = float(np.linalg.norm(problem.operator@native_fine-rhs)/max(np.linalg.norm(rhs), 1e-30))
                mapped = solver.evaluate(native_fine, data.coordinates)
                error = float(data.relative_l2(data.u_scat[index], mapped))
                if not np.isfinite(error) or not np.isfinite(residual) or residual > 1e-7:
                    raise ValueError('Native fine CG2 diagnostic solve failed')
                fine_cases.append(dict(method='Native CG2 discretization diagnostic', index=index, k=float(k), theta=theta,
                                       error_fem_l2=error, error_percent=100*error, dofs_fine=problem.ndofs,
                                       solve_seconds=fine_solve_seconds, factorization_seconds=record['fine_native_factorization_seconds'],
                                       linear_residual=residual))
            del fine_factor, native_fine
            case_rows = []
            # Construct one reduced system at a time to bound memory.
            for m in modes:
                system = _ReducedSystem(problem, space, m, ordering=sparse_ordering)
                record['systems'].append(dict(modes=m, matrix_projection_seconds=system.matrix_projection_seconds,
                                               factorization_seconds=system.factorization_seconds))
                for raw_index in chosen:
                    index = int(raw_index)
                    theta = float(data.theta[index])
                    started = perf_counter()
                    rhs = solver.problem(float(k), theta).rhs
                    rhs_seconds = perf_counter()-started
                    current = replace(problem, rhs=rhs, direction=(float(np.cos(theta)), float(np.sin(theta))))
                    prediction, timing = system.solve(current)
                    started = perf_counter()
                    mapped = solver.evaluate(prediction, data.coordinates)
                    evaluation_seconds = perf_counter()-started
                    error = float(data.relative_l2(data.u_scat[index], mapped))
                    if not np.isfinite(error):
                        raise ValueError('Nonfinite GMsFEM error')
                    case_rows.append(dict(method=f'GMsFEM CG2 Type2 {n_coarse}x{n_coarse} M={m}',
                                          index=index, k=float(k), theta=theta, basis_type=2,
                                          n_coarse=n_coarse, modes=m, dofs_fine=problem.ndofs, fine_degree=2,
                                          dofs_support=len(data.coordinates), dofs_native_truth=reference_info.get('native_dofs', 0),
                                          support_evaluation_seconds=evaluation_seconds,
                                          error_fem_l2=error, error_percent=100*error,
                                          offline_seconds=space.build_seconds,
                                          assembly_seconds=assembly_seconds, rhs_assembly_seconds=rhs_seconds,
                                          **timing))
                del system
            shard = dict(spec=spec, k=float(k), rows=case_rows, fine_reference_rows=fine_cases, record=record)
            _write_json(shard_path, shard)
            del space, problem
        rows.extend(shard['rows'])
        fine_rows.extend(shard['fine_reference_rows'])
        records.append(shard['record'])
        rows.sort(key=lambda row: (row['modes'], row['index']))
        payload = dict(rows=rows, fine_reference_rows=fine_rows, metadata=dict(protocol=spec, frequencies=records,
                       complete=len(records) == len(frequencies),
                       unique_offline_seconds=sum(r['offline_seconds'] for r in records),
                       timing_note='Case setup timings repeat; count frequency records once. Offline Mmax shared by nested M.'))
        _write_json(output/'baselines.json', payload)
        _csv(output/'baselines.csv', rows)
        _csv(output/'native_cg2_diagnostic.csv', fine_rows)
        print(f'Completed GMsFEM frequency {position+1}/{len(frequencies)} k={k:.9g}', flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-coarse', type=int, default=20)
    parser.add_argument('--modes', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--sparse-ordering',default='COLAMD',choices=['COLAMD','MMD_AT_PLUS_A','ND','ND2','ND_NOPIVOT'])
    parser.add_argument('--fine-mesh', required=True)
    parser.add_argument('--indices', type=int, nargs='+', help='Explicit held-out smoke subset; use separate output')
    args = parser.parse_args()
    evaluate(args.data_dir, args.output_dir, n_coarse=args.n_coarse, modes=args.modes, indices=args.indices, fine_mesh=args.fine_mesh,sparse_ordering=args.sparse_ordering)


if __name__ == '__main__':
    main()
