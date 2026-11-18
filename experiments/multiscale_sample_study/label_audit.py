"""Check every saved FEM label against an independently rebuilt discrete PDE.

Run in the complex dolfinx environment. Only the temporary reference setup
performs an anchor LU solve; the 1,200 audit queries assemble operators and
right-hand sides and never solve a linear system.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import numpy as np
import scipy

from experiments.reference_sno_gmsfem.dataset import (
    _reference_problem, _sha256, _write_json,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESIDUAL_THRESHOLD = 1e-8


def normalized_residual(operator, rhs, field):
    """Euclidean residual for the complex discrete scattered-field equation."""
    rhs, field = np.asarray(rhs), np.asarray(field)
    if rhs.ndim != 1 or field.shape != rhs.shape or not np.isfinite(field).all():
        raise ValueError('Invalid FEM field dimensions or nonfinite values')
    denominator = float(np.linalg.norm(rhs))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('FEM right-hand side norm must be finite and positive')
    residual = float(np.linalg.norm(operator@field-rhs)/denominator)
    if not np.isfinite(residual):
        raise ValueError('Nonfinite normalized FEM residual')
    return residual


def audit_labels(studyroot=HERE):
    """Reassemble all frozen physical queries and commit a full residual audit."""
    root = Path(studyroot).expanduser().resolve()
    protocol = json.loads((root/'protocol.json').read_text())
    data = root/'data'
    started = perf_counter()
    provenance_hash = _sha256(data/'provenance.json')
    hashes = {}
    for name, expected in protocol['dataset_sha256'].items():
        actual = _sha256(data/name)
        if actual != expected:
            raise ValueError(f'Frozen FEM dataset checksum changed: {name}')
        hashes[name] = actual
    info = json.loads((data/'dataset_info.json').read_text())
    if info.get('complete') is not True or info.get('samples') != 1200:
        raise ValueError('Expected all 1200 committed FEM labels')
    completed = np.load(data/'completed.npy', allow_pickle=False)
    if completed.shape != (1200,) or completed.dtype != bool or not completed.all():
        raise ValueError('Incomplete FEM label bitmap')
    with np.load(data/'samples.npz', allow_pickle=False) as saved:
        k0, theta = saved['k0'], saved['theta']
        frequency_id, split = saved['frequency_id'], saved['split']
    if (k0.shape != (1200,) or theta.shape != (1200,) or
            len(np.unique(k0)) != 1200 or len(np.unique(frequency_id)) != 1200 or
            not np.array_equal(np.bincount(split, minlength=3), [1000,100,100])):
        raise ValueError('Invalid physical query counts, identities, or split')
    fields = np.load(data/'u_scat.npy', mmap_mode='r', allow_pickle=False)
    if fields.shape != (1200,info['nodes']) or fields.dtype != np.complex128:
        raise ValueError('Invalid complex FEM label storage')
    rows = []
    with tempfile.TemporaryDirectory(prefix='scaling-fem-label-audit-') as temporary:
        setup_started = perf_counter()
        # This recreates and independently checks the exact problem and anchor.
        # Never pass the original data directory: _reference_problem writes its
        # own mesh, materials archive and provenance into this temporary folder.
        solver, problem = _reference_problem(Path(temporary))
        setup_seconds = perf_counter()-setup_started
        with np.load(data/'reference_problem.npz', allow_pickle=False) as saved:
            names = ('coordinates','triangles','cell_tags','circles','A','n','A_cells','n_cells')
            for name in names:
                if not np.array_equal(problem[name], saved[name]):
                    raise ValueError(f'Rebuilt FEM geometry or materials differ: {name}')
        anchor_provenance = json.loads((Path(temporary)/'provenance.json').read_text())
        assembly_started = perf_counter()
        for index, (k, angle) in enumerate(zip(k0, theta, strict=True)):
            assembled = solver.build_problem(problem['A_cells'][0], problem['n_cells'][0],
                                             float(k), float(angle))
            residual = normalized_residual(assembled.operator, assembled.rhs, fields[index])
            if residual >= RESIDUAL_THRESHOLD:
                raise ValueError(f'FEM label {index} at k={k:.17g} has residual {residual:.6g} >= {RESIDUAL_THRESHOLD:g}')
            rows.append(dict(index=index,k=float(k),theta=float(angle),residual=residual))
            count = index+1
            if count == 10 or count % 100 == 0:
                seconds = perf_counter()-assembly_started
                estimate = seconds*1200/count
                print(f'FEM label residual audit: {count}/1200, {seconds:.1f}s; estimated assembly total {estimate:.1f}s', flush=True)
            del assembled
        assembly_seconds = perf_counter()-assembly_started
    if _sha256(data/'provenance.json') != provenance_hash:
        raise ValueError('Original FEM provenance changed during audit')
    residuals = np.array([row['residual'] for row in rows])
    import dolfinx
    value = dict(schema_version=1,passed=True,ncases=len(rows),threshold=RESIDUAL_THRESHOLD,
                 residual_definition='Euclidean ||H(k,theta)u_scat-f(k,theta)|| / ||f(k,theta)||',
                 minimum=float(residuals.min()),mean=float(residuals.mean()),
                 p95=float(np.percentile(residuals,95)),maximum=float(residuals.max()),
                 setup_and_anchor_seconds=setup_seconds,assembly_seconds=assembly_seconds,
                 total_seconds=perf_counter()-started,
                 per_query_linear_solves=0,temporary_independent_anchor_solve=True,
                 original_provenance_untouched=True,original_provenance_sha256=provenance_hash,
                 residual_statistics=dict(min=float(residuals.min()),mean=float(residuals.mean()),
                                          p95=float(np.percentile(residuals,95)),max=float(residuals.max())),
                 anchor_verification=anchor_provenance['anchor'],
                 dataset_sha256=hashes,protocol_sha256=_sha256(root/'protocol.json'),
                 source_sha256={str(path.relative_to(REPO)):_sha256(path) for path in (
                     Path(__file__),REPO/'experiments/reference_sno_gmsfem/dataset.py',
                     REPO/'neural_operator/fem.py',REPO/'gmsfem/fem.py',REPO/'gmsfem/geometry.py')},
                 runtime=dict(python=platform.python_version(),system=platform.system(),
                              machine=platform.machine(),numpy=np.__version__,scipy=scipy.__version__,
                              dolfinx=dolfinx.__version__),
                 completed_at=datetime.now(timezone.utc).isoformat())
    output = root/'results'
    output.mkdir(parents=True,exist_ok=True)
    csv_path = output/'fem_label_audit.csv'
    temporary_csv = csv_path.with_suffix('.csv.tmp')
    with temporary_csv.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['index','k','theta','residual'])
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)
    value['rows_csv'] = dict(path=csv_path.name,sha256=_sha256(csv_path))
    _write_json(output/'fem_label_audit.json',value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('studyroot',nargs='?',type=Path,default=HERE)
    value = audit_labels(parser.parse_args().studyroot)
    print(f"FEM label audit passed: {value['ncases']} cases; maximum residual {value['maximum']:.6g}")


if __name__ == '__main__':
    main()
