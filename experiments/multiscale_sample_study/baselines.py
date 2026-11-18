"""Reproduce or verify the deterministic common-test solver supplement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.reference_sno_gmsfem.dataset import _sha256, _write_json, load_dataset

HERE = Path(__file__).resolve().parent


def run(root=HERE):
    root=Path(root).resolve()
    protocol=json.loads((root/'protocol.json').read_text())
    for name,digest in protocol['dataset_sha256'].items():
        if _sha256(root/'data'/name)!=digest:
            raise ValueError('Frozen physical dataset differs')
    data=load_dataset(root/'data')
    test=data.split_indices()[2]
    directory=root/'results/gmsfem_m16'
    path=directory/'baselines.json'
    manifest_path=root/'results/gmsfem_manifest.json'
    if not path.exists():
        from experiments.reference_sno_gmsfem.baselines import evaluate_baselines
        evaluate_baselines(data,directory,test,coarse_grids=(20,),basis_types=(1,2),
                           modes=16,reproduce=False,save_fields=False)
    value=json.loads(path.read_text())
    metadata=value['metadata']
    expected={(method,int(i)) for i in test for method in
              ('FEM h=0.04','GMsFEM 20x20 Type 1 M=16','GMsFEM 20x20 Type 2 M=16')}
    rows=value['rows']
    if (metadata['sample_indices']!=test.tolist() or metadata['modes']!=16
            or metadata['coarse_grids']!=[20] or metadata['basis_types']!=[1,2]
            or len(rows)!=len(expected) or {(r['method'],r['index']) for r in rows}!=expected):
        raise ValueError('Unpaired or incomplete solver baseline')
    for row in rows:
        i=row['index']
        if row['k']!=float(data.k0[i]) or row['theta']!=float(data.theta[i]):
            raise ValueError('Solver physical query differs')
        for field in ('scattered_e_complex_ref','total_e_complex_ref'):
            if not np.isfinite(row[field]) or row[field]<0:
                raise ValueError('Invalid solver metric')
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if (manifest['dataset_sha256']!=protocol['dataset_sha256']
                or manifest['baselines_sha256']!=_sha256(path)):
            raise ValueError('Frozen solver manifest differs')
    else:
        _write_json(manifest_path,dict(dataset_sha256=protocol['dataset_sha256'],
            baselines_sha256=_sha256(path),source_runtime='complex DOLFINx/PETSc CPU2 in pinned Linux ARM64 Docker',
            timing='Fresh paired100case baseline; separate from native CPU/MPS neural training'))
    print(f'Paired solver supplement verified: {len(rows)} rows on {len(test)} test queries',flush=True)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path,nargs='?',default=HERE)
    run(parser.parse_args().root)


if __name__=='__main__':
    main()
