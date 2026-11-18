"""Wait for native datasets, train, replay, and produce the mesh-stability report."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def status(root,stage,**extra):
    path=root/'results/execution.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(stage=stage,**extra),indent=2)+'\n');temp.replace(path)
    print(stage,flush=True)

def freeze(root):
    paths=list(root.glob('*.py'))+[root/'protocol.json',root/'queries.json']
    paths += [REPO/name for name in ('experiments/highk_svd_gmsfem/training.py',
        'experiments/highk_svd_gmsfem/highorder.py','experiments/highk_svd_gmsfem/mesh.py',
        'experiments/highk_svd_gmsfem/gms_highorder.py','experiments/gms_sno_100/model.py',
        'experiments/gms_sno_100/pod.py','experiments/multiscale_sample_study/sno.py',
        'experiments/multiscale_sample_study/training.py','experiments/reference_sno_gmsfem/dataset.py',
        'neural_operator/mesh_transfer.py','gmsfem/fem.py','gmsfem/geometry.py')]
    value=dict(source_sha256={str(p.relative_to(REPO)):sha(p) for p in paths},
        datasets={str(i):json.loads((root/f'data/level{i}/dataset_info.json').read_text()) for i in range(3)})
    path=root/'frozen_inputs.json'
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Frozen experiment inputs changed')
    path.write_text(json.dumps(value,indent=2)+'\n')
    return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['all','train','report','status']);args=parser.parse_args()
    root=ROOT;(root/'results').mkdir(exist_ok=True)
    if args.stage=='status':
        for i in range(3):
            p=root/f'data/level{i}/progress.json'
            print('level',i,len(json.loads(p.read_text())['completed']) if p.exists() else 0,'/1400')
        p=root/'results/execution.json'
        if p.exists():print(p.read_text())
        return
    try:
        if args.stage in ('all','train'):
            status(root,'waiting_for_three_native_datasets')
            while not all((root/f'data/level{i}/dataset_info.json').exists() for i in range(3)):time.sleep(15)
            frozen=freeze(root)
            from .training import train_all,verify
            protocol=json.loads((root/'protocol.json').read_text())
            status(root,'training_nine_independent_models');train_all(root,protocol)
            status(root,'verifying_checkpoints');verify(root,protocol)
        if args.stage in ('all','report'):
            status(root,'waiting_for_independent_cg3_reference')
            while not (root/'data/reference/dataset_info.json').exists():time.sleep(15)
            frozen=freeze(root)
            for name,digest in frozen['source_sha256'].items():
                if sha(REPO/name)!=digest:raise ValueError(f'Frozen source changed: {name}')
            status(root,'evaluating_common_physical_quadrature')
            code='from pathlib import Path; import json; from experiments.svd_sno_mesh_stability_2026_11_16.report import build_report; r=Path("experiments/svd_sno_mesh_stability_2026_11_16"); build_report(r,json.loads((r/"protocol.json").read_text()))'
            subprocess.run(['bash','scripts/docker.sh','compose','exec','-T','-e','OPENBLAS_NUM_THREADS=2',
                '-e','OMP_NUM_THREADS=2','dolfinx-complex','python3','-u','-c',code],cwd=REPO,check=True)
            status(root,'primary_complete',confirmation='2000 samples permitted only after the predeclared stability criteria pass')
    except Exception as exc:
        status(root,'failed',error=str(exc));raise

if __name__=='__main__':main()
