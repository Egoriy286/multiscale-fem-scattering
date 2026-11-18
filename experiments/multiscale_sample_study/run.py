"""Generate, train, replay and report the nested four-size comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import torch

from experiments.reference_sno_gmsfem.dataset import _sha256, _write_json, load_dataset
from .dataset import generate_dataset, query_hash

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def freeze_protocol(root, args):
    root = Path(root)
    info = json.loads((root/'data/dataset_info.json').read_text())
    # Hash source content, rather than platform-dependent assembled mass bytes.
    paths = [HERE/name for name in ('dataset.py','sno.py','fno.py','training.py','run.py')]
    paths += [REPO/name for name in ('experiments/gms_sno_100/model.py',
        'experiments/gms_sno_100/pod.py','experiments/mscale_fno_100/model.py',
        'experiments/mscale_fno_100/training.py','neural_operator/data.py',
        'neural_operator/mesh_transfer.py','experiments/reference_sno_gmsfem/dataset.py')]
    actual = {name:_sha256(root/'data'/name) for name in info['sha256']}
    if actual != info['sha256']:
        raise ValueError('Dataset differs from frozen generation manifest')
    protocol = dict(schema_version=1, train_sizes=[400,600,800,1000], n_validation=100,n_test=100,
        methods=['sno_gelu','mscale_sno','fno_gelu','mscale_fno'], seeds=[11,29,47],
        epochs=args.epochs, batch_size=50, validation_interval=5, patience_checks=15,
        device=args.device, sno_device='cpu', threads=2, learning_rate=.001, weight_decay=.0001,
        scheduler='cosine per optimizer update; horizon epochs*ceil(Ntrain/batch_size)',
        depth=4, resolution=64, fourier_modes=8, fno_multiscale_width=2, fno_single_width=6,
        sno_width=8, pod_rank=20, scales=[1,40,80,100,120,140,180,200],
        primary_metric='mean individual relative complex scattered consistent FEM L2; percentages in reports',
        validation_dtype='float64', online_dtype='float32 on MPS; double CPU quadratic otherwise',
        checkpoint_selection='minimum evaluated validation mean; first epoch wins exact ties',
        selection_evaluations='epoch 1, every 5 epochs, and final budget epoch',
        dataset_sha256=actual, source_sha256={str(p.relative_to(REPO)):_sha256(p) for p in paths},
        runtime=dict(python=platform.python_version(), machine=platform.machine(), system=platform.system(),
                     numpy=np.__version__,scipy=scipy.__version__,torch=str(torch.__version__)),
        resource_adaptation='compact width-2 eight-branch FNO and matched width-6 single FNO; not the 256M-parameter journal experiment',
        no_test_tuning=True)
    if args.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was selected but is unavailable')
    path = root/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Frozen study protocol differs; use a new study directory')
    _write_json(path,protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','train','verify','report','all','status','live'))
    parser.add_argument('--root',type=Path,default=HERE)
    parser.add_argument('--device',choices=('mps','cpu'),default='mps')
    parser.add_argument('--epochs',type=int,default=100)
    args = parser.parse_args()
    if args.epochs<1:
        parser.error('--epochs must be positive')
    root = args.root.resolve()
    (root/'results').mkdir(parents=True,exist_ok=True)
    if args.stage in ('prepare','all'):
        if (root/'data/dataset_info.json').exists():
            # The committed physical query manifest is authoritative across
            # NumPy/OS versions; recomputing a floating-point planner can alter
            # last bits without indicating a change to saved PDE cases.
            info = json.loads((root/'data/dataset_info.json').read_text())
            recipe = json.loads((root/'data/config.json').read_text())
            for name,digest in info['sha256'].items():
                if _sha256(root/'data'/name)!=digest:
                    raise ValueError('Completed dataset checksum changed')
            if recipe['generator_sha256']!=_sha256(HERE/'dataset.py'):
                raise ValueError('Frozen dataset generator changed')
            with np.load(root/'data/samples.npz',allow_pickle=False) as archive:
                if query_hash({k:archive[k] for k in archive.files})!=recipe['query_sha256']:
                    raise ValueError('Frozen physical query plan changed')
            load_dataset(root/'data')
        else:
            generate_dataset(root/'data')
    if args.stage in ('train','all'):
        from .training import train_all
        data = load_dataset(root/'data')
        protocol = freeze_protocol(root,args)
        _write_json(root/'results/execution.json',dict(status='running',
            started_at=datetime.now(timezone(timedelta(hours=9))).isoformat()))
        train_all(root,data,protocol)
    if args.stage in ('verify','all'):
        from .training import verify
        data = load_dataset(root/'data')
        protocol = json.loads((root/'protocol.json').read_text())
        # Re-check the original source and label fingerprints before replay.
        for path,digest in protocol['source_sha256'].items():
            if _sha256(REPO/path)!=digest:
                raise ValueError('Frozen source changed')
        for path,digest in protocol['dataset_sha256'].items():
            if _sha256(root/'data'/path)!=digest:
                raise ValueError('Frozen physical data changed')
        verify(root,data,protocol)
    if args.stage in ('report','all','live'):
        from .report import build_report
        build_report(root,allow_partial=args.stage=='live')
        if args.stage!='live':
            _write_json(root/'results/execution.json',dict(status='passed',
                finished_at=datetime.now(timezone(timedelta(hours=9))).isoformat()))
    if args.stage=='status':
        info = root/'results/training_summary.json'
        print(f"Completed {len(json.loads(info.read_text())['runs']) if info.exists() else 0}/48 runs")
        state = root/'results/execution.json'
        if state.exists():
            print(state.read_text())


if __name__=='__main__':
    main()
