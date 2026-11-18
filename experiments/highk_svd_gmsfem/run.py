"""Execute the isolated high-k fine-tuning and GMsFEM comparison."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import platform

from experiments.reference_sno_gmsfem.dataset import _sha256, _write_json
from .dataset import generate_dataset, verify_dataset, TRAIN_SIZES

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def freeze_protocol(root, *, device='mps', epochs=1000):
    root=Path(root)
    data=verify_dataset(root/'data')
    info=json.loads((root/'data/dataset_info.json').read_text())
    recipe=json.loads((root/'data/config.json').read_text())
    if recipe['generator_sha256']!=_sha256(HERE/'dataset.py'):
        raise ValueError('Dataset generator changed since label generation')
    paths=[HERE/name for name in ('dataset.py','mesh.py','highorder.py','training.py','run.py')]
    paths += [REPO/name for name in ('experiments/gms_sno_100/model.py',
        'experiments/gms_sno_100/pod.py','experiments/multiscale_sample_study/sno.py',
        'neural_operator/data.py','neural_operator/fem.py','neural_operator/mesh_transfer.py')]
    paths += [HERE/name for name in ('gms_highorder.py','reduced.py','baselines.py','baseline_parallel.py','gms_refinement.py')]
    sources={str(p.relative_to(REPO)):_sha256(p) for p in paths}
    old=REPO/'experiments/multiscale_sample_study'
    oldpaths=[old/f'runs/n1000/mscale_sno/seed_{seed}/model.pt' for seed in (11,29,47)]
    oldpaths += [old/'cache/n1000/sno'/name for name in ('pod.npz','sno_preprocessing.npz','sno_config.json')]
    oldpaths += [old/'data/reference_problem.npz',old/'data/samples.npz']
    gms_selection=json.loads((root/'results/gms_mesh_selection.json').read_text())
    if not gms_selection.get('passed'):
        raise ValueError('GMsFEM fine-mesh refinement checks have not passed')
    gms_config=dict(gms_selection['configuration'],workers=2,worker_threads=2)
    if _sha256(root/gms_config['fine_mesh'])!=gms_config['fine_mesh_sha256']:
        raise ValueError('Selected GMsFEM fine mesh changed')
    import numpy as np
    import torch
    import scipy
    protocol=dict(schema_version=1,experiment='highk_svd_gmsfem',
        methods=['mscale_sno','GMsFEM Type2'],seeds=[11,29,47],
        train_sizes=list(TRAIN_SIZES),
        n_train=len(data.split_indices()[0]),n_validation=len(data.split_indices()[1]),
        n_test=len(data.split_indices()[2]),k_range=[.05,20.],theta=0.,
        fixed_pretraining_train_samples=1000,pretraining_k_range=[.1,4.],
        sample_size_interpretation='N is additional new fine-tuning samples; total training observations include the fixed 1000-query historical pretraining set; this is not a from-scratch sample-efficiency study',
        epochs=epochs,batch_size=100,learning_rate=.001,weight_decay=.0001,
        validation_interval=5,patience_checks=40,device=device,threads=2,
        scheduler='cosine per optimizer update; epochs*ceil(Ntrain/batch_size); no warm restart',
        pod_max_rank=128,pod_expansion_rank=512,pod_rank_candidates=[32,64,96,128,192,256,384,512],
        pod_relative_tolerance=.01,pod_batch_size=16,
        pod_rank_selection='smallest declared candidate with train-only RMS individual relative FEM projection error <=1%; refuse if 512 cap insufficient',
        sno_width=8,depth=4,scales=[1,40,80,100,120,140,180,200],
        old_prior_gate=dict(enabled=True,start=4.,stop=5.),
        transfer='joint fine tuning of corresponding old seed model plus a zero-output-initialized full-rank multiscale residual; old decoder projected into new train-only mass POD',
        gmsfem=gms_config,gms_refinement_selection_sha256=_sha256(root/'results/gms_mesh_selection.json'),
        metric='mean individual relative scattered complex consistent P1-support FEM mass L2; percent',
        validation_dtype='float64 CPU',training_dtype='float32',
        checkpoint_selection='lowest evaluated validation mean; test evaluated only after selection',
        k_zero='excluded: singular static Neumann operator and zero scattered-field denominator',
        dataset_sha256=info['sha256'],source_sha256=sources,
        inherited_source_sha256={str(p.relative_to(REPO)):_sha256(p) for p in oldpaths},
        reference=info['reference'],
        runtime=dict(python=platform.python_version(),platform=platform.system(),
                     numpy=np.__version__,scipy=scipy.__version__,torch=str(torch.__version__)),
        interpretation='new fixed-medium interpolation study; low-k success does not establish high-k superiority; all predeclared results retained')
    if device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Selected MPS backend unavailable')
    path=root/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Frozen high-k protocol changed; create an explicitly new experiment')
    _write_json(path,protocol)
    return protocol


def verify_sources(root):
    protocol=json.loads((Path(root)/'protocol.json').read_text())
    for group in ('source_sha256','inherited_source_sha256'):
        for name,digest in protocol[group].items():
            if _sha256(REPO/name) != digest:raise ValueError(f'Frozen source changed: {name}')
    if _sha256(Path(root)/protocol['gmsfem']['fine_mesh'])!=protocol['gmsfem']['fine_mesh_sha256']:
        raise ValueError('Selected GMsFEM fine mesh changed')
    if _sha256(Path(root)/'results/gms_mesh_selection.json')!=protocol['gms_refinement_selection_sha256']:
        raise ValueError('GMsFEM refinement selection changed')
    verify_dataset(Path(root)/'data')
    return protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','train','verify','audit-labels','baselines','report','status'))
    p.add_argument('--root',type=Path,default=HERE)
    p.add_argument('--device',choices=('cpu','mps'),default='mps')
    p.add_argument('--epochs',type=int,default=1000)
    a=p.parse_args();root=a.root.resolve();(root/'results').mkdir(parents=True,exist_ok=True)
    if a.epochs<1:p.error('--epochs must be positive')
    if a.stage=='prepare':generate_dataset(root)
    elif a.stage=='train':
        from .training import train_all
        protocol=freeze_protocol(root,device=a.device,epochs=a.epochs)
        _write_json(root/'results/execution.json',dict(status='training',
            started_at=datetime.now(timezone(timedelta(hours=9))).isoformat()))
        train_all(root,verify_dataset(root/'data'),protocol)
    elif a.stage=='verify':
        from .training import verify
        protocol=verify_sources(root);verify(root,verify_dataset(root/'data'),protocol)
    elif a.stage=='audit-labels':
        from .label_audit import run
        run(root)
    elif a.stage=='baselines':
        from .baseline_parallel import evaluate_parallel
        protocol=verify_sources(root);cfg=protocol['gmsfem']
        evaluate_parallel(root/'data',root/'results',n_coarse=cfg['n_coarse'],modes=cfg['modes'],
                          fine_mesh=root/cfg['fine_mesh'],sparse_ordering=cfg['sparse_ordering'],
                          workers=cfg['workers'],worker_threads=cfg['worker_threads'])
    elif a.stage=='report':
        from .report import build_report
        from .reference_report import build as reference_report
        verify_sources(root)
        if not json.loads((root/'results/verification.json').read_text())['passed']:
            raise ValueError('Checkpoint verification has not passed')
        if not json.loads((root/'results/fem_label_audit.json').read_text())['passed']:
            raise ValueError('Independent native FEM label audit has not passed')
        reference_report(root)
        build_report(root/'data',root/'results',root/'runs')
        _write_json(root/'results/execution.json',dict(status='passed',
            completed_at=datetime.now(timezone(timedelta(hours=9))).isoformat()))
    else:
        for name in ('data/dataset_info.json','results/execution.json','results/training_summary.json'):
            path=root/name
            if path.exists():
                value=json.loads(path.read_text())
                print(name,json.dumps({k:v for k,v in value.items() if k not in ('sha256','reference','runs')}))
                if 'runs' in value:print('Completed neural runs:',len(value['runs']))
        path=root/'data/completed.npy'
        if path.exists():
            import numpy as np
            bitmap=np.load(path);print('Committed FEM labels:',int(bitmap.sum()),'/',len(bitmap))


if __name__=='__main__':main()
