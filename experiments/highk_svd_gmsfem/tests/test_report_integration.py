"""Small end-to-end artifact check for nested fine-tuning report layouts."""
import json
from pathlib import Path
import numpy as np
from neural_operator.data import Dataset
from experiments.highk_svd_gmsfem import report
from experiments.reference_sno_gmsfem.dataset import _sha256


def test_nested_report(tmp_path,monkeypatch):
    root=tmp_path;results=root/'results';results.mkdir();runs=root/'runs';runs.mkdir();directory=root/'data';directory.mkdir()
    n=2400;split=np.r_[np.ones(200,int),np.full(200,2),np.zeros(2000,int)]
    data=Dataset(A=np.ones((n,3)),n=np.ones((n,3)),u_scat=np.full((n,3),2+0j),
        coordinates=np.array([[0.,0.],[1.,0.],[0.,1.]]),triangles=np.array([[0,1,2]]),
        k0=np.linspace(.05,20,n),theta=np.zeros(n),medium_id=np.zeros(n,int),split=split,
        frequency_id=np.arange(n),split_strategy='frequency')
    monkeypatch.setattr(report,'load_dataset',lambda _:data)
    def save(path,value):path.write_text(json.dumps(value))
    sizes=[400,600,800,1000,2000];seeds=[11,22,33]
    save(root/'protocol.json',dict(seeds=seeds,train_sizes=sizes,gmsfem=dict(n_coarse=20,modes=[16,32],fine_mesh_sha256='abc')))
    save(directory/'dataset_info.json',dict(reference=dict(native_dofs=377530)))
    test=data.split_indices()[2];allruns=[]
    for size in sizes:
        for seed in seeds:
            d=runs/f'n{size}'/'mscale_sno'/f'seed_{seed}';d.mkdir(parents=True)
            (d/'model.pt').write_bytes(b'checkpoint')
            np.save(d/'predictions.npy',np.full((len(test),3),2.2+0j))
            save(d/'metrics.json',[dict(index=int(i),k=float(data.k0[i]),theta=0.,scattered_pct=10.) for i in test])
            save(d/'initial_validation.json',dict(mean_relative_l2=.5))
            save(d/'history.json',[dict(epoch=1,optimizer_steps=3,elapsed_seconds=1.,train_relative_l2=.2,val_relative_l2=.2),dict(epoch=2,optimizer_steps=6,elapsed_seconds=2.,train_relative_l2=.1,val_relative_l2=.1)])
            allruns.append(dict(n_train=size,seed=seed,best_epoch=2,epochs_completed=2,parameters=100,r_complex=12,pod_real_unknowns=24,
                predictions_sha256=_sha256(d/'predictions.npy'),checkpoint_sha256=_sha256(d/'model.pt')))
    save(results/'training_summary.json',dict(runs=allruns))
    rows=[dict(method=f'GMsFEM CG2 Type2 20x20 M={m}',index=int(i),k=float(data.k0[i]),theta=0.,
        basis_type=2,n_coarse=20,modes=m,dofs_coarse=441*m,dofs_fine=167925,dofs_support=3,fine_degree=2,
        galerkin_residual=1e-12,error_percent=20.,error_fem_l2=.2) for m in [16,32] for i in test]
    fine=[dict(method='Native CG2 discretization diagnostic',index=int(i),k=float(data.k0[i]),theta=0.,error_percent=1.) for i in test]
    save(results/'baselines.json',dict(rows=rows,fine_reference_rows=fine,metadata=dict(complete=True,
        protocol=dict(indices=test.tolist(),basis_type=2,n_coarse=20,modes=[16,32],fine_degree=2,fine_mesh_sha256='abc',dataset_sha256={}),frequencies=[])))
    save(results/'gms_mesh_selection.json',dict(passed=True,configuration=dict(fine_mesh_sha256='abc')))
    save(results/'gms_refinement.json',dict(complete=True,rows=[dict(level=0,method='native_cg2',k=20.,error_percent=1.)],differences=[]))
    summaries=report.build_report(directory,results,runs)
    assert len(summaries)==7
    assert (results/'sample_efficiency.png').exists()
    assert (results/'convergence_optimizer_steps.png').exists()
    dofs=(results/'dofs.csv').read_text()
    assert 'pod_complex_rank' in dofs and 'network_parameters' in dofs
    markdown=(results/'REPORT.md').read_text()
    assert '|---|---:|---:|---:|---:|---:|\n| ' in markdown
    assert '1000+N' in markdown and 'exactly zero' in markdown
