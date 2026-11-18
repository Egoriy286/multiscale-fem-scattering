"""Reporting must reject missing, stale, and unpaired scientific evidence."""
import csv
import hashlib
import itertools
import json

import numpy as np
import pytest

from experiments.multiscale_sample_study import report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def study(tmp_path):
    protocol=dict(methods=list(report.NAMES),train_sizes=[400,600,800,1000],seeds=[11,29,47],
                  epochs=10,patience_checks=2,validation_interval=5)
    write(tmp_path/'protocol.json',protocol)
    return tmp_path,protocol


def complete(study):
    root,p=study
    verification=[]
    metrics=[]
    for n,m,s in itertools.product(p['train_sizes'],p['methods'],p['seeds']):
        d=root/'runs'/f'n{n}'/m/f'seed_{s}'
        d.mkdir(parents=True)
        rows=[dict(epoch=e,optimizer_steps=e*10,train_relative_l2=.2,val_relative_l2=v,
                   elapsed_seconds=e*.5,learning_rate=.001) for e,v in [(5,.2),(10,.1)]]
        write(d/'history.json',rows)
        (d/'model.pt').write_bytes(f'{n}{m}{s}'.encode())
        np.savez(d/'predictions.npz',indices=np.arange(100),k=np.arange(100)*.01,
                 theta=np.zeros(100),prediction=np.ones((100,2),dtype=complex))
        hashes={k:hashlib.sha256((d/f).read_bytes()).hexdigest() for f,k in
                [('model.pt','checkpoint_sha256'),('predictions.npz','predictions_sha256')]}
        identity=dict(n_train=n,method=m,seed=s)
        value=float(s)
        write(d/'result.json',dict(**identity,**hashes,parameters=10,epochs_completed=10,best_epoch=10,
                                  best_validation_relative_l2=.1,training_seconds=5.,stop_reason='epochs_completed',
                                  scattered_mean_pct=value,scattered_p95_pct=value,total_mean_pct=value/2,total_p95_pct=value/2))
        verification.append(dict(**identity,**hashes))
        metrics.extend(dict(**identity,index=i,k=i*.01,theta=0.,scattered_pct=value,total_pct=value/2) for i in range(100))
    write(root/'results'/'verification.json',dict(passed=True,runs=verification))
    with (root/'results'/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    return root


def test_missing_runs_do_not_publish_final(study, monkeypatch):
    root,_=study
    with pytest.raises(ValueError,match='Incomplete study'):
        report.build_report(root)
    monkeypatch.setattr(report,'_plots',lambda *args: [])
    result=report.build_report(root,allow_partial=True)
    assert result['status']=='partial' and result['completed']==0
    assert 'unfinished' in (root/'results'/'LIVE_REPORT.md').read_text()
    assert not (root/'results'/'REPORT.md').exists()


def test_final_report_aggregates_real_seed_sd_and_observed_thresholds(study,monkeypatch):
    root=complete(study)
    monkeypatch.setattr(report,'_plots',lambda *args: [])
    write(root/'results'/'physical_audit.json', {'status':'fixture'})
    write(root/'results'/'representation_diagnostics.json', {'status':'fixture'})
    result=report.build_report(root)
    text=(root/'results'/'REPORT.md').read_text()
    assert '(physical_audit.json)' in text and '(representation_diagnostics.json)' in text
    assert 'G-projection' in text and 'Trainable real weights' in text
    assert result['completed']==48 and result['status']=='complete'
    with (root/'results'/'summary.csv').open() as f:
        row=next(csv.DictReader(f))
    assert float(row['scattered_mean_pct'])==29
    assert float(row['scattered_seed_sd_pct'])==18
    with (root/'results'/'validation_thresholds.csv').open() as f:
        thresholds=list(csv.DictReader(f))
    assert len(thresholds)==48*4
    assert thresholds[0]['reached']=='False' and thresholds[0]['epoch']==''
    assert thresholds[1]['epoch']=='10'
    with (root/'results'/'paired_errors.csv').open() as f:
        assert len(list(csv.DictReader(f)))==4*3*6*100


def test_stale_weights_rejected(study,monkeypatch):
    root=complete(study)
    (root/'runs'/'n400'/'sno_gelu'/'seed_11'/'model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        report.build_report(root)
    assert not (root/'results'/'REPORT.md').exists()


def test_missing_verification_rejected(study):
    root=complete(study)
    (root/'results'/'verification.json').unlink()
    with pytest.raises(ValueError,match='verification.json'):
        report.build_report(root)


def test_duplicate_query_rejected(study):
    root=complete(study)
    path=root/'results'/'metrics.csv'
    with path.open() as f: rows=list(csv.DictReader(f))
    rows[1]['index']=rows[0]['index']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='Duplicate test query'):
        report.build_report(root)


def test_history_cannot_invent_or_reorder_epochs(study):
    root,_=study
    path=root/'runs'/'n400'/'sno_gelu'/'seed_11'/'history.json'
    row=dict(epoch=5,optimizer_steps=10,train_relative_l2=.2,val_relative_l2=.1,elapsed_seconds=1.,learning_rate=.001)
    write(path,[row,row])
    with pytest.raises(ValueError,match='Nonmonotone'):
        report.build_report(root,allow_partial=True)


def test_live_plot_uses_actual_single_observation(study, monkeypatch):
    root,_=study
    path=root/'runs'/'n400'/'sno_gelu'/'seed_11'/'history.json'
    write(path,[dict(epoch=5,optimizer_steps=200,train_relative_l2=.2,val_relative_l2=.1,
                     elapsed_seconds=3.,learning_rate=.001)])
    from matplotlib.axes import Axes
    original = Axes.set_xscale
    original_y = Axes.set_yscale
    scales = []
    yscales = []
    def capture_scale(self, value, **kwargs):
        scales.append(value)
        return original(self, value, **kwargs)
    def capture_yscale(self, value, **kwargs):
        yscales.append(value)
        return original_y(self, value, **kwargs)
    monkeypatch.setattr(Axes, 'set_xscale', capture_scale)
    monkeypatch.setattr(Axes, 'set_yscale', capture_yscale)
    result=report.build_report(root,allow_partial=True)
    assert 'log' in scales
    assert yscales.count('log') == 3
    assert result['completed']==0
    for axis in ['epoch','time','steps']:
        assert (root/'results'/f'live_convergence_{axis}.png').stat().st_size>1000
    assert not (root/'results'/'sample_size_error.png').exists()


def baseline_fixture(study):
    root,p=study
    p['dataset_sha256']={'etal.msh':'referencehash','u_scat.npy':'fieldhash'}
    rows=[]
    for method in ['FEM h=0.04','GMsFEM 20x20 Type 1 M=16','GMsFEM 20x20 Type 2 M=16']:
        rows.extend(dict(method=method,index=i,sample_index=i,k=i*.01,theta=0.,
                         scattered_e_complex_ref=2.,total_e_complex_ref=1.) for i in range(100,200))
    payload=dict(rows=rows,metadata=dict(case='notebook',coarse_grids=[20],basis_types=[1,2],modes=16,
                 sample_indices=list(range(100,200)),reference_mesh_sha256='referencehash',
                 fine_mesh_sha256='a8af14fc965d4823fcb5be5d6f102e7225ff2d553b1c4b3a71325834e30807de'))
    path=root/'results'/'gmsfem_m16'/'baselines.json'
    write(path,payload)
    manifest=dict(dataset_sha256=p['dataset_sha256'],baselines_sha256=report._hash(path),source_runtime='Docker CPU')
    write(root/'results'/'gmsfem_manifest.json',manifest)
    metrics={(400,'sno_gelu',11):{i:dict(k=i*.01,theta=0.) for i in range(100,200)}}
    return root,p,path,payload,manifest,metrics


def test_fresh_baseline_supplement(study):
    root,p,_,_,_,metrics=baseline_fixture(study)
    rows=report._baselines(root,p,metrics)
    assert len(rows)==3 and all(r['test_queries']==100 for r in rows)
    assert rows[0]['scattered_mean_pct']==2. and rows[0]['total_p95_pct']==1.


@pytest.mark.parametrize('damage',['missing_manifest','stale_dataset','changed_baseline','duplicate','wrong_query','missing_method'])
def test_baseline_rejects_unverified_or_unpaired_data(study,damage):
    root,p,path,payload,manifest,metrics=baseline_fixture(study)
    mp=root/'results'/'gmsfem_manifest.json'
    if damage=='missing_manifest':
        mp.unlink()
    elif damage=='stale_dataset':
        manifest['dataset_sha256']={'etal.msh':'oldhash'}
        write(mp,manifest)
    elif damage=='changed_baseline':
        payload['rows'][0]['scattered_e_complex_ref']=3.
        write(path,payload)
    else:
        if damage=='duplicate': payload['rows'][1]=payload['rows'][0].copy()
        if damage=='wrong_query': payload['rows'][0]['k']=9.
        if damage=='missing_method': payload['rows']=payload['rows'][:200]
        write(path,payload)
        manifest['baselines_sha256']=report._hash(path)
        write(mp,manifest)
    with pytest.raises(ValueError):
        report._baselines(root,p,metrics)


def frequency_fixture():
    p=dict(train_sizes=[400],methods=['sno_gelu'],seeds=[11,29,47])
    # Deliberately shuffled insertion order and nonconsecutive physical IDs.
    queries={5:1.8,2:.2,9:3.4}
    errors={11:{5:2.,2:1.,9:3.},29:{5:8.,2:4.,9:6.},47:{5:5.,2:7.,9:9.}}
    metrics={(400,'sno_gelu',seed):{
        i:dict(k=k,theta=0.,scattered_pct=errors[seed][i],total_pct=errors[seed][i]/2)
        for i,k in queries.items()} for seed in p['seeds']}
    return p,metrics


def test_frequency_profiles_use_physical_k_and_all_seed_errors():
    p,metrics=frequency_fixture()
    rows=report._frequency_profiles(p,metrics)
    assert [r['index'] for r in rows]==[2,5,9]
    assert [r['k'] for r in rows]==[.2,1.8,3.4]
    np.testing.assert_allclose([r['scattered_mean_pct'] for r in rows],[4.,5.,6.])
    np.testing.assert_allclose([r['scattered_min_pct'] for r in rows],[1.,2.,3.])
    np.testing.assert_allclose([r['scattered_max_pct'] for r in rows],[7.,8.,9.])
    np.testing.assert_allclose([r['total_mean_pct'] for r in rows],[2.,2.5,3.])
    assert all(r['seed_count']==3 for r in rows)


@pytest.mark.parametrize('damage',['query','missing','nonfinite'])
def test_frequency_profiles_reject_mismatched_cases_and_invalid_values(damage):
    p,metrics=frequency_fixture()
    row=metrics[400,'sno_gelu',29]
    if damage=='query':row[5]['k']=1.9
    elif damage=='missing':row.pop(5)
    else:row[5]['scattered_pct']=float('nan')
    with pytest.raises(ValueError,match='paired physical|Invalid frequency'):
        report._frequency_profiles(p,metrics)


def test_frequency_plot_preserves_solver_per_k_values(tmp_path,monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    p,metrics=frequency_fixture()
    profiles=report._frequency_profiles(p,metrics)
    solver=[]
    for method in report.BASELINE_STYLES:
        for j,k in enumerate([.2,1.8,3.4]):
            row=dict(n_train='',method=method,index=j,k=k,theta=0.,seed_count=0)
            for field,factor in [('scattered',1.),('total',.5)]:
                for statistic in ('mean','min','max'):
                    row[f'{field}_{statistic}_pct']=(j+1)*factor
            solver.append(row)
    observed=[]
    original=Axes.plot
    def capture(self,x,y,*args,**kwargs):
        observed.append((kwargs.get('label'),list(x),list(y)))
        return original(self,x,y,*args,**kwargs)
    monkeypatch.setattr(Axes,'plot',capture)
    paths=report._frequency_plots(tmp_path,p,profiles,solver)
    assert set(paths)=={'test_scattered_error_vs_k.png','test_total_error_vs_k.png','gmsfem_error_vs_k.png'}
    method='GMsFEM 20x20 Type 2 M=16'
    assert (method,[.2,1.8,3.4],[1.,2.,3.]) in observed
    assert (method,[.2,1.8,3.4],[.5,1.,1.5]) in observed
    assert all((tmp_path/name).stat().st_size>1000 for name in paths)
