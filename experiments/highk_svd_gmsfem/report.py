"""Strictly paired high-k report from raw completed runs; no interpolated data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.reference_sno_gmsfem.dataset import _sha256, _write_json
from .dataset import load_dataset
from .baselines import _csv


def _read(path):
    return json.loads(Path(path).read_text())


def _stats(values):
    a = np.asarray(values, dtype=float)
    if not a.size or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Expected finite, nonnegative, nonempty errors')
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), maximum=float(a.max()))


def validate_paired(rows, indices, data):
    """Require one score for every method/seed/case, with exact physical queries."""
    groups = {}
    for row in rows:
        key = (row['method'], row.get('seed', 'deterministic'))
        groups.setdefault(key, []).append(row)
        i = int(row['index'])
        if (i not in indices or row['k'] != float(data.k0[i])
                or row['theta'] != float(data.theta[i])
                or not np.isfinite(row['error_percent']) or row['error_percent'] < 0):
            raise ValueError('Unpaired or invalid raw comparison row')
    for key, values in groups.items():
        actual = [r['index'] for r in values]
        if len(actual) != len(indices) or set(actual) != set(map(int, indices)):
            raise ValueError(f'Missing or duplicate held-out cases for {key}')
    if not groups:
        raise ValueError('No comparison rows')


def build_report(data_dir, results_dir, run_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_dir, results, run_root = map(Path, (data_dir, results_dir, run_root))
    data = load_dataset(data_dir)
    test = data.split_indices()[2]
    protocol = _read(run_root.parent/'protocol.json')
    summary = _read(results/'training_summary.json')
    runs = summary['runs']
    expected_seeds = list(protocol['seeds'])
    train_sizes = list(protocol['train_sizes'])
    expected_runs = {(n, seed) for n in train_sizes for seed in expected_seeds}
    actual_runs = [(r['n_train'], r['seed']) for r in runs]
    if len(actual_runs) != len(expected_runs) or set(actual_runs) != expected_runs:
        raise ValueError('Incomplete or duplicated neural seed runs')
    baseline_path = results/'baselines.json'
    baseline = _read(baseline_path)
    if not baseline['metadata']['complete']:
        raise ValueError('GMsFEM run is incomplete')
    bspec = baseline['metadata']['protocol']
    gms = protocol['gmsfem']
    nc = gms['n_coarse']
    if (bspec['indices'] != test.tolist() or bspec['basis_type'] != 2
            or bspec['n_coarse'] != nc or bspec['modes'] != gms['modes'] or bspec['fine_degree'] != 2
            or bspec['fine_mesh_sha256'] != gms['fine_mesh_sha256']
            or bspec.get('sparse_ordering','COLAMD') != gms.get('sparse_ordering','COLAMD')):
        raise ValueError('GMsFEM evaluated another test split or basis type')
    for name, expected in bspec['dataset_sha256'].items():
        if _sha256(data_dir/name) != expected:
            raise ValueError(f'GMsFEM physical data hash differs: {name}')
    expected_gms = {f'GMsFEM CG2 Type2 {nc}x{nc} M={m}' for m in gms['modes']}
    if set(r['method'] for r in baseline['rows']) != expected_gms:
        raise ValueError('Missing or unexpected predeclared GMsFEM configuration')
    for row in baseline['rows']:
        if (row['basis_type'] != 2 or row['n_coarse'] != nc or row['modes'] not in gms['modes']
                or row['dofs_coarse'] != (nc+1)**2*row['modes'] or row['dofs_support'] != len(data.coordinates)
                or not 0 <= row['galerkin_residual'] <= 1e-7):
            raise ValueError('GMsFEM row configuration/residual differs')
    rows = [dict(row, seed='deterministic') for row in baseline['rows']]
    fine_diagnostic = [dict(row, seed='deterministic') for row in baseline['fine_reference_rows']]
    validate_paired(fine_diagnostic, test, data)
    reference_info = _read(data_dir/'dataset_info.json')
    _write_json(results/'reference_metadata.json', reference_info)
    learning_checks=[]
    histories, timing_rows, artifact_hashes = [], [], {'baselines.json': _sha256(baseline_path)}
    for run in runs:
        seed = run['seed']
        n_train = run['n_train']
        method = f'Multiscale POD SVD-SNO new N={n_train}'
        directory = run_root/f'n{n_train}'/run.get('variant', 'mscale_sno')/f'seed_{seed}'
        if not directory.exists() and 'directory' in run:
            directory = Path(run['directory'])
        path = directory/'predictions.npy'
        metric_path = directory/'metrics.json'
        saved_metrics = _read(metric_path)
        if isinstance(saved_metrics, dict):
            saved_metrics = saved_metrics['rows']
        if len(saved_metrics) != len(test):
            raise ValueError('Incomplete neural metric rows')
        by_index = {int(r['index']): r for r in saved_metrics}
        if len(by_index) != len(test) or set(by_index) != set(map(int, test)):
            raise ValueError('Invalid neural sample identity manifest')
        predictions = np.load(path, mmap_mode='r', allow_pickle=False)
        if predictions.shape != (len(test), len(data.coordinates)):
            raise ValueError('Invalid neural prediction shape')
        for j, i in enumerate(test):
            manifest = by_index[int(i)]
            if manifest['k'] != float(data.k0[i]) or manifest['theta'] != float(data.theta[i]):
                raise ValueError('Neural saved physical query mismatch')
            error = float(data.relative_l2(data.u_scat[i], predictions[j]))
            if not np.isclose(100*error, manifest['scattered_pct'], rtol=2e-5, atol=1e-6):
                raise ValueError('Saved neural scores differ from prediction field recomputation')
            rows.append(dict(method=method, seed=seed, n_train=n_train, fixed_pretraining_train=1000, total_unique_training_queries=1000+n_train, index=int(i),
                             k=float(data.k0[i]), theta=float(data.theta[i]),
                             error_fem_l2=error, error_percent=100*error))
        artifact_hashes[str(metric_path)] = _sha256(metric_path)
        artifact_hashes[str(path)] = _sha256(path)
        if run.get('predictions_sha256') != artifact_hashes[str(path)]:
            raise ValueError('Neural prediction hash differs from completed run')
        if run.get('checkpoint_sha256') != _sha256(directory/'model.pt'):
            raise ValueError('Neural checkpoint hash differs from completed run')
        initial_path=directory/'initial_validation.json'
        initial_validation=_read(initial_path)
        artifact_hashes[str(initial_path)]=_sha256(initial_path)
        history_path = directory/'history.json'
        history = _read(history_path)
        if not history:
            raise ValueError('Empty neural history')
        selected_epoch = min(history, key=lambda h: h['val_relative_l2'])['epoch']
        initial_mean=float(initial_validation['mean_relative_l2'])
        if not np.isfinite(initial_mean) or initial_mean<0:
            raise ValueError('Invalid initial validation score')
        learning_checks.append(dict(n_train=n_train,seed=seed,initial_validation_percent=100*initial_mean,
            best_validation_percent=100*min(h['val_relative_l2'] for h in history),best_epoch=selected_epoch,
            old_weights_l2_change=run.get('old_weights_l2_change'))) 
        if selected_epoch != run['best_epoch'] or run['epochs_completed'] != history[-1]['epoch']:
            raise ValueError('Neural checkpoint selection differs from validation history')
        for h in history:
            if not all(np.isfinite(h[key]) and h[key] >= 0 for key in ('train_relative_l2', 'val_relative_l2')):
                raise ValueError('Nonfinite/negative convergence error')
            histories.append(dict(seed=seed, n_train=n_train, **h))
        artifact_hashes[str(history_path)] = _sha256(history_path)
        timing_rows.append(dict(method=method, seed=seed, n_train=n_train,
                                **{k: v for k, v in run.items()
                                   if ('seconds' in k or 'dof' in k or 'parameter' in k or k in ('r_complex','pod_complex_rank','pod_real_unknowns','support_real_unknowns')) and np.isscalar(v)}))
        del predictions
    validate_paired(rows, test, data)
    methods = sorted(set(r['method'] for r in rows))
    # Aggregation preserves the paired cases; neural uncertainty is across seeds.
    summaries, bins, per_k = [], [], []
    for method in methods:
        values = [r for r in rows if r['method'] == method]
        summaries.append(dict(method=method, **_stats([r['error_percent'] for r in values])))
        for lower, upper in zip((0, 4, 8, 12, 16), (4, 8, 12, 16, 20)):
            selected = [r['error_percent'] for r in values
                        if lower <= r['k'] <= upper and (lower == 0 or r['k'] > lower)]
            if selected:
                bins.append(dict(method=method, k_lower=lower, k_upper=upper, **_stats(selected)))
        for k in sorted(set(r['k'] for r in values)):
            scores = [r['error_percent'] for r in values if r['k'] == k]
            per_k.append(dict(method=method, k=k, mean=float(np.mean(scores)),
                              minimum=float(min(scores)), maximum=float(max(scores)), count=len(scores)))
    for summary_row in summaries:
        method=summary_row['method']
        means=[np.mean([r['error_percent'] for r in rows if r['method']==method and r['seed']==seed])
               for seed in sorted({r['seed'] for r in rows if r['method']==method},key=str)]
        summary_row.update(seed_count=len(means),seed_mean_sd=float(np.std(means,ddof=1)) if len(means)>1 else None)
    for record in baseline['metadata']['frequencies']:
        for system in record['systems']:
            timing_rows.append(dict(method=f"GMsFEM CG2 Type2 {bspec['n_coarse']}x{bspec['n_coarse']} M={system['modes']}",
                                    k=record['k'], dofs_coarse=(bspec['n_coarse']+1)**2*system['modes'],
                                    shared_offline_seconds=record['offline_seconds'], **system))
    seed_summaries = []
    for method, seed in sorted({(r['method'], str(r['seed'])) for r in rows}):
        values = [r['error_percent'] for r in rows if r['method'] == method and str(r['seed']) == seed]
        seed_summaries.append(dict(method=method, seed=seed, **_stats(values)))
    _csv(results/'per_seed_summary.csv', seed_summaries)
    _csv(results/'paired_errors.csv', rows)
    _csv(results/'summary.csv', summaries)
    _csv(results/'native_cg2_diagnostic_summary.csv', [dict(method='Native CG2 discretization diagnostic', **_stats([r['error_percent'] for r in fine_diagnostic]))])
    _csv(results/'bands.csv', bins)
    _csv(results/'per_k.csv', per_k)
    _csv(results/'convergence_raw.csv', histories)
    _csv(results/'fine_tuning_validation.csv',learning_checks)
    _csv(results/'timings_dofs.csv', timing_rows)
    native_truth_dofs=reference_info.get('reference',{}).get('native_dofs',0)
    if not isinstance(native_truth_dofs,int) or native_truth_dofs<=0:
        raise ValueError('Reference metadata must declare measured native DOFs')
    dof_rows=[]
    for run in runs:
        dof_rows.append(dict(method='Multiscale POD SVD-SNO',n_train=run['n_train'],seed=run['seed'],
            dofs_native_truth=native_truth_dofs,dofs_native_truth_real=2*native_truth_dofs,dofs_support=len(data.coordinates),
            support_real_unknowns=2*len(data.coordinates),pod_complex_rank=run.get('r_complex',run.get('pod_complex_rank')),
            pod_real_unknowns=run.get('pod_real_unknowns'),network_parameters=run['parameters']))
    for method in expected_gms:
        row=next(r for r in rows if r['method']==method)
        dof_rows.append(dict(method=method,dofs_native_truth=native_truth_dofs,dofs_native_truth_real=2*native_truth_dofs,dofs_support=len(data.coordinates),
            dofs_gms_full=row['dofs_fine'],dofs_gms_full_real=2*row['dofs_fine'],fine_degree=row['fine_degree'],dofs_reduced=row['dofs_coarse'],dofs_reduced_real=2*row['dofs_coarse']))
    _csv(results/'dofs.csv',dof_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in methods:
        points = [r for r in per_k if r['method'] == method]
        x = [r['k'] for r in points]
        ax.plot(x, [r['mean'] for r in points], marker='.', markersize=4, linewidth=1, label=method)
        if points[0]['count'] > 1:
            ax.fill_between(x, [r['minimum'] for r in points], [r['maximum'] for r in points], alpha=.18)
    ordered_fine = sorted(fine_diagnostic, key=lambda row: row['k'])
    ax.plot([r['k'] for r in ordered_fine], [r['error_percent'] for r in ordered_fine], ':', color='gray', linewidth=1.5, label='Native full CG2 diagnostic vs stored reference')
    ax.set(xlabel='Wavenumber k', ylabel='Scattered consistent FEM L2 error (%)', title='Paired held-out cases; neural mean and seed min–max')
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(results/'error_vs_k.png', dpi=180); fig.savefig(results/'error_vs_k.svg'); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(17,9),sharex=True)
    for axis,n_train in zip(axes.ravel(),train_sizes):
        allowed=expected_gms|{f'Multiscale POD SVD-SNO new N={n_train}'}
        for method in sorted(allowed):
            points=[r for r in per_k if r['method']==method]
            x=[r['k'] for r in points]
            axis.plot(x,[r['mean'] for r in points],'.-',markersize=2,linewidth=.8,label=method)
            if points[0]['count']>1:axis.fill_between(x,[r['minimum'] for r in points],[r['maximum'] for r in points],alpha=.17)
        axis.set(title=f'Additional new N={n_train}',xlabel='k',ylabel='Scattered L2 error (%)')
        axis.set_yscale('log');axis.grid(alpha=.25);axis.legend(fontsize=6)
    axis=axes.ravel()[-1]
    axis.plot([r['k'] for r in ordered_fine],[r['error_percent'] for r in ordered_fine],'.-',markersize=2,color='gray')
    axis.set(title='Native full CG2 discretization diagnostic',xlabel='k',ylabel='Scattered L2 error (%)')
    axis.set_yscale('log');axis.grid(alpha=.25);fig.tight_layout();fig.savefig(results/'error_vs_k_panels.png',dpi=180);plt.close(fig)
    fig, axes = plt.subplots(len(train_sizes), 1, figsize=(10, 3*len(train_sizes)), squeeze=False)
    for axis, n_train in zip(axes[:,0], train_sizes):
        for seed in expected_seeds:
            h = [r for r in histories if r['seed'] == seed and r['n_train'] == n_train]
            for split, style in (('train', '--'), ('val', '-')):
                key = f'{split}_relative_l2'
                axis.plot([r['epoch'] for r in h], [100*r[key] for r in h], style, linewidth=1, label=f'{split}, seed {seed}')
        axis.set(xlabel='Epoch', ylabel='Relative L2 mean (%)', title=f'Additional new training N={n_train}; fixed old pretraining 1000')
        axis.set_yscale('log'); axis.grid(alpha=.25); axis.legend(fontsize=7,ncol=3)
    fig.tight_layout(); fig.savefig(results/'convergence.png', dpi=180); plt.close(fig)
    fig,axes=plt.subplots(len(train_sizes),1,figsize=(10,3*len(train_sizes)),squeeze=False)
    for axis,n_train in zip(axes[:,0],train_sizes):
        for seed in expected_seeds:
            h=[r for r in histories if r['n_train']==n_train and r['seed']==seed]
            for split,style in (('train','--'),('val','-')):
                axis.plot([r['optimizer_steps'] for r in h],[100*r[f'{split}_relative_l2'] for r in h],style,label=f'{split}, seed {seed}',linewidth=1)
        axis.set(xlabel='Optimizer steps',ylabel='Relative L2 mean (%)',title=f'Additional new training N={n_train}')
        axis.set_yscale('log');axis.grid(alpha=.25);axis.legend(fontsize=7,ncol=3)
    fig.tight_layout();fig.savefig(results/'convergence_optimizer_steps.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(len(train_sizes),1,figsize=(10,3*len(train_sizes)),squeeze=False)
    for axis,n_train in zip(axes[:,0],train_sizes):
        for seed in expected_seeds:
            h=[r for r in histories if r['n_train']==n_train and r['seed']==seed]
            for split,style in (('train','--'),('val','-')):
                axis.plot([r['elapsed_seconds'] for r in h],[100*r[f'{split}_relative_l2'] for r in h],style,label=f'{split}, seed {seed}',linewidth=1)
        axis.set(xlabel='Measured training wall time (seconds)',ylabel='Relative L2 mean (%)',title=f'Additional new training N={n_train}')
        axis.set_yscale('log');axis.grid(alpha=.25);axis.legend(fontsize=7,ncol=3)
    fig.tight_layout();fig.savefig(results/'convergence_wall_time.png',dpi=180);plt.close(fig)
    efficiency = []
    for n_train in train_sizes:
        method = f'Multiscale POD SVD-SNO new N={n_train}'
        seed_means = [r['mean'] for r in seed_summaries if r['method'] == method]
        efficiency.append(dict(n_train=n_train,fixed_pretraining_train=1000,total_unique_training_queries=1000+n_train,
            mean=float(np.mean(seed_means)),seed_mean_sd=float(np.std(seed_means,ddof=1)),minimum=float(min(seed_means)),maximum=float(max(seed_means))))
    _csv(results/'sample_efficiency.csv',efficiency)
    fig,axis=plt.subplots(figsize=(8,5))
    axis.plot([r['n_train'] for r in efficiency],[r['mean'] for r in efficiency],'.-',label='Neural mean over seeds')
    axis.fill_between([r['n_train'] for r in efficiency],[r['minimum'] for r in efficiency],[r['maximum'] for r in efficiency],alpha=.2,label='Seed min–max')
    for row in summaries:
        if row['method'] in expected_gms:
            axis.axhline(row['mean'],linestyle='--' if 'M=16' in row['method'] else '-.',
                         color='#dc2626' if 'M=16' in row['method'] else '#64748b',label=row['method'])
    axis.set(xlabel='Additional new training queries (fixed pretraining: 1000)',ylabel='Mean individual scattered L2 error (%)',title='Fine-tuning sample efficiency on one frozen test set')
    axis.set_yscale('log')
    axis.grid(alpha=.25,which='both');axis.legend(fontsize=7);fig.tight_layout();fig.savefig(results/'sample_efficiency.png',dpi=180);plt.close(fig)
    audit_path=results/'gms_refinement.json'
    selection_path=results/'gms_mesh_selection.json'
    if not selection_path.exists() or not _read(selection_path).get('passed'):
        raise ValueError('GMsFEM fine-discretization selection audit has not passed')
    selection=_read(selection_path)
    if selection['configuration']['fine_mesh_sha256']!=bspec['fine_mesh_sha256']:
        raise ValueError('GMsFEM selection and executed fine mesh differ')
    artifact_hashes[str(selection_path)]=_sha256(selection_path)
    if not audit_path.exists() or not _read(audit_path).get('complete'):
        raise ValueError('Native CG2 fine-grid refinement audit is incomplete')
    audit=_read(audit_path);artifact_hashes[str(audit_path)]=_sha256(audit_path)
    _csv(results/'gms_refinement_errors.csv',[{k:v for k,v in r.items() if k!='diagnostics'} for r in audit['rows']])
    _csv(results/'gms_refinement_differences.csv',audit['differences'])
    fig,axis=plt.subplots(figsize=(9,5))
    for level,method in sorted({(r['level'],r['method']) for r in audit['rows']}):
        measured=sorted([r for r in audit['rows'] if r['level']==level and r['method']==method],key=lambda r:r['k'])
        axis.plot([r['k'] for r in measured],[r['error_percent'] for r in measured],'.-',label=f'{method}, fine level {level}')
    axis.set(xlabel='Anchor k',ylabel='Native field relative L2 error (%)',title='Actual CG2 GMsFEM fine-grid refinement vs native CG3')
    axis.grid(alpha=.25);axis.legend(fontsize=7);fig.tight_layout();fig.savefig(results/'gms_refinement.png',dpi=180);plt.close(fig)
    mesh_path = results/'mesh_convergence/summary.json'
    mesh_note = 'Mesh convergence artifact absent.'
    if mesh_path.exists():
        mesh = _read(mesh_path)
        artifact_hashes[str(mesh_path)] = _sha256(mesh_path)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pairs = sorted({(r['coarse_level'], r['fine_level']) for r in mesh['rows']})
        for coarse, fine in pairs:
            measured = sorted([r for r in mesh['rows'] if (r['coarse_level'], r['fine_level']) == (coarse, fine)], key=lambda r: r['k'])
            ax.plot([r['k'] for r in measured], [100*r['scattered_relative_l2'] for r in measured], '.-', label=f'level {coarse} → {fine}')
        ax.axhline(100*mesh['tolerance'], color='black', linestyle=':', label='Declared tolerance')
        ax.set(xlabel='Wavenumber k', ylabel='Successive scattered FEM L2 difference (%)', title='Historical P1 refinement diagnostic (not the selected reference)')
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(results/'mesh_convergence.png', dpi=180); plt.close(fig)
        mesh_note = (f"Historical P1 mesh convergence passed: {mesh['passed']}; maximum latest successive scattered difference "
                     f"{100*mesh['max_successive_scattered_relative_l2']:.5g}%. " + mesh['limitation'])
    lines = ['# High-k multiscale POD SVD-SNO / GMsFEM', '',
             f'Fresh split: {np.sum(data.split == 0)} train, {np.sum(data.split == 1)} validation, {len(test)} test; {len(expected_seeds)} neural seeds.',
             'Errors below are percentages of the scattered complex consistent P1 FEM L2 norm, averaged over individual cases. Neural rows pool equally sized seed runs. The shaded curve spans seed minima and maxima at each exact test frequency.', '',
             'The inherited model contribution is gated to exactly zero for k≥5; the full-new-rank residual learns that range. Lower-frequency inherited weights are jointly fine-tuned. Test endpoints 0.05 and 20 lie slightly outside the new training range; this includes small endpoint extrapolations.', '']
    lines += ['', 'Every neural model inherits a seed-matched checkpoint pretrained on 1000 old low-k training queries and selected using old validation data. N denotes additional disjoint high-k fine-tuning queries: total unique training queries are 1000+N. This measures fine-tuning sample efficiency. The inherited 20-mode SNO and full-new-rank residual SNO are jointly trained; capacity varies with the new training-only POD rank.', '']
    lines += ['| Method | Mean % | Seed-mean SD % | Median % | P95 % | Max % |', '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['method']} | {r['mean']:.5g} | {format(r['seed_mean_sd'],'.5g') if r['seed_mean_sd'] is not None else '—'} | {r['median']:.5g} | {r['p95']:.5g} | {r['maximum']:.5g} |" for r in summaries]
    lines += ['', 'Selected reference audit: [reference convergence](reference_convergence.png) and [reference DOF table](reference_dofs.csv). The failed historical P1 study is shown separately from the selected CG3 checks.', '', 'Actual native CG2 GMsFEM fine-grid refinement and the native fine-solve discretization adequacy are recorded in `gms_refinement.json` and `gms_mesh_selection.json`. Reduction errors are retained even when large; fine-grid adequacy does not imply coarse-space accuracy.', '', 'GMsFEM uses full boundary snapshots and a fresh frequency-dependent Type2 space per test k. The declared M values are nested truncations of one maximum-M construction; timings include shared setup and must not be summed across repeated case rows.', '',
              'Raw artifacts: `paired_errors.csv`, `per_k.csv`, `bands.csv`, `convergence_raw.csv`, and `timings_dofs.csv`. Mesh accuracy is documented independently by the mesh convergence experiment; report errors compare against the chosen discrete reference.']
    fine_stats = _stats([r['error_percent'] for r in fine_diagnostic])
    lines += ['', f"Native full CG2 diagnostic versus the stored reference: mean {fine_stats['mean']:.5g}%, P95 {fine_stats['p95']:.5g}%, maximum {fine_stats['maximum']:.5g}%. This quantifies the source fine discretization discrepancy; it is not a lower bound on every reduced-space error.", '',
              'When high-order reference fields are sampled onto the P1 support mesh, all reported primary errors use those stored P1 fields. The independent native high-order refinement error and high-order-to-P1 representation error must be read separately in `reference_metadata.json` and the physics convergence artifacts. These errors are not interchangeable.', '', mesh_note]
    lines += ['', 'Validation before and after fine-tuning (selection uses validation only):', '',
              '| New N | Seed | Initial validation % | Best validation % | Selected epoch | Old weight L2 change |',
              '|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['n_train']} | {r['seed']} | {r['initial_validation_percent']:.5g} | {r['best_validation_percent']:.5g} | {r['best_epoch']} | {r['old_weights_l2_change']} |" for r in learning_checks]
    lines += ['', 'Seed-mean SD uses the three seed-level test means with ddof=1. It describes training-seed variability, not a query-resampling confidence interval.', '',
              '| Method / additional new N | Native truth complex DOFs | Support complex DOFs | GMs full complex DOFs | Reduced complex DOFs / POD rank | POD real unknowns | Network parameters |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for n_train in train_sizes:
        selected=[r for r in dof_rows if r.get('n_train')==n_train]
        def observed(key):
            values={r.get(key) for r in selected}
            if None in values:raise ValueError(f'Missing measured DOF/parameter field {key}')
            return ', '.join(map(str,sorted(values)))
        lines.append(f"| Neural new N={n_train} | {native_truth_dofs} | {len(data.coordinates)} | — | {observed('pod_complex_rank')} | {observed('pod_real_unknowns')} | {observed('network_parameters')} |")
    for row in dof_rows:
        if 'dofs_reduced' in row:lines.append(f"| {row['method']} | {native_truth_dofs} | {len(data.coordinates)} | {row['dofs_gms_full']} | {row['dofs_reduced']} | — | — |")
    lines += ['', 'Complex DOF/rank counts and real coefficient unknown counts describe the discretized fields; network parameters are separate trainable quantities.', '',
              '![Paired per-k errors](error_vs_k_panels.png)', '', '![Fine-tuning sample efficiency](sample_efficiency.png)', '',
              '![Training and validation by epoch](convergence.png)', '', '![Training and validation by optimizer step](convergence_optimizer_steps.png)', '',
              '![Training and validation by measured wall time](convergence_wall_time.png)', '',
              '![Native CG2 fine-grid audit](gms_refinement.png)', '', '![Independent reference checks](reference_convergence.png)']
    (results/'REPORT.md').write_text('\n'.join(lines)+'\n')
    _write_json(results/'report_manifest.json', dict(passed=True, test_samples=len(test), seeds=expected_seeds, train_sizes=train_sizes,
                 methods=methods, reference_metadata_sha256=_sha256(results/'reference_metadata.json'), artifact_sha256=artifact_hashes, metric='mean individual scattered consistent P1 FEM L2 percent'))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    build_report(args.data_dir, args.results_dir, args.run_root)


if __name__ == '__main__':
    main()
