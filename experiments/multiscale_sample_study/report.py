"""Actual-history reporting, gated on complete runs and checkpoint verification."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

NAMES = {'sno_gelu': 'SVD-SNO', 'mscale_sno': 'Multiscale SVD-SNO',
         'fno_gelu': 'FNO', 'mscale_fno': 'MscaleFNO'}
COLORS = dict(zip(NAMES, ('#2563eb', '#08916a', '#d97706', '#a333b6')))
BASELINE_STYLES = {
    'FEM h=0.04': dict(color='#111827', linestyle=':'),
    'GMsFEM 20x20 Type 1 M=16': dict(color='#dc2626', linestyle='--'),
    'GMsFEM 20x20 Type 2 M=16': dict(color='#64748b', linestyle='-.'),
}


def _read(path):
    return json.loads(Path(path).read_text())


def _key(row):
    return int(row['n_train']), row['method'], int(row['seed'])


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _history(path):
    rows = _read(path) if path.exists() else []
    previous = (0, -1, -1.)
    for r in rows:
        values = [r[k] for k in ('epoch', 'optimizer_steps', 'train_relative_l2',
                                 'val_relative_l2', 'elapsed_seconds', 'learning_rate')]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError(f'Nonfinite or negative history: {path}')
        current = int(r['epoch']), int(r['optimizer_steps']), float(r['elapsed_seconds'])
        if current[0] <= previous[0] or current[1] <= previous[1] or current[2] < previous[2]:
            raise ValueError(f'Nonmonotone history coordinates: {path}')
        previous = current
    return rows


def _collect(root, protocol):
    dimensions = [protocol[k] for k in ('train_sizes', 'methods', 'seeds')]
    if any(not d or len(d) != len(set(d)) for d in dimensions):
        raise ValueError('Empty or duplicate protocol dimensions')
    if set(protocol['methods']) != set(NAMES):
        raise ValueError('The four main methods are required')
    if set(protocol['train_sizes']) != {400, 600, 800, 1000} or set(protocol['seeds']) != {11, 29, 47}:
        raise ValueError('The frozen four sizes and three seeds are required')
    expected = set(itertools.product(*dimensions))
    runs, histories, missing = {}, {}, []
    for n, method, seed in sorted(expected):
        directory = root/'runs'/f'n{n}'/method/f'seed_{seed}'
        history = _history(directory/'history.json')
        if history:
            histories[n, method, seed] = history
        result_path = directory/'result.json'
        if not result_path.exists():
            missing.append((n, method, seed))
            continue
        result = _read(result_path)
        if _key(result) != (n, method, seed):
            raise ValueError(f'Result identity mismatch: {directory}')
        if not history or result['epochs_completed'] != history[-1]['epoch']:
            raise ValueError(f'Result/history mismatch: {directory}')
        if result['epochs_completed'] > protocol['epochs']:
            raise ValueError(f'Epoch budget exceeded: {directory}')
        best = min(history, key=lambda h: h['val_relative_l2'])
        if result['best_epoch'] != best['epoch'] or not np.isclose(
                result['best_validation_relative_l2'], best['val_relative_l2'], rtol=1e-8):
            raise ValueError(f'Invalid validation checkpoint: {directory}')
        reason = result['stop_reason']
        if reason not in {'epochs_completed', 'validation_patience'}:
            raise ValueError(f'Unfinished stop reason: {directory}')
        if reason == 'epochs_completed' and result['epochs_completed'] != protocol['epochs']:
            raise ValueError(f'Incomplete epoch budget: {directory}')
        if reason == 'validation_patience':
            best_index = history.index(best)
            if len(history)-best_index-1 < protocol['patience_checks']:
                raise ValueError(f'Insufficient patience evidence: {directory}')
        runs[n, method, seed] = result
    return expected, runs, histories, missing


def _verify(root, expected, runs):
    path = root/'results'/'verification.json'
    if not path.exists():
        raise ValueError('Final report requires verification.json')
    verification = _read(path)
    entries = verification.get('runs', [])
    if verification.get('passed') is not True or len(entries) != len(expected) or {_key(r) for r in entries} != expected:
        raise ValueError('Checkpoint verification is incomplete or failed')
    for row in entries:
        key = _key(row)
        n, method, seed = key
        directory = root/'runs'/f'n{n}'/method/f'seed_{seed}'
        for filename, field in [('model.pt', 'checkpoint_sha256'), ('predictions.npz', 'predictions_sha256')]:
            actual = _hash(directory/filename)
            if row.get(field) != actual or runs[key].get(field) != actual:
                raise ValueError(f'Verification hash mismatch: {directory/filename}')


def _metrics(root, expected, runs):
    with (root/'results'/'metrics.csv').open(newline='') as stream:
        raw = list(csv.DictReader(stream))
    groups = {}
    for r in raw:
        key = _key(r)
        if key not in expected:
            raise ValueError('Unexpected metric run')
        item = dict(index=int(r['index']), k=float(r['k']), theta=float(r['theta']),
                    scattered_pct=float(r['scattered_pct']), total_pct=float(r['total_pct']))
        if not np.isfinite(list(item.values())).all() or min(item['scattered_pct'], item['total_pct']) < 0:
            raise ValueError('Nonfinite or negative test metric')
        if item['index'] in groups.setdefault(key, {}):
            raise ValueError('Duplicate test query')
        groups[key][item['index']] = item
    if set(groups) != expected:
        raise ValueError('Incomplete test metrics')
    reference = None
    for key, values in groups.items():
        signature = {(i, r['k'], r['theta']) for i, r in values.items()}
        if len(signature) != 100 or (reference is not None and signature != reference):
            raise ValueError('Test queries are not the common 100-case set')
        reference = signature
        n, method, seed = key
        directory = root/'runs'/f'n{n}'/method/f'seed_{seed}'
        with np.load(directory/'predictions.npz') as saved:
            ids = np.asarray(saved['indices'])
            if len(ids) != 100 or len(set(ids.tolist())) != 100 or set(ids.tolist()) != set(values):
                raise ValueError('Prediction/test metric identities differ')
            if not np.isfinite(saved['prediction']).all():
                raise ValueError('Nonfinite predicted fields')
            for i, k, theta in zip(ids, saved['k'], saved['theta'], strict=True):
                if float(k) != values[int(i)]['k'] or float(theta) != values[int(i)]['theta']:
                    raise ValueError('Prediction physical query mismatch')
        for metric, prefix in [('scattered_pct', 'scattered'), ('total_pct', 'total')]:
            a = np.array([r[metric] for r in values.values()])
            for suffix, actual in [('mean_pct', a.mean()), ('p95_pct', np.percentile(a, 95))]:
                if not np.isclose(runs[key][prefix+'_'+suffix], actual, rtol=1e-6, atol=1e-7):
                    raise ValueError('Saved summary disagrees with test metrics')
    return groups


def _baselines(root, protocol, metrics):
    """Validate the optional freshly solved deterministic 100-query supplement."""
    path = root/'results'/'gmsfem_m16'/'baselines.json'
    if not path.exists():
        return []
    manifest_path = root/'results'/'gmsfem_manifest.json'
    if not manifest_path.exists():
        raise ValueError('GMsFEM baselines require gmsfem_manifest.json')
    manifest = _read(manifest_path)
    if (not protocol.get('dataset_sha256') or manifest.get('dataset_sha256') != protocol['dataset_sha256']
            or manifest.get('baselines_sha256') != _hash(path) or not manifest.get('source_runtime')):
        raise ValueError('GMsFEM manifest dataset/hash/runtime mismatch')
    payload = _read(path)
    metadata = payload['metadata']
    expected_methods = {'FEM h=0.04', 'GMsFEM 20x20 Type 1 M=16', 'GMsFEM 20x20 Type 2 M=16'}
    expected_queries = next(iter(metrics.values()))
    expected_signature = {(i, r['k'], r['theta']) for i, r in expected_queries.items()}
    if (metadata.get('case') != 'notebook' or metadata.get('coarse_grids') != [20]
            or metadata.get('basis_types') != [1, 2] or metadata.get('modes') != 16
            or metadata.get('reference_mesh_sha256') != protocol['dataset_sha256'].get('etal.msh')
            or metadata.get('fine_mesh_sha256') != 'a8af14fc965d4823fcb5be5d6f102e7225ff2d553b1c4b3a71325834e30807de'
            or len(metadata.get('sample_indices', [])) != 100
            or set(metadata['sample_indices']) != set(expected_queries)):
        raise ValueError('GMsFEM metadata differs from the physical test problem')
    groups = {}
    for row in payload['rows']:
        method = row['method']
        index = int(row['index'])
        if method not in expected_methods or index in groups.setdefault(method, {}):
            raise ValueError('Unexpected or duplicate GMsFEM baseline row')
        if row.get('sample_index', index) != index:
            raise ValueError('GMsFEM sample identity mismatch')
        values = [float(row[k]) for k in ['k', 'theta', 'scattered_e_complex_ref', 'total_e_complex_ref']]
        if not np.isfinite(values).all() or min(values[2:]) < 0:
            raise ValueError('Invalid GMsFEM physical query/error')
        groups[method][index] = row
    if set(groups) != expected_methods:
        raise ValueError('Missing GMsFEM baseline method')
    summary = []
    for method, rows in sorted(groups.items()):
        signature = {(i, float(r['k']), float(r['theta'])) for i, r in rows.items()}
        if len(rows) != 100 or signature != expected_signature:
            raise ValueError('GMsFEM baselines are not the common 100 test queries')
        item = dict(method=method, test_queries=100, source_runtime=manifest['source_runtime'])
        for prefix in ['scattered', 'total']:
            values = [r[prefix+'_e_complex_ref'] for r in rows.values()]
            item[prefix+'_mean_pct'] = float(np.mean(values))
            item[prefix+'_p95_pct'] = float(np.percentile(values, 95))
        summary.append(item)
    return summary


def _frequency_profiles(protocol, metrics):
    """Pair physical queries before averaging errors across model seeds."""
    profiles = []
    for n, method in itertools.product(protocol['train_sizes'], protocol['methods']):
        groups = [metrics[n, method, seed] for seed in protocol['seeds']]
        reference = groups[0]
        signature = {(i, row['k'], row['theta']) for i, row in reference.items()}
        if any({(i, row['k'], row['theta']) for i, row in group.items()} != signature for group in groups):
            raise ValueError('Frequency profiles require paired physical queries across seeds')
        for index in sorted(reference, key=lambda i: (reference[i]['k'], reference[i]['theta'], i)):
            query = reference[index]
            row = dict(n_train=n, method=method, index=index, k=query['k'], theta=query['theta'],
                       seed_count=len(groups))
            for field in ('scattered', 'total'):
                values = np.asarray([group[index][field+'_pct'] for group in groups], dtype=float)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Invalid frequency-profile error')
                row.update({field+'_mean_pct': float(values.mean()),
                            field+'_min_pct': float(values.min()),
                            field+'_max_pct': float(values.max())})
            profiles.append(row)
    return profiles


def _baseline_profiles(root, baselines):
    """Per-query solver values, after _baselines has validated their manifest."""
    if not baselines:
        return []
    payload = _read(root/'results/gmsfem_m16/baselines.json')
    rows = []
    for source in sorted(payload['rows'], key=lambda r: (r['method'], r['k'], r['theta'], r['index'])):
        row = dict(n_train='', method=source['method'], index=int(source['index']),
                   k=float(source['k']), theta=float(source['theta']), seed_count=0)
        for field in ('scattered', 'total'):
            value = float(source[field+'_e_complex_ref'])
            row.update({field+'_mean_pct': value, field+'_min_pct': value, field+'_max_pct': value})
        rows.append(row)
    return rows


def _frequency_plots(root, protocol, profiles, baseline_profiles):
    """Actual test error versus physical k, with identical solver cases in each panel."""
    import matplotlib.pyplot as plt
    paths = []
    for field in ('scattered', 'total'):
        fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), squeeze=False)
        for ax, n in zip(axes.flat, protocol['train_sizes']):
            for method in protocol['methods']:
                selected = [row for row in profiles if row['n_train']==n and row['method']==method]
                ax.plot([r['k'] for r in selected], [r[field+'_mean_pct'] for r in selected],
                        color=COLORS[method], linewidth=1.5, label=NAMES[method])
                ax.fill_between([r['k'] for r in selected], [r[field+'_min_pct'] for r in selected],
                                [r[field+'_max_pct'] for r in selected], color=COLORS[method], alpha=.10)
            for method in BASELINE_STYLES:
                selected = [row for row in baseline_profiles if row['method']==method]
                if selected:
                    ax.plot([r['k'] for r in selected], [r[field+'_mean_pct'] for r in selected],
                            **BASELINE_STYLES[method], linewidth=1.45, label=method)
            values = [r[field+'_min_pct'] for r in profiles if r['n_train']==n]
            values += [r[field+'_min_pct'] for r in baseline_profiles]
            if values and min(values)>0:
                ax.set_yscale('log')
            ax.set(title=f'{n} training queries', xlabel='Physical wavenumber k',
                   ylabel=f'Test {field} complex FEM L2 (%)')
            ax.grid(alpha=.22, which='both')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9,
                   bbox_to_anchor=(.5, .015), frameon=False)
        fig.suptitle(f'Test {field} error versus k — same 100 physical queries\n'
                     'Neural curves: mean and min–max across 3 seeds; solvers: deterministic', fontsize=12)
        fig.tight_layout(rect=(0, .13, 1, .94))
        path=root/f'test_{field}_error_vs_k.png'
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path.name)
    if baseline_profiles:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
        for ax, field in zip(axes, ('scattered', 'total')):
            for method, style in BASELINE_STYLES.items():
                selected = [row for row in baseline_profiles if row['method']==method]
                ax.plot([r['k'] for r in selected], [r[field+'_mean_pct'] for r in selected],
                        **style, linewidth=1.6, label=method)
            if min(row[field+'_mean_pct'] for row in baseline_profiles)>0:
                ax.set_yscale('log')
            ax.set(xlabel='Physical wavenumber k', ylabel=f'Test {field} complex FEM L2 (%)')
            ax.grid(alpha=.22, which='both')
        handles, labels=axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, frameon=False)
        fig.suptitle('FEM and GMsFEM Type 1/2, 20x20, M=16 — actual errors at 100 common test frequencies')
        fig.tight_layout(rect=(0, .11, 1, .92))
        path=root/'gmsfem_error_vs_k.png'
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path.name)
    return paths


def _plots(root, protocol, histories, summary, partial, baselines=None, profiles=None, baseline_profiles=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    prefix = 'live_' if partial else ''
    paths = []
    for coordinate, label, suffix in [('epoch', 'Epoch', 'epoch'), ('elapsed_seconds', 'Elapsed training time (s)', 'time'),
                                       ('optimizer_steps', 'Optimizer updates', 'steps')]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
        for ax, n in zip(axes.flat, protocol['train_sizes']):
            for method in protocol['methods']:
                sequences = [histories[n, method, seed] for seed in protocol['seeds'] if (n, method, seed) in histories]
                for rows in sequences:
                    ax.plot([r[coordinate] for r in rows], [100*r['val_relative_l2'] for r in rows],
                            color=COLORS[method], alpha=.27, linewidth=.8, marker='.' if len(rows)==1 else None)
                # Aggregate at genuinely common evaluated epochs only. Wall-time is the
                # median observed time at that epoch; there is no invented resampling.
                common = set.intersection(*[{r['epoch'] for r in rows} for rows in sequences]) if sequences else set()
                if common:
                    lookups = [{r['epoch']: r for r in rows} for rows in sequences]
                    x = [np.median([d[e][coordinate] for d in lookups]) for e in sorted(common)]
                    values = np.array([[100*d[e]['val_relative_l2'] for e in sorted(common)] for d in lookups])
                    ax.plot(x, np.median(values, axis=0), color=COLORS[method], label=f'{NAMES[method]} ({len(sequences)} seeds)')
                    if len(sequences)>1:
                        lo, hi = np.percentile(values, [25, 75], axis=0)
                        ax.fill_between(x, lo, hi, color=COLORS[method], alpha=.12)
            panel_label = label
            observed_times = [r['elapsed_seconds'] for (size, _, _), rows in histories.items() if size == n for r in rows]
            if coordinate == 'elapsed_seconds' and observed_times and min(observed_times) > 0:
                ax.set_xscale('log')
                panel_label = 'Elapsed training time (seconds, logarithmic scale)'
            observed_errors = [r['val_relative_l2'] for (size, _, _), rows in histories.items() if size == n for r in rows]
            error_label = 'Validation scattered FEM error (%)'
            if observed_errors and min(observed_errors) > 0:
                ax.set_yscale('log')
                error_label = 'Validation scattered FEM error (%, logarithmic scale)'
            elif observed_errors:
                error_label += ' — linear scale (zero observed)'
            ax.set(title=f'{n} training queries', xlabel=panel_label, ylabel=error_label)
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=7)
        fig.suptitle(('LIVE — incomplete study; ' if partial else '')+'Raw seeds; median/IQR at observed epochs; positive validation errors on logarithmic scale')
        fig.tight_layout()
        path = root/f'{prefix}convergence_{suffix}.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path.name)
    if summary:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5))
        for ax, metric in zip(axes, ['scattered', 'total']):
            for method in protocol['methods']:
                rows = [r for r in summary if r['method']==method]
                ax.errorbar([r['n_train'] for r in rows], [r[metric+'_mean_pct'] for r in rows],
                            yerr=[r[metric+'_seed_sd_pct'] for r in rows], color=COLORS[method], label=NAMES[method], marker='o', capsize=3)
            for baseline in baselines or []:
                ax.axhline(baseline[metric+'_mean_pct'], **BASELINE_STYLES[baseline['method']],
                           linewidth=1.4, label=baseline['method'])
            ax.set(xlabel='Training queries', ylabel=f'Test {metric} FEM error (%)', title='Mean ± sample SD across initialization seeds')
            ax.grid(alpha=.2)
        handles, labels=axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9,
                   bbox_to_anchor=(.5, .015), frameon=False)
        fig.tight_layout(rect=(0, .21, 1, 1))
        path=root/'sample_size_error.png'
        fig.savefig(path,dpi=160)
        plt.close(fig)
        paths.append(path.name)
    if profiles and not partial:
        paths.extend(_frequency_plots(root, protocol, profiles, baseline_profiles or []))
    return paths


def build_report(root, allow_partial=False):
    """Write final artifacts only after exact run completeness and replay verification.

    Root verification producer writes results/verification.json with passed=true and one
    {n_train, method, seed, checkpoint_sha256, predictions_sha256} entry per run.
    Partial mode writes only LIVE_REPORT.md and live convergence PNGs.
    """
    root=Path(root)
    output=root/'results'
    output.mkdir(parents=True, exist_ok=True)
    protocol=_read(root/'protocol.json')
    expected,runs,histories,missing=_collect(root,protocol)
    if allow_partial:
        plots=_plots(output,protocol,histories,[],True)
        lines=['# LIVE — unfinished experiment', '', f'{len(runs)}/{len(expected)} runs have completion records; {len(histories)} have measured histories.',
               '', 'This is a progress view, not a final result. Raw points are actual evaluated epochs. Positive validation errors are shown in percent on a logarithmic y-axis; a panel with an observed zero uses a labeled linear axis. Shading is the interquartile range, not a confidence interval.', '',
               '## Unfinished runs', '']+[f'- n={n}, {m}, seed={s}' for n,m,s in missing]
        lines+=['']+[f'![Live convergence]({p})' for p in plots]
        path=output/'LIVE_REPORT.md'
        path.write_text('\n'.join(lines)+'\n')
        return {'status':'partial','completed':len(runs),'expected':len(expected),'report':str(path)}
    if missing:
        raise ValueError(f'Incomplete study: {len(missing)} unfinished runs; use allow_partial=True for progress')
    _verify(root,expected,runs)
    metrics=_metrics(root,expected,runs)
    baselines=_baselines(root,protocol,metrics)
    summary=[]
    for n,method in itertools.product(protocol['train_sizes'],protocol['methods']):
        records=[runs[n,method,s] for s in protocol['seeds']]
        row=dict(n_train=n,method=method,seeds=len(records))
        for prefix in ['scattered','total']:
            means=np.array([r[prefix+'_mean_pct'] for r in records])
            row[prefix+'_mean_pct']=float(means.mean())
            row[prefix+'_seed_sd_pct']=float(means.std(ddof=1)) if len(means)>1 else 0.
            row[prefix+'_mean_case_p95_pct']=float(np.mean([r[prefix+'_p95_pct'] for r in records]))
        row['training_seconds_mean']=float(np.mean([r['training_seconds'] for r in records]))
        row['parameters_min']=min(r['parameters'] for r in records)
        row['parameters_max']=max(r['parameters'] for r in records)
        summary.append(row)
    threshold_rows=[]
    for key,rows in sorted(histories.items()):
        n,method,seed=key
        for threshold in [5,10,20,50]:
            hit=next((r for r in rows if 100*r['val_relative_l2']<=threshold),None)
            threshold_rows.append(dict(n_train=n,method=method,seed=seed,threshold_pct=threshold,reached=hit is not None,
                                       epoch=hit['epoch'] if hit else '',optimizer_steps=hit['optimizer_steps'] if hit else '',
                                       elapsed_seconds=hit['elapsed_seconds'] if hit else ''))
    paired=[]
    for n,seed in itertools.product(protocol['train_sizes'],protocol['seeds']):
        for a,b in itertools.combinations(protocol['methods'],2):
            for index in sorted(metrics[n,a,seed]):
                left,right=metrics[n,a,seed][index],metrics[n,b,seed][index]
                paired.append(dict(n_train=n,seed=seed,index=index,method_a=a,method_b=b,
                                   scattered_delta_pct=left['scattered_pct']-right['scattered_pct'],total_delta_pct=left['total_pct']-right['total_pct']))
    _csv(output/'summary.csv',summary)
    if baselines:
        _csv(output/'gmsfem_summary.csv',baselines)
    _csv(output/'validation_thresholds.csv',threshold_rows)
    _csv(output/'paired_errors.csv',paired)
    profiles=_frequency_profiles(protocol,metrics)
    solver_profiles=_baseline_profiles(root,baselines)
    _csv(output/'frequency_error_profiles.csv',profiles+solver_profiles)
    plots=_plots(output,protocol,histories,summary,False,baselines,profiles,solver_profiles)
    lines=['# Completed sample-size study','',f'All {len(expected)} runs passed the supplied checkpoint replay verification and artifact hash checks.',
           '', 'Common consistent-mass complex FEM errors; validation-selected checkpoints. Seed SD measures initialization variability on one fixed dataset. Case p95 is averaged across seeds. No test tuning is permitted by the protocol.', '',
           '| Train | Method | Scattered mean ± seed SD (%) | Mean case p95 (%) | Total mean ± seed SD (%) | Trainable real weights | Mean training s |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for r in summary:
        lines.append(f"| {r['n_train']} | {NAMES[r['method']]} | {r['scattered_mean_pct']:.3f} ± {r['scattered_seed_sd_pct']:.3f} | {r['scattered_mean_case_p95_pct']:.3f} | {r['total_mean_pct']:.3f} ± {r['total_seed_sd_pct']:.3f} | {r['parameters_min']:,}–{r['parameters_max']:,} | {r['training_seconds_mean']:.1f} |")
    lines += ['', f"Resource-limited training used a {protocol['epochs']}-epoch cap, with validation every {protocol.get('validation_interval', 5)} epochs (and the recorded boundary evaluations). Reaching the cap does not establish convergence.", '',
              'SVD-SNO runs use CPU; FFT-FNO runs use MPS. Wall-time comparisons across those families include different hardware/runtime paths and do not isolate architecture efficiency. Ordinary SVD-SNO uses GELU whereas multiscale SVD-SNO uses sine: this is a whole-method comparison, so differences cannot be attributed solely to scaling.', '',
              'The trainable-weight count excludes the train-fitted POD mean/modes and fixed grid/FEM transfers. Both SNO variants share the same decoder for each training size. Representation fitting/projection timings are recorded in cache/n*/{sno,fno}/cache.json; FEM generation timings are in data/dataset_info.json. These offline costs are outside the optimizer/validation training timer.', '']
    if baselines:
        lines += ['## Fresh deterministic common-test supplement', '',
                  'These fresh FEM/GMsFEM solves use the same 100 physical test queries and reference-normalized complex errors. They have no training-sample count or initialization-seed SD. The deterministic solver fields are evaluated directly at reference quadrature; neural predictions are P1 reconstructions on the reference mesh. This supplement was not used for neural tuning.', '',
                  '| Solver | Scattered mean (%) | Scattered p95 (%) | Total mean (%) | Total p95 (%) |',
                  '| --- | --- | --- | --- | --- |']
        for b in baselines:
            lines.append(f"| {b['method']} | {b['scattered_mean_pct']:.3f} | {b['scattered_p95_pct']:.3f} | {b['total_mean_pct']:.3f} | {b['total_p95_pct']:.3f} |")
        lines += ['', '[Baseline summary and runtime](gmsfem_summary.csv).', '']
    lines += ['', '## Error profiles versus physical wavenumber', '',
              'The k plots use all 100 common test queries, sorted by the actual physical wavenumber. Each neural point is the arithmetic mean of the three validation-selected models; shading is their observed min–max, not a confidence interval. FEM and GMsFEM curves use the same per-query reference-normalized complex errors, with Type 1/2 on the 20x20 grid and M=16. Resonant peaks and nonmonotonicity are preserved: each k defines a different scattering problem.', '',
              'Horizontal solver lines in the training-size plot are the means over that same test set. They are independent of neural training-set size. Training/validation convergence plots retain their own validation metric; the solver curves and horizontal lines belong to the common-test comparisons.', '',
              '[All per-frequency values and seed ranges](frequency_error_profiles.csv).', '']
    lines+=['', 'Convergence error axes show percentages on a logarithmic scale when all observed errors are positive; a panel containing zero uses a labeled linear scale. Raw seed curves and median/interquartile summaries use common actually evaluated epochs; shading is not a confidence interval. Time and update coordinates are medians at those same epochs. Threshold crossings are first observed validation evaluations, with no interpolation; blank crossings mean not reached.', '',
            'See [protocol](../PROTOCOL.md), [article comparison](../ARTICLE_COMPARISON.md), [summary](summary.csv), [paired case differences](paired_errors.csv), and [observed threshold crossings](validation_thresholds.csv).','']
    if (output/'physical_audit.json').exists():
        lines += ['', '[Independent physical metric audit](physical_audit.json).']
    if (output/'fem_label_audit.json').exists():
        label_audit=_read(output/'fem_label_audit.json')
        if not label_audit.get('passed') or label_audit.get('ncases')!=1200:
            raise ValueError('Full FEM label residual audit did not pass')
        lines += ['', f"[All 1,200 FEM label residuals](fem_label_audit.json): maximum normalized discrete-system residual {label_audit['maximum']:.3e}. This checks the saved discrete PDE solutions, not their continuum discretization error."]
    if (output/'representation_diagnostics.json').exists():
        lines += ['', '[Representation diagnostics](representation_diagnostics.json). The optimal consistent-mass G-projection residual is a lower bound for predictions restricted to the fixed bilinear grid-to-FEM decoder: it minimizes the FEM error over that decoder’s entire output space. The separate P1 FEM-to-grid-to-FEM round-trip error measures one particular transfer and is not a strict lower bound. Neither diagnostic guarantees optimizer convergence.']
    lines += [f'![Study plot]({p})' for p in plots]
    path=output/'REPORT.md'
    path.write_text('\n'.join(lines)+'\n')
    return {'status':'complete','completed':len(runs),'expected':len(expected),'report':str(path)}
