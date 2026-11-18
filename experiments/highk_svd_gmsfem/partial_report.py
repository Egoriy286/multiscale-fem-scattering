"""Reproduce the explicitly partial neural-only report from saved per-case metrics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
SIZES = (400, 600, 800)
SEEDS = (11, 29, 47)
colors = ('#2464a0', '#db7d24', '#39885b')
series = {}
reference = None
for n in SIZES:
    rows_by_seed = []
    for seed in SEEDS:
        path = ROOT / f'runs/n{n}/mscale_sno/seed_{seed}/metrics.json'
        rows = json.loads(path.read_text())
        assert len(rows) == 200 and all(r['n_train'] == n and r['seed'] == seed for r in rows)
        rows.sort(key=lambda r: r['index'])
        cases = [(r['index'], r['k'], r['theta']) for r in rows]
        if reference is None:
            reference = cases
        assert cases == reference, f'Test cases differ: {path}'
        errors = np.array([r['scattered_pct'] for r in rows])
        assert np.isfinite(errors).all() and (errors > 0).all()
        rows_by_seed.append(errors)
    series[n] = np.stack(rows_by_seed)
k = np.array([r[1] for r in reference])
assert np.all(np.diff(k) > 0) and np.isclose(k[0], .05) and np.isclose(k[-1], 20)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
for n, color in zip(SIZES, colors):
    vals = series[n]
    ax.fill_between(k, vals.min(axis=0), vals.max(axis=0), color=color, alpha=.15, linewidth=0)
    ax.plot(k, vals.mean(axis=0), color=color, linewidth=1.3, label=f'N = {n}')
ax.set_yscale('log')
ax.set(xlabel='Wavenumber k', ylabel='Relative scattered-field L2 error (%)', title='PARTIAL: neural error across 200 shared test cases')
ax.grid(True, which='both', alpha=.2)
ax.legend(title='Training queries')
fig.text(.5, -.025, 'Lines: mean across 3 seeds; shading: seed min–max at each case. 9 of 15 planned runs complete.', ha='center', fontsize=9)
fig.savefig(OUT / 'partial_error_vs_k.png', dpi=180, bbox_inches='tight')
plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
means, sds = [], []
for n in SIZES:
    seed_means = series[n].mean(axis=1)
    means.append(seed_means.mean()); sds.append(seed_means.std(ddof=1))
    ax.scatter(np.full(3, n), seed_means, s=32, zorder=3, color='#2464a0', alpha=.65)
ax.errorbar(SIZES, means, yerr=sds, color='#c66b1d', marker='o', capsize=5, linewidth=1.8, label='Mean ± sample SD across seeds')
ax.set(xlabel='Additional training queries N', ylabel='Mean individual scattered-field L2 error (%)', title='PARTIAL: error versus training sample size', xticks=SIZES)
ax.grid(True, axis='y', alpha=.25)
ax.legend()
fig.text(.5, -.025, 'Each dot averages 200 shared test cases; 3 seeds per N. 9 of 15 planned runs complete.', ha='center', fontsize=9)
fig.savefig(OUT / 'partial_sample_size.png', dpi=180, bbox_inches='tight')
plt.close(fig)
lines = ['# PARTIAL report: completed neural runs', '', '**Status: 9 of 15 planned neural runs complete.** N = 1000 and 2000 are unfinished. This is not the final study report; no GMsFEM aggregate is compared here.', '', 'The nine completed runs cover N = 400, 600, and 800, each with seeds 11, 29, and 47. All are evaluated on the same 200 test cases, with k from 0.05 to 20. Values below are the arithmetic mean of individual relative scattered-field L2 errors, in percent. The across-seed standard deviation uses the sample convention (ddof = 1).', '', '| Training queries | Seed 11 | Seed 29 | Seed 47 | Mean ± sample SD |', '|---:|---:|---:|---:|---:|']
for n, mean, sd in zip(SIZES, means, sds):
    sm = series[n].mean(axis=1)
    lines.append(f'| {n} | {sm[0]:.4f}% | {sm[1]:.4f}% | {sm[2]:.4f}% | {mean:.4f}% ± {sd:.4f} percentage points |')
lines += ['', '![Partial error by wavenumber](partial_error_vs_k.png)', '', 'Lines show the mean over seeds for each case; shaded bands span the minimum to maximum seed error. The vertical axis is logarithmic.', '', '![Partial sample-size results](partial_sample_size.png)', '', 'Dots show the 200-case mean for each seed; bars show mean ± sample standard deviation across the three seed means.', '', 'Generated from saved `metrics.json` files by `partial_report.py`; test case identifiers, wavenumbers, and angles are checked for equality across all nine runs.']
(OUT / 'PARTIAL_REPORT.md').write_text('\n'.join(lines) + '\n')
for n, mean, sd in zip(SIZES, means, sds):
    print(n, series[n].mean(axis=1), mean, sd)
