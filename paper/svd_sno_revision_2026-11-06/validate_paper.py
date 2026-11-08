#!/usr/bin/env python3
"""Check manuscript claims against small, immutable evidence snapshots.

Needs NumPy; no FEM backend, model training or network access. Run after build.sh.
Use --refresh-evidence only when deliberately selecting new source artifacts.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parents[1]
EVIDENCE = PAPER / 'evidence/revision_2026-11-08'
SOURCES = {
 'pilot_low_samples.csv': 'neural_operator/runs/comparison/neural_full_test.csv',
 'pilot_low_summary.csv': 'neural_operator/runs/comparison/neural_full_test_summary.csv',
 'pilot_high_samples.csv': 'neural_operator/runs/high_frequency/comparison/frequency_samples.csv',
 'pilot_high_summary.csv': 'neural_operator/runs/high_frequency/comparison/frequency_summary.csv',
 'pilot_paired_samples.csv': 'neural_operator/runs/comparison/paired_samples.csv',
 'pilot_paired_summary.csv': 'neural_operator/runs/comparison/paired_summary.csv',
 'pilot_refinement.csv': 'neural_operator/runs/high_frequency/comparison/refinement.csv',
 'pilot_available_svd_metrics.json': 'neural_operator/runs/low_frequency/svd/metrics.json',
 'pilot_available_svd_config.json': 'neural_operator/runs/low_frequency/svd/config.json',
 'static_v1_low_metrics.json': 'neural_operator/randomized/runs/low_frequency_static_1000x8/svd/metrics.json',
 'static_v1_high_metrics.json': 'neural_operator/randomized/runs/high_frequency_static_1000x8/svd/metrics.json',
 'static_fno_low_metrics.json': 'neural_operator/randomized/runs/low_frequency_static_1000x8/fno/metrics.json',
 'static_chebyshev_low_metrics.json': 'neural_operator/runs/revision_2026-11-08/static_low/chebyshev/metrics.json',
 'operator_revision.json': 'neural_operator/docs/operator_revision.json',
 'static_v2_low_metrics.json': 'neural_operator/runs/revision_2026-11-08/static_low/svd/metrics.json',
 'static_fno_low_verification.json': 'neural_operator/docs/fno_static_low_verification_2026-11-08.json',
 'static_results.json': 'neural_operator/randomized/static_results.json',
 'static_low_config.json': 'neural_operator/data/static_low_frequency_1000x8/config.json',
 'static_high_config.json': 'neural_operator/data/static_high_frequency_1000x8/config.json',
 'static_low_geometry.npz': 'neural_operator/data/static_low_frequency_1000x8/geometry.npz',
}

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_json(name):
    return json.loads((EVIDENCE/name).read_text())

def csv_rows(name):
    with (EVIDENCE/name).open(newline='') as stream:
        return list(csv.DictReader(stream))

def stats(values):
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(np.max(values)))

def pct(number):
    return f'{100*float(number):.4f}'.replace('.', ',')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-evidence', action='store_true')
    parser.add_argument('--visual-pages-reviewed', type=int, default=0)
    args = parser.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    manifest_path = EVIDENCE/'sources.json'
    if args.refresh_evidence:
        manifest = {}
        for name, relative in SOURCES.items():
            source = REPO/relative
            shutil.copyfile(source, EVIDENCE/name)
            manifest[name] = dict(source=relative, sha256=sha(source), size_bytes=source.stat().st_size)
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    manifest = json.loads(manifest_path.read_text())
    for name, record in manifest.items():
        assert sha(EVIDENCE/name) == record['sha256'], name
    tex = (PAPER/'main.tex').read_text()
    labels = re.findall(r'\\label\{([^}]+)\}', tex)
    assert len(labels) == len(set(labels)), 'Duplicate LaTeX labels'
    for name in re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', tex):
        assert name in labels, f'Undefined label: {name}'
    bib = set(re.findall(r'\\bibitem\{([^}]+)\}', tex))
    for entry in re.findall(r'\\cite\{([^}]+)\}', tex):
        assert set(entry.split(',')) <= bib, entry
    figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', tex)
    for name in figures:
        assert (PAPER/name).is_file(), name
    # Archived pilot aggregates are reconstructed from their actual per-sample rows.
    verified_rows = []
    for samples_name, summary_name, group_keys in (
        ('pilot_low_samples.csv', 'pilot_low_summary.csv', ('method',)),
        ('pilot_high_samples.csv', 'pilot_high_summary.csv', ('method', 'regime')),
        ('pilot_paired_samples.csv', 'pilot_paired_summary.csv', ('method',)),
    ):
        samples, summaries = csv_rows(samples_name), csv_rows(summary_name)
        for row in summaries:
            selected = [float(s['error_fem_l2']) for s in samples
                        if all(s[k] == row[k] for k in group_keys)]
            assert selected, row
            actual = stats(selected)
            for key, value in actual.items():
                np.testing.assert_allclose(value, float(row[key]), atol=1e-12, rtol=1e-12)
            verified_rows.append(dict(source=summary_name,
                                      group={k: row[k] for k in group_keys}, statistics=actual))
            if summary_name == 'pilot_low_summary.csv':
                lines = [x for x in tex.splitlines() if x.startswith(row['method']+' &')]
                assert any(all(pct(row[key]) in line for key in ('mean', 'median', 'p95', 'max'))
                           for line in lines), row['method']
            if summary_name == 'pilot_high_summary.csv':
                regime = 'frozen' if row['regime'] == 'frozen_low_band' else 'retrained'
                line = next(x for x in tex.splitlines() if x.startswith(row['method']+' & '+regime))
                assert pct(row['mean']) in line and pct(row['p95']) in line
    # Static table rows, each tied to its own saved run and version.
    for name, band, model in (
        ('static_v1_low_metrics.json', '[1,2]', 'SVD-SNO'),
        ('static_v1_high_metrics.json', '[4,8]', 'SVD-SNO'),
        ('static_fno_low_metrics.json', '[1,2]', 'FNO'),
        ('static_v2_low_metrics.json', '[1,2]', 'SVD-SNO v2'),
        ('static_chebyshev_low_metrics.json', '[1,2]', 'Chebyshev-SNO'),
    ):
        metric = read_json(name)
        assert metric['metric'] == 'consistent_complex_FEM_L2'
        assert metric['field'] == 'scattered' and metric['test_samples'] == 1280
        prefix = r'\('+band+r'\) & '+model+' &'
        line = next(x for x in tex.splitlines() if x.startswith(prefix))
        for key in ('mean', 'median', 'p95', 'max'):
            assert pct(metric['test_fem_l2'][key]) in line, (name, key)
    static = read_json('static_results.json')
    frozen = static['high_band_comparison'][0]
    for value in (frozen['mean'], frozen['p95'], frozen['max'], frozen['pod_projection_mean']):
        assert pct(value) in tex, ('static frozen', value)
    for row in static['refinement']:
        assert pct(row['error_fem_l2']) in tex
    # Exact DCT-I convention and both stated parameter counts.
    dct_errors = []
    for m in (3, 8, 32):
        j = np.arange(m); s = np.cos(np.pi*(1-j[:, None]/(m-1))*j[None, :])
        d = np.ones(m); d[[0, -1]] = .5
        a = 2/(m-1)*d[:, None]*s.T*d[None, :]
        error = float(np.max(np.abs(a@s-np.eye(m))))
        assert error < 1e-12
        dct_errors.append(dict(size=m, inverse_max_error=error))
    assert 384+4*(71**2+2*(64**2+64))+130 == 53958
    assert (7*24+24)+3*(24**2+24+2*8**2)+(24*2+2) == 2426
    # Historical JAX statistics are checked against saved outputs, not rerun.
    historical = json.loads((PAPER/'evidence/notebook_extract.json').read_text())
    output_text = '\n'.join(str(o.get('text', o.get('text_plain', '')))
                            for cell in historical['cells'] for o in cell.get('outputs', []))
    saved_numbers = [float(x) for x in re.findall(r'[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?', output_text)]
    jax_rows = [
        ('SVD-проекция & Узловая', [.009221193, .007497699, .02282793, .03502893]),
        ('SVD-SNO & Узловая', [.068832680583, .062592789531, .133105203509, .206574067473]),
        ('SVD-проекция & \\(L^2\\)', [.006960822080, .005726345924, .016452080801, .026688281573]),
        ('SVD-SNO & \\(L^2\\)', [.064548673516, .059376394700, .122804294585, .193277788602]),
    ]
    for prefix, values in jax_rows:
        line = next(x for x in tex.splitlines() if x.startswith(prefix))
        for value in values:
            assert any(np.isclose(value, number, rtol=1e-6, atol=1e-12) for number in saved_numbers)
            assert pct(value) in line, (prefix, value)
    # Local Markdown links are checked for the current paper entry points only.
    markdown_links = 0
    for name in ('README.md', 'REVIEW.md', 'REPRODUCIBILITY.md'):
        for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', (PAPER/name).read_text()):
            if '://' in target or target.startswith('#'):
                continue
            assert (PAPER/target.split('#')[0]).exists(), (name, target)
            markdown_links += 1
    # The originals remain byte-for-byte identical to supplied evidence.
    original = PAPER/'original/main_original.tex'
    assert sha(original) == 'fe0266a45fc98ad418c40ba590cbc6a73e8095197d4fb6fde1ebc536710e343f'
    log = PAPER/'build/main.log'
    warnings = {}
    if log.is_file():
        output = log.read_text(errors='replace')
        for name, pattern in {
            'latex_errors': r'^!|LaTeX Error|Emergency stop',
            'overfull_boxes': r'Overfull \\[hv]box',
            'undefined_references': r'undefined references|Reference .* undefined|Citation .* undefined',
            'missing_glyphs': r'Missing character:',
        }.items():
            warnings[name] = bool(re.search(pattern, output, re.M))
            assert not warnings[name], name
    pdf_pages = None
    try:
        from pypdf import PdfReader
        pdf_pages = len(PdfReader(PAPER/'main.pdf').pages)
    except ImportError:
        try:
            info = subprocess.run(['pdfinfo', str(PAPER/'main.pdf')], capture_output=True,
                                  check=True, text=True).stdout
            pdf_pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    if args.visual_pages_reviewed:
        assert args.visual_pages_reviewed == pdf_pages, (args.visual_pages_reviewed, pdf_pages)
    report = dict(
        checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Manuscript and archived per-sample statistics; no retraining or FEM solve',
        source_snapshots=len(manifest), verified_aggregate_rows=verified_rows,
        labels=len(labels), bibliography_entries=len(bib), figures=figures,
        pdf_pages=pdf_pages, historical_jax_metric_values_checked=16, markdown_links_checked=markdown_links,
        dct_inverse_checks=dct_errors, originals_unchanged=True,
        historical_jax_training_rerun=False, historical_low_pilot_svd_checkpoint_recovered=False,
        visual_pages_reviewed=args.visual_pages_reviewed, latex=warnings,
        tex_sha256=sha(PAPER/'main.tex'),
        pdf_sha256=sha(PAPER/'main.pdf') if (PAPER/'main.pdf').is_file() else None,
    )
    rendered_report = json.dumps(report, indent=2, ensure_ascii=False)+'\n'
    (EVIDENCE/'validation_report.json').write_text(rendered_report)
    (PAPER/'evidence/validation_report.json').write_text(rendered_report)
    print(json.dumps(dict(source_snapshots=len(manifest), aggregate_rows=len(verified_rows),
                         labels=len(labels), figures=len(figures), checks='passed'), ensure_ascii=False))

if __name__ == '__main__':
    main()
