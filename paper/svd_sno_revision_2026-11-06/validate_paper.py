#!/usr/bin/env python3
"""Validate the current one-medium, three-model manuscript without training.

Needs NumPy. --refresh-evidence deliberately snapshots the three selected runs.
After the final PDF is visually checked, pass its actual page count with
--visual-pages-reviewed N. This records hashes, not a new visual inspection.
"""
from __future__ import annotations
import argparse
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
EVIDENCE = PAPER/'evidence/current_static_low'
RUNS = {
 'sno': ('SVD-SNO v2', 'neural_operator/runs/revision_2026-11-08/static_low/svd'),
 'fno': ('FNO', 'neural_operator/randomized/runs/low_frequency_static_1000x8/fno'),
 'chebyshev': ('Chebyshev-SNO', 'neural_operator/runs/revision_2026-11-08/static_low/chebyshev'),
}
FINGERPRINT = '786ee3849e909083d98213653affa47907312d0f626e8a60da76e88c70de0ca5'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def sources():
    paths = {}
    for key, (_, run) in RUNS.items():
        for filename in ('metrics.json', 'config.json', 'history.json'):
            paths[f'{key}_{filename}'] = f'{run}/{filename}'
    paths.update({
      'dataset_config.json': 'neural_operator/data/static_low_frequency_1000x8/config.json',
      'geometry.npz': 'neural_operator/data/static_low_frequency_1000x8/geometry.npz',
      'chebyshev_decoder_validation.json': 'neural_operator/docs/chebyshev_decoder_validation_2026-11-08.json',
      'fno_verification.json': 'neural_operator/docs/fno_static_low_verification_2026-11-08.json',
    })
    return paths

def pct(value):
    return f'{100*float(value):.4f}'.replace('.', ',')

def read(name):
    return json.loads((EVIDENCE/name).read_text())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-evidence', action='store_true')
    parser.add_argument('--visual-pages-reviewed', type=int, default=0)
    args = parser.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if args.refresh_evidence:
        manifest = {}
        for name, relative in sources().items():
            source = REPO/relative
            shutil.copyfile(source, EVIDENCE/name)
            manifest[name] = dict(source=relative, sha256=sha(source), size_bytes=source.stat().st_size)
        (EVIDENCE/'sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
    manifest = read('sources.json')
    for name, record in manifest.items():
        assert sha(EVIDENCE/name) == record['sha256'], name
    tex = (PAPER/'main.tex').read_text()
    metric_values_checked = 0
    for key, (label, _) in RUNS.items():
        metrics = read(f'{key}_metrics.json')
        assert metrics['dataset_fingerprint'] == FINGERPRINT
        assert metrics['metric'] == 'consistent_complex_FEM_L2' and metrics['field'] == 'scattered'
        assert [metrics[x] for x in ('train_samples', 'val_samples', 'test_samples')] == [5360, 1360, 1280]
        assert metrics['split_strategy'] == 'frequency' and metrics['physical_media'] == 1
        line = next(line for line in tex.splitlines() if line.startswith(label+' &'))
        for statistic in ('mean', 'median', 'p95', 'max'):
            assert pct(metrics['test_fem_l2'][statistic]) in line, (label, statistic)
            metric_values_checked += 1
        assert str(metrics['parameters']).replace('4022', '4\\,022').replace('445010', '445\\,010').replace('2426', '2\\,426') in line
        history = read(f'{key}_history.json')
        assert len(history['val']) == metrics['completed_epochs']
        assert int(np.argmin(history['val']))+1 == metrics['best_epoch']
        if key == 'sno':
            assert metrics['config']['sno_architecture_version'] == 2
            assert metrics['ranks'] == {'A': 0, 'n': 0, 'real': 10, 'imag': 10}
            assert pct(metrics['consistent_projection_floor_fem_l2']['mean']) in tex
    decoder = read('chebyshev_decoder_validation.json')
    assert decoder['dataset_fingerprint'] == FINGERPRINT and decoder['validation_samples'] == 1360
    eight = next(row for row in decoder['records'] if row['modes_per_axis'] == 8)
    assert pct(eight['validation_fem_l2']['mean']) in tex
    assert pct(eight['validation_fem_l2']['p95']) in tex
    labels = re.findall(r'\\label\{([^}]+)\}', tex)
    assert len(labels) == len(set(labels))
    for name in re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', tex):
        assert name in labels, name
    bib = set(re.findall(r'\\bibitem\{([^}]+)\}', tex))
    for names in re.findall(r'\\cite\{([^}]+)\}', tex):
        assert set(names.split(',')) <= bib, names
    figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', tex)
    for name in figures:
        assert (PAPER/name).is_file(), name
    markdown_links = 0
    for filename in ('README.md', 'REVIEW.md', 'REPRODUCIBILITY.md'):
        for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', (PAPER/filename).read_text()):
            if '://' in target or target.startswith('#'):
                continue
            if target in ('main.pdf', 'evidence/validation_report.json', 'evidence/current_static_low/validation_report.json') and not (PAPER/target).exists():
                continue
            assert (PAPER/target.split('#')[0]).exists(), (filename, target)
            markdown_links += 1
    dct_errors = []
    for m in (3, 8, 32):
        j = np.arange(m)
        synthesis = np.cos(np.pi*(1-j[:, None]/(m-1))*j[None, :])
        d = np.ones(m); d[[0, -1]] = .5
        analysis = 2/(m-1)*d[:, None]*synthesis.T*d[None, :]
        error = float(np.max(np.abs(analysis@synthesis-np.eye(m))))
        assert error < 1e-12
        dct_errors.append(dict(size=m, inverse_max_error=error))
    assert (5*24+24)+3*(24**2+24+10**2+24**2)+(24*2+2) == 4022
    assert (7*24+24)+3*(24**2+24+2*8**2)+(24*2+2) == 2426
    provenance = json.loads((PAPER/'images/static_low_predictions_provenance.json').read_text())
    assert provenance['dataset_fingerprint'] == FINGERPRINT
    assert provenance['sample']['position_in_test'] == 0 and set(provenance['models']) == set(RUNS)
    assert provenance['training_epochs_performed'] == provenance['fem_solves_performed'] == 0
    for key, record in provenance['models'].items():
        assert record['checked_test_samples'] == 1280
        for statistic, value in record['test_fem_l2'].items():
            np.testing.assert_allclose(value, read(f'{key}_metrics.json')['test_fem_l2'][statistic], rtol=2e-6, atol=1e-9)
    original = PAPER/'original/main_original.tex'
    assert sha(original) == 'fe0266a45fc98ad418c40ba590cbc6a73e8095197d4fb6fde1ebc536710e343f'
    warnings = {}
    log = PAPER/'build/main.log'
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
    pdf = PAPER/'main.pdf'
    pdf_pages = None
    if pdf.is_file():
        try:
            from pypdf import PdfReader
            pdf_pages = len(PdfReader(pdf).pages)
        except ImportError:
            try:
                info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, check=True, text=True).stdout
                pdf_pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
    if args.visual_pages_reviewed:
        assert pdf_pages == args.visual_pages_reviewed
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(),
      scope='Current static-low dataset and three selected completed runs',
      source_snapshots=len(manifest), metric_table_values_checked=metric_values_checked,
      labels=len(labels), bibliography_entries=len(bib), figures=figures,
      markdown_links_checked=markdown_links, dct_inverse_checks=dct_errors,
      original_unchanged=True, pdf_pages=pdf_pages, visual_pages_reviewed=args.visual_pages_reviewed,
      latex=warnings, tex_sha256=sha(PAPER/'main.tex'), pdf_sha256=sha(pdf) if pdf.is_file() else None)
    encoded = json.dumps(report, indent=2, ensure_ascii=False)+'\n'
    (EVIDENCE/'validation_report.json').write_text(encoded)
    (PAPER/'evidence/validation_report.json').write_text(encoded)
    print(json.dumps(dict(source_snapshots=len(manifest), metric_values_checked=metric_values_checked,
                         figures=len(figures), pdf_pages=pdf_pages, checks='passed')))

if __name__ == '__main__':
    main()
