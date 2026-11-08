#!/usr/bin/env python3
"""Verify three current static-low runs and plot one common fixed test query.

Run inside dolfinx-complex from the repository root. No training or FEM solve.
The first saved test index is selected before examining any model errors.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parents[1]
sys.path.insert(0, str(REPO))
from neural_operator.data import load_dataset
from neural_operator.training import dataset_fingerprint, load_predictor, summarize

RUNS = {
 'sno': ('SVD-SNO v2', 'neural_operator/runs/revision_2026-11-08/static_low/svd'),
 'fno': ('FNO', 'neural_operator/randomized/runs/low_frequency_static_1000x8/fno'),
 'chebyshev': ('Chebyshev-SNO', 'neural_operator/runs/revision_2026-11-08/static_low/chebyshev'),
}

def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            d.update(block)
    return d.hexdigest()

def main():
    dataset = load_dataset(REPO/'neural_operator/data/static_low_frequency_1000x8', mmap=True)
    fingerprint = dataset_fingerprint(dataset)
    test = dataset.split_indices()[2]
    assert len(test) == 1280 and dataset.split_strategy == 'frequency'
    sample = int(test[0])
    reference = np.asarray(dataset.u_scat[sample])
    predictions, records = {}, {}
    for key, (label, relative) in RUNS.items():
        run = REPO/relative
        metrics = json.loads((run/'metrics.json').read_text())
        assert metrics['dataset_fingerprint'] == fingerprint
        with np.load(run/'test_predictions.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['indices'], test)
            errors = dataset.relative_l2(dataset.u_scat[test], saved['prediction'])
            np.testing.assert_allclose(errors, saved['fem_l2'], rtol=2e-6, atol=1e-9)
            for statistic, value in summarize(errors).items():
                np.testing.assert_allclose(value, metrics['test_fem_l2'][statistic], rtol=2e-6, atol=1e-9)
            saved_sample = np.asarray(saved['prediction'][0])
        predictor = load_predictor(run/'model.pt', device='cpu')
        if key == 'sno':
            assert predictor.model_kwargs['architecture_version'] == 2
        np.testing.assert_array_equal(predictor.coordinates, dataset.coordinates)
        np.testing.assert_array_equal(predictor.triangles, dataset.triangles)
        prediction = predictor.predict_dataset(dataset, [sample])[0]
        np.testing.assert_allclose(prediction, saved_sample, rtol=2e-5, atol=2e-7)
        predictions[key] = prediction
        records[key] = dict(label=label, run=relative,
            checked_test_samples=len(test), test_fem_l2=summarize(errors),
            sample_fem_l2=float(dataset.relative_l2(reference, prediction)),
            checkpoint_sample_max_abs_difference=float(np.max(np.abs(prediction-saved_sample))),
            files={name: dict(path=str((run/name).relative_to(REPO)), sha256=sha(run/name))
                   for name in ('model.pt', 'metrics.json', 'test_predictions.npz')})
    images = PAPER/'images'
    images.mkdir(exist_ok=True)
    triangulation = mtri.Triangulation(dataset.coordinates[:, 0], dataset.coordinates[:, 1], dataset.triangles)
    transforms = (np.real, np.imag, np.abs)
    field_ranges, error_ranges = [], []
    for transform in transforms:
        values = [transform(reference), *(transform(pred) for pred in predictions.values())]
        field_ranges.append((float(min(v.min() for v in values)), float(max(v.max() for v in values))))
        error_ranges.append((0., float(max(np.abs(transform(pred)-transform(reference)).max()
                                          for pred in predictions.values()))))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42})
    for key, prediction in predictions.items():
        fig, axes = plt.subplots(3, 3, figsize=(10.2, 10.0), layout='constrained')
        for row, (transform, component) in enumerate(zip(transforms, ('Re', 'Im', 'Abs'))):
            truth, pred = transform(reference), transform(prediction)
            for col, values in enumerate((truth, pred, np.abs(pred-truth))):
                low, high = error_ranges[row] if col == 2 else field_ranges[row]
                artist = axes[row, col].tripcolor(triangulation, values, shading='gouraud',
                      cmap='magma' if col == 2 else ('viridis' if row == 2 else 'RdBu_r'),
                      vmin=low, vmax=high)
                axes[row, col].set_aspect('equal')
                axes[row, col].set_title((f'{component}: FEM', f'{component}: {records[key]["label"]}',
                                          f'{component}: absolute error')[col])
                axes[row, col].set(xlim=(-2, 2), ylim=(-2, 2), xlabel='x', ylabel='y')
                fig.colorbar(artist, ax=axes[row, col], shrink=.82, pad=.025)
        title = f'{records[key]["label"]}; sample {sample}; k={dataset.k0[sample]:.6f}; theta={dataset.theta[sample]:.6f}'
        fig.suptitle(title+f'\nComplex FEM L2 error: {100*records[key]["sample_fem_l2"]:.4f}%')
        for extension in ('pdf', 'png'):
            fig.savefig(images/f'static_low_{key}_components.{extension}', dpi=200)
        plt.close(fig)
    provenance = dict(scope='One fixed medium, k in [1,2], three current completed runs',
        dataset='neural_operator/data/static_low_frequency_1000x8', dataset_fingerprint=fingerprint,
        selection='First saved test position, chosen before inspecting model errors',
        sample=dict(index=sample, position_in_test=0, k=float(dataset.k0[sample]),
                    theta=float(dataset.theta[sample])),
        models=records, field_ranges=field_ranges, error_ranges=error_ranges,
        training_epochs_performed=0, fem_solves_performed=0,
        generator_sha256=sha(Path(__file__)))
    (images/'static_low_predictions_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(sample=provenance['sample'],
                         errors={key: value['sample_fem_l2'] for key, value in records.items()}), indent=2))

if __name__ == '__main__':
    main()
