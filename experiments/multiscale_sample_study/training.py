"""One frozen training and replay protocol for both operator families."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.gms_sno_100.model import SpectralSurrogate
from experiments.reference_sno_gmsfem.dataset import _sha256, _write_json
from .dataset import training_view
from .fno import FNOAdapter, FEMGridLoss, MPSStencilLoss, make_model, model_config as fno_config
from .sno import SNOAdapter, model_config as sno_config, _norm2

METHODS = ('sno_gelu', 'mscale_sno', 'fno_gelu', 'mscale_fno')


def run_directory(root, n_train, method, seed):
    if method not in METHODS:
        raise ValueError('Unknown method')
    return Path(root)/'runs'/f'n{int(n_train)}'/method/f'seed_{int(seed)}'


def _save(path, value):
    temporary = Path(str(path)+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def parameter_count(model):
    return sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())


def _new_config(method, prep, protocol):
    if 'sno' in method:
        return sno_config(prep.pod.rank, method, protocol['sno_width'], protocol['depth'])
    width = protocol['fno_multiscale_width'] if method == 'mscale_fno' else protocol['fno_single_width']
    return fno_config(method, width=width, modes=protocol['fourier_modes'], depth=protocol['depth'])


def prepare_caches(root, data, protocol):
    """Fit each training-size-specific decoder/normalizer once, never across sizes."""
    root = Path(root)
    for n_train in protocol['train_sizes']:
        view = training_view(data, n_train)
        train = view.split_indices()[0]
        for family in ('sno', 'fno'):
            directory = root/'cache'/f'n{n_train}'/family
            spec = dict(n_train=n_train, family=family, dataset_sha256=protocol['dataset_sha256'],
                        source_sha256=protocol['source_sha256'],
                        pod_rank=protocol['pod_rank'], resolution=protocol['resolution'])
            marker = directory/'cache.json'
            if marker.exists():
                old = json.loads(marker.read_text())
                if old['specification'] != spec:
                    raise ValueError('Frozen preprocessing specification differs')
                for name, digest in old['files'].items():
                    if _sha256(directory/name) != digest:
                        raise ValueError('Cached preprocessing fingerprint differs')
                continue
            directory.mkdir(parents=True, exist_ok=True)
            started = perf_counter()
            print(f'Preparing {family} train-only representation, n={n_train}', flush=True)
            if family == 'sno':
                adapter = SNOAdapter(view, train, rank=protocol['pod_rank'],
                                     width=protocol['sno_width'], depth=protocol['depth'])
            else:
                adapter = FNOAdapter(view, train, variant='mscale_fno',
                    config=fno_config('mscale_fno', width=protocol['fno_multiscale_width'],
                                      modes=protocol['fourier_modes'], depth=protocol['depth']),
                    resolution=protocol['resolution'])
            adapter.save_preprocessing(directory)
            floor = adapter.prep['floor_relative_l2']
            diagnostics = dict(preprocessing_seconds=perf_counter()-started,
                rank=adapter.pod.rank if family == 'sno' else None,
                projection_floor_mean_pct={name:float(100*floor[ids].mean())
                    for name, ids in zip(('train', 'validation', 'test'), view.split_indices())},
                projection_floor_max_pct={name:float(100*floor[ids].max())
                    for name, ids in zip(('train', 'validation', 'test'), view.split_indices())})
            files = {p.name:_sha256(p) for p in directory.iterdir() if p.is_file() and p.name != 'cache.json'}
            _write_json(marker, dict(specification=spec, files=files, diagnostics=diagnostics))
            del adapter
            gc.collect()


def load_adapter(root, n_train, method, protocol, *, seed=None):
    family = 'sno' if 'sno' in method else 'fno'
    device = protocol.get('sno_device',protocol['device']) if family=='sno' else protocol['device']
    directory = Path(root)/'cache'/f'n{n_train}'/family
    adapter = (SNOAdapter if family == 'sno' else FNOAdapter).load(directory, device)
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'mps':
            torch.mps.manual_seed(seed)
    adapter.config = _new_config(method, adapter, protocol)
    adapter.model = (SpectralSurrogate(**adapter.config) if family == 'sno'
                     else make_model(adapter.config)).to(device)
    if family == 'fno':
        adapter.validation_loss = FEMGridLoss(adapter.prep, 'cpu')
        adapter.loss = (MPSStencilLoss(adapter.prep, device) if device == 'mps'
                        else FEMGridLoss(adapter.prep, device))
    return adapter


def _model_state(model):
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}


@torch.no_grad()
def validation_values(adapter, ids, batch_size):
    adapter.model.eval()
    return np.concatenate([adapter.validation_loss(adapter.model(adapter.inputs[ids[start:start+batch_size]]),
                                                   ids[start:start+batch_size]).cpu().numpy()
                           for start in range(0, len(ids), batch_size)])


def _total_norms(data, indices):
    """Analytic incident wave in reference quadrature, as in the common scorer."""
    from gmsfem.metrics import BARYCENTRIC, WEIGHTS
    vertices = data.coordinates[data.triangles]
    edges = vertices[:, 1:]-vertices[:, :1]
    areas = abs(edges[:, 0, 0]*edges[:, 1, 1]-edges[:, 0, 1]*edges[:, 1, 0])/2
    points = np.einsum('qj,cjd->cqd', BARYCENTRIC, vertices)
    weights = areas[:, None]*WEIGHTS[None, :]
    result = []
    for i in indices:
        direction = np.array([np.cos(data.theta[i]), np.sin(data.theta[i])])
        incident = np.exp(1j*data.k0[i]*(points@direction))
        total = np.asarray(data.u_scat[i])[data.triangles]@BARYCENTRIC.T + incident
        result.append(float(np.sum(abs(total)**2*weights)))
    return np.array(result)


def _score(data, indices, prediction, total_norm2):
    error2 = _norm2(prediction-data.u_scat[indices], data.mass_matrix)
    true2 = _norm2(data.u_scat[indices], data.mass_matrix)
    return 100*np.sqrt(error2/true2), 100*np.sqrt(error2/total_norm2)


def train_one(root, data, protocol, n_train, method, seed, total_norm2):
    root = Path(root)
    directory = run_directory(root, n_train, method, seed)
    directory.mkdir(parents=True, exist_ok=True)
    view = training_view(data, n_train)
    train, val, test = view.split_indices()
    spec = dict(protocol_sha256=_json_hash(protocol), n_train=n_train, method=method, seed=seed)
    if (directory/'result.json').exists():
        if json.loads((directory/'config.json').read_text())['specification'] != spec:
            raise ValueError('Completed run specification differs')
        return json.loads((directory/'result.json').read_text())
    adapter = load_adapter(root, n_train, method, protocol, seed=seed)
    optimizer = torch.optim.Adam(adapter.model.parameters(), lr=protocol['learning_rate'],
                                  weight_decay=protocol['weight_decay'], foreach=False)
    batches = int(np.ceil(n_train/protocol['batch_size']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol['epochs']*batches)
    rng = np.random.default_rng(seed)
    history, best, best_epoch, bad_checks, elapsed, first, steps = [], float('inf'), 0, 0, 0., 1, 0
    best_state = None
    config = dict(specification=spec, model=adapter.config, parameters=parameter_count(adapter.model),
                  family='sno' if 'sno' in method else 'fno',
                  online_dtype='float32' if protocol['device']=='mps' or 'sno' in method else 'float64',
                  validation_dtype='float64', preprocessing='train-only; cached once per N/family')
    if (directory/'config.json').exists() and json.loads((directory/'config.json').read_text()) != config:
        raise ValueError('Incomplete run specification differs')
    _write_json(directory/'config.json', config)
    if (directory/'resume.pt').exists():
        state = torch.load(directory/'resume.pt', weights_only=False, map_location='cpu')
        if state['specification'] != spec:
            raise ValueError('Resume protocol differs')
        adapter.model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        if protocol['device']=='mps':
            torch.mps.set_rng_state(state['mps_rng'])
        history, best, best_epoch, bad_checks = (state[k] for k in ('history','best','best_epoch','bad_checks'))
        elapsed, first, steps = state['elapsed'], state['epoch']+1, state['steps']
        best_state = state['best_state']
        _save(directory/'model.pt', dict(state_dict=best_state, best_epoch=best_epoch, config=config))
    started = perf_counter()
    stop_reason = 'epochs_completed'
    for epoch in range(first, protocol['epochs']+1):
        if bad_checks >= protocol['patience_checks']:
            stop_reason = 'validation_patience'
            break
        adapter.model.train()
        losses = []
        for ids in np.array_split(rng.permutation(train), batches):
            optimizer.zero_grad(set_to_none=True)
            values = adapter.loss(adapter.model(adapter.inputs[ids]), ids)
            values.mean().backward()
            optimizer.step()
            scheduler.step()
            losses.append(values.detach())
            steps += 1
        values = torch.cat(losses).cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite training loss')
        if epoch != 1 and epoch % protocol['validation_interval'] and epoch != protocol['epochs']:
            continue
        validation = validation_values(adapter, val, protocol['batch_size'])
        if not np.isfinite(validation).all():
            raise FloatingPointError('Nonfinite validation loss')
        score = float(validation.mean())
        history.append(dict(epoch=epoch, optimizer_steps=steps,
            train_relative_l2=float(values.mean()), val_relative_l2=score,
            val_p95=float(np.percentile(validation,95)), elapsed_seconds=elapsed+perf_counter()-started,
            learning_rate=optimizer.param_groups[0]['lr']))
        if score < best:
            best, best_epoch, bad_checks = score, epoch, 0
            best_state = _model_state(adapter.model)
            _save(directory/'model.pt', dict(state_dict=best_state, best_epoch=best_epoch, config=config))
        else:
            bad_checks += 1
        state = dict(specification=spec, model=adapter.model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), numpy_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state() if protocol['device']=='mps' else None,
            epoch=epoch, steps=steps, history=history, best=best, best_epoch=best_epoch,
            best_state=best_state, bad_checks=bad_checks, elapsed=history[-1]['elapsed_seconds'])
        _save(directory/'resume.pt', state)
        _write_json(directory/'history.json', history)
        print(f'n={n_train} {method} seed={seed} epoch={epoch}/{protocol["epochs"]} '
              f'val={100*score:.4f}% best={100*best:.4f}% seconds={history[-1]["elapsed_seconds"]:.1f}', flush=True)
    if bad_checks >= protocol['patience_checks'] and history[-1]['epoch'] < protocol['epochs']:
        stop_reason = 'validation_patience'
    adapter.model.load_state_dict(best_state)
    prediction = adapter.predict_fields(test, protocol['batch_size'])
    scattered, total = _score(view, test, prediction, total_norm2)
    with (directory/'predictions.npz').open('wb') as stream:
        np.savez_compressed(stream, indices=view.original_indices[test], k=view.k0[test], theta=view.theta[test],
                            prediction=prediction)
    rows = [dict(n_train=n_train, method=method, seed=seed, index=int(i), k=float(view.k0[i]),
                 theta=float(view.theta[i]), scattered_pct=float(s), total_pct=float(t))
            for i,s,t in zip(test, scattered, total)]
    _write_json(directory/'metrics.json', rows)
    result = dict(n_train=n_train, method=method, seed=seed, parameters=config['parameters'],
        epochs_completed=history[-1]['epoch'], best_epoch=best_epoch, best_validation_relative_l2=best,
        training_seconds=history[-1]['elapsed_seconds'], stop_reason=stop_reason,
        scattered_mean_pct=float(scattered.mean()), scattered_p95_pct=float(np.percentile(scattered,95)),
        total_mean_pct=float(total.mean()), total_p95_pct=float(np.percentile(total,95)),
        checkpoint_sha256=_sha256(directory/'model.pt'), predictions_sha256=_sha256(directory/'predictions.npz'))
    _write_json(directory/'result.json', result)
    del adapter, optimizer, scheduler
    gc.collect()
    if protocol['device']=='mps':
        torch.mps.empty_cache()
    return result


def _publish(root):
    import csv
    root = Path(root)
    runs, rows = [], []
    for path in sorted((root/'runs').glob('n*/*/seed_*/result.json')):
        runs.append(json.loads(path.read_text()))
        rows.extend(json.loads((path.parent/'metrics.json').read_text()))
    _write_json(root/'results/training_summary.json', dict(runs=runs))
    if rows:
        with (root/'results/metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return runs


def train_all(root, data, protocol):
    import fcntl
    root = Path(root)
    (root/'runs').mkdir(parents=True, exist_ok=True)
    with (root/'runs/.training.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another study trainer is running') from None
        torch.set_num_threads(protocol['threads'])
        prepare_caches(root, data, protocol)
        test = data.split_indices()[2]
        norm_file = root/'cache/test_total_norms.npy'
        if not norm_file.exists():
            np.save(norm_file, _total_norms(data,test), allow_pickle=False)
        norms = np.load(norm_file, allow_pickle=False)
        for n_train in protocol['train_sizes']:
            for method in protocol['methods']:
                for seed in protocol['seeds']:
                    train_one(root, data, protocol, n_train, method, seed, norms)
                    _publish(root)
                    from .report import build_report
                    build_report(root, allow_partial=True)
        return _publish(root)


def verify(root, data, protocol):
    """Replay every checkpoint; compare saved fields and independent mass metrics."""
    root = Path(root)
    records = []
    for n_train in protocol['train_sizes']:
        view = training_view(data,n_train)
        test = view.split_indices()[2]
        for method in protocol['methods']:
            for seed in protocol['seeds']:
                directory = run_directory(root,n_train,method,seed)
                result = json.loads((directory/'result.json').read_text())
                adapter = load_adapter(root,n_train,method,protocol)
                checkpoint = torch.load(directory/'model.pt', weights_only=False, map_location='cpu')
                adapter.model.load_state_dict(checkpoint['state_dict'])
                history = json.loads((directory/'history.json').read_text())
                selected = min(history,key=lambda r:r['val_relative_l2'])['epoch']
                if selected != checkpoint['best_epoch'] or selected != result['best_epoch']:
                    raise ValueError('Checkpoint was not selected by validation')
                prediction = adapter.predict_fields(test,protocol['batch_size'])
                with np.load(directory/'predictions.npz',allow_pickle=False) as saved:
                    if not np.array_equal(saved['indices'],view.original_indices[test]):
                        raise ValueError('Unpaired heldout indices')
                    if not np.array_equal(saved['k'],view.k0[test]) or not np.array_equal(saved['theta'],view.theta[test]):
                        raise ValueError('Unpaired physical queries')
                    if not np.allclose(prediction,saved['prediction'],rtol=3e-5,atol=2e-6):
                        raise ValueError('Checkpoint field replay failed')
                scores,_ = _score(view,test,prediction,np.load(root/'cache/test_total_norms.npy'))
                if not np.isclose(scores.mean(),result['scattered_mean_pct'],rtol=1e-6,atol=1e-6):
                    raise ValueError('Independent mass metric disagrees')
                records.append(dict(n_train=n_train,method=method,seed=seed,
                    checkpoint_sha256=_sha256(directory/'model.pt'),
                    predictions_sha256=_sha256(directory/'predictions.npz')))
                del adapter
                gc.collect()
                if protocol['device']=='mps':
                    torch.mps.empty_cache()
    value = dict(passed=True, runs=records, protocol_sha256=_sha256(root/'protocol.json'))
    _write_json(root/'results/verification.json',value)
    return value
