"""Frozen-model inference on re-meshed versions of the same physical problem.

Mapped POD evaluates the original learned P1 span on target points. This
study measures interpolation and FEM discretization effects, not learning
new physics or establishing mesh-native invariance.
"""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import re
import numpy as np
from .data import mass_matrix, update_array_digest
from .fem import HelmholtzSolver, generate_mesh
from .mesh_transfer import P1Transfer


def material_table(cell_tags, alpha, n):
    """Recover exact constant materials for the unique physical circle tags."""
    tags,alpha,n = np.asarray(cell_tags),np.asarray(alpha),np.asarray(n)
    if tags.shape != alpha.shape or tags.shape != n.shape:
        raise ValueError('DG0 material arrays and tags must have matching shapes')
    table = {}
    for tag in np.unique(tags):
        cells = tags == tag
        if np.ptp(alpha[cells]) != 0 or np.ptp(n[cells]) != 0:
            raise ValueError('A physical surface must carry a constant material')
        table[int(tag)] = (float(alpha[cells][0]),float(n[cells][0]))
    return table


def apply_materials(cell_tags, table):
    tags = np.asarray(cell_tags)
    missing = set(map(int,np.unique(tags)))-table.keys()
    if missing:
        raise ValueError(f'Target has unknown physical tags: {sorted(missing)}')
    alpha,n = np.empty(len(tags)),np.empty(len(tags))
    for tag,(a_value,n_value) in table.items():
        alpha[tags == tag],n[tags == tag] = a_value,n_value
    return alpha,n


def select_heldout_cases(frequency_id, split, k0, theta, n_frequencies=3, n_directions=2):
    """Spread test frequencies across their band and retain spaced directions."""
    if n_frequencies < 1 or n_directions < 1:
        raise ValueError('Positive frequency and direction counts required')
    frequency_id,split,k0,theta = map(np.asarray,(frequency_id,split,k0,theta))
    ids = np.unique(frequency_id[split == 2])
    if len(ids) < n_frequencies:
        raise ValueError('Not enough held-out frequency groups')
    ids = sorted(ids,key=lambda fid: float(k0[np.flatnonzero(frequency_id == fid)[0]]))
    selected = np.asarray(ids)[np.rint(np.linspace(0,len(ids)-1,n_frequencies)).astype(int)]
    indices = []
    for fid in selected:
        group = np.flatnonzero((frequency_id == fid)&(split == 2))
        group = group[np.argsort(theta[group])]
        if len(group) < n_directions:
            raise ValueError('Not enough directions in held-out frequency group')
        # Eight directions, two requested: j=0,j=2, as in the study protocol.
        stride = max(1,len(group)//(2*n_directions))
        indices.extend(group[np.arange(n_directions)*stride])
    return np.asarray(indices,dtype=np.int64)


def fem_norm(values, mass):
    values = np.asarray(values)
    flat = values.reshape(-1,values.shape[-1])
    squared = np.real(np.sum(flat.conj()*(mass@flat.T).T,axis=1))
    return np.sqrt(np.maximum(squared,0)).reshape(values.shape[:-1])


def relative_fem_error(reference, prediction, mass):
    numerator,denominator = fem_norm(np.asarray(prediction)-reference,mass),fem_norm(reference,mass)
    result = np.divide(numerator,denominator,out=np.full_like(numerator,np.inf,dtype=float),where=denominator > 0)
    return np.where((denominator == 0)&(numerator == 0),0.,result)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry_metadata(coordinates, triangles):
    return dict(coordinates_sha256=update_array_digest(hashlib.sha256(),coordinates).hexdigest(),
                triangles_sha256=update_array_digest(hashlib.sha256(),triangles).hexdigest(),
                coordinates_dtype=str(coordinates.dtype),triangles_dtype=str(triangles.dtype),
                nodes=len(coordinates),triangles=len(triangles),training_epochs=0,refit=False)


def _mapping_metadata(transfer):
    matrix = transfer.matrix
    counts = np.diff(matrix.indptr)
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    return dict(coverage=float(np.mean(counts > 0)) if len(counts) else 1.,
                max_abs_rowsum_minus_one=float(np.max(np.abs(row_sum-1))) if len(row_sum) else 0.,
                max_nnz_row=int(counts.max()) if len(counts) else 0)


def _solve_batch(solver,alpha,n,k0,theta):
    fields,timings = [],[]
    for k,angle in zip(k0,theta):
        started = perf_counter()
        fields.append(solver.solve(alpha,n,k,angle))
        timings.append(perf_counter()-started)
    return np.asarray(fields),np.asarray(timings)


def run_cross_mesh_study(checkpoints, dataset_dirs, output_dir, n_frequencies=3,
                         n_directions=2, *, mesh_specs=None):
    """Re-solve held-out incidents and evaluate checkpoints without fitting.

    Timing columns describe per-case solves and amortized batch setup/inference.
    Returns rows, grouped summaries, representative fields and mesh metadata.
    """
    from .training import load_predictor
    from .diagnostics import write_csv
    if set(checkpoints) != set(dataset_dirs) or not checkpoints:
        raise ValueError('Checkpoint and dataset band keys must agree')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    specs = mesh_specs if mesh_specs is not None else {
        'coarse':dict(h=.2,n_coarse=4),
        'retriangulated':dict(h=.1,n_coarse=3),
        'fine':dict(h=.05,n_coarse=4)}
    solvers,mesh_info = {},{}
    for name,spec in specs.items():
        if not re.fullmatch(r'[A-Za-z0-9_]+',name) or name == 'permuted':
            raise ValueError('Mesh names must be identifiers other than permuted')
        started = perf_counter()
        mesh_dir = output_dir/'meshes'/name
        mesh_dir.mkdir(parents=True,exist_ok=True)
        path = generate_mesh(mesh_dir/'etal.msh',**spec)
        solver = HelmholtzSolver(path)
        solvers[name] = solver
        mesh_info[name] = dict(spec,**_geometry_metadata(solver.coordinates,solver.triangles),build_seconds=perf_counter()-started)
    rows,fields,checkpoint_hashes = [],{},{}
    source_geometry,selected_queries = {},{}
    for band,checkpoint in checkpoints.items():
        if not re.fullmatch(r'[A-Za-z0-9_]+',band):
            raise ValueError('Band names must be identifiers')
        checkpoint = Path(checkpoint)
        before = _sha256(checkpoint)
        started = perf_counter()
        predictor = load_predictor(checkpoint)
        load_seconds = perf_counter()-started
        directory = Path(dataset_dirs[band])
        with np.load(directory/'geometry.npz',allow_pickle=False) as geometry:
            coordinates,triangles,tags = geometry['dof_coords'][:,:2],geometry['triangles'],geometry['cell_tags']
        if not np.array_equal(coordinates,predictor.coordinates) or not np.array_equal(triangles,predictor.triangles):
            raise ValueError('Source dataset mesh differs from checkpoint mesh')
        with np.load(directory/'fem_dataset.npz',allow_pickle=False) as archive:
            if str(archive['split_strategy'].item()) != 'frequency':
                raise ValueError('Cross-mesh study requires a fixed-medium frequency-split dataset')
            alpha,n = archive['A_cells'][0],archive['n_cells'][0]
            frequency_id,split = archive['frequency_id'],archive['split']
            all_k,all_theta = archive['k0'],archive['theta']
        indices = select_heldout_cases(frequency_id,split,all_k,all_theta,n_frequencies,n_directions)
        k0,theta = all_k[indices],all_theta[indices]
        source_geometry[band] = _geometry_metadata(coordinates,triangles)
        selected_queries[band] = dict(sample_indices=indices.tolist(),frequency_ids=frequency_id[indices].tolist(),k0=k0.tolist(),theta=theta.tolist())
        count = len(indices)
        table = material_table(tags,alpha,n)
        started = perf_counter()
        source = HelmholtzSolver(directory/'etal.msh')
        source_setup_seconds = perf_counter()-started
        if not np.array_equal(source.triangles,triangles) or not np.array_equal(source.coordinates,coordinates) or not np.array_equal(source.cell_tags,tags):
            raise ValueError('Source mesh reader differs from saved DOF ordering')
        source_fem,source_solve = _solve_batch(source,alpha,n,k0,theta)
        source_A,source_n = source.nodal_coefficients(alpha),source.nodal_coefficients(n)
        started = perf_counter()
        source_prediction = predictor.predict(np.broadcast_to(source_A,(count,len(source_A))),np.broadcast_to(source_n,(count,len(source_n))),theta,k0)
        source_inference = perf_counter()-started
        source_mass = mass_matrix(coordinates,triangles)
        source_error = relative_fem_error(source_fem,source_prediction,source_mass)
        permutation = np.random.default_rng(2026).permutation(len(coordinates))
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(len(permutation))
        targets = {name:(solver.coordinates,solver.triangles,solver,None) for name,solver in solvers.items()}
        targets['permuted'] = (coordinates[permutation],inverse[triangles],None,permutation)
        mesh_info.setdefault('permuted',dict(**_geometry_metadata(coordinates[permutation],inverse[triangles]),build_seconds=0.,description='Training mesh with a coordinate permutation and inverse connectivity'))
        for name,(target_xy,target_tri,solver,perm) in targets.items():
            started = perf_counter()
            forward = P1Transfer(coordinates,triangles,target_xy)
            reverse = P1Transfer(target_xy,target_tri,coordinates)
            mass = mass_matrix(target_xy,target_tri)
            mapping_seconds = perf_counter()-started
            mesh_info[name].setdefault('mapping_by_band',{})[band] = dict(forward=_mapping_metadata(forward),reverse=_mapping_metadata(reverse))
            started = perf_counter()
            mapped_predictor = predictor.on_mesh(target_xy,target_tri)
            offline_seconds = perf_counter()-started
            if solver is None:
                target_fem = source_fem[:,perm]
                target_solve = np.zeros(count)
                target_A,target_n = source_A[perm],source_n[perm]
            else:
                target_alpha,target_refractive = apply_materials(solver.cell_tags,table)
                target_fem,target_solve = _solve_batch(solver,target_alpha,target_refractive,k0,theta)
                target_A,target_n = solver.nodal_coefficients(target_alpha),solver.nodal_coefficients(target_refractive)
            started = perf_counter()
            static_inputs = (hasattr(predictor,'bases') and all(predictor.bases[key].rank == 0 for key in ('A','n')))
            if static_inputs:
                prediction = mapped_predictor.predict_static(theta,k0)
            else:
                prediction = mapped_predictor.predict(np.broadcast_to(target_A,(count,len(target_A))),np.broadcast_to(target_n,(count,len(target_n))),theta,k0)
            inference_seconds = perf_counter()-started
            started = perf_counter()
            transferred_fem,transferred_prediction = forward.apply(source_fem),forward.apply(source_prediction)
            roundtrip = reverse.apply(transferred_fem)
            transfer_seconds = perf_counter()-started
            metrics = dict(error_model_vs_target_fem=relative_fem_error(target_fem,prediction,mass),
                           error_source_fem_vs_target_fem=relative_fem_error(target_fem,transferred_fem,mass),
                           error_model_vs_transferred_source_fem=relative_fem_error(transferred_fem,prediction,mass),
                           source_model_vs_source_fem=source_error,
                           roundtrip_source_fem=relative_fem_error(source_fem,roundtrip,source_mass),
                           commutation_error_relative=relative_fem_error(transferred_prediction,prediction,mass),
                           commutation_error_absolute=fem_norm(prediction-transferred_prediction,mass))
            for j,index in enumerate(indices):
                rows.append(dict(band=band,mesh=name,sample_index=int(index),frequency_id=int(frequency_id[index]),k0=float(k0[j]),theta=float(theta[j]),source_nodes=len(coordinates),target_nodes=len(target_xy),**{metric:float(value[j]) for metric,value in metrics.items()},checkpoint_load_seconds=load_seconds/count,source_setup_seconds=source_setup_seconds/count,mapping_seconds=mapping_seconds/count,offline_seconds=offline_seconds/count,transfer_seconds=transfer_seconds/count,source_fem_seconds=float(source_solve[j]),target_fem_seconds=float(target_solve[j]),source_inference_seconds=source_inference/count,inference_seconds=inference_seconds/count))
            prefix = f'{band}__{name}'
            fields.update({prefix+'__coordinates':target_xy,prefix+'__triangles':target_tri,
                           prefix+'__reference':target_fem,prefix+'__prediction':prediction,
                           prefix+'__source_fem_transferred':transferred_fem,
                           prefix+'__k0':k0,prefix+'__theta':theta,
                           prefix+'__sample_indices':indices})
        after = _sha256(checkpoint)
        if before != after:
            raise RuntimeError(f'Checkpoint changed during frozen inference: {checkpoint}')
        checkpoint_hashes[band] = dict(path=str(checkpoint.resolve()),before=before,after=after,unchanged=True)
    summaries = []
    for band in checkpoints:
        for name in (*solvers,'permuted'):
            cases = [row for row in rows if row['band'] == band and row['mesh'] == name]
            summary = dict(band=band,mesh=name,cases=len(cases))
            for key,value in cases[0].items():
                if key.startswith(('error_','commutation_','roundtrip_','source_model_')) or key.endswith('_seconds'):
                    values = [row[key] for row in cases]
                    summary[key+'_mean'],summary[key+'_max'] = float(np.mean(values)),float(np.max(values))
            summaries.append(summary)
    metadata = dict(description=__doc__,cases=len(rows),frequencies_per_band=n_frequencies,directions_per_frequency=n_directions,saved_samples_per_band_mesh=n_frequencies*n_directions,Ns=n_frequencies*n_directions,selected_samples_per_band=n_frequencies*n_directions,checkpoints=checkpoint_hashes,source_geometry=source_geometry,selected_queries=selected_queries,training_epochs=0,refit=False)
    fields['metadata_json'] = np.asarray(json.dumps(metadata))
    write_csv(output_dir/'per_case.csv',rows)
    (output_dir/'summary.json').write_text(json.dumps(dict(metadata=metadata,summaries=summaries),indent=2)+'\n')
    (output_dir/'mesh_info.json').write_text(json.dumps(mesh_info,indent=2)+'\n')
    np.savez_compressed(output_dir/'fields.npz',**fields)
    return rows,summaries,fields,mesh_info
