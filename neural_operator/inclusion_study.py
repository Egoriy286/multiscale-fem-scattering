"""Diagnostic changes to inclusion centers/materials with frozen static models.

Zero-rank material encoders cannot respond to a new physical medium. This
study reports that limitation beside independent FEM re-solves; it does not
establish generalization to new inclusions.
"""
from pathlib import Path
from time import perf_counter
import json
import re
import numpy as np
from .data import mass_matrix
from .fem import HelmholtzSolver
from .mesh_transfer import P1Transfer
from .mesh_study import (material_table,apply_materials,select_heldout_cases,
    relative_fem_error,fem_norm,_sha256,_geometry_metadata,_solve_batch)


def jitter_centers(circles,seed=2026,amplitude=.035):
    """Move each center deterministically while keeping disks inside and disjoint."""
    result = np.array(circles,dtype=float,copy=True)
    rng = np.random.default_rng(seed)
    for index,circle in enumerate(result):
        original = circle[:2].copy()
        others = np.arange(len(result)) != index
        for _ in range(2000):
            proposed = original+rng.uniform(-amplitude,amplitude,2)
            inside = np.all(np.abs(proposed)+circle[2] < 2-1e-6)
            distances = np.linalg.norm(result[others,:2]-proposed,axis=1)
            if inside and np.all(distances >= result[others,2]+circle[2]+1e-6):
                result[index,:2] = proposed
                break
        else:
            raise ValueError('Could not find a safe distinct jitter for every center')
    return result


def default_inclusion_cases(circles,alpha,n,seed=2026):
    circles,alpha,n = np.asarray(circles),np.asarray(alpha),np.asarray(n)
    active = (alpha != 1)|(n != 1)
    changed_A,changed_n = alpha.copy(),n.copy()
    changed_A[active],changed_n[active] = 1.5-alpha[active],4.5-n[active]
    shifted = circles.copy()
    shifted[:,:2] += np.array([.18,-.12])
    def case(xy,a,refractive):
        return dict(circles=np.array(xy,copy=True),A=np.array(a,copy=True),n=np.array(refractive,copy=True))
    return dict(baseline=case(circles,alpha,n),
                centers_shifted_all=case(shifted,alpha,n),
                centers_jittered=case(jitter_centers(circles,seed),alpha,n),
                A_changed=case(circles,changed_A,n),
                n_changed=case(circles,alpha,changed_n),
                shifted_A_n=case(shifted,changed_A,changed_n))


def _validate_case(case,count):
    case = {key:np.asarray(case[key],dtype=float) for key in ('circles','A','n')}
    if case['circles'].shape != (count,3) or case['A'].shape != (count,) or case['n'].shape != (count,):
        raise ValueError('Cases require circles[N,3], A[N], n[N]')
    if not all(np.isfinite(value).all() for value in case.values()) or np.any(case['A'] <= 0) or np.any(case['n'] <= 0):
        raise ValueError('Finite geometry and positive materials required')
    return case


def run_inclusion_study(checkpoints,dataset_dirs,output_dir,n_frequencies=3,n_directions=2,*,cases=None):
    """Return per-query rows, grouped summaries, all fields, and reproducibility metadata.

    Reads only compact DG0 data and query parameters, and never archived u_scat.
    A/n controls reuse the source mesh; equal center arrangements reuse one mesh.
    """
    from .training import load_predictor
    from .inclusion_geometry import generate_inclusion_mesh
    from .diagnostics import write_csv
    if not checkpoints or set(checkpoints) != set(dataset_dirs):
        raise ValueError('Checkpoint and dataset band keys must agree')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    rows,fields,case_records = [],{},{}
    metadata = dict(description=__doc__,training_epochs=0,refit=False,checkpoints={},selected_queries={},source_geometry={},meshes={},Ns=n_frequencies*n_directions)
    geometry_cache = {}
    for band,checkpoint in checkpoints.items():
        if not re.fullmatch(r'[A-Za-z0-9_]+',band):
            raise ValueError('Band names must be identifiers')
        before = _sha256(checkpoint)
        predictor = load_predictor(checkpoint)
        source_dir = Path(dataset_dirs[band])
        with np.load(source_dir/'geometry.npz',allow_pickle=False) as geometry:
            xy,triangles,tags = geometry['dof_coords'][:,:2],geometry['triangles'],geometry['cell_tags']
            circles = geometry['etal_circles'][:,:3]
        if not np.array_equal(xy,predictor.coordinates) or not np.array_equal(triangles,predictor.triangles):
            raise ValueError('Dataset mesh and checkpoint mesh must match exactly')
        with np.load(source_dir/'fem_dataset.npz',allow_pickle=False) as archive:
            if archive['split_strategy'].item() != 'frequency':
                raise ValueError('Inclusion study requires a fixed-medium frequency split')
            source_a,source_n = archive['A_cells'][0],archive['n_cells'][0]
            frequency_id,split,k0,theta = (archive[key] for key in ('frequency_id','split','k0','theta'))
        indices = select_heldout_cases(frequency_id,split,k0,theta,n_frequencies,n_directions)
        k0,theta = k0[indices],theta[indices]
        count = len(indices)
        table = material_table(tags,source_a,source_n)
        inclusion_A = np.array([table[j+2][0] for j in range(len(circles))])
        inclusion_n = np.array([table[j+2][1] for j in range(len(circles))])
        variants = default_inclusion_cases(circles,inclusion_A,inclusion_n) if cases is None else dict(baseline=dict(circles=circles,A=inclusion_A,n=inclusion_n),**{name:value for name,value in cases.items() if name != 'baseline'})
        source = HelmholtzSolver(source_dir/'etal.msh')
        if not np.array_equal(source.coordinates,xy) or not np.array_equal(source.triangles,triangles) or not np.array_equal(source.cell_tags,tags):
            raise ValueError('Source mesh DOF/cell ordering differs from saved geometry')
        source_fem,source_times = _solve_batch(source,source_a,source_n,k0,theta)
        nodal_A,nodal_n = source.nodal_coefficients(source_a),source.nodal_coefficients(source_n)
        source_batch_A,source_batch_n = np.broadcast_to(nodal_A,(count,len(xy))),np.broadcast_to(nodal_n,(count,len(xy)))
        source_features = predictor.encode(source_batch_A,source_batch_n,theta,k0)
        source_prediction = predictor.predict(source_batch_A,source_batch_n,theta,k0)
        ranks = {key:int(predictor.bases[key].rank) for key in ('A','n')} if hasattr(predictor,'bases') else None
        static_rank_zero = ranks is not None and ranks['A'] == ranks['n'] == 0
        metadata['source_geometry'][band] = _geometry_metadata(xy,triangles)
        metadata['selected_queries'][band] = dict(sample_indices=indices.tolist(),frequency_ids=frequency_id[indices].tolist(),k0=k0.tolist(),theta=theta.tolist())
        case_records[band] = {}
        for name,value in variants.items():
            if not re.fullmatch(r'[A-Za-z0-9_]+',name):
                raise ValueError('Case names must be identifiers')
            value = _validate_case(value,len(circles))
            arrangement,a_values,n_values = value['circles'],value['A'],value['n']
            same_centers = np.array_equal(arrangement,circles)
            started = perf_counter()
            if same_centers:
                solver = source
            else:
                key = arrangement.tobytes()
                if key not in geometry_cache:
                    directory = output_dir/'meshes'/name
                    directory.mkdir(parents=True,exist_ok=True)
                    path = generate_inclusion_mesh(directory/'etal.msh',arrangement,h=.1,n_coarse=4)
                    geometry_cache[key] = HelmholtzSolver(path)
                solver = geometry_cache[key]
            target_xy,target_tri = solver.coordinates,solver.triangles
            mapped = predictor.on_mesh(target_xy,target_tri)
            forward = P1Transfer(xy,triangles,target_xy)
            reverse = P1Transfer(target_xy,target_tri,xy)
            mass = mass_matrix(target_xy,target_tri)
            setup_seconds = perf_counter()-started
            target_table = {1:(1.,1.)}
            target_table.update({j+2:(a_values[j],n_values[j]) for j in range(len(arrangement))})
            target_a,target_n = apply_materials(solver.cell_tags,target_table)
            if name == 'baseline':
                if not np.array_equal(target_a,source_a) or not np.array_equal(target_n,source_n):
                    raise ValueError('Baseline must preserve exact original DG0 materials')
                reference,solve_times = source_fem,source_times
            else:
                reference,solve_times = _solve_batch(solver,target_a,target_n,k0,theta)
            A_node,n_node = solver.nodal_coefficients(target_a),solver.nodal_coefficients(target_n)
            A_batch,n_batch = np.broadcast_to(A_node,(count,len(target_xy))),np.broadcast_to(n_node,(count,len(target_xy)))
            started = perf_counter()
            # Deliberately exercise the changed material inputs, even for zero ranks.
            prediction = mapped.predict(A_batch,n_batch,theta,k0)
            inference_seconds = perf_counter()-started
            encoded = predictor.encode(reverse.apply(A_batch),reverse.apply(n_batch),theta,k0)
            feature_change = np.max(np.abs(encoded-source_features),axis=1)
            transferred_source = forward.apply(source_fem)
            transferred_prediction = forward.apply(source_prediction)
            metrics = dict(error_model_vs_target_fem=relative_fem_error(reference,prediction,mass),
                           physical_change_gap=relative_fem_error(reference,transferred_source,mass),
                           model_invariance_relative=relative_fem_error(transferred_prediction,prediction,mass),
                           model_invariance_absolute=fem_norm(prediction-transferred_prediction,mass),
                           input_encoding_change_max=feature_change)
            changes = dict(center_displacement_max=float(np.linalg.norm(arrangement[:,:2]-circles[:,:2],axis=1).max()),radius_change_max=float(np.max(np.abs(arrangement[:,2]-circles[:,2]))),A_change_max=float(np.max(np.abs(a_values-inclusion_A))),n_change_max=float(np.max(np.abs(n_values-inclusion_n))))
            for j,index in enumerate(indices):
                rows.append(dict(band=band,case=name,sample_index=int(index),frequency_id=int(frequency_id[index]),k0=float(k0[j]),theta=float(theta[j]),target_nodes=len(target_xy),static_rank_zero=static_rank_zero,**changes,**{key:float(metric[j]) for key,metric in metrics.items()},setup_seconds=setup_seconds/count,fem_solve_seconds=float(solve_times[j]),inference_seconds=inference_seconds/count))
            prefix = f'{band}__{name}'
            fields.update({prefix+'__coordinates':target_xy,prefix+'__triangles':target_tri,prefix+'__reference':reference,prefix+'__prediction':prediction,prefix+'__source_reference_transferred':transferred_source,prefix+'__A_cells':target_a,prefix+'__n_cells':target_n,prefix+'__A':A_node,prefix+'__n':n_node,prefix+'__sample_indices':indices,prefix+'__k0':k0,prefix+'__theta':theta})
            case_records[band][name] = dict(**{key:array.tolist() for key,array in value.items()},**changes,background=dict(A=1.,n=1.))
            metadata['meshes'][prefix] = _geometry_metadata(target_xy,target_tri)
        after = _sha256(checkpoint)
        if after != before:
            raise RuntimeError('Checkpoint changed during frozen-model inclusion study')
        metadata['checkpoints'][band] = dict(path=str(Path(checkpoint).resolve()),before=before,after=after,unchanged=True,input_ranks=ranks,static_rank_zero=static_rank_zero)
    summaries = []
    for band in checkpoints:
        for name in case_records[band]:
            group = [row for row in rows if row['band'] == band and row['case'] == name]
            summary = dict(band=band,case=name,queries=len(group),static_rank_zero=group[0]['static_rank_zero'])
            for key in ('error_model_vs_target_fem','physical_change_gap','model_invariance_relative','model_invariance_absolute','input_encoding_change_max','fem_solve_seconds','inference_seconds'):
                summary[key+'_mean'] = float(np.mean([row[key] for row in group]))
                summary[key+'_max'] = float(np.max([row[key] for row in group]))
            summaries.append(summary)
    metadata['cases'] = case_records
    fields['metadata_json'] = np.asarray(json.dumps(metadata))
    write_csv(output_dir/'per_case.csv',rows)
    (output_dir/'cases.json').write_text(json.dumps(case_records,indent=2)+'\n')
    (output_dir/'summary.json').write_text(json.dumps(dict(metadata=metadata,summaries=summaries),indent=2)+'\n')
    np.savez_compressed(output_dir/'fields.npz',**fields)
    return rows,summaries,fields,metadata
