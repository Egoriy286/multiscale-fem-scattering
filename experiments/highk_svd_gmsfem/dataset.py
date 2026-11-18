"""Independent high-frequency queries and resumable, memory-mapped FEM labels.

The high-order reference backend is selected only after measured refinement.
Its values are represented on a declared P1 support mesh for the common POD
decoder and consistent-mass comparison. Representation error is audited
separately from the native reference's refinement error.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from neural_operator.data import Dataset
from experiments.reference_sno_gmsfem.dataset import (
    _sha256, _write_json, _save_npy, _save_npz,
)
from experiments.multiscale_sample_study.dataset import query_hash

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRAIN_SIZES = (400,600,800,1000,2000)


def make_queries(n_train=2000, n_validation=200, n_test=200, seed=2026111401,
                 k_min=.05, k_max=20., historical_frequencies=(), train_sizes=TRAIN_SIZES):
    """Stratified independent frequency queries; test includes both endpoints.

    Exclude every historical query to prevent inherited training observations
    appearing verbatim in the new test set. No solutions enter this planner.
    """
    for name, value, minimum in (("n_train", n_train, 2),
                                 ("n_validation", n_validation, 1),
                                 ("n_test", n_test, 3), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not np.isfinite([k_min, k_max]).all() or not 0 < k_min < k_max:
        raise ValueError('Require 0 < k_min < k_max; k=0 is a singular static problem')
    sizes=tuple(int(n) for n in train_sizes)
    if tuple(sorted(set(sizes)))!=sizes or not sizes or sizes[0]<2 or sizes[-1]!=n_train:
        raise ValueError('Increasing train_sizes must end at n_train')
    rng = np.random.default_rng(seed)
    used = set(map(float, historical_frequencies))
    if k_min in used or k_max in used:
        raise ValueError('New test endpoints overlap historical queries')
    used.update((float(k_min), float(k_max)))

    def stratified(count):
        values = k_min+(k_max-k_min)*(np.arange(count)+rng.uniform(.05,.95,count))/count
        for j in range(count):
            while float(values[j]) in used:
                values[j] = np.nextafter(values[j], k_max)
            used.add(float(values[j]))
        return values

    validation = stratified(n_validation)
    test = np.sort(np.r_[k_min, stratified(n_test-2), k_max])
    training=list(k_min+(k_max-k_min)*(np.arange(sizes[0])+rng.uniform(.45,.55,sizes[0]))/sizes[0])
    used.update(map(float,training))
    for size in sizes[1:]:
        while len(training)<size:
            ordered=np.sort(training);gaps=np.diff(ordered)
            priority=np.lexsort((rng.random(len(gaps)),-np.round(gaps,13)))
            chosen=priority[:min(size-len(training),len(gaps))]
            new=ordered[chosen]+gaps[chosen]*rng.uniform(.45,.55,len(chosen))
            for value in np.sort(new):
                while float(value) in used:value=np.nextafter(value,k_max)
                training.append(float(value));used.add(float(value))
    # The row order makes the full training set a zero-copy contiguous slice.
    k = np.r_[validation, test, training]
    return dict(frequencies=k.copy(), k0=k, theta=np.zeros(len(k)),
                frequency_id=np.arange(len(k),dtype=np.int32),
                medium_id=np.zeros(len(k),dtype=np.int32),
                split=np.r_[np.ones(n_validation,dtype=np.int8),
                            np.full(n_test,2,dtype=np.int8),
                            np.zeros(n_train,dtype=np.int8)])


def training_view(data,n_train):
    """Zero-copy heldout-first view; prefixes have independently fitted PODs."""
    if isinstance(n_train,bool) or n_train not in TRAIN_SIZES:
        raise ValueError('Training count is not a declared nested prefix')
    heldout=int(np.count_nonzero(data.split!=0));stop=heldout+n_train
    if stop>len(data.k0) or not np.all(data.split[:heldout]!=0) or not np.all(data.split[heldout:]==0):
        raise ValueError('Incorrect nested heldout-first dataset layout')
    fields=('A','n','u_scat','theta','k0','medium_id','split','A_cells','n_cells','frequency_id')
    values={name:None if getattr(data,name) is None else getattr(data,name)[:stop] for name in fields}
    values.update({name:getattr(data,name) for name in ('coordinates','triangles','cell_tags','circles','directory')})
    view=Dataset(**values,split_strategy='frequency');view._mass=data._mass
    view.original_indices=np.arange(stop);view.n_train=n_train
    return view


def load_dataset(directory):
    """Read the new mesh-specific storage; never assume the old mesh hash."""
    directory=Path(directory).resolve()
    info=json.loads((directory/'dataset_info.json').read_text())
    if not info.get('complete') or info.get('schema_version') != 1:
        raise ValueError('High-k dataset is incomplete or unsupported')
    for name in ('etal.msh','reference_problem.npz','samples.npz'):
        if _sha256(directory/name) != info['sha256'][name]:
            raise ValueError(f'Changed high-k dataset file: {name}')
    with np.load(directory/'reference_problem.npz',allow_pickle=False) as archive:
        values={name:archive[name] for name in ('coordinates','triangles','cell_tags','circles')}
        materials={name:archive[name] for name in ('A','n','A_cells','n_cells')}
    with np.load(directory/'samples.npz',allow_pickle=False) as archive:
        values.update({name:archive[name] for name in ('k0','theta','medium_id','split','frequency_id')})
    count=info['samples']
    values['u_scat']=np.load(directory/'u_scat.npy',mmap_mode='r',allow_pickle=False)
    if values['u_scat'].shape != (count,info['nodes']) or values['u_scat'].dtype != np.complex128:
        raise ValueError('Incorrect complex128 label shape')
    bitmap=np.load(directory/'completed.npy',allow_pickle=False)
    if bitmap.shape != (count,) or bitmap.dtype != bool or not bitmap.all():
        raise ValueError('Uncommitted high-k labels')
    for name,array in materials.items():
        if array.shape[0] != 1:
            raise ValueError('Expected a single fixed material')
        values[name]=np.broadcast_to(array,(count,array.shape[1]))
    return Dataset(**values,split_strategy='frequency',directory=directory)


def verify_dataset(directory, *, labels=True):
    directory=Path(directory)
    info=json.loads((directory/'dataset_info.json').read_text())
    for name,digest in info['sha256'].items():
        if name=='u_scat.npy' and not labels:
            continue
        if _sha256(directory/name) != digest:
            raise ValueError(f'Frozen high-k file changed: {name}')
    data=load_dataset(directory)
    with np.load(directory/'samples.npz',allow_pickle=False) as archive:
        queries={name:archive[name] for name in archive.files}
    if query_hash(queries) != info['query_sha256']:
        raise ValueError('Physical query fingerprint differs')
    if len(np.unique(data.k0)) != len(data.k0) or np.any(data.k0<=0):
        raise ValueError('Duplicated or degenerate physical queries')
    return data


def generate_dataset(root=HERE, *, n_train=2000, n_validation=200, n_test=200,
                     seed=2026111401, reference=None):
    """Solve every new frequency with the independently checked FEM backend.

    ``reference`` supplies a mesh-specific native solver, its P1 support mesh,
    and JSON provenance; this separation also permits higher-order truth.
    """
    root=Path(root).resolve();directory=root/'data'
    if (directory/'dataset_info.json').exists():
        return verify_dataset(directory)
    if reference is None:
        from .highorder import selected_reference
        reference=selected_reference(root)
    # The backend contract is explicit and records both meshes and polynomial
    # degree. The solver returns nodal values in support_solver DOF order.
    solver=reference['solver'];support=reference['support_solver']
    metadata=reference['metadata'];mesh=Path(reference['support_mesh'])
    historical=REPO/'experiments/multiscale_sample_study/data/samples.npz'
    with np.load(historical,allow_pickle=False) as archive:
        old_k=archive['k0']
    queries=make_queries(n_train,n_validation,n_test,seed,historical_frequencies=old_k)
    recipe=dict(schema_version=1,experiment='highk_svd_gmsfem',
                n_train=n_train,train_sizes=list(TRAIN_SIZES),n_validation=n_validation,n_test=n_test,seed=seed,
                k_range=[.05,20.],theta=0.,medium='fixed notebook geometry and materials',
                k_zero='excluded: zero RHS and singular Neumann stiffness; relative scattered error undefined',
                sample_unit='one independently solved physical frequency query',
                query_sha256=query_hash(queries),historical_queries_sha256=_sha256(historical),
                support_mesh_sha256=_sha256(mesh),reference=metadata,
                generator_sha256=_sha256(Path(__file__)))
    directory.mkdir(parents=True,exist_ok=True)
    config=directory/'config.json'
    if config.exists() and json.loads(config.read_text()) != recipe:
        raise ValueError('Frozen generation recipe differs')
    _write_json(config,recipe)
    if not (directory/'etal.msh').exists():
        shutil.copyfile(mesh,directory/'etal.msh')
    if _sha256(directory/'etal.msh') != recipe['support_mesh_sha256']:
        raise ValueError('Support mesh differs')
    if (directory/'samples.npz').exists():
        with np.load(directory/'samples.npz',allow_pickle=False) as archive:
            if query_hash({name:archive[name] for name in archive.files}) != recipe['query_sha256']:
                raise ValueError('Saved physical queries differ')
    else:
        _save_npz(directory/'samples.npz',**queries)
    from experiments.reference_sno_gmsfem.baselines import _materials
    from gmsfem.geometry import inclusion_circles
    alpha,n=_materials(support.cell_tags)
    if not (directory/'reference_problem.npz').exists():
        _save_npz(directory/'reference_problem.npz',coordinates=support.coordinates,
                  triangles=support.triangles,cell_tags=support.cell_tags,circles=inclusion_circles(),
                  A=support.nodal_coefficients(alpha)[None],n=support.nodal_coefficients(n)[None],
                  A_cells=alpha[None],n_cells=n[None])
    count=len(queries['k0']);shape=(count,len(support.coordinates));path=directory/'u_scat.npy'
    fields=(np.load(path,mmap_mode='r+',allow_pickle=False) if path.exists() else
            np.lib.format.open_memmap(path,mode='w+',dtype=np.complex128,shape=shape))
    if fields.shape != shape or fields.dtype != np.complex128:
        raise ValueError('Unexpected existing high-k label array')
    bitmap_path=directory/'completed.npy';time_path=directory/'generation_times.npy'
    residual_path=directory/'linear_residuals.npy'
    stats_path=directory/'native_solve_stats.json'
    completed=np.load(bitmap_path) if bitmap_path.exists() else np.zeros(count,dtype=bool)
    timings=np.load(time_path) if time_path.exists() else np.zeros(count)
    residuals=np.load(residual_path) if residual_path.exists() else np.zeros(count)
    native_stats=json.loads(stats_path.read_text()) if stats_path.exists() else {}
    if completed.shape != (count,) or completed.dtype != bool:
        raise ValueError('Invalid high-k completion bitmap')
    started=perf_counter();before=int(completed.sum())
    # Nearby-frequency preconditioner reuse is permitted; every right-hand
    # side is still solved and checked against its own native FEM equation.
    for i in np.argsort(queries['k0']):
        k=queries['k0'][i]
        if completed[i]:
            if not np.isfinite(fields[i]).all():raise ValueError('Nonfinite committed label')
            continue
        tick=perf_counter()
        result=solver.solve_on_support(float(k),0.)
        fields[i]=result['field'];residuals[i]=result['relative_residual']
        timings[i]=perf_counter()-tick
        if not np.isfinite(fields[i]).all() or not 0<=residuals[i]<1e-8:
            raise FloatingPointError(f'Invalid native FEM label at k={k}: {residuals[i]}')
        native_stats[str(int(i))]=result.get('native_solve',dict(residual=float(residuals[i])))
        fields.flush();_save_npy(time_path,timings);_save_npy(residual_path,residuals)
        _write_json(stats_path,native_stats)
        completed[i]=True;_save_npy(bitmap_path,completed)
        if completed.sum()==before+1 or completed.sum()%10==0 or completed.all():
            print(f'High-k independent FEM labels {completed.sum()}/{count}; k={k:.6g}; '
                  f'{timings[i]:.2f}s; residual={residuals[i]:.2g}',flush=True)
    names=('config.json','etal.msh','reference_problem.npz','samples.npz','u_scat.npy',
           'completed.npy','generation_times.npy','linear_residuals.npy','native_solve_stats.json')
    info=dict(schema_version=1,complete=True,samples=count,nodes=shape[1],
              triangles=len(support.triangles),frequency_groups=count,new_fem_labels=count,
              inherited_fem_labels=0,query_sha256=recipe['query_sha256'],reference=metadata,
              splits=[dict(name=name,samples=int(np.sum(queries['split']==j)))
                      for j,name in enumerate(('train','validation','test'))],
              normalized_native_residual_max=float(residuals.max()),
              solve_seconds=float(timings.sum()),current_session_seconds=perf_counter()-started,
              sha256={name:_sha256(directory/name) for name in names})
    _write_json(directory/'dataset_info.json',info)
    return load_dataset(directory)
