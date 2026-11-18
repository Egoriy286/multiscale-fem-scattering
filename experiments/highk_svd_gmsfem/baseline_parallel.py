"""Two isolated FEM workers with frequency-disjoint, strictly validated merging."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from .dataset import load_dataset
from .baselines import _csv
from experiments.reference_sno_gmsfem.dataset import _sha256,_write_json


def merge_worker_results(payloads,indices,k,theta):
    """Pure validation/merge; never trust worker completeness or row counts alone."""
    indices=np.asarray(indices,dtype=int);expected=set(map(int,indices))
    if not payloads or not len(indices) or len(expected)!=len(indices):raise ValueError('Invalid merge inputs')
    protocols=[];seen=set();rows=[];fine=[];frequencies=[];frequency_seen=set()
    for payload in payloads:
        metadata=payload['metadata'];protocol=dict(metadata['protocol'])
        subset=protocol.pop('indices')
        if not metadata.get('complete') or len(set(subset))!=len(subset) or not subset:
            raise ValueError('Incomplete or duplicate worker sample manifest')
        if seen.intersection(subset) or not set(subset)<=expected:raise ValueError('Overlapping or unexpected worker samples')
        seen.update(subset);protocols.append(protocol)
        if protocol!=protocols[0]:raise ValueError('Worker physical/source/configuration fingerprints differ')
        modes=protocol['modes'];nc=protocol['n_coarse']
        identity=[]
        for row in payload['rows']:
            i=row['index'];m=row['modes'];identity.append((i,m))
            if (i not in subset or m not in modes or row['basis_type']!=2 or row['n_coarse']!=nc
                    or row['dofs_coarse']!=(nc+1)**2*m or row['fine_degree']!=2
                    or row['method']!=f'GMsFEM CG2 Type2 {nc}x{nc} M={m}'):
                raise ValueError('Worker row configuration/identity differs')
            _physical_row(row,i,k,theta)
            if not np.isfinite(row['galerkin_residual']) or not 0<=row['galerkin_residual']<=1e-7:
                raise ValueError('Invalid worker reduced residual')
        if sorted(identity)!=sorted((i,m) for i in subset for m in modes):raise ValueError('Missing or duplicate method/case rows')
        fine_ids=[]
        for row in payload['fine_reference_rows']:
            i=row['index'];fine_ids.append(i);_physical_row(row,i,k,theta)
            if i not in subset or not np.isfinite(row['linear_residual']) or not 0<=row['linear_residual']<=1e-7:
                raise ValueError('Invalid native fine diagnostic')
        if sorted(fine_ids)!=sorted(subset):raise ValueError('Missing or duplicate native diagnostics')
        observed=set()
        for record in metadata['frequencies']:
            wave=record['k']
            if wave in frequency_seen or wave in observed or wave not in set(map(float,np.asarray(k)[subset])):
                raise ValueError('Overlapping or unexpected worker frequencies')
            observed.add(wave)
            if sorted(s['modes'] for s in record['systems'])!=sorted(modes):raise ValueError('Incomplete frequency systems')
            if not np.isfinite(record['offline_seconds']) or record['offline_seconds']<0:raise ValueError('Invalid offline time')
        if observed!=set(map(float,np.asarray(k)[subset])):raise ValueError('Incomplete frequency diagnostics')
        frequency_seen.update(observed);rows.extend(payload['rows']);fine.extend(payload['fine_reference_rows']);frequencies.extend(metadata['frequencies'])
    if seen!=expected:raise ValueError('Missing expected held-out samples')
    protocol=dict(protocols[0],indices=indices.tolist())
    rows.sort(key=lambda r:(r['modes'],r['index']));fine.sort(key=lambda r:r['index']);frequencies.sort(key=lambda r:r['k'])
    return dict(rows=rows,fine_reference_rows=fine,metadata=dict(protocol=protocol,frequencies=frequencies,
        complete=True,workers=len(payloads),unique_offline_seconds=sum(r['offline_seconds'] for r in frequencies),
        timing_note='Worker frequency sets are disjoint. Setup is counted once per frequency; case setup fields repeat. Parallel wall time is distinct from summed worker work.'))


def _physical_row(row,i,k,theta):
    if (i<0 or i>=len(k) or row['k']!=float(k[i]) or row['theta']!=float(theta[i])
            or not np.isfinite(row['error_fem_l2']) or row['error_fem_l2']<0
            or not np.isfinite(row['error_percent'])
            or not np.isclose(row['error_percent'],100*row['error_fem_l2'],rtol=1e-12,atol=1e-12)):
        raise ValueError('Worker physical query or error metric differs')


def evaluate_parallel(data_dir,output_dir,*,n_coarse=20,modes=(16,32),fine_mesh=None,
                      sparse_ordering='COLAMD',workers=2,worker_threads=2,indices=None):
    """Run independent subprocesses; resume serial frequency shards in each worker."""
    if isinstance(workers,bool) or int(workers)!=workers or not 1<=workers<=2:raise ValueError('Use one or two bounded FEM workers')
    if isinstance(worker_threads,bool) or int(worker_threads)!=worker_threads or worker_threads<1:raise ValueError('Positive worker_threads required')
    if fine_mesh is None:raise ValueError('Declare the independent CG2 GMsFEM fine mesh explicitly')
    if not modes or any(isinstance(m,bool) or int(m)!=m or m<1 for m in modes):raise ValueError('Modes must be positive integers')
    modes=tuple(sorted(set(map(int,modes))))
    data_dir,output_dir=Path(data_dir).resolve(),Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    data=load_dataset(data_dir);selected=data.split_indices()[2] if indices is None else np.asarray(indices)
    if (selected.ndim!=1 or selected.dtype.kind not in 'iu' or len(set(map(int,selected)))!=len(selected) or not len(selected)
            or np.any(selected<0) or np.any(selected>=len(data.k0)) or np.any(data.split[selected]!=2)):
        raise ValueError('Expected distinct held-out indices')
    waves=np.unique(data.k0[selected]);workers=min(int(workers),len(waves))
    partitions=[selected[np.isin(data.k0[selected],waves[j::workers])].tolist() for j in range(workers)]
    k,theta=data.k0.copy(),data.theta.copy();del data
    env=dict(os.environ,OMP_NUM_THREADS=str(worker_threads),OPENBLAS_NUM_THREADS=str(worker_threads),MKL_NUM_THREADS=str(worker_threads))
    processes=[];logs=[];start=time.perf_counter()
    _write_json(output_dir/'parallel_status.json',dict(status='running',workers=workers,worker_threads=worker_threads,partitions=partitions))
    try:
        for worker,subset in enumerate(partitions):
            folder=output_dir/f'worker_{worker}';folder.mkdir(exist_ok=True)
            command=[sys.executable,'-m','experiments.highk_svd_gmsfem.baselines','--data-dir',str(data_dir),
                     '--output-dir',str(folder),'--fine-mesh',str(Path(fine_mesh).resolve()),'--n-coarse',str(n_coarse),
                     '--modes',*map(str,modes),'--sparse-ordering',sparse_ordering,'--indices',*map(str,subset)]
            stream=(folder/'worker.log').open('w');logs.append(stream)
            processes.append(subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env))
            print(f'GMsFEM worker {worker}: {len(subset)} held-out cases',flush=True)
        codes=[process.wait() for process in processes]
    finally:
        for stream in logs:stream.close()
    wall=time.perf_counter()-start
    if any(code!=0 for code in codes):
        _write_json(output_dir/'parallel_status.json',dict(status='failed',return_codes=codes,workers=workers,wall_seconds=wall))
        raise RuntimeError(f'GMsFEM worker failure {codes}; inspect worker logs and resume unchanged protocol')
    paths=[output_dir/f'worker_{j}'/'baselines.json' for j in range(workers)]
    payloads=[json.loads(path.read_text()) for path in paths]
    merged=merge_worker_results(payloads,selected,k,theta)
    merged['metadata'].update(parallel_wall_seconds=wall,worker_threads=worker_threads,
        worker_artifact_sha256={str(path.relative_to(output_dir)):_sha256(path) for path in paths},
        parent_source_sha256=_sha256(Path(__file__)))
    prior=output_dir/'baselines.json'
    if prior.exists():
        old=json.loads(prior.read_text())
        if old.get('metadata',{}).get('protocol')==merged['metadata']['protocol'] and old['metadata'].get('complete'):
            merged['metadata']['parallel_wall_seconds']=old['metadata'].get('parallel_wall_seconds',wall)
            merged['metadata']['resume_check_wall_seconds']=wall
    _write_json(prior,merged);_csv(output_dir/'baselines.csv',merged['rows']);_csv(output_dir/'native_cg2_diagnostic.csv',merged['fine_reference_rows'])
    _write_json(output_dir/'parallel_status.json',dict(status='complete',workers=workers,worker_threads=worker_threads,wall_seconds=wall))
    return merged


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--fine-mesh',required=True);p.add_argument('--n-coarse',type=int,default=20);p.add_argument('--modes',type=int,nargs='+',default=[16,32])
    p.add_argument('--workers',type=int,default=2);p.add_argument('--worker-threads',type=int,default=2);p.add_argument('--sparse-ordering',default='COLAMD')
    p.add_argument('--indices',type=int,nargs='+');a=p.parse_args()
    evaluate_parallel(a.data_dir,a.output_dir,n_coarse=a.n_coarse,modes=a.modes,fine_mesh=a.fine_mesh,
        sparse_ordering=a.sparse_ordering,workers=a.workers,worker_threads=a.worker_threads,indices=a.indices)

if __name__=='__main__':main()
