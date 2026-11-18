"""Native finite-element comparisons on common physical quadrature points.

All errors are complex scattered-field L2 relative to the named reference.
Geometry is affine, and each field is evaluated in its own native Lagrange
basis. Triangles and field batches are processed in bounded chunks.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import roots_jacobi, roots_legendre
from matplotlib.tri import Triangulation
from gmsfem.metrics import BARYCENTRIC, WEIGHTS


def quadrature(degree=5):
    if degree==5: return BARYCENTRIC.copy(), WEIGHTS.copy()
    if degree!=7: raise ValueError('Quadrature degree must be 5 or 7')
    # Duffy transform: integral (1-r) dr ds. Four nodes integrate degree 7.
    x,wx=roots_jacobi(4,1,0); y,wy=roots_legendre(4)
    r=(x+1)/2; s=(y+1)/2
    bary=np.array([[1-a-(1-a)*b,a,(1-a)*b] for a in r for b in s])
    weights=np.array([u*v/4 for u in wx for v in wy])
    return bary,weights


class NativeEvaluator:
    """Evaluate degree 2 or 3 nodal Lagrange fields on affine triangles."""
    def __init__(self,vertices,vertex_triangles,coordinates,triangles,degree):
        self.vertices=np.asarray(vertices); self.vertex_triangles=np.asarray(vertex_triangles)
        self.coordinates=np.asarray(coordinates); self.triangles=np.asarray(triangles); self.degree=degree
        xy=self.vertices[self.vertex_triangles]; edge=xy[:,1:]-xy[:,:1]
        det=edge[:,0,0]*edge[:,1,1]-edge[:,0,1]*edge[:,1,0]
        if np.any(abs(det)<1e-20): raise ValueError('Degenerate cells')
        oriented=self.vertex_triangles.copy(); oriented[det<0]=oriented[det<0][:,[0,2,1]]
        self.finder=Triangulation(self.vertices[:,0],self.vertices[:,1],oriented).get_trifinder()
        self.origin=xy[:,0]; self.edge=edge; self.det=det
        nodes=self.coordinates[self.triangles]; delta=nodes-self.origin[:,None,:]
        l1=(delta[:,:,0]*edge[:,None,1,1]-delta[:,:,1]*edge[:,None,1,0])/det[:,None]
        l2=(edge[:,None,0,0]*delta[:,:,1]-edge[:,None,0,1]*delta[:,:,0])/det[:,None]
        bary=np.stack((1-l1-l2,l1,l2),axis=-1)
        self.indices=np.rint(degree*bary).astype(np.int8)
        if not np.allclose(degree*bary,self.indices,atol=1e-7) or not np.all(self.indices.sum(axis=-1)==degree):
            raise ValueError('Native dofs are not affine Lagrange nodes')

    def prepare(self,points):
        points=np.asarray(points).reshape(-1,2); cells=self.finder(points[:,0],points[:,1]).astype(np.int32)
        if np.any(cells<0): raise ValueError('Quadrature point outside native mesh')
        d=points-self.origin[cells]; e=self.edge[cells]; det=self.det[cells]
        l1=(d[:,0]*e[:,1,1]-d[:,1]*e[:,1,0])/det
        l2=(e[:,0,0]*d[:,1]-e[:,0,1]*d[:,0])/det
        bary=np.stack((1-l1-l2,l1,l2),axis=-1); idx=self.indices[cells]
        basis=np.ones(idx.shape[:2])
        for dim in range(3):
            for m in range(self.degree):
                basis*=np.where(idx[:,:,dim]>m,(self.degree*bary[:,None,dim]-m)/(m+1),1.)
        return self.triangles[cells],basis

    @staticmethod
    def evaluate_prepared(field,prepared):
        cells,basis=prepared
        return np.sum(np.asarray(field)[cells]*basis,axis=-1)


def physical_errors(evaluators,fields,pairs,vertices,triangles,degree=5,chunk_cells=1024):
    """Compare one query across fields using bounded chunks and direct FE evaluation.

    pairs maps metric name to (candidate field key, reference field key).
    Returns relative L2 fractions, with zero-reference norms rejected.
    """
    bary,weights=quadrature(degree); num={k:0. for k in pairs}; den={k:0. for k in pairs}
    for start in range(0,len(triangles),chunk_cells):
        xy=vertices[triangles[start:start+chunk_cells]]; e=xy[:,1:]-xy[:,:1]
        area=abs(e[:,0,0]*e[:,1,1]-e[:,0,1]*e[:,1,0])/2
        points=np.einsum('qj,cjd->cqd',bary,xy).reshape(-1,2); w=(area[:,None]*weights).ravel()
        prepared={}
        values={}
        for name,field in fields.items():
            evaluator=evaluators[name]; key=id(evaluator)
            if key not in prepared: prepared[key]=evaluator.prepare(points)
            values[name]=evaluator.evaluate_prepared(field,prepared[key])
        for key,(candidate,reference) in pairs.items():
            num[key]+=float(np.dot(w,abs(values[candidate]-values[reference])**2))
            den[key]+=float(np.dot(w,abs(values[reference])**2))
    if any(v<=1e-28 for v in den.values()): raise ValueError('Reference field norm is zero')
    return {k:math.sqrt(num[k]/den[k]) for k in pairs}


def summary(values):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all(): raise ValueError('Missing or nonfinite metric')
    return dict(mean=float(a.mean()),p95=float(np.quantile(a,.95)),max=float(a.max()))


class DolfinxEvaluator:
    """Evaluate a native DOLFINx function, retaining its actual CG3 basis."""
    def __init__(self,domain,space,vertices,triangles):
        from dolfinx import fem
        self.function=fem.Function(space); self._last_field=None
        xy=vertices[triangles]; e=xy[:,1:]-xy[:,:1]
        det=e[:,0,0]*e[:,1,1]-e[:,0,1]*e[:,1,0]
        oriented=triangles.copy(); oriented[det<0]=oriented[det<0][:,[0,2,1]]
        self.finder=Triangulation(vertices[:,0],vertices[:,1],oriented).get_trifinder()
    def prepare(self,points):
        points=np.asarray(points).reshape(-1,2); cells=self.finder(points[:,0],points[:,1]).astype(np.int32)
        if np.any(cells<0): raise ValueError('Quadrature point outside CG3 mesh')
        return np.column_stack((points,np.zeros(len(points)))),cells
    def evaluate_prepared(self,field,prepared):
        if field is not self._last_field:
            self.function.x.array[:]=field; self._last_field=field
        return self.function.eval(*prepared).reshape(-1)


def _reference_evaluator(mesh):
    from mpi4py import MPI
    from dolfinx import fem
    from dolfinx.io import gmsh as gmshio
    domain=gmshio.read_from_msh(str(mesh),MPI.COMM_SELF,gdim=2).mesh
    low=fem.functionspace(domain,('Lagrange',1)); high=fem.functionspace(domain,('Lagrange',3))
    n=domain.topology.index_map(2).size_local
    return DolfinxEvaluator(domain,high,low.tabulate_dof_coordinates()[:,:2],
        np.array([low.dofmap.cell_dofs(c) for c in range(n)]))


def build_report(root,protocol):
    """Verify completed artifacts and produce the paired empirical stability report."""
    from .physics import digest
    root=Path(root); raw=json.loads((root/'queries.json').read_text()); queries=raw['queries'] if isinstance(raw,dict) else raw
    ids=[i for i,q in enumerate(queries) if q['split'] in (2,'test')]
    if len(ids)!=200: raise ValueError('Expected exactly 200 test rows')
    train_ids=[i for i,q in enumerate(queries) if q['split'] in (0,'train')]
    validation_ids=[i for i,q in enumerate(queries) if q['split'] in (1,'validation','val')]
    reference=root/'data/reference'; ri=json.loads((reference/'dataset_info.json').read_text())
    if ri['identity']['query_indices']!=ids or ri['completed_count']!=len(ids): raise ValueError('Reference test identity mismatch')
    if ri['identity']['manifest_sha256']!=digest(root/'queries.json') or ri['identity']['mesh_sha256']!=digest(root/'meshes/level1.msh'): raise ValueError('Reference manifest or mesh changed')
    if digest(reference/'fields.npy')!=ri['file_sha256']['fields.npy']: raise ValueError('Reference field hash mismatch')
    ref=np.load(reference/'fields.npy',mmap_mode='r'); evaluators={}; truth={}; predictions={}; native_rows={}; runs={}
    for level in protocol['levels']:
        folder=root/'data'/f'level{level}'; info=json.loads((folder/'dataset_info.json').read_text())
        if info['identity']['manifest_sha256']!=digest(root/'queries.json') or info['completed_count']!=len(queries): raise ValueError('Incomplete native snapshots')
        if any(digest(folder/name)!=info['file_sha256'][name] for name in ('fields.npy','geometry.npz','mass.npz')): raise ValueError('Native dataset hash mismatch')
        cache=root/'cache'/f'level{level}'; preprocessing=json.loads((cache/'preprocessing.json').read_text())
        spec=preprocessing['specification']
        if spec['train_ids']!=train_ids or spec['rank']!=protocol['pod_rank'] or spec['queries_sha256']!=digest(root/'queries.json') or spec['fields_sha256']!=info['file_sha256']['fields.npy'] or spec['geometry_sha256']!=info['file_sha256']['geometry.npz'] or spec['mass_sha256']!=info['file_sha256']['mass.npz']:
            raise ValueError('POD training split or data identity mismatch')
        for name in ('pod.npz','floor_relative_l2.npy'):
            if digest(cache/name)!=preprocessing['files'][name]: raise ValueError('POD artifact hash mismatch')
        with np.load(folder/'geometry.npz') as z:
            evaluator=NativeEvaluator(z['vertices'],z['vertex_triangles'],z['coordinates'],z['triangles'],2)
            if level==2: vertices=z['vertices'].copy(); triangles=z['vertex_triangles'].copy()
        evaluators[f't{level}']=evaluator; truth[level]=np.load(folder/'fields.npy',mmap_mode='r')
        for seed in protocol['seeds']:
            run=root/'runs'/f'level{level}'/f'seed_{seed}'; result=json.loads((run/'result.json').read_text())
            if result['specification']['test_ids']!=ids or result['specification']['training_ids']!=train_ids or result['specification']['validation_ids']!=validation_ids or result['specification']['queries_sha256']!=digest(root/'queries.json'): raise ValueError('Run query identity mismatch')
            if result['predictions_sha256']!=digest(run/'predictions.npy'): raise ValueError('Prediction hash mismatch')
            name=f'p{level}_{seed}'; evaluators[name]=evaluator; predictions[level,seed]=np.load(run/'predictions.npy',mmap_mode='r')
            if predictions[level,seed].shape!=(len(ids),truth[level].shape[1]): raise ValueError('Prediction shape mismatch')
            metrics=json.loads((run/'metrics.json').read_text())
            if [r['query_id'] for r in metrics]!=ids: raise ValueError('Native metric query identity mismatch')
            native_rows[level,seed]=metrics; runs[level,seed]=result
    evaluators['ref']=_reference_evaluator(root/'meshes/level1.msh')
    pairs={f'fem{l}_vs_cg3':(f't{l}','ref') for l in protocol['levels']}
    for l in protocol['levels']:
        for seed in protocol['seeds']: pairs[f'pred{l}_{seed}_vs_cg3']=(f'p{l}_{seed}','ref')
    for l in (0,1):
        for seed in protocol['seeds']: pairs[f'pred{l}_vs_{l+1}_{seed}']=(f'p{l}_{seed}',f'p{l+1}_{seed}')
    out=root/'results'; out.mkdir(exist_ok=True); progress=out/'mesh_stability_progress.json'
    identity=dict(queries_sha256=digest(root/'queries.json'), reference_sha256=ri['file_sha256']['fields.npy'],
                  report_sha256=digest(Path(__file__)),
                  predictions={f'{l}_{s}':runs[l,s]['predictions_sha256'] for l in protocol['levels'] for s in protocol['seeds']},
                  metrics={f'{l}_{s}':digest(root/'runs'/f'level{l}'/f'seed_{s}'/'metrics.json') for l in protocol['levels'] for s in protocol['seeds']},
                  preprocessing={str(l):digest(root/'cache'/f'level{l}'/'preprocessing.json') for l in protocol['levels']},
                  geometry={str(l):digest(root/'data'/f'level{l}'/'geometry.npz') for l in protocol['levels']},
                  native={str(l):digest(root/'data'/f'level{l}'/'fields.npy') for l in protocol['levels']})
    if progress.exists():
        state=json.loads(progress.read_text())
        if state['identity']!=identity: raise ValueError('Report inputs changed; refusing unsafe resume')
        rows=state['rows']; audit=state['audit']
        if [r['query_id'] for r in rows]!=ids[:len(rows)]: raise ValueError('Invalid report progress')
    else: rows=[]; audit=[]
    for j,i in enumerate(ids):
        if j<len(rows): continue
        fields={'ref':ref[j]}; fields.update({f't{l}':truth[l][i] for l in protocol['levels']})
        fields.update({f'p{l}_{s}':predictions[l,s][j] for l in protocol['levels'] for s in protocol['seeds']})
        row=dict(query_id=i,k=float(queries[i]['k']),theta=float(queries[i]['theta']))
        row.update(physical_errors(evaluators,fields,pairs,vertices,triangles)); rows.append(row)
        # Fixed, predeclared spread in manifest order; reports quadrature sensitivity,
        # not a convergence theorem across all possible fields.
        if j in (0,49,99,149,199):
            higher=physical_errors(evaluators,fields,pairs,vertices,triangles,degree=7)
            audit.append(dict(query_id=i,degree5={key:row[key] for key in pairs},degree7=higher,
                              max_absolute_difference=max(abs(row[key]-higher[key]) for key in pairs)))
        tmp=progress.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(identity=identity,rows=rows,audit=audit),indent=2)+'\n'); tmp.replace(progress)
        print(f'Common quadrature {j+1}/{len(ids)}',flush=True)
    summaries={key:summary([r[key] for r in rows]) for key in pairs}
    native={f'level{l}_seed{s}':{key:summary([r[key] for r in native_rows[l,s]]) for key in ('fem_error_pct','pod_floor_pct','neural_in_span_error_pct')} for l in protocol['levels'] for s in protocol['seeds']}
    criteria={}
    for s in protocol['seeds']:
        difference=summaries[f'pred1_vs_2_{s}']['mean']
        change=100*abs(summaries[f'pred1_{s}_vs_cg3']['mean']-summaries[f'pred2_{s}_vs_cg3']['mean'])
        criteria[str(s)]=dict(finest_prediction_mean_relative_l2=difference,test_mean_error_change_percentage_points=change,
                              passed=bool(difference<=protocol['criteria']['finest_prediction_mean_relative_l2_per_seed'] and change<=protocol['criteria']['finest_test_mean_error_change_percentage_points']))
    result=dict(metric='complex scattered field relative L2; denominator named reference; prediction pair denominator finer prediction',
                quadrature='degree 5, finest native CG2 affine mesh; both native functions at same physical points',
                quadrature_audit=dict(degree=7,rows=audit,max_absolute_difference=max(a['max_absolute_difference'] for a in audit),
                                      tolerance_fraction=1e-4,passed=max(a['max_absolute_difference'] for a in audit)<=1e-4),
                native_metrics_percent=native,common_metrics_fraction=summaries,criteria_per_seed=criteria,
                stable=all(x['passed'] for x in criteria.values()) and max(a['max_absolute_difference'] for a in audit)<=1e-4,runs={f'level{l}_seed{s}':runs[l,s] for l in protocol['levels'] for s in protocol['seeds']},
                rows=rows,interpretation='Empirical paired test results, not a uniform bound. Relative errors can be sensitive when the reference scattered-field norm is small, especially near k=0.05.')
    (out/'mesh_stability.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
