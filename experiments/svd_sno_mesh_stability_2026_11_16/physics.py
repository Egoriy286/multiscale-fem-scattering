"""Independent native CG2 mesh-stability snapshots and bounded pilot.

Each level is a separate solve on a graded, material-fitted mesh. All fields
are native DOLFINx CG2 coefficients, with the corresponding consistent mass.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import resource
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import sparse
from experiments.highk_svd_gmsfem.mesh import generate_graded_mesh
from experiments.highk_svd_gmsfem.gms_highorder import P2Backend

ROOT = Path(__file__).resolve().parent
LEVELS = ((.08, .01), (.04, .005), (.02, .0025))

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def peak_rss_mb():
    # Linux reports KiB; macOS reports bytes.
    import sys
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform == 'linux' else 1024**2)

def mesh_path(level, root=ROOT):
    hb, hi = LEVELS[level]
    return generate_graded_mesh(root / 'meshes' / f'level{level}.msh', h_background=hb,
                                h_inclusion=hi, circle_points=256)

def build_level(level, root=ROOT):
    root = Path(root); out = root / 'data' / f'level{level}'; out.mkdir(parents=True, exist_ok=True)
    start = perf_counter(); mesh = mesh_path(level, root); mesh_seconds = perf_counter()-start
    backend = P2Backend(mesh); native = backend.native
    geometry=dict(coordinates=backend.coordinates, triangles=backend.triangles,
                  cell_tags=backend.tags, vertices=native.coordinates, vertex_triangles=native.triangles)
    gp=out/'geometry.npz'; mp=out/'mass.npz'
    if gp.exists() or mp.exists():
        if not gp.exists() or not mp.exists(): raise ValueError('Incomplete saved geometry')
        with np.load(gp,allow_pickle=False) as z:
            for key,value in geometry.items():
                if not np.array_equal(z[key],value): raise ValueError(f'Existing geometry differs: {key}')
        delta=sparse.load_npz(mp)-backend.M0
        if delta.nnz and sparse.linalg.norm(delta)>1e-12*sparse.linalg.norm(backend.M0):
            raise ValueError('Existing mass matrix differs')
    else:
        np.savez(gp,**geometry); sparse.save_npz(mp,backend.M0)
    meta = dict(level=level, h_background=LEVELS[level][0], h_inclusion=LEVELS[level][1],
                circle_points=256, degree=2, mesh_sha256=digest(mesh), mesh_seconds=mesh_seconds,
                setup_seconds=backend.setup_seconds, dofs=len(backend.coordinates),
                cells=len(backend.triangles), assembly_errors=backend.assembly_errors,
                peak_rss_mb=peak_rss_mb())
    (out/'geometry.json').write_text(json.dumps(meta, indent=2)+'\n')
    return backend, out, meta

def pilot_level(level, root=ROOT):
    backend, out, meta = build_level(level, root)
    native = backend.native; fields=[]; stats=[]
    for k in (4., 10., 20.):
        u, stat = native.solve(k, 0.); fields.append(u); stat['k']=k; stat['peak_rss_mb']=peak_rss_mb(); stats.append(stat)
        print(json.dumps(stat), flush=True)
    np.savez(out/'pilot.npz', fields=np.asarray(fields), frequencies=[4.,10.,20.], theta=0.)
    meta['solves']=stats; meta['peak_rss_mb']=peak_rss_mb()
    (out/'pilot.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta

def compare_levels(coarse, fine, root=ROOT, chunk_cells=4000):
    """Compare native CG2 at identical 7-point quadrature on fine cells."""
    from experiments.highk_svd_gmsfem.highorder import HighOrderSolver
    from gmsfem.metrics import BARYCENTRIC, WEIGHTS
    root=Path(root); coarse_dir=root/'data'/f'level{coarse}'; fine_dir=root/'data'/f'level{fine}'
    with np.load(coarse_dir/'pilot.npz') as z: ca=z['fields']; frequencies=z['frequencies']
    with np.load(fine_dir/'pilot.npz') as z:
        fa=z['fields']
        if not np.array_equal(frequencies,z['frequencies']): raise ValueError('Pilot frequencies differ')
    # The reference quadrature uses the fine mesh, independently of both FE evaluations.
    with np.load(fine_dir/'geometry.npz') as z: coords=z['vertices']; triangles=z['vertex_triangles']
    a=HighOrderSolver(mesh_path(coarse,root),2); b=HighOrderSolver(mesh_path(fine,root),2)
    num=np.zeros(len(frequencies)); den=np.zeros_like(num); start=perf_counter()
    for s in range(0,len(triangles),chunk_cells):
        xy=coords[triangles[s:s+chunk_cells]]; e=xy[:,1:]-xy[:,:1]
        area=np.abs(e[:,0,0]*e[:,1,1]-e[:,0,1]*e[:,1,0])/2
        points=np.einsum('qj,cjd->cqd',BARYCENTRIC,xy).reshape(-1,2); w=area[:,None]*WEIGHTS
        for i in range(len(frequencies)):
            av=a.evaluate(ca[i],points).reshape(len(xy),-1); bv=b.evaluate(fa[i],points).reshape(len(xy),-1)
            num[i]+=np.sum(abs(av-bv)**2*w); den[i]+=np.sum(abs(bv)**2*w)
    rows=[dict(k=float(k),relative_scattered_l2=float(np.sqrt(n/d))) for k,n,d in zip(frequencies,num,den)]
    result=dict(coarse=coarse, fine=fine, quadrature='fine affine triangles, 7-point degree-five rule; original native CG2 evaluations',
                rows=rows, seconds=perf_counter()-start, peak_rss_mb=peak_rss_mb())
    (fine_dir/f'pilot_compare_{coarse}_{fine}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def generate(level, manifest, root=ROOT):
    """Generate restartable native coefficients, one row per manifest query."""
    root=Path(root); manifest=Path(manifest); queries=json.loads(manifest.read_text())
    if isinstance(queries,dict): queries=queries['queries']
    if not queries: raise ValueError('Empty query manifest')
    backend,out,meta=build_level(level,root); native=backend.native; native.iterative=True
    shape=(len(queries),meta['dofs']); path=out/'fields.npy'; progress=out/'progress.json'
    repo=ROOT.parents[1]
    sources=[Path(__file__),repo/'experiments/highk_svd_gmsfem/highorder.py',
             repo/'experiments/highk_svd_gmsfem/gms_highorder.py',repo/'experiments/highk_svd_gmsfem/mesh.py',
             repo/'gmsfem/fem.py',repo/'gmsfem/geometry.py',repo/'gmsfem/geometry/scattering.geo']
    # Geometry module locates the CAD asset; include its actual source path.
    from gmsfem.geometry import _GEO_DIR
    sources[-1]=_GEO_DIR/'scattering.geo'
    identity=dict(manifest_sha256=digest(manifest),mesh_sha256=meta['mesh_sha256'],shape=list(shape),dtype='complex128',
                  geometry_sha256=digest(out/'geometry.npz'),mass_sha256=digest(out/'mass.npz'),
                  sources_sha256={str(p.relative_to(repo)):digest(p) for p in sources})
    if progress.exists():
        state=json.loads(progress.read_text())
        if state['identity']!=identity: raise ValueError('Existing snapshots have a different manifest or mesh')
        fields=np.lib.format.open_memmap(path,mode='r+')
        if fields.shape!=shape or fields.dtype!=np.complex128: raise ValueError('Snapshot shape or dtype mismatch')
    else:
        fields=np.lib.format.open_memmap(path,mode='w+',dtype=np.complex128,shape=shape)
        state=dict(identity=identity,completed=[],stats={})
        progress.write_text(json.dumps(state,indent=2)+'\n')
    complete=set(state['completed'])
    # Nearby frequencies maximise safe LU reuse; storage remains manifest ordered.
    for i in sorted(range(len(queries)),key=lambda j:float(queries[j]['k'])):
        if i in complete: continue
        q=queries[i]; field,stat=native.solve(float(q['k']),float(q['theta']))
        fields[i]=field; fields.flush(); state['completed'].append(i); state['stats'][str(i)]=stat
        temp=progress.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(progress)
        print(f'level {level}: {len(state["completed"])}/{len(queries)} query {i}, {stat["seconds"]:.2f}s residual {stat["residual"]:.2e}',flush=True)
    if any(not np.isfinite(v['residual']) or v['residual']>=1e-10 for v in state['stats'].values()):
        raise ValueError('Dataset contains an unacceptable original-equation residual')
    files=['fields.npy','geometry.npz','mass.npz','progress.json']
    info=dict(identity=identity,completed_count=len(state['completed']),
              maximum_original_equation_residual=max(v['residual'] for v in state['stats'].values()),
              file_sha256={name:digest(out/name) for name in files})
    (out/'dataset_info.json').write_text(json.dumps(info,indent=2)+'\n')
    return state

def audit_geometry(root=ROOT):
    """Check identical material-interface segments and areas at all levels.

    Coordinates are rounded to 12 decimal places solely for the comparison;
    no mesh coordinates are modified. Sorted endpoints and material labels
    ensure this verifies connectivity as well as vertex positions.
    """
    root=Path(root); interfaces=[]; rows=[]
    for level in range(3):
        with np.load(root/'data'/f'level{level}'/'geometry.npz') as z:
            xy=z['vertices']; tri=z['vertex_triangles']; tags=z['cell_tags']
        edges=np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1)
        materials=np.tile(tags,3); order=np.lexsort((edges[:,1],edges[:,0]))
        edges=edges[order]; materials=materials[order]
        mask=np.all(edges[1:]==edges[:-1],axis=1)&(materials[1:]!=materials[:-1])
        coords=np.round(xy[edges[:-1][mask]],12)
        reverse=(coords[:,0,0]>coords[:,1,0])|((coords[:,0,0]==coords[:,1,0])&(coords[:,0,1]>coords[:,1,1]))
        coords[reverse]=coords[reverse][:,::-1]
        pairs=np.sort(np.column_stack((materials[:-1][mask],materials[1:][mask])),axis=1)
        segments=np.column_stack((coords.reshape(-1,4),pairs))
        segments=segments[np.lexsort(tuple(segments[:,i] for i in reversed(range(6))))]
        interfaces.append(segments)
        e=xy[tri[:,1:]]-xy[tri[:,:1]]
        area=np.abs(e[:,0,0]*e[:,1,1]-e[:,0,1]*e[:,1,0])/2
        lengths=np.linalg.norm(xy[tri]-np.roll(xy[tri],1,axis=1),axis=2)
        rows.append(dict(level=level,interface_edges=len(segments),material_areas={str(int(t)):float(area[tags==t].sum()) for t in np.unique(tags)}, max_cell_edge_by_material={str(int(t)):float(lengths[tags==t].max()) for t in np.unique(tags)}))
    same=all(np.array_equal(interfaces[0],other) for other in interfaces[1:])
    result=dict(identical_material_interface_segments_at_1e_12=same,rows=rows,
                interpretation='Each material interface has the same polygonal segments and adjacent material labels; independently graded interiors are not nested.')
    (root/'data'/'geometry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if not same: raise ValueError('Material-interface polygons differ between mesh levels')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','compare','generate','audit']);parser.add_argument('--level',type=int,choices=range(3));parser.add_argument('--coarse',type=int);parser.add_argument('--fine',type=int);parser.add_argument('--manifest',type=Path);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.stage=='audit': result=audit_geometry(args.root)
    elif args.stage=='pilot': result=pilot_level(args.level,args.root)
    elif args.stage=='compare': result=compare_levels(args.coarse,args.fine,args.root)
    else: result=generate(args.level,args.manifest,args.root)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
