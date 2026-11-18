"""Material-aware P1 meshes and measured Helmholtz mesh convergence.

The high-index inclusions have local wavenumber sqrt(n/alpha)*k. Meshes
therefore refine the entire inclusion interiors, with a graded transition
outside. Convergence compares original P1 fields at common quadrature points;
no reference-node interpolation is used in the reported errors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import gc
import numpy as np

HERE = Path(__file__).resolve().parent
LEVELS = ((.04,.005),(.02,.0025),(.01,.00125))
FREQUENCIES = (4.,8.,12.,16.,20.)


def generate_graded_mesh(path, *, h_background, h_inclusion, n_coarse=20, transition=.08, circle_points=0):
    """Return material-tagged mesh path; cache by recipe and CAD checksum."""
    from gmsfem.geometry import inclusion_circles, _assign_materials, _GEO_DIR
    import gmsh
    path=Path(path).expanduser().resolve()
    if not 0<h_inclusion<=h_background or transition<=0:
        raise ValueError('Require 0 < inclusion size <= background size and positive transition')
    recipe=dict(h_background=float(h_background),h_inclusion=float(h_inclusion),
                n_coarse=int(n_coarse),transition=float(transition),circle_points=int(circle_points),
                cad_sha256=hashlib.sha256((_GEO_DIR/'scattering.geo').read_bytes()).hexdigest(),
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                sizing='constant inside each exact disk; linear radial grading outside; minimum over disks')
    info=path.with_suffix('.mesh_info.json')
    if path.exists() and info.exists():
        old=json.loads(info.read_text())
        if old.get('recipe')==recipe and old.get('sha256')==hashlib.sha256(path.read_bytes()).hexdigest():
            return path
    path.parent.mkdir(parents=True,exist_ok=True)
    own=not gmsh.isInitialized()
    if own: gmsh.initialize()
    previous=gmsh.model.getCurrent()
    t=perf_counter()
    try:
        gmsh.option.setNumber('General.Verbosity',1)
        gmsh.onelab.setNumber('GMSFEM/n_coarse',[int(n_coarse)])
        gmsh.onelab.setNumber('GMSFEM/h',[float(h_background)])
        gmsh.open(str(_GEO_DIR/'scattering.geo'))
        areas=_assign_materials(gmsh)
        fields=[]
        for x,y,r,_ in inclusion_circles():
            f=gmsh.model.mesh.field.add('MathEval')
            # Max prevents coarsening inside disks; Min limits far-field sizes.
            expression=(f'Min({h_background:.17g},{h_inclusion:.17g}+'
                        f'{(h_background-h_inclusion)/transition:.17g}*'
                        f'Max(0,Sqrt((x-({x:.17g}))^2+(y-({y:.17g}))^2)-{r:.17g}))')
            gmsh.model.mesh.field.setString(f,'F',expression);fields.append(f)
        f=gmsh.model.mesh.field.add('Min')
        gmsh.model.mesh.field.setNumbers(f,'FieldsList',fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
        for name,value in [('MeshSizeMin',min(h_inclusion,2*np.pi*.02/circle_points) if circle_points else h_inclusion),('MeshSizeMax',h_background),
                           ('MeshSizeFromPoints',0),('MeshSizeFromCurvature',circle_points),
                           ('MeshSizeExtendFromBoundary',0)]:
            gmsh.option.setNumber('Mesh.'+name,float(value))
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber('Mesh.MshFileVersion',4.1)
        gmsh.option.setNumber('Mesh.Binary',0)
        gmsh.write(str(path))
        nodes=len(gmsh.model.mesh.getNodes()[0])
        info.write_text(json.dumps(dict(recipe=recipe,nodes=nodes,material_areas=areas,
                         generation_seconds=perf_counter()-t,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
    finally:
        if own: gmsh.finalize()
        else:
            gmsh.model.remove()
            if previous:gmsh.model.setCurrent(previous)
    return path


def solve_mesh(mesh, output, *, frequencies=FREQUENCIES, theta=0.):
    """Save independent direct P1 solves and residuals; return archive path."""
    from neural_operator.fem import HelmholtzSolver
    from experiments.reference_sno_gmsfem.dataset import _contrast_rhs
    from experiments.reference_sno_gmsfem.baselines import _materials
    output=Path(output); mesh=Path(mesh)
    signature=dict(mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
                   frequencies=list(frequencies),theta=float(theta))
    if output.exists():
        with np.load(output,allow_pickle=False) as z:
            if json.loads(str(z['signature']))==signature:return output
    started=perf_counter();solver=HelmholtzSolver(mesh)
    # Existing backend quadrature is unchanged, and exact zero-contrast terms omitted.
    solver.rhs=lambda a,n,k,theta:_contrast_rhs(solver,a,n,k,theta)
    alpha,n=_materials(solver.cell_tags)
    print(f'mesh {mesh.name}: {len(solver.coordinates)} nodes, {len(solver.triangles)} cells',flush=True)
    fields=[];timings=[];residuals=[]
    for k in frequencies:
        t=perf_counter();u=solver.solve(alpha,n,float(k),theta);timings.append(perf_counter()-t)
        problem=solver.build_problem(alpha,n,float(k),theta)
        residuals.append(float(np.linalg.norm(problem.operator@u-problem.rhs)/np.linalg.norm(problem.rhs)))
        fields.append(u)
        print(f'  k={k:g}: {timings[-1]:.2f}s, residual={residuals[-1]:.2g}',flush=True)
        del problem
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(output,coordinates=solver.coordinates,triangles=solver.triangles,
             cell_tags=solver.cell_tags,fields=fields,frequencies=frequencies,theta=theta,
             solve_seconds=timings,residuals=residuals,signature=json.dumps(signature),
             total_seconds=perf_counter()-started)
    del solver;gc.collect()
    return output


def compare_meshes(coarse_archive,fine_archive,*,chunk_cells=15000):
    """Integrate both P1 fields directly on finer-mesh 7-point quadrature.

    Summation is chunked to bound memory. Incident waves remain analytic.
    Intersecting the meshes is unnecessary for convergence diagnostics; the
    common quadrature is degree-five exact within each smooth overlap and is
    substantially finer than the coarser P1 kinks it samples.
    """
    from matplotlib.tri import Triangulation
    from gmsfem.metrics import BARYCENTRIC,WEIGHTS
    with np.load(coarse_archive,allow_pickle=False) as z:c={k:z[k] for k in z.files}
    with np.load(fine_archive,allow_pickle=False) as z:f={k:z[k] for k in z.files}
    if not np.array_equal(c['frequencies'],f['frequencies']) or c['theta']!=f['theta']:
        raise ValueError('Mesh solutions must use the same physical queries')
    num=np.zeros(len(f['frequencies']));den=np.zeros_like(num);totalden=np.zeros_like(num)
    theta=float(f['theta']);direction=np.array([np.cos(theta),np.sin(theta)])
    source_tri=c['triangles'].copy();source_xy=c['coordinates'][source_tri]
    source_e=source_xy[:,1:]-source_xy[:,:1]
    reverse=source_e[:,0,0]*source_e[:,1,1]-source_e[:,0,1]*source_e[:,1,0]<0
    source_tri[reverse]=source_tri[reverse][:,[0,2,1]]
    locator=Triangulation(c['coordinates'][:,0],c['coordinates'][:,1],source_tri).get_trifinder()
    t=perf_counter()
    for start in range(0,len(f['triangles']),chunk_cells):
        tri=f['triangles'][start:start+chunk_cells]; xy=f['coordinates'][tri]
        e=xy[:,1:]-xy[:,:1];areas=np.abs(e[:,0,0]*e[:,1,1]-e[:,0,1]*e[:,1,0])/2
        points=np.einsum('qj,cjd->cqd',BARYCENTRIC,xy)
        flat_points=points.reshape(-1,2)
        located=locator(flat_points[:,0],flat_points[:,1])
        if np.any(located<0):raise ValueError('Quadrature point outside source triangles')
        containing=source_tri[located];vertices=c['coordinates'][containing]
        uv=np.linalg.solve(np.swapaxes(vertices[:,1:]-vertices[:,:1],1,2),(flat_points-vertices[:,0])[...,None])[...,0]
        bary=np.column_stack((1-uv.sum(axis=1),uv))
        cq=np.einsum('fpj,pj->fp',c['fields'][:,containing],bary).reshape(len(num),len(tri),-1)
        fq=np.einsum('fcj,qj->fcq',f['fields'][:,tri],BARYCENTRIC)
        weight=areas[:,None]*WEIGHTS
        num+=np.sum(abs(cq-fq)**2*weight,axis=(1,2));den+=np.sum(abs(fq)**2*weight,axis=(1,2))
        incident=np.exp(1j*f['frequencies'][:,None,None]*(points@direction)[None])
        totalden+=np.sum(abs(fq+incident)**2*weight,axis=(1,2))
        del cq,fq,incident
    return [dict(k=float(k),theta=theta,coarse_nodes=len(c['coordinates']),fine_nodes=len(f['coordinates']),
                 scattered_relative_l2=float(np.sqrt(a/b)),total_relative_l2=float(np.sqrt(a/d)),
                 comparison_seconds=perf_counter()-t)
            for k,a,b,d in zip(f['frequencies'],num,den,totalden)]


def run_convergence(root=HERE, *, levels=LEVELS, frequencies=FREQUENCIES, tolerance=.02):
    """Measure successive mesh convergence; never imply passage after a failure."""
    import csv
    root=Path(root).resolve();out=root/'results/mesh_convergence';out.mkdir(parents=True,exist_ok=True)
    archives=[];meshes=[];rows=[]
    for i,(hb,hi) in enumerate(levels):
        mesh=generate_graded_mesh(root/f'meshes/graded_{i}.msh',h_background=hb,h_inclusion=hi)
        archive=solve_mesh(mesh,out/f'level_{i}_fields.npz',frequencies=frequencies)
        archives.append(archive);meshes.append(str(mesh.relative_to(root)))
        if i:
            pairs=compare_meshes(archives[i-1],archive)
            for row in pairs:row.update(coarse_level=i-1,fine_level=i)
            rows.extend(pairs)
            print('successive convergence:',json.dumps(pairs),flush=True)
            (out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    latest=rows[-len(frequencies):]
    result=dict(levels=[list(v) for v in levels],frequencies=list(frequencies),
                meshes=meshes,selected_mesh=meshes[-1],tolerance=tolerance,
                passed=bool(latest and max(r['scattered_relative_l2'] for r in latest)<=tolerance),
                max_successive_scattered_relative_l2=max(r['scattered_relative_l2'] for r in latest),
                rows=rows,metric='P1 fields evaluated directly at common fine-mesh degree-five quadrature; analytic incident wave',
                limitation='Successive refinement is an empirical error indicator, not a rigorous continuum error bound; fixed first-order absorbing boundary remains part of the physical model.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'convergence.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def selected_mesh(root=HERE, *, require_pass=True):
    """Return finest tested mesh, requiring the measured tolerance by default."""
    root=Path(root).resolve();data=json.loads((root/'results/mesh_convergence/summary.json').read_text())
    if require_pass and not data['passed']:raise ValueError('Mesh convergence tolerance has not passed')
    return root/data['selected_mesh']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=HERE)
    args=parser.parse_args();print(json.dumps(run_convergence(args.root),indent=2))
