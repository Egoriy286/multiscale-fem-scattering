"""Plot measured reference checks and publish unambiguous complex DOF counts."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .dataset import HERE


def build(root=HERE):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'results'
    selected=json.loads((out/'highorder/selected_reference.json').read_text())
    p1=json.loads((out/'mesh_convergence/summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.7))
    for level in sorted({r['fine_level'] for r in p1['rows']}):
        rows=sorted((r for r in p1['rows'] if r['fine_level']==level),key=lambda r:r['k'])
        a,b=rows[0]['coarse_nodes'],rows[0]['fine_nodes']
        axes[0].plot([r['k'] for r in rows],[100*r['scattered_relative_l2'] for r in rows],
                     'o-',label=f'P1: {a:,} → {b:,} complex DOFs')
    axes[0].set_title('Measured P1 refinement differences')
    checks=[('geometry256_degree_convergence.json','CG2 → CG3, same mesh','relative_scattered_l2'),
            ('geometry256_spatial_convergence.json','CG3 → independently refined CG2','relative_scattered_l2'),
            ('geometry256_representation.json','CG3 → stored P1 observation representation','p1_representation_relative_l2')]
    for filename,label,quantity in checks:
        rows=sorted(json.loads((out/'highorder'/filename).read_text()),key=lambda r:r['k'])
        axes[1].plot([r['k'] for r in rows],[100*r[quantity] for r in rows],'o-',label=label)
    axes[1].set_title('Selected CG3 reference: separate checks')
    for ax in axes:
        ax.axhline(100*selected['separate_relative_l2_tolerance'],color='#475569',linestyle='--',label='Declared 2% check threshold')
        ax.set(xlabel='Physical wavenumber k',ylabel='Relative scattered L2 difference (%)',yscale='log')
        ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=8,loc='best')
    fig.suptitle('Actual mesh/order checks at five anchors; empirical differences, not a uniform error bound',fontsize=11)
    fig.tight_layout();fig.savefig(out/'reference_convergence.png',dpi=180);plt.close(fig)
    rows=[]
    for level,(hb,hi) in enumerate(p1['levels']):
        with np.load(out/f'mesh_convergence/level_{level}_fields.npz',allow_pickle=False) as archive:
            vertices=len(archive['coordinates']);cells=len(archive['triangles'])
        rows.append(dict(stage=f'P1 mesh level {level}',degree=1,vertices=vertices,triangles=cells,
                         dofs_complex=vertices,real_unknowns=2*vertices,
                         requested_h_background=hb,requested_h_inclusion=hi))
    geom=selected['geometry']['native'];dofs=selected['native_dofs']
    rows.append(dict(stage='Selected native CG3 reference',degree=3,vertices=geom['vertices'],
                     triangles=geom['triangles'],dofs_complex=dofs,real_unknowns=2*dofs,
                     requested_h_background=.04,requested_h_inclusion=.005))
    with (out/'reference_dofs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


if __name__=='__main__':build()
