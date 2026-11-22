"""Recompute tables, paired metrics, and figures from the four frozen test records."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from neural_operator.data import mass_matrix

MODELS={'svd':'SVD-SNO','fno':'FNO','chebyshev':'Chebyshev-SNO','fourier':'Fourier-SNO'}
COLORS=('#235678','#bb5436','#277362','#805591')
SIZES=(400,600,800,1000)
def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))
def number(x):
    return f'{float(x):.2f}'.replace('.',',')
def rows(name,data):
    (ROOT/'tables'/f'{name}.tex').write_text('\n'.join(' & '.join(row)+r' \\' for row in data)+'\n')
def norm(x,m):
    return np.sqrt(np.sum(x.conj()*(m@x.T).T,axis=1).real)

def main():
    q=np.load(ROOT/'data/queries.npz')
    g=np.load(ROOT/'data/reference_problem.npz')
    geom=g['coordinates']; tri=g['triangles']
    mass=mass_matrix(geom,tri)
    ids=np.flatnonzero(q['split']==2)
    actual=np.load(ROOT/'data/fields.npy',mmap_mode='r')
    sample=ids[np.flatnonzero(q['k'][ids]==4)[0]]
    truth=actual[sample]
    test_norms=norm(actual[ids],mass)
    k=q['k'][ids]
    for split in (0,1,2):
        mine=set(q['medium'][q['split']==split])
        assert mine.isdisjoint(q['medium'][q['split']!=split])
    triangle_xy=geom[tri]
    edges=triangle_xy[:,1:]-triangle_xy[:,:1]
    area=np.abs(np.linalg.det(edges))/2
    def element_norm(v):
        z=v[tri]
        return np.sqrt(np.dot(area,abs(z.sum(axis=1))**2+(abs(z)**2).sum(axis=1))/12)
    abs_rows=[]
    near_rows=[]
    nq=np.load(ROOT/"data/nearzero.npz")
    groups=np.tile(np.repeat(np.arange(5),2),len(ids)//10)
    assert len(ids)==120 and np.all(k[groups==0]==.05) and np.all(k[groups==4]==4)
    cmp=np.load(ROOT/'data/mesh_check.npz')
    assert np.array_equal(cmp['rows'],ids)
    mesh_err=100*norm(cmp['values']-actual[ids],mass)/norm(actual[ids],mass)
    measures=[]; bysize=[]; byband=[]; examples=[]; eall=[]; size_series=[]
    for key,label in MODELS.items():
        path=ROOT/key
        cases=read(path/'cases.csv'); summary=read(path/'summary.csv')
        means=[]
        for n in SIZES:
            records=[v for v in cases if int(v['n'])==n]
            records.sort(key=lambda v:int(v['row']))
            assert [int(v['row']) for v in records]==ids.tolist()
            e=np.array([float(v['error_pct']) for v in records])
            assert np.isfinite(e).all() and (e>=0).all()
            sm,=[v for v in summary if int(v['n'])==n]
            metric=[e.mean(),np.median(e),np.percentile(e,95)]
            np.testing.assert_allclose(metric,[float(sm[key]) for key in ('mean_pct','median_pct','p95_pct')],rtol=2e-5,atol=2e-5)
            means.append(metric[0])
        size_series.append(means)
        bysize.append([label]+[number(x) for x in means])
        measures.append([label, f"{int(sm['parameters']):,}".replace(',',r'\,'),number(sm['val_pct'])]+[number(x) for x in metric+[e.max()]])
        bands=[e[groups==j].mean() for j in range(5)]
        byband.append([label]+[number(x) for x in bands])
        eall.append(bands)
        ex=np.load(path/'example.npz')
        assert int(ex['row'])==sample
        field=ex['field'].reshape(-1)
        physical=100*norm((field-truth)[None],mass)[0]/norm(truth[None],mass)[0]
        np.testing.assert_allclose(physical,e[np.where(ids==sample)[0][0]],rtol=1e-4,atol=.002)
        analytic=100*element_norm(field-truth)/element_norm(truth)
        np.testing.assert_allclose(analytic,physical,rtol=1e-10)
        abs_rows.append([label, f'{(test_norms*e/100).mean():.5f}'.replace('.',','),
                         number(min(e.reshape(12,10).mean(axis=1))),
                         number(max(e.reshape(12,10).mean(axis=1)))])
        probe=sorted(read(path/'nearzero.csv'),key=lambda v:int(v['row']))
        assert [int(v['row']) for v in probe]==list(range(6))
        for feature in ('k','theta'):
            np.testing.assert_allclose([float(v[feature]) for v in probe],nq[feature])
        probe_error=np.array([float(v['error_pct']) for v in probe])
        assert np.isfinite(probe_error).all() and (probe_error>=0).all()
        near_rows.append([label]+[number(x) for x in probe_error.reshape(-1,2).mean(axis=1)])
        examples.append(field)
    rows('nearzero',near_rows); rows('absolute',abs_rows); rows('size',bysize); rows('main',measures); rows('band',byband)
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig,(a,b)=plt.subplots(1,2,figsize=(7.2,2.75),layout='constrained')
    for name,series,bands,c,marker in zip(MODELS.values(),size_series,eall,COLORS,('o','s','^','D')):
        a.plot(SIZES,series,'-',marker=marker,color=c,label=name,ms=4)
        b.plot(range(5),bands,'-',marker=marker,color=c,ms=4)
    a.set(xlabel='Число обучающих полей',xticks=SIZES,ylabel='Средняя погрешность, %')
    b.set(xlabel='Группа волновых чисел',xticks=range(5),xticklabels=['0,05','I','II','III','4'])
    for ax in (a,b): ax.grid(axis='y',alpha=.25);ax.set_yscale('log');ax.spines[['top','right']].set_visible(False)
    a.legend(fontsize=7,frameon=False)
    fig.savefig(ROOT/'figures/errors.pdf',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,5,figsize=(8.1,3.65),layout='constrained')
    fields=[truth]+examples
    amp=max(abs(f).max() for f in fields)
    err=max(abs(f-truth).max() for f in examples)
    for i,(label,v) in enumerate(zip(['МКЭ',*MODELS.values()],fields)):
        axs[0,i].set_title(label,fontsize=9)
        for j in (0,1):
            ax=axs[j,i]
            if i==0 and j==1:
                ax.axis('off');continue
            im=ax.tripcolor(geom[:,0],geom[:,1],tri,abs(v) if j==0 else abs(v-truth),
                            shading='gouraud',cmap='viridis' if j==0 else 'magma',vmin=0,
                            vmax=amp if j==0 else err,rasterized=True)
            ax.set(aspect='equal',xticks=[-2,0,2],yticks=[-2,0,2]);ax.tick_params(labelsize=6)
            if i==4:fig.colorbar(im,ax=list(axs[j]),shrink=.82,aspect=15)
    fig.savefig(ROOT/'figures/fields.pdf',bbox_inches='tight',dpi=180);plt.close(fig)
    info={'mesh_mean':mesh_err.mean(),'mesh_max':mesh_err.max(),'mesh_at_four':mesh_err[groups==4].mean(),
          'sample':int(sample),'medium':int(q['medium'][sample]),'angle':q['theta'][sample],
          'sample_errors':[100*norm((f-truth)[None],mass)[0]/norm(truth[None],mass)[0] for f in examples],
          'means':dict(zip(MODELS.values(),[x[-1] for x in size_series])),
          'bands':dict(zip(MODELS.values(),eall))}
    (ROOT/'audit.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
