"""Build the manuscript's illustrations and numerical tables from the case records."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, Rectangle

HERE = Path(__file__).resolve().parent
ART = HERE / 'figures'
ART.mkdir(exist_ok=True)
GEN = HERE / 'tables'
GEN.mkdir(exist_ok=True)
BLUE, RED, TEAL = '#264a73', '#ae4837', '#277765'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'axes.unicode_minus':True})

def finish(fig,name):
    fig.savefig(ART/f'{name}.pdf',bbox_inches='tight',pad_inches=.06)
    plt.close(fig)

def geometry():
    g = np.load(HERE/'data/reference_problem.npz')
    xy,tr = g['coordinates'],g['triangles']
    fig,(a,b) = plt.subplots(1,2,figsize=(6.4,2.5),layout='constrained')
    for x,y,r,_ in g['circles']:
        a.add_patch(Circle((x,y),r,color=BLUE))
    a.add_patch(Rectangle((-.36,.24),.32,.32,fill=False,edgecolor=RED,lw=1.2))
    a.set(xlim=(-2,2),ylim=(-2,2),xticks=[-2,-1,0,1,2],yticks=[-2,-1,0,1,2],
          title='а) Расчётная область',xlabel='$x_1$',ylabel='$x_2$',aspect='equal')
    a.spines[['right','top']].set_visible(True)
    verts=xy[tr]
    mask=((verts[:,:,0].max(1)>-.36)&(verts[:,:,0].min(1)<-.04)&
          (verts[:,:,1].max(1)>.24)&(verts[:,:,1].min(1)<.56))
    b.add_collection(PolyCollection(verts[mask],facecolors=np.where(
        g['cell_tags'][mask]==1,'#ffffff','#c3d6e6'),edgecolors='#5f7684',linewidths=.32))
    b.set(xlim=(-.36,-.04),ylim=(.24,.56),xticks=[-.35,-.25,-.15,-.05],
          yticks=[.25,.35,.45,.55],title='б) Фрагмент эталонной сетки',
          xlabel='$x_1$',ylabel='$x_2$',aspect='equal')
    finish(fig,'mesh')

def read(path):
    with path.open() as file:
        return list(csv.DictReader(file))

def fmt(value):
    return f'{float(value):.2f}'.replace('.',',')

def intfmt(value):
    return f'{int(value):,}'.replace(',',r'\,')

def table(name,rows):
    (GEN/f'{name}.tex').write_text('\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n')

def report():
    s = read(HERE/'sno_summary.csv')
    g = read(HERE/'gms/summary.csv')
    base = [v for v in s if v['r']=='24' and v['w']=='48']
    table('samples',[[v['n'],str(int(v['n'])//8),fmt(v['projection_pct']),fmt(v['mean_pct']),
                      fmt(v['median_pct']),fmt(v['p95_pct'])] for v in base])
    table('modes',[[v['r'],v['w'],intfmt(v['parameters']),fmt(v['projection_pct']),fmt(v['val_pct']),
                    fmt(v['mean_pct']),fmt(v['p95_pct'])] for v in s if v['n']=='1000'])
    table('gms',[[('МКЭ' if v['basis_type']=='0' else 'GMsFEM, тип '+v['basis_type']),
                  v['modes'] if v['basis_type']!='0' else r'\textendash',intfmt(v['dofs_complex']),
                  fmt(v['mean_percent']),fmt(v['median_percent']),fmt(v['p95_percent'])] for v in g])
    selected = min((v for v in s if v['n']=='1000'), key=lambda row: float(row['val_pct']))
    assert (selected['r'], selected['w']) == ('48','48')
    table('paired_sno', [['SVD-SNO',r'\textendash',r'\textendash',
                         fmt(selected['mean_pct']),fmt(selected['median_pct']),fmt(selected['p95_pct'])]])
    cases = read(HERE/'sno_cases.csv')
    gr = read(HERE/'gms/per_case.csv')
    fig,(a,b) = plt.subplots(1,2,figsize=(7.0,2.85),layout='constrained')
    n=[int(v['n']) for v in base]
    a.plot(n,[float(v['mean_pct']) for v in base],'-o',color=BLUE,label='SNO')
    a.plot(n,[float(v['projection_pct']) for v in base],'--s',color=TEAL,label='Проекция SVD')
    a.set(xlabel='Число обучающих полей',ylabel='Средняя погрешность, %',xticks=n,
          title='а) 24 моды, ширина 48')
    def kcurve(axis,rows,col,style,color,label):
        freqs=np.array([float(v['k']) for v in rows])
        vals=np.array([float(v[col]) for v in rows])
        f=np.unique(freqs)
        axis.plot(f,[vals[freqs==k].mean() for k in f],style,color=color,
                  label=label,linewidth=1.3,markersize=3)
    kcurve(b,[v for v in cases if (v['n'],v['r'],v['w'])==('1000','24','48')],
           'error_pct','-o',BLUE,'SNO')
    kcurve(b,[v for v in gr if (v['basis_type'],v['modes'])==('2','16')],
           'error_percent','-s',RED,'GMsFEM, тип 2')
    kcurve(b,[v for v in gr if v['basis_type']=='0'],'error_percent','--',TEAL,'МКЭ, $h=0.04$')
    b.set(xlabel='Волновое число $k$',ylabel='Средняя погрешность, %',
          title='б) Восемь направлений на частоту')
    for ax in (a,b):
        if ax is b:
            ax.set_yscale('log')
        else:
            ax.set_ylim(0,40)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_: f'{x:g}'))
        ax.grid(axis='y',which='both',color='#dddddd',lw=.45)
        ax.legend(frameon=False,fontsize=8)
    finish(fig,'errors')
    fig,axs=plt.subplots(1,3,figsize=(7.05,2.38),layout='constrained')
    e=np.load(HERE/'runs/example.npz')
    geom=np.load(HERE/'data/reference_problem.npz')
    xy,tr=geom['coordinates'],geom['triangles']
    u,p=e['reference'],e['prediction']
    gm = np.load(HERE/'gms/example.npz')
    assert gm['row'] == e['row'] and gm['k'] == e['k'] and gm['theta'] == e['theta']
    fields = [abs(u),abs(p),abs(gm['field'])]
    peak = max(f.max() for f in fields)
    for i,(axis,field,title) in enumerate(zip(axs,fields,
             [r'а) МКЭ, $|u_h|$',r'б) SNO, $|\widehat u|$',r'в) GMsFEM, $|u_{\rm ms}|$'])):
        cm=axis.tripcolor(xy[:,0],xy[:,1],tr,field,shading='gouraud',
                          cmap='viridis',vmin=0,vmax=peak,
                          rasterized=True)
        axis.set(aspect='equal',xticks=[-2,0,2],yticks=[-2,0,2],
                 xlabel='$x_1$',title=title)
        if i==0: axis.set_ylabel('$x_2$')
        if i==2: fig.colorbar(cm,ax=list(axs),location='bottom',shrink=.66,aspect=55,pad=.03)
    finish(fig,'field')
if __name__ == '__main__':
    geometry()
    if (HERE/'sno_summary.csv').exists() and (HERE/'gms/summary.csv').exists():
        report()
