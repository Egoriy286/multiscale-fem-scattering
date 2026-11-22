"""Field plots from the new FEM labels and medium manifest."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
BLUE,RED,GREEN='#365f86','#b05033','#267066'

def design(q,g):
    fig,(a,b)=plt.subplots(1,2,figsize=(6.65,2.65),layout='constrained')
    for x,y,r,tag in g['circles']:
        a.add_patch(Circle((x,y),r,facecolor=RED if tag==4 else BLUE,edgecolor='none'))
    a.set(xlim=(-.9,.9),ylim=(-1.,1.),aspect='equal',xlabel='$x_1$',ylabel='$x_2$',
          title='а) Включения в центре области')
    a.legend(handles=[Patch(color=BLUE,label='Группа 1'),Patch(color=RED,label='Группа 2')],
             loc='lower center',fontsize=7,frameon=False,ncol=2,bbox_to_anchor=(.5,-.52))
    for s,color,mark,label in ((0,'#a8b5bd','o','Обучение'),(1,GREEN,'^','Валидация'),(2,RED,'s','Тест')):
        ids=np.flatnonzero(q['split']==s)[::8]
        b.scatter(q['alpha1'][ids],q['n1'][ids],s=12,c=color,marker=mark,linewidth=0,label=label)
    b.set(xlabel='$a_1$',ylabel='$n_1$',xlim=(.38,.92),ylim=(1.85,6.15),title='б) Параметры первой группы')
    b.legend(loc='lower center',fontsize=7,frameon=False,ncol=3,bbox_to_anchor=(.5,-.52))
    fig.savefig(OUT/'design.pdf',bbox_inches='tight',pad_inches=.04)
    plt.close(fig)

def increments(q,g):
    u=np.load(ROOT/'data/fields.npy',mmap_mode='r')
    ids=np.flatnonzero(q['split']==3)[::8]
    assert len(ids)==3 and np.unique(q['theta'][ids]).size==1
    delta=[abs(u[i]-u[ids[0]]) for i in ids[1:]]
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.25),layout='constrained')
    vmax=max(z.max() for z in delta)
    for a,z,title in zip(axs,delta,('а) Изменение $a_1$','б) Изменение $n_1$')):
        xy=g['coordinates']
        cm=a.tripcolor(xy[:,0],xy[:,1],g['triangles'],z,shading='gouraud',
                       cmap='viridis',vmin=0,vmax=vmax,rasterized=True)
        a.set(aspect='equal',xticks=[-2,0,2],yticks=[-2,0,2],xlabel='$x_1$',ylabel='$x_2$',title=title)
    fig.colorbar(cm,ax=axs,shrink=.88,label=r'$|\Delta u_h|$')
    fig.savefig(OUT/'increments.pdf',bbox_inches='tight',pad_inches=.04,dpi=220)
    plt.close(fig)
if __name__=='__main__':
    with np.load(ROOT/'data/queries.npz') as q, np.load(ROOT/'data/reference_problem.npz') as g:
        design(q,g)
        if (ROOT/'data/complete.json').exists():
            increments(q,g)
