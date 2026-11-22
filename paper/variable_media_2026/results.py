"""Compose paper tables from measured results. Percentages are pointwise FEM errors."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'tables'

def read(name):
    with (ROOT/name).open() as stream:
        return list(csv.DictReader(stream))

def num(value):
    return f'{float(value):.2f}'.replace('.',',')

def integer(value):
    return f'{int(value):,}'.replace(',',r'\,')

def table(name,rows):
    (OUT/f'{name}.tex').write_text('\n'.join(' & '.join(r)+r' \\' for r in rows)+'\n')

def main():
    OUT.mkdir(exist_ok=True)
    summary=read('sno_summary.csv')
    s={(int(x['n']),int(x['rank']),int(x['width'])):x for x in summary if x['model']=='materials'}
    candidates=[s[1000,r,48] for r in (8,16,24)]
    best=min(candidates,key=lambda x:float(x['val_pct']))
    control,=[x for x in summary if x['model']=='wave_only']
    assert (best['width'],best['rank'],best['parameters'])==(control['width'],control['rank'],control['parameters'])
    table('size',[[str(n),str(n//8)]+[num(s[n,r,48]['mean_pct']) for r in (8,16,24)] for n in (400,600,800,1000)])
    configs=[s[1000,8,48],s[1000,16,24],s[1000,16,48],s[1000,24,48],control]
    table('rank',[[('Среда и угол' if x['model']=='materials' else 'Только угол'),x['rank'],x['width'],integer(x['parameters'])]+
                  [num(x[col]) for col in ('projection_pct','val_pct','mean_pct','p95_pct')] for x in configs])
    gms=read('gms/summary.csv')
    table('gms',[[('МКЭ' if x['basis_type']=='0' else 'GMsFEM, тип '+x['basis_type']),
                    x['modes'] if x['basis_type']!='0' else r'\textendash',integer(x['dofs_complex'])]+
                   [num(x[key]) for key in ('mean_percent','median_percent','p95_percent')] for x in gms]+
                 [['SVD-SNO',r'\textendash',r'\textendash']+
                   [num(best[key]) for key in ('mean_pct','median_pct','p95_pct')]])
    sens=read('sensitivity.csv')
    both=[x for x in sens if x['model']=='materials']
    table('change',[[r'$a_1:\ 0{,}45\to0{,}85$' if x['changed']=='alpha1' else r'$n_1:\ 2{,}4\to5{,}6$']+
                   [num(x[key]) for key in ('true_change_pct','predicted_change_pct','change_error_pct')] for x in both])
    det,=[x for x in gms if x['basis_type']=='2' and x['modes']=='16']
    g={int(x['index']):float(x['error_percent']) for x in read('gms/per_case.csv') if x['basis_type']=='2' and x['modes']=='16'}
    sn={int(x['row']):float(x['error_pct']) for x in read('sno_cases.csv') if
        (x['model'],x['n'],x['rank'],x['width'])==('materials','1000',best['rank'],'48')}
    assert len(g)==128 and g.keys()==sn.keys()
    fig,(a,b)=plt.subplots(1,2,figsize=(6.9,2.5),layout='constrained')
    for r,color in zip((8,16,24),('#a55136','#377e73','#365f86')):
        counts=(400,600,800,1000)
        a.plot(counts,[float(s[n,r,48]['mean_pct']) for n in counts],'-o',
               markersize=3,color=color,label=f'$r={r}$')
    a.set(xlabel='Число обучающих полей',ylabel='Средняя погрешность, %',
          xticks=counts,title='а) Обучающая выборка')
    a.legend(frameon=False,fontsize=8)
    with np.load(ROOT/'data/queries.npz') as plan:
        ids=np.array(sorted(g))
        med=plan['medium'][ids]
    gx=np.array([g[i] for i in ids])
    sy=np.array([sn[i] for i in ids])
    b.scatter(gx,sy,s=9,color='#b7c8d2',label='Отдельное поле')
    b.scatter([gx[med==m].mean() for m in np.unique(med)],
              [sy[med==m].mean() for m in np.unique(med)],
              s=16,marker='D',color='#275576',label='Среднее по среде')
    top=1.08*max(gx.max(),sy.max())
    b.plot([0,top],[0,top],color='#666666',linewidth=.8,linestyle='--',zorder=0)
    b.set(xlim=(0,top),ylim=(0,top),xlabel='Погрешность GMsFEM, %',
          ylabel='Погрешность SNO, %',title='б) Одни и те же задачи')
    b.legend(frameon=False,fontsize=7,loc='upper right')
    for ax in (a,b):
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.savefig(ROOT/'figures/paired.pdf',bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    macros={'BestRank':best['rank'],'BestMean':num(best['mean_pct']),'BlindMean':num(control['mean_pct']),
          'BestFloor':num(best['projection_pct']),'GmsMean':num(det['mean_percent']),
          'BestMedian':num(best['median_pct']),'BestTail':num(best['p95_pct']),
          'WidthSmall':num(s[1000,16,24]['mean_pct']),'WidthLarge':num(s[1000,16,48]['mean_pct']),
          'Wins':str(sum(g[j]<sn[j] for j in sn))}
    for x in both:
        name='Alpha' if x['changed']=='alpha1' else 'Index'
        for key,suffix in (('true_change_pct','True'),('predicted_change_pct','Pred'),('change_error_pct','Error')):
            macros[name+suffix]=num(x[key])
    (OUT/'values.tex').write_text(''.join('\\newcommand{\\'+key+'}{'+value+'}\n' for key,value in macros.items()))
if __name__=='__main__':
    main()
