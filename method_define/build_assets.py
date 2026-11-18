from pathlib import Path
import json,csv,hashlib,shutil
from datetime import datetime,timezone,timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent;A=ROOT/'assets';A.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16})
def box(ax,x,y,w,h,t,c='#eaf1f6',fs=17):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',fc=c,ec='#334155',lw=1.4));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arrow(ax,x,y,xx,yy):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=16,lw=1.3,color='#334155'))
def canvas(w,h):
 fig,ax=plt.subplots(figsize=(w,h));ax.set(xlim=(0,w),ylim=(0,h));ax.axis('off');return fig,ax
def save(fig,name):
 fig.savefig(A/(name+'.png'),dpi=210,bbox_inches='tight',facecolor='white');fig.savefig(A/(name+'.svg'),bbox_inches='tight');plt.close(fig)
f,ax=canvas(12,7.5);box(ax,.15,3,2,1.1,'p\n[B, 3]',fs=20)
scales=[1,40,80,100,120,140,180,200]
for j,s in enumerate(scales):
 y=6.6-j*.8;box(ax,3,y,3,.55,f's{j+1}={s}   SNO{j+1}',fs=17);arrow(ax,2.2,3.55,2.9,y+.27);arrow(ax,6.1,y+.27,7.1,3.6)
box(ax,7.2,3.1,1.4,1.0,'Σ γⱼ',fs=22);arrow(ax,8.7,3.6,9.1,3.6);box(ax,9.2,3,2.5,1.2,'â [B,R,2]\nRe / Im',fs=19);arrow(ax,10.4,2.9,10.4,1.85);box(ax,8.5,.6,3.2,1.2,'c = s â\nû = μ + Φ c',fs=19)
ax.text(.3,.35,'J = 8 независимых ветвей; sⱼ и γⱼ обучаются',fontsize=17);save(f,'architecture')
f,ax=canvas(12,5.5)
box(ax,.1,2.1,1.3,.8,'Zℓ');box(ax,2.6,3.4,2.7,.8,'Zℓ Wloc + b',fs=17);box(ax,2.6,1,2.1,.8,'Zℓ Wch',fs=17);box(ax,5.2,1,1.6,.8,'Bℓ (·)',fs=17)
arrow(ax,1.5,2.5,2.5,3.8);arrow(ax,1.5,2.5,2.5,1.4);arrow(ax,4.8,1.4,5.1,1.4)
box(ax,7.3,2.1,1.1,.8,'Σ',fs=22);arrow(ax,5.4,3.8,7.3,2.8);arrow(ax,6.9,1.4,7.3,2.2)
box(ax,9,2.1,2.6,.8,'ρ sin(·) + Zℓ',fs=17);arrow(ax,8.5,2.5,8.9,2.5)
arrow(ax,.7,3,.7,4.9);arrow(ax,.7,4.9,10.2,4.9);arrow(ax,10.2,4.9,10.2,3)
ax.text(5.8,5.05,'residual',ha='center');ax.text(.2,.2,'Bℓ: смешивание мод; W: смешивание каналов; ρ = 1/√D.',fontsize=15);save(f,'coefficient_block')
f,ax=canvas(9,8.5);items=[('Только train-поля U [N,nₕ]',7.2),('X [nₕ,N] = (U − μ)ᵀ',5.9),('G = X* M X  [N,N]',4.6),('G V = V Λ; выбор ранга R',3.3),('Φ = X Vᵣ Λᵣ⁻¹ᐟ²  [nₕ,R]',2),('c = Φ* M (u − μ); û = μ + Φ ĉ',.6)]
for k,(t,y) in enumerate(items):
 box(ax,.6,y,7.8,.85,t,fs=19)
 if k<len(items)-1:arrow(ax,4.5,y-.02,4.5,items[k+1][1]+.95)
save(f,'pod_pipeline')
f,ax=canvas(11,7);box(ax,.1,3,1.6,1,'k, θ',fs=20);box(ax,3,4.8,3.3,1.1,'Старая SNO\nRold = 20',fs=18);box(ax,3,.8,3.3,1.1,'Новая MS-SNO\nR мод',fs=18);arrow(ax,1.8,3.5,3,5.35);arrow(ax,1.8,3.5,3,1.35);box(ax,7.2,4.8,3.3,1.1,'Проекция × g(k)\n1 до 4; 0 от 5',fs=17);arrow(ax,6.4,5.35,7.1,5.35);box(ax,8,2.7,2.5,1.0,'Сумма ĉ\nновый POD Φ',fs=19);arrow(ax,8.9,4.7,9.2,3.8);arrow(ax,6.4,1.35,9.2,2.6);ax.text(.3,.1,'Старые веса дообучаются; новая residual-сеть действует на всём диапазоне.',fontsize=14);save(f,'highk_transfer')
f,ax=canvas(11,8);labels=[('SVD-SNO','POD → c → SNO → поле u'),('Multiscale SVD-SNO','POD → J ветвей → Σ → поле u'),('FNO','Сетка → FFT / локальная ветвь → сетка'),('Chebyshev-SNO','Lobatto → DCT-I → SNO → синтез'),('GMsFEM Type 1 / 2','Локальные задачи → P → Галёркин')]
for j,(n,t) in enumerate(labels):
 y=6.7-j*1.35;box(ax,.1,y,3.2,.85,n,fs=16);box(ax,3.9,y,6.8,.85,t,fs=17);arrow(ax,3.4,y+.42,3.8,y+.42)
save(f,'method_map')
lowpath=REPO/'experiments/multiscale_sample_study/results/summary.csv';low=list(csv.DictReader(lowpath.open()));gmspath=REPO/'experiments/multiscale_sample_study/results/gmsfem_summary.csv';gms=list(csv.DictReader(gmspath.open()))
f,ax=plt.subplots(figsize=(9,5));names={'sno_gelu':'SVD-SNO','mscale_sno':'Multiscale SVD-SNO','fno_gelu':'FNO','mscale_fno':'MscaleFNO'}
for m,n in names.items():
 rows=[r for r in low if r['method']==m];x=[int(r['n_train']) for r in rows];y=[float(r['scattered_mean_pct']) for r in rows];e=[float(r['scattered_seed_sd_pct']) for r in rows];ax.errorbar(x,y,yerr=e,marker='o',capsize=3,label=n)

for row in gms:
 if row['method'].startswith('GMsFEM'):
  ax.axhline(float(row['scattered_mean_pct']),linestyle='--',label=row['method'])
ax.set(xlabel='Train N',ylabel='Mean complex scattered L2 error (%)',yscale='log',title='Общий тест: k=0.1–4; 100 задач; 3 seed');ax.legend(fontsize=11);ax.grid(alpha=.2);save(f,'low_comparison')
f,ax=plt.subplots(figsize=(8,4.8));ax.bar(['SVD-SNO v2','FNO','Chebyshev-SNO'],[.9475,1.6599,7.9364],color=['#2563eb','#d97706','#059669']);ax.set(ylabel='Mean complex FEM L2 error (%)',title='Отдельный static-low: k=1–2; 1280 test; 1 seed');ax.grid(axis='y',alpha=.2);save(f,'static_low_comparison')
for source,name in [(REPO/'experiments/highk_svd_gmsfem/results/partial_sample_size.png','highk_partial.png'),(REPO/'experiments/highk_svd_gmsfem/results/reference_convergence.png','reference_convergence.png'),(REPO/'experiments/multiscale_sample_study/results/convergence_epoch.png','low_convergence.png')]:shutil.copyfile(source,A/name)
partial=json.loads((REPO/'experiments/highk_svd_gmsfem/results/partial_verification.json').read_text())
staticdoc=REPO/'neural_operator/docs/operator_revision.md'
metrics={'snapshot_at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'low':low,'gms_low':gms,'highk_completed':partial['per_n_seed_summary'],'highk_status':'partial 9/15; Ns1000,2000 and GMs incomplete at snapshot','static_low':{'range':[1,2],'train':5360,'validation':1360,'test':1280,'seeds':1,'metrics':[{'method':'SVD-SNO v2','mean_pct':.9475,'parameters':4022},{'method':'FNO','mean_pct':1.6599,'parameters':445010},{'method':'Chebyshev-SNO modes8','mean_pct':7.9364,'parameters':2426}],'chebyshev_validation_floor_mean_pct':7.6143},'source_sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [lowpath,gmspath,staticdoc,REPO/'experiments/highk_svd_gmsfem/results/partial_verification.json']}}
(A/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
print('Assets created; GMs low summary:',gms)
# Print-sized versions keep legends readable on an A4 page.
fig,axes=plt.subplots(2,1,figsize=(8,8));root=REPO/'experiments/highk_svd_gmsfem/results'
p1=json.loads((root/'mesh_convergence/summary.json').read_text())
for lev in sorted({r['fine_level'] for r in p1['rows']}):
 rows=[r for r in p1['rows'] if r['fine_level']==lev];axes[0].plot([r['k'] for r in rows],[100*r['scattered_relative_l2'] for r in rows],'o-',label=f'P1 level {lev-1} → {lev}')
for name,label,key in [('geometry256_degree_convergence.json','CG2 → CG3','relative_scattered_l2'),('geometry256_spatial_convergence.json','Spatial refinement','relative_scattered_l2'),('geometry256_representation.json','CG3 → P1 representation','p1_representation_relative_l2')]:
 rows=json.loads((root/'highorder'/name).read_text());axes[1].plot([r['k'] for r in rows],[100*r[key] for r in rows],'o-',label=label)
for ax in axes:
 ax.axhline(2,color='gray',linestyle='--',label='2% check threshold');ax.set(xlabel='k',ylabel='L2 difference (%)',yscale='log');ax.grid(alpha=.2);ax.legend(fontsize=11)
axes[0].set_title('P1 refinement: not the selected truth',fontsize=13);axes[1].set_title('Selected CG3: separate anchor checks',fontsize=13);fig.tight_layout();save(fig,'reference_convergence')
fig,ax=plt.subplots(figsize=(8,5.5))
for m,label in names.items():
 hist=[json.loads((REPO/f'experiments/multiscale_sample_study/runs/n1000/{m}/seed_{seed}/history.json').read_text()) for seed in (11,29,47)]
 common=sorted(set.intersection(*[set(row['epoch'] for row in h) for h in hist]));means=[]
 for ep in common:means.append(100*np.mean([next(row['val_relative_l2'] for row in h if row['epoch']==ep) for h in hist]))
 ax.plot(common,means,'o-',markersize=3,label=label)
ax.set(xlabel='Epoch',ylabel='Validation complex FEM L2 error (%)',yscale='log',title='N=1000, k=0.1–4; mean across 3 seeds');ax.grid(alpha=.2);ax.legend(fontsize=11);fig.tight_layout();save(fig,'low_convergence')
