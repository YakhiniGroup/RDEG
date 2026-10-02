from pathlib import Path
import os
import argparse
from paths import analysis_root
PACKAGE=analysis_root()
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
ROOT=OUT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
source=(PACKAGE/'manuscript/rerun_sections.tex').read_text()
joint=source.split(r'\label{tab:joint_main}',1)[1].split(r'\end{table}',1)[0]
groups=[('Breast','WRS'),('Lung','WRS'),('Breast','Pooled'),('Lung','Pooled')];plot=[]
for cohort,test in groups:
    line=next(l for l in joint.splitlines() if l.startswith(cohort+' & '+test+' &'))
    cells=line.replace('\\\\','').split(' & ')
    for method,idx in [('BH',3),('ERDEG',4),('RDEG',6)]:
        h,n=map(int,cells[idx].split('/'));plot.append(dict(cohort=cohort,test=test,method=method,selected=n,hits=h,precision=h/n))
df=pd.DataFrame(plot);df.to_csv(ROOT/'precision_figure_data.csv',index=False)
fig,axes=plt.subplots(2,1,figsize=(7.2,5.8),layout='constrained')
for j,(method,color) in enumerate(zip(['BH','ERDEG','RDEG'],['#7d8792','#247ba0','#d97735'])):
    rows=[df[df.cohort.eq(c)&df.test.eq(t)&df.method.eq(method)].iloc[0] for c,t in groups];pos=np.arange(4)+(j-1)*.25
    bars=axes[0].bar(pos,[r.precision*100 for r in rows],.24,label=method,color=color);axes[0].bar_label(bars,fmt='%.1f',fontsize=8.5,padding=2)
    bars=axes[1].bar(pos,[r.selected for r in rows],.24,label=method,color=color);axes[1].bar_label(bars,fmt='%d',fontsize=8.5,padding=2)
axes[0].set(ylabel='Reference precision (%)',ylim=(0,87),title='A   Fraction agreeing with the reference')
axes[1].set(ylabel='Selected directional hypotheses',ylim=(0,2000),title='B   Discovery count')
for ax in axes:
    ax.set_xticks(range(4),['Breast\nWRS','Lung\nWRS','Breast\nPooled Student','Lung\nPooled Student']);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.17);ax.legend(frameon=False,ncol=3,loc='upper left')
for ext in ['png','svg']:fig.savefig(OUT/f'precision_and_size.{ext}',dpi=300)
plt.close(fig)

b=pd.read_csv(PACKAGE/'results/rdeg-no-tcga-review/budget_curves/summary.csv');colors={'WRS':'#247ba0','Pooled':'#c56b26'}
fig,axs=plt.subplots(2,3,figsize=(8.8,5.4),sharex=True,layout='constrained')
for col,direction in enumerate(['greater','less','joint']):
    for test,color in colors.items():
        d=b[(b.test==test)&(b.family==direction)].sort_values('k')
        axs[0,col].plot(d.k,d.selected,color=color,marker='o',markersize=4,lw=1.8,label=test)
        axs[1,col].plot(d.k,100*d.reference_precision,color=color,marker='o',markersize=4,lw=1.8)
        axs[1,col].plot(d.k,100*d.matched_reference_precision,color=color,ls='--',marker='x',markersize=5,lw=1.5)
    axs[0,col].set_title(['Separate greater','Separate less','Joint family'][col],fontweight='bold');axs[0,col].set_ylim(bottom=-10)
    axs[1,col].set_xlabel('Label-flip budget k');axs[1,col].set_ylim(0,103)
    for row in range(2):axs[row,col].set_xticks(range(6));axs[row,col].grid(alpha=.2)
axs[0,0].set_ylabel('Selected hypotheses');axs[1,0].set_ylabel('Reference precision (%)')
handles=[Line2D([0],[0],color=c,lw=2,label=t) for t,c in colors.items()]+[Line2D([0],[0],color='#444',marker='o',lw=1.5,label='ERDEG'),Line2D([0],[0],color='#444',ls='--',marker='x',lw=1.5,label='Equal-size original-p')]
fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False,fontsize=9)
for ext in ['png','svg']:fig.savefig(OUT/f'breast_budget.{ext}',dpi=300)
plt.close(fig)

# Refresh the optional project figure; it is not included in either document.
alltables=source.split(r'\label{tab:direction_main}')[1].split(r'\end{table}')[0]
fig,axs=plt.subplots(1,3,figsize=(11,4.2),sharey=True,layout='constrained')
equal=[]
for ax,mode in zip(axs,['greater','less','joint']):
    for j,(method,idx,color) in enumerate([('ERDEG',4,'#247ba0'),('Equal-size original-p',5,'#d97735')]):
        vals=[];labs=[]
        for c,t in groups:
            tab=joint if mode=='joint' else alltables;label=t if mode=='joint' else t+' '+mode
            line=next(l for l in tab.splitlines() if l.startswith(c+' & '+label+' &'));cells=line.replace('\\\\','').split(' & ');h,n=map(int,cells[idx].split('/'))
            vals.append(h/n*100);labs.append(f'{h}/{n}');equal.append(dict(cohort=c,test=t,direction=mode,method=method,hits=h,selected=n))
        bars=ax.bar(np.arange(4)+(j-.5)*.36,vals,width=.35,color=color,label=method)
        ax.bar_label(bars,labels=labs,fontsize=6.4,rotation=90,padding=3)
    ax.set_title(mode.capitalize());ax.set_ylim(0,110);ax.set_xticks(range(4),['Breast\nWRS','Lung\nWRS','Breast\nPooled','Lung\nPooled'],fontsize=8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axs[0].set_ylabel('Reference precision (%)');fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
for ext in ['png','svg']:fig.savefig(OUT/f'matched_size_comparison.{ext}',dpi=300)
pd.DataFrame(equal).to_csv(ROOT/'matched_figure_data.csv',index=False)
print('Figures regenerated from current manuscript tables and budget results.')

b=pd.read_csv(PACKAGE/'results/rdeg-lung-variance-review/lung/budget_summary.csv')
fig,axs=plt.subplots(2,3,figsize=(8.5,5.5),sharex='col',layout='constrained')
for j,direction in enumerate(['greater','less','joint']):
    for test in ['WRS','Pooled']:
        d=b[b.test.eq(test)&b.direction.eq(direction)].sort_values('k')
        axs[0,j].plot(d.k,d.selected,color=colors[test],marker='o',markersize=3,label=test)
        axs[1,j].plot(d.k,d.precision*100,color=colors[test],marker='o',markersize=3)
        axs[1,j].plot(d.k,d.matched_precision*100,color=colors[test],linestyle='--',linewidth=1.4)
    axs[0,j].set_title(['A   Greater','B   Less','C   Joint'][j])
    axs[0,j].set_ylim(bottom=0)
    axs[1,j].set(xlabel='Label-flip budget k',ylim=(35,103),xticks=[0,2,4,6,8])
    axs[0,j].grid(alpha=.18);axs[1,j].grid(alpha=.18)
axs[0,0].set_ylabel('ERDEG selections')
axs[1,0].set_ylabel('Reference precision (%)')
axs[0,0].legend(frameon=False)
axs[1,2].legend(handles=[Line2D([0],[0],color='#555555',label='ERDEG'),
    Line2D([0],[0],color='#555555',linestyle='--',label='Equal-size original-p')],frameon=False,loc='lower right',fontsize=8)
for ext in ['png','svg']:fig.savefig(OUT/f'lung_budget.{ext}',dpi=300)
plt.close(fig)

import json
from analysis_core import WRS
raw=np.load(PACKAGE/'data/breast/gt_10.npz',allow_pickle=False)
d=pd.DataFrame({'ALG5':raw['x'][:,list(raw['genes']).index('ALG5')],'group':raw['group']})
a=WRS(d[['ALG5']].values,d.group.values);lo,hi=a.candidates(1)
pp=np.array([pv[0,0] for idx,pv in a.relabelings(1)])
illustration=dict(cohort='MAINZ',gene='ALG5',samples=len(d),group_LumA=int(d.group.sum()),
                  alternative='greater',p=float(a.p[0,0]),p_L=float(lo[0,0]),p_U=float(hi[0,0]),labelings=len(pp))
(OUT/'ALG5_stability.json').write_text(json.dumps(illustration,indent=2))
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for mask,color,label in [(d.group.values,'#0072B2','LumA to other'),(~d.group.values,'#D55E00','Other to LumA')]:
    axes[0].scatter(a.R[mask,0],pp[1:][mask],s=17,color=color,label=label)
axes[0].set_xlabel('Pooled rank of relabeled patient');axes[0].set_ylabel('One-sided p-value');axes[0].legend(frameon=False,fontsize=9)
axes[1].hist(pp,bins=18,color='#56B4E9',edgecolor='white');axes[1].set_xlabel('One-sided p-value');axes[1].set_ylabel('Number of labelings')
for ax in axes:
    if ax==axes[0]:ax.axhline(a.p[0,0],color='black',ls='--',lw=1)
    else:
        for value,color in [(a.p[0,0],'black'),(lo[0,0],'#009E73'),(hi[0,0],'#009E73')]:ax.axvline(value,color=color,ls='--',lw=1)
fig.suptitle('MAINZ ALG5: fixed greater alternative, all 201 labelings',x=.08,ha='left',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'ALG5_stability.png',dpi=220);fig.savefig(OUT/'ALG5_stability.svg');plt.close(fig)
