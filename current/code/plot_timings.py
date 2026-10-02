"""Create figures and a concrete source/Markdown review proposal after timing."""
from pathlib import Path
import difflib, hashlib, json, os
import argparse
from paths import analysis_root
PACKAGE=analysis_root()
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
ROOT=PACKAGE/'results/rdeg-linear-benchmark'
DEST=args.out.resolve();DEST.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(DEST/'matplotlib-cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

s=pd.read_csv(ROOT/'summary.csv',dtype={'labelings':str})
checks=json.loads((ROOT/'endpoint_checks.json').read_text())
validation=json.loads((ROOT/'validation.json').read_text())
assert len(s)==98 and len(checks)==180 and len(validation)==56
max_error=max(r['max_absolute_error'] for r in checks)
max_direct=max(r[a+'_error'] for r in validation for a in ['Linear','Quadratic','Enumeration'])
def sci(x):
    mantissa,exponent=f'{x:.2e}'.split('e')
    return mantissa+r'\times10^{'+str(int(exponent))+'}'

colors={'Linear':'#0072B2','Quadratic':'#D55E00','Enumeration':'#636363'}
styles={'Linear':('-','o'),'Quadratic':('--','s'),'Enumeration':(':','^')}
labels={'Linear':'Linear sweep','Quadratic':'Quadratic search','Enumeration':'Exhaustive enumeration'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
    'axes.labelsize':10,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False})

def curve(ax,d,metric,axis,algorithm):
    d=d[d.algorithm.eq(algorithm)].sort_values(axis)
    med=d[metric+'_median'].to_numpy();lo=d[metric+'_min'].to_numpy();hi=d[metric+'_max'].to_numpy()
    ls,marker=styles[algorithm]
    ax.errorbar(d[axis],med,yerr=np.stack((np.maximum(med-lo,0),np.maximum(hi-med,0))),
        color=colors[algorithm],linestyle=ls,marker=marker,markersize=4,capsize=2,
        linewidth=1.6,label=labels[algorithm])

def real_figure(metric,filename,algorithms):
    fig,axs=plt.subplots(2,2,figsize=(8.2,6.2),layout='constrained')
    for idx,(ax,cohort,test) in enumerate([(axs[0,0],'Micma','WRS'),(axs[0,1],'Micma','Pooled'),(axs[1,0],'GSE19188','WRS'),(axs[1,1],'GSE19188','Pooled')]):
        d=s[s.experiment.eq('real_budget')&s.cohort.eq(cohort)&s.test.eq(test)]
        for algorithm in algorithms:curve(ax,d,metric,'k',algorithm)
        cohort_title='Breast: n = 104, N = 9,532' if cohort=='Micma' else 'Lung: n = 91, N = 5,000'
        test_title='WRS' if test=='WRS' else 'Pooled Student'
        ax.set(title=f'{chr(65+idx)}  {cohort_title}\n{test_title}',xlabel='Label-flip budget k',ylabel='Seconds, including preparation' if metric=='total_seconds' else 'Seconds, excluding preparation')
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks(sorted(d.k.unique()))
        ax.xaxis.set_major_formatter(ScalarFormatter());ax.minorticks_off();ax.grid(alpha=.2)
        ax.legend(frameon=False,loc='upper right' if metric=='total_seconds' else 'best',fontsize=8.5)
    for ext in ['png','svg']:fig.savefig(DEST/f'{filename}.{ext}',dpi=220)
    plt.close(fig)

real_figure('total_seconds','endpoint_methods',['Linear','Quadratic','Enumeration'])

real_figure('search_seconds','endpoint_search',['Linear','Quadratic'])
fig,axs=plt.subplots(1,2,figsize=(8.2,3.3),layout='constrained')
for idx,(ax,test) in enumerate(zip(axs,['WRS','Pooled'])):
    d=s[s.experiment.eq('sample_size')&s.test.eq(test)]
    for algorithm in ['Linear','Quadratic','Enumeration']:curve(ax,d,'total_seconds','samples',algorithm)
    ax.set(title=f'{chr(65+idx)}  {"WRS" if test=="WRS" else "Pooled Student"}: N = 1,000, k = 2',xlabel='Samples n',ylabel='Seconds, including preparation')
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(sorted(d.samples.unique()))
    ax.xaxis.set_major_formatter(ScalarFormatter());ax.minorticks_off();ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
for ext in ['png','svg']:fig.savefig(DEST/f'endpoint_sample_size.{ext}',dpi=220)
plt.close(fig)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
    'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':8})
fig,axs=plt.subplots(1,2,figsize=(8.0,3.7),layout='constrained')
for ax,cohort,title in zip(axs,['Micma','GSE19188'],['A  Breast: n = 104, N = 9,532','B  Lung: n = 91, N = 5,000']):
    for test,color in [('WRS','#0072B2'),('Pooled','#D55E00')]:
        for method,style,marker,label in [('Linear','-','o','efficient'),('Enumeration','--','s','exhaustive')]:
            d=s[s.cohort.eq(cohort)&s.test.eq(test)&s.algorithm.eq(method)&s.k.le(3)].sort_values('k')
            med=d.total_seconds_median.to_numpy();lo=d.total_seconds_min.to_numpy();hi=d.total_seconds_max.to_numpy()
            ax.errorbar(d.k,med,yerr=np.stack((med-lo,hi-med)),color=color,linestyle=style,
                marker=marker,markersize=4,capsize=2,linewidth=1.6,label=f'{test} {label}')
    ax.set(title=title,xlabel='Label-flip budget k',ylabel='Endpoint time (seconds)',xticks=[1,2,3])
    ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(loc='upper left',frameon=False)
for ext in ['png','svg']:fig.savefig(DEST/f'endpoint_scalability.{ext}',dpi=220)
plt.close(fig)
