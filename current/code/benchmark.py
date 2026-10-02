"""Sequential measured benchmark for the item 6–7 proposal, not live edits."""
from pathlib import Path
from itertools import combinations, islice
import argparse, datetime, hashlib, importlib.util, json, math, os, platform, sys, time
import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr, stdtr
from linear_intervals import LinearIntervals
from reference.analysis_core import WRS
from reference.efficient_ttest import Moments

ROOT = Path(__file__).resolve().parent
from paths import analysis_root
INPUT_ROOT=analysis_root()
SEED, REPEATS, BATCH = 20260923, 3, 64
sys.path.insert(0, str(ROOT/'reference'))
spec = importlib.util.spec_from_file_location('previous', ROOT/'reference/previous_benchmark.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Statistics:
    """Enumeration preparation needs pooled ranks/moments, but no group sort."""
    def __init__(self, x, g, test):
        self.test=test; self.n,self.N=x.shape; self.m=int(g.sum())
        self.base=WRS(x,g) if test=='WRS' else Moments(x,g)
        self.original_sum=self.base.W if test=='WRS' else self.base.sa
        self.values=self.base.R if test=='WRS' else self.base.x
        self.valid=self.base.valid
        self.delta=1-2*g.astype(int)
    _stat=LinearIntervals._stat

def tails(o, low, high):
    cdf=ndtr if o.test=='WRS' else lambda t: stdtr(o.n-2,t)
    lo=np.stack((cdf(-high),cdf(low)))
    hi=np.stack((cdf(-low),cdf(high)))
    lo[:,~o.valid]=1;hi[:,~o.valid]=1
    return lo,hi

def quadratic(o,k):
    """Same preparation/stat/tail work as linear; enumerate all count pairs."""
    zero=np.zeros((1,o.N))
    def prefix(v):return np.concatenate((zero,np.cumsum(v[:k],axis=0)))
    amin,amax,bmin,bmax=[prefix(v) for v in (o.a,o.a[::-1],o.b,o.b[::-1])]
    low=np.full(o.N,np.inf);high=np.full(o.N,-np.inf)
    for i in range(k+1):
        for j in range(k+1-i):
            m=o.m-i+j
            low=np.minimum(low,o._stat(o.original_sum+(bmin[j]-amax[i]),m))
            high=np.maximum(high,o._stat(o.original_sum+(bmax[j]-amin[i]),m))
    return tails(o,low,high)

def exhaustive(o,k):
    low=o._stat(o.original_sum,o.m);high=low.copy();count=1
    for distance in range(1,k+1):
        iterator=combinations(range(o.n),distance)
        while True:
            block=list(islice(iterator,BATCH))
            if not block:break
            ii=np.asarray(block,dtype=int)
            sums=np.broadcast_to(o.original_sum,(len(ii),o.N)).copy()
            sizes=np.full(len(ii),o.m,dtype=int)
            for j in range(distance):
                change=o.delta[ii[:,j]]
                sums+=o.values[ii[:,j]]*change[:,None]
                sizes+=change
            stat=o._stat(sums,sizes[:,None])
            low=np.minimum(low,stat.min(axis=0));high=np.maximum(high,stat.max(axis=0))
            count+=len(ii)
    assert count==sum(math.comb(o.n,j) for j in range(k+1))
    return tails(o,low,high)

def prepare(x,g,test,algorithm):
    return Statistics(x,g,test) if algorithm=='Enumeration' else LinearIntervals(x,g,test)

def run(o,k,algorithm):
    return o.bounds(k) if algorithm=='Linear' else quadratic(o,k) if algorithm=='Quadratic' else exhaustive(o,k)

def compare(a,b):return previous.compare(a,b)

def validate():
    rng=np.random.default_rng(SEED);rows=[]
    cases=[]
    for n,m in [(10,5),(12,5)]:
        for kind in ['normal','rounded','skewed']:
            x=rng.normal(size=(n,16))
            if kind=='rounded':x=np.round(x)
            if kind=='skewed':x=np.exp(x)
            x[:,0]=3;x[0,1]=np.nan
            cases.append((kind,x,np.arange(n)<m))
    g=np.arange(10)<5
    cases.append(('two_constant_groups',np.where(g[:,None],np.array([1.,.1]),np.array([2.,.3])),g))
    for kind,x,g in cases:
        for test in ['WRS','Pooled']:
            for k in range(4):
                direct,count=previous.direct_bounds(x,g,k,test)
                row=dict(kind=kind,n=len(g),features=x.shape[1],k=k,test=test,labelings=count)
                for algorithm in ['Linear','Quadratic','Enumeration']:
                    row[algorithm+'_error']=compare(run(prepare(x,g,test,algorithm),k,algorithm),direct)
                row['Frozen_quadratic_error']=compare(previous.candidate(x,g,k,test),direct)
                rows.append(row)
    (OUT/'validation.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('VALIDATED',len(rows),'small cases',flush=True)

def load_real():
    matrices=[];inputs=[]
    for cohort,kind,name,maximum in [('Micma','breast','cohort_data',28),('GSE19188','lung','GSE19188',43)]:
        path=INPUT_ROOT/'data'/kind/(name+'.npz');d=np.load(path,allow_pickle=False)
        matrices.append((cohort,d['x'],d['group'],maximum))
        inputs.append(dict(cohort=cohort,path=str(path.relative_to(INPUT_ROOT)),sha256=sha(path)))
    (OUT/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    return matrices

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
    validate()
    matrices=load_real();rng=np.random.default_rng(SEED)
    aa,bb=rng.normal(size=(192,1000)),rng.normal(size=(192,1000))
    cases=[]
    for cohort,x,g,maximum in matrices:
        for k in [1,2,3,5,10,20,maximum]:
            for test in ['WRS','Pooled']:cases.append(('real_budget',cohort,x,g,k,test))
    for n in [24,48,96,192,384]:
        x=np.vstack((aa[:n//2],bb[:n//2]));g=np.arange(n)<n//2
        for test in ['WRS','Pooled']:cases.append(('sample_size','Balanced_normal',x,g,2,test))
    metadata=dict(date=datetime.date.today().isoformat(),seed=SEED,repeats=REPEATS,batch_size=BATCH,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
        platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
        thread_environment={key:os.environ.get(key) for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']},
        timing='Sequential perf_counter, fresh preparation then search segments; total is their sum; BH excluded',
        order='Cyclic shifts by case and repetition; no concurrent benchmark jobs',
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'linear_intervals.py',*sorted((ROOT/'reference').glob('*.py'))]})
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    warm=rng.normal(size=(24,1000));wg=np.arange(24)<12
    for test in ['WRS','Pooled']:
        for algorithm in ['Linear','Quadratic','Enumeration']:run(prepare(warm,wg,test,algorithm),2,algorithm)
    rows=[];checks=[]
    for case_index,(experiment,cohort,x,g,k,test) in enumerate(cases):
        key=f'{experiment}/{cohort}/{test}/n{len(g)}/k{k}'
        algorithms=['Linear','Quadratic']+(['Enumeration'] if experiment=='sample_size' or k<=3 else [])
        print('START',key,flush=True)
        for rep in range(REPEATS):
            shift=(case_index+rep)%len(algorithms)
            order=algorithms[shift:]+algorithms[:shift];answers={}
            for position,algorithm in enumerate(order):
                t0=time.perf_counter();o=prepare(x,g,test,algorithm);t1=time.perf_counter()
                answer=run(o,k,algorithm);t2=time.perf_counter()
                answers[algorithm]=answer
                rows.append(dict(experiment=experiment,cohort=cohort,test=test,samples=len(g),group_true=int(g.sum()),features=x.shape[1],k=k,
                    labelings=str(sum(math.comb(len(g),j) for j in range(k+1))),algorithm=algorithm,repetition=rep+1,position=position+1,
                    preprocessing_seconds=t1-t0,search_seconds=t2-t1,total_seconds=t2-t0))
                del o
            for algorithm in algorithms[1:]:
                err=compare(answers['Linear'],answers[algorithm])
                checks.append(dict(case=key,repetition=rep+1,baseline=algorithm,max_absolute_error=err))
            pd.DataFrame(rows).to_csv(OUT/'timings.csv',index=False)
            (OUT/'endpoint_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
            print('DONE',key,'rep',rep+1,{r['algorithm']:round(r['total_seconds'],6) for r in rows[-len(order):]},flush=True)
    d=pd.DataFrame(rows)
    keys=['experiment','cohort','test','samples','group_true','features','k','labelings','algorithm']
    summary=d.groupby(keys)[['preprocessing_seconds','search_seconds','total_seconds']].agg(['median','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT/'summary.csv',index=False)
    print('COMPLETE',len(cases),'cases',len(rows),'runs',len(checks),'endpoint comparisons',flush=True)

if __name__=='__main__':main()
