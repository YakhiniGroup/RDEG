"""Independent structural, distributional and current-data checks for item 6."""
from pathlib import Path
from itertools import combinations,combinations_with_replacement,product
import argparse,hashlib,json,platform,sys,time,warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu,ttest_ind,rankdata,false_discovery_control

ROOT=Path(__file__).resolve().parent
from paths import analysis_root
INPUT_ROOT=analysis_root()
sys.path.insert(0,str(ROOT/'reference'))
from linear_intervals import LinearIntervals,net_extremes,SweepAudit
from reference.analysis_core import WRS,bh_adjusted,vectors,identifiers
from reference.efficient_ttest import Moments

SEED=20260925
ATOL=5e-13
RTOL=3e-10

def compare(a,b):
    error=0.
    for aa,bb in zip(a,b):
        assert np.isfinite(aa).all() and np.isfinite(bb).all()
        np.testing.assert_allclose(aa,bb,atol=ATOL,rtol=RTOL)
        error=max(error,float(np.max(np.abs(aa-bb))))
    return error

def quadratic(x,g,k,test):
    return WRS(x,g).candidates(k) if test=='WRS' else Moments(x,g).pooled_candidates(k)

def direct(x,g,k,test):
    valid=np.isfinite(x).all(axis=0)
    xx=np.where(valid[None,:],x,0)
    valid &= np.ptp(xx,axis=0)>0
    lo=np.ones((2,x.shape[1]));hi=np.zeros_like(lo);count=0
    for distance in range(k+1):
        for ids in combinations(range(len(g)),distance):
            gg=g.copy();gg[list(ids)]=~gg[list(ids)]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                if test=='WRS':
                    p=np.stack([mannwhitneyu(xx[gg],xx[~gg],axis=0,alternative=t,method='asymptotic',use_continuity=False).pvalue for t in ['greater','less']])
                else:
                    p=np.stack([ttest_ind(xx[gg],xx[~gg],axis=0,equal_var=True,alternative=t).pvalue for t in ['greater','less']])
            p[:,~valid]=1
            assert np.isfinite(p).all(),(test,ids)
            lo=np.minimum(lo,p);hi=np.maximum(hi,p);count+=1
    return (lo,hi),count

def check_core():
    batches=0;features=0;candidates=0;max_comparisons_ratio=0.;max_inner_moves=0
    # Full cross-product of sorted multisets; no random selection here.
    for m,b in product(range(1,6),repeat=2):
        aa=list(combinations_with_replacement([-1.,0.,1.],m))
        bb=list(combinations_with_replacement([-1.,0.,1.],b))
        pairs=list(product(aa,bb))
        A=np.array([a for a,_ in pairs]).T
        B=np.array([b for _,b in pairs]).T
        N=len(pairs)
        for k in range(min(m,b)+1):
            for maximize in [False,True]:
                remove,add=(A,B[::-1]) if maximize else (A[::-1],B)
                pr=np.r_[np.zeros((1,N)),np.cumsum(remove,axis=0)]
                pa=np.r_[np.zeros((1,N)),np.cumsum(add,axis=0)]
                audit=SweepAudit(0,np.zeros(N,dtype=int),np.zeros(N,dtype=int))
                for d,changes,q in net_extremes(remove,add,k,maximize=maximize,audit=audit):
                    indices=np.arange(max(0,-d),(k-d)//2+1)
                    all_changes=pa[indices+d]-pr[indices]
                    pos=np.argmax(all_changes,axis=0) if maximize else np.argmin(all_changes,axis=0)
                    best=all_changes[pos,np.arange(N)]
                    np.testing.assert_array_equal(changes,best)
                    np.testing.assert_array_equal(q,indices[pos])
                    candidates+=N
                assert audit.net_changes==2*k+1
                assert np.all(audit.decrements==k)
                assert np.all(audit.comparisons<=3*k+1)
                max_comparisons_ratio=max(max_comparisons_ratio,float(audit.comparisons.max()/(3*k+1)))
                max_inner_moves=max(max_inner_moves,int(audit.decrements.max()))
                batches+=1;features+=N
    return dict(batches=batches,feature_endpoint_cases=features,net_change_extrema_checked=candidates,
                smallest_argmin_or_argmax_exact=True,integer_sums_exact=True,
                max_comparison_fraction_of_bound=max_comparisons_ratio,max_decrements=max_inner_moves)

def check_small():
    rng=np.random.default_rng(SEED);rows=[]
    for n,m in [(4,2),(5,2),(7,3),(8,4),(9,3),(10,5)]:
        for kind in ['normal','tied','skewed','two_constants']:
            x=rng.normal(size=(n,12));g=np.arange(n)<m
            if kind=='tied': x=np.round(x)
            elif kind=='skewed': x=np.exp(x)
            elif kind=='two_constants':
                x=np.where(g[:,None],np.arange(12)[None,:]*.125,np.arange(12)[None,:]*.375)
            x[:,0]=3; x[0,1]=np.nan; x[-1,2]=np.inf
            for test in ['WRS','Pooled']:
                obj=LinearIntervals(x,g,test);prev=None
                for k in range(obj.limit+1):
                    new=obj.bounds(k,audit=True);old=quadratic(x,g,k,test)
                    raw,count=direct(x,g,k,test)
                    err_old=compare(new[:2],old);err_direct=compare(new[:2],raw)
                    if prev:
                        assert np.all(new[0]<=prev[0]+ATOL) and np.all(new[1]>=prev[1]-ATOL)
                    prev=new
                    rows.append(dict(n=n,m=m,features=x.shape[1],kind=kind,test=test,k=k,
                        labelings=count,quadratic_error=err_old,direct_error=err_direct))
    return rows

def exact_wrs_check():
    rows=[]
    for values in [np.arange(1.,9.),np.array([0.,0.,1.,1.,1.,2.,3.,3.])]:
        n=len(values);ranks=rankdata(values)
        null={r:np.array([ranks[list(ii)].sum() for ii in combinations(range(n),r)]) for r in range(1,n)}
        # Every original binary labeling with group sizes 2,3,4.
        for m in [2,3,4]:
            for ix in combinations(range(n),m):
                g=np.zeros(n,dtype=bool);g[list(ix)]=True
                A=np.sort(ranks[g])[:,None];B=np.sort(ranks[~g])[:,None];S=ranks[g].sum()
                for k in range(min(m,n-m)):
                    bestlo=np.ones(2);besthi=np.zeros(2)
                    for mini,maxi in zip(net_extremes(A[::-1],B,k),net_extremes(A,B[::-1],k,maximize=True)):
                        d,ds,_=mini;e,dt,_=maxi;assert d==e;r=m+d
                        for s in [S+ds[0],S+dt[0]]:
                            pp=np.array([(null[r]>=s).mean(),(null[r]<=s).mean()])
                            bestlo=np.minimum(bestlo,pp);besthi=np.maximum(besthi,pp)
                    truthlo=np.ones(2);truthhi=np.zeros(2);visited=0
                    for dist in range(k+1):
                        for flips in combinations(range(n),dist):
                            gg=g.copy();gg[list(flips)]=~gg[list(flips)]
                            r=int(gg.sum());s=ranks[gg].sum()
                            pp=np.array([(null[r]>=s).mean(),(null[r]<=s).mean()])
                            truthlo=np.minimum(truthlo,pp);truthhi=np.maximum(truthhi,pp);visited+=1
                    np.testing.assert_array_equal(bestlo,truthlo)
                    np.testing.assert_array_equal(besthi,truthhi)
                    rows.append(dict(tied=len(np.unique(values))<n,m=m,k=k,labelings=visited))
    return dict(cases=len(rows),labelings=sum(r['labelings'] for r in rows),endpoint_values_identical=True,
                tied_cases=sum(r['tied'] for r in rows),method='Enumerated fixed-size permutation nulls, inclusive tails')

def check_invariance_and_batching():
    rng=np.random.default_rng(SEED+1);rows=[]
    x=rng.integers(-10,11,size=(28,173)).astype(float);g=np.arange(28)<11
    for test in ['WRS','Pooled']:
        obj=LinearIntervals(x,g,test)
        for k in [0,1,2,obj.limit]:
            orig=obj.bounds(k,audit=True)
            one=[LinearIntervals(x[:,[i]],g,test).bounds(k) for i in range(x.shape[1])]
            merged=tuple(np.concatenate([z[j] for z in one],axis=1) for j in range(2))
            error=compare(orig[:2],merged)
            perm=rng.permutation(len(g));col=rng.permutation(x.shape[1])
            error=max(error,compare(orig[:2],LinearIntervals(x[perm],g[perm],test).bounds(k)))
            compare(tuple(z[:,col] for z in orig[:2]),LinearIntervals(x[:,col],g,test).bounds(k))
            affine=LinearIntervals(x*8+32,g,test).bounds(k)
            error=max(error,compare(orig[:2],affine))
            reversed_groups=LinearIntervals(x,~g,test).bounds(k)
            error=max(error,compare(tuple(z[::-1] for z in orig[:2]),reversed_groups))
            rows.append(dict(test=test,k=k,features=x.shape[1],max_error=error))
    bad_cases=0
    for test in ['WRS','Pooled']:
        obj=LinearIntervals(x,g,test)
        for k in [-1,obj.limit+1,1.5]:
            try:obj.bounds(k)
            except ValueError:bad_cases+=1
            else:raise AssertionError('Invalid budget accepted')
    for g_bad in [np.zeros(len(g)),np.full(len(g),2),g[:-1]]:
        try:LinearIntervals(x,g_bad)
        except ValueError:bad_cases+=1
        else:raise AssertionError('Invalid labels accepted')
    return dict(validations=rows,invalid_inputs_rejected=bad_cases)

def load_real():
    rows=[]
    for label,kind,cohort,tau,k in [('Micma','breast','cohort_data',.001,5),('GSE19188','lung','GSE19188',.01,8)]:
        path=INPUT_ROOT/'data'/kind/(cohort+'.npz')
        a=np.load(path,allow_pickle=False)
        rows.append((label,a['x'],a['group'],a['genes'],tau,k,path))
    return rows

def check_real():
    rows=[];inputs=[];selections=[]
    breast_budget=pd.read_csv(INPUT_ROOT/'results/rdeg-no-tcga-review/budget_curves/summary.csv')
    lung_budget=pd.read_csv(INPUT_ROOT/'results/rdeg-lung-variance-review/lung/budget_summary.csv')
    for cohort,x,g,genes,tau,largest_published,path in load_real():
        inputs.append(dict(cohort=cohort,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           n=len(g),m=int(g.sum()),features=len(genes),tau=tau))
        for test in ['WRS','Pooled']:
            print('REAL START',cohort,test,flush=True)
            obj=LinearIntervals(x,g,test)
            if cohort=='Micma':
                olddir=INPUT_ROOT/f'results/rdeg-no-tcga-review/{test}/own_reference'
            else: olddir=INPUT_ROOT/f'results/rdeg-lung-variance-review/lung/{test}/{test}_reference'
            saved={d:pd.read_csv(olddir/f'{cohort}_{d}_k1_features.csv.gz') for d in ['greater','less','joint']}
            for d,f in saved.items():np.testing.assert_array_equal(f.hypothesis,identifiers(genes,d))
            for k in sorted(set(range(largest_published+1))|{obj.limit}):
                start=time.perf_counter()
                new=obj.bounds(k,audit=True);old=quadratic(x,g,k,test)
                err=compare(new[:2],old)
                direct_error=None;labelings=None
                if k==1:
                    oracle,labelings=direct(x,g,1,test)
                    direct_error=compare(new[:2],oracle)
                if cohort=='Micma' and k<=5:
                    ep=np.load(INPUT_ROOT/f'results/rdeg-budget-curves/{cohort}_{test}_endpoints.npz')
                    np.testing.assert_array_equal(ep['genes'].astype(str),genes)
                    compare(new[:2],(ep['p_L'][k],ep['p_U'][k]))
                for direction,pv in vectors(new[1]).items():
                    q=bh_adjusted(pv);qo=bh_adjusted(vectors(old[1])[direction])
                    np.testing.assert_allclose(q,qo,atol=ATOL,rtol=RTOL)
                    np.testing.assert_allclose(q,false_discovery_control(pv),atol=ATOL,rtol=RTOL)
                    selected=q<=tau
                    np.testing.assert_array_equal(selected,qo<=tau)
                    if k in [0,1]:
                        s=saved[direction]
                        np.testing.assert_array_equal(selected,s.BH if k==0 else s.ERDEG)
                        np.testing.assert_allclose(pv,s.p if k==0 else s.p_U,atol=ATOL,rtol=RTOL)
                    if k==2 and cohort=='Micma' and test=='WRS':
                        s=pd.read_csv(olddir/f'{cohort}_{direction}_k2_features.csv.gz')
                        np.testing.assert_array_equal(selected,s.ERDEG)
                    hits=int((selected&saved[direction].consensus_reference.to_numpy(bool)).sum())
                    count=int(selected.sum())
                    if k<=largest_published:
                        if cohort=='Micma':
                            expected=breast_budget[(breast_budget.test==test)&(breast_budget.family==direction)&(breast_budget.k==k)].iloc[0]
                            h=int(expected.reference_overlap)
                        else:
                            expected=lung_budget[(lung_budget.test==test)&(lung_budget.direction==direction)&(lung_budget.k==k)].iloc[0]
                            h=int(expected.hits)
                        assert count==int(expected.selected) and hits==h,(cohort,test,k,direction,count,hits)
                    selections.append(dict(cohort=cohort,test=test,k=k,direction=direction,selected=count,hits=hits))
                rows.append(dict(cohort=cohort,test=test,k=k,features=len(genes),quadratic_error=err,
                                 direct_error=direct_error,direct_labelings=labelings,
                                 max_pointer_decrements=int(max(a.decrements.max() for a in new[2].values())),
                                 max_pointer_comparisons=int(max(a.comparisons.max() for a in new[2].values())),
                                 wall_seconds_for_all_checks=time.perf_counter()-start))
                print('REAL PASS',cohort,test,k,'error',err,flush=True)
                (OUT/'real_checks.json').write_text(json.dumps(rows,indent=2))
                pd.DataFrame(selections).to_csv(OUT/'selection_checks.csv',index=False)
    (OUT/'inputs.json').write_text(json.dumps(inputs,indent=2))
    return rows

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--real-only',action='store_true');parser.add_argument('--small-only',action='store_true');parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
    report={}
    if not args.real_only:
        for name,fn in [('core',check_core),('small',check_small),('exact_wrs',exact_wrs_check),('invariance',check_invariance_and_batching)]:
            print('START',name,flush=True);start=time.perf_counter();report[name]=fn();print('PASS',name,'seconds',time.perf_counter()-start,flush=True)
            (OUT/'small_checks.json').write_text(json.dumps(report,indent=2))
    if not args.small_only:report['real']=check_real()
    meta=dict(seed=SEED,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,platform=platform.platform(),atol=ATOL,rtol=RTOL,
              files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py')})
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2));print('ALL REQUESTED CHECKS PASSED',flush=True)

if __name__=='__main__':main()
