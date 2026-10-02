"""Portable reproduction of the frozen RDEG breast/lung benchmark.

Inputs are read from RDEG_ANALYSIS_ROOT (or the current code package). Results are written only to --out; no network is used.
Run from any directory: python /path/to/package/code/reproduce.py --all --out /tmp/rdeg-run
"""
from pathlib import Path
import argparse,json,time,warnings,platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu,ttest_ind,false_discovery_control
from scipy.special import stdtr
from analysis_core import WRS,bh_adjusted,vectors,identifiers
from efficient_ttest import Moments
from linear_intervals import LinearIntervals
from paths import analysis_root
ROOT=analysis_root()
TESTS=('WRS','Pooled','Welch'); MODES=('greater','less','joint')
CHECKS=[]; ROWS=[]

def close(a,b,label):
    np.testing.assert_allclose(a,b,rtol=5e-10,atol=5e-13,err_msg=label)

def equal(a,b,label):
    np.testing.assert_array_equal(a,b,err_msg=label)

def q(p):
    v=bh_adjusted(p)
    close(v,false_discovery_control(p),'BH vs SciPy')
    return v

def load(kind,cohort):
    with np.load(ROOT/'data'/kind/f'{cohort}.npz',allow_pickle=False) as f:
        return {k:f[k] for k in f.files}

def direct(x,g,test):
    valid=np.isfinite(x).all(axis=0)
    xx=np.where(valid[None,:],x,0)
    valid &= np.ptp(xx,axis=0)>0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        if test=='WRS':
            p=np.stack([mannwhitneyu(xx[g],xx[~g],axis=0,alternative=d,method='asymptotic',use_continuity=False).pvalue for d in ('greater','less')])
        else:
            fit=ttest_ind(xx[g],xx[~g],axis=0,equal_var=test=='Pooled',alternative='greater')
            p=np.stack([stdtr(fit.df,-fit.statistic),stdtr(fit.df,fit.statistic)])
            valid &= (~np.isnan(fit.statistic) if test=='Pooled' else np.isfinite(fit.statistic)) & np.isfinite(fit.df) & (fit.df>0)
    p[:,~valid]=1
    assert np.isfinite(p).all()
    return p

def family_reference(ps,focus,mode,tau,common=None):
    return np.stack([q(vectors(p if common is None else p[:,common])[mode])<=tau for i,p in enumerate(ps) if i!=focus]).all(axis=0)

def selections(p,pu,rob,genes,mode,tau):
    pv=vectors(p)[mode]; ids=identifiers(genes,mode)
    q0=q(pv);qu=q(vectors(pu)[mode]); r=rob[mode]
    masks=dict(BH=q0<=tau,ERDEG=qu<=tau,RDEG=r<=tau)
    order=np.lexsort((ids,pv))
    for method in ('ERDEG','RDEG'):
        m=np.zeros(len(pv),bool);m[order[:masks[method].sum()]]=True
        masks['matched_'+method]=m
    assert not np.any(masks['ERDEG'] & ~masks['RDEG'])
    assert not np.any(masks['RDEG'] & ~masks['BH'])
    return dict(hypothesis=ids,p=pv,BH_adjusted_p=q0,p_U=vectors(pu)[mode],BH_adjusted_p_U=qu,max_BH_adjusted_p=r),masks

def verify_features(path,columns,masks,ref,case,out):
    expected=pd.read_csv(path)
    if 'reference_0p01' in expected:
        expected=expected.rename(columns={'reference_0p01':'consensus_reference',**{m+'_0p01':m for m in ('BH','ERDEG','RDEG','matched_ERDEG','matched_RDEG')}})
    equal(expected.hypothesis,columns['hypothesis'],str(path))
    for name,val in columns.items():
        if name!='hypothesis':close(expected[name],val,f'{path.name} {name}')
    equal(expected.consensus_reference,ref,'reference '+str(path))
    for name,mask in masks.items():
        equal(expected[name],mask,f'{path.name} {name}')
        ROWS.append(dict(case=case,method=name,selected=int(mask.sum()),hits=int((mask&ref).sum()),reference_size=int(ref.sum())))
    pd.DataFrame({**columns,'consensus_reference':ref,**masks}).to_csv(out/(case.replace('/','_')+'.csv.gz'),index=False)
    CHECKS.append(dict(check='feature_file',case=case,hypotheses=len(ref),all_memberships_identical=True))

def budget(obj,p,refs,genes,tau,expected,test,kind,out):
    modecol='family' if 'family' in expected else 'direction'
    hitcol='reference_overlap' if 'reference_overlap' in expected else 'hits'
    mcol='matched_reference_overlap' if 'matched_reference_overlap' in expected else 'matched_hits'
    rows=[]; endpoints=[]
    for k in sorted(expected[expected.test==test].k.unique()):
        pl,pu=obj.bounds(int(k));endpoints.append(pu)
        for mode in MODES:
            e=expected[(expected.test==test)&(expected[modecol]==mode)&(expected.k==k)]
            assert len(e)==1;e=e.iloc[0]
            mask=q(vectors(pu)[mode])<=tau
            ids=identifiers(genes,mode);order=np.lexsort((ids,vectors(p)[mode]))
            match=np.zeros(len(mask),bool);match[order[:mask.sum()]]=True
            ref=refs[mode]
            actual=[mask.sum(),(mask&ref).sum(),(match&ref).sum(),ref.sum()]
            equal(actual,[e.selected,e[hitcol],e[mcol],e.reference_size],f'{kind} {test} {mode} k={k}')
            rows.append(dict(test=test,direction=mode,k=int(k),selected=int(mask.sum()),hits=int((mask&ref).sum()),matched_hits=int((match&ref).sum())))
    pd.DataFrame(rows).to_csv(out/f'{kind}_{test}_budgets.csv',index=False)
    np.savez_compressed(out/f'{kind}_{test}_endpoints.npz',p_U=np.array(endpoints),genes=genes)
    CHECKS.append(dict(check='budget_curve',case=kind+'/'+test,rows=len(rows),counts_identical=True))

def run(kind,out,full):
    breast=kind=='breast'
    base=ROOT/'results'/('rdeg-no-tcga-review' if breast else 'rdeg-lung-'+('iqr' if kind=='lung_iqr' else 'variance')+'-review/lung')
    focusname='cohort_data' if breast else 'GSE19188';label='Micma' if breast else focusname;tau=.001 if breast else .01
    allps={}; archives={}
    # Recompute each reference test from observations, independently of interval code.
    for test in TESTS:
        a=dict(np.load(base/test/('breast_reference_pvalues.npz' if breast else 'lung_reference_pvalues.npz'),allow_pickle=False))
        ps=[]
        for c in a['cohorts']:
            d=load(kind,str(c));equal(d['genes'],a['genes'],'gene order')
            ps.append(direct(d['x'],d['group'],test))
        ps=np.array(ps);close(ps,a['p'],kind+'/'+test+' ordinary reference p-values')
        allps[test]=ps;archives[test]=a
        print('verified reference p-values:',kind,test,flush=True)
    focus=archives['WRS']['cohorts'].tolist().index(focusname)
    d=load(kind,focusname);x,g,genes=d['x'],d['group'],d['genes']
    for test in TESTS:
        p=allps[test][focus]
        enum=WRS(x,g).relabelings(1) if test=='WRS' else Moments(x,g).relabelings(1,equal_var=test=='Pooled')
        lo=np.ones_like(p);hi=np.zeros_like(p);rob={m:np.zeros(len(v)) for m,v in vectors(p).items()}
        common=np.load(ROOT/'data/breast/frozen_common_mask.npy') if breast and test=='WRS' and full else None
        rc={m:np.zeros(len(v)) for m,v in vectors(p[:,common]).items()} if common is not None else None
        nlabels=0
        for ix,pp in enum:
            if not ix:close(pp,p,'original vs moment/rank implementation')
            lo=np.minimum(lo,pp);hi=np.maximum(hi,pp)
            for m,pv in vectors(pp).items():rob[m]=np.maximum(rob[m],bh_adjusted(pv))
            if rc is not None:
                for m,pv in vectors(pp[:,common]).items():rc[m]=np.maximum(rc[m],bh_adjusted(pv))
            nlabels+=1
        assert nlabels==len(g)+1
        obj=None
        if test!='Welch':
            obj=LinearIntervals(x,g,test);pl,pu=obj.bounds(1);close(pl,lo,'linear lower vs exhaustive');close(pu,hi,'linear upper vs exhaustive')
        refs={m:family_reference(allps[test],focus,m,tau) for m in MODES}
        for mode in MODES:
            columns,masks=selections(p,hi,rob,genes,mode,tau)
            folder='own_reference' if breast else test+'_reference'
            verify_features(base/test/folder/f'{label}_{mode}_k1_features.csv.gz',columns,masks,refs[mode],f'{kind}/{test}/{mode}/k1',out)
            # Cross-test reference controls change only the evaluation reference.
            if full:
                for reftest in TESTS:
                    folder='fixed_'+reftest+'_reference' if breast else reftest+'_reference'
                    path=base/test/folder/f'{label}_{mode}_k1_features.csv.gz'
                    if reftest!=test and path.exists():
                        ref=family_reference(allps[reftest],focus,mode,tau)
                        verify_features(path,columns,masks,ref,f'{kind}/{test}/{mode}/reference_{reftest}',out)
            if breast and full:
                columns2,masks2=selections(p,hi,rob,genes,mode,.01)
                ref2=family_reference(allps[test],focus,mode,.01)
                path=ROOT/'results/rdeg-breast-tau-0p01'/test/f'{mode}_k1_features.csv.gz'
                verify_features(path,columns2,masks2,ref2,f'breast_0p01/{test}/{mode}/k1',out)
                if common is not None:
                    cols,masks3=selections(p[:,common],hi[:,common],rc,genes[common],mode,tau)
                    ref3=family_reference(allps[test],focus,mode,tau,common)
                    verify_features(base/test/'common_family'/f'Micma_common_{mode}_k1_features.csv.gz',cols,masks3,ref3,f'breast_common/{test}/{mode}/k1',out)
        if obj is not None:
            expected=pd.read_csv(base/('budget_curves/summary.csv' if breast else 'budget_summary.csv'))
            budget(obj,p,refs,genes,tau,expected,test,kind,out)
        CHECKS.append(dict(check='exhaustive_k1',case=kind+'/'+test,labelings=nlabels))
        print('verified focus selections and budgets:',kind,test,flush=True)
    if breast:
        cert=LinearIntervals(x,g,'Pooled');grid=np.stack([q(cert.bounds(k)[1].reshape(-1)) for k in range(6)])
        ids=identifiers(genes,'joint');q0=q(allps['Pooled'][focus].reshape(-1));rows=[]
        for gene,want in [('NAT1',2),('CHRNA6',0)]:
            i=list(ids).index(gene+'|greater');passed=np.flatnonzero(grid[:,i]<=tau);largest=int(passed[-1]) if len(passed) else -1
            assert largest==want
            rows.append(dict(gene=gene,q_original=float(q0[i]),qU_k1=float(grid[1,i]),qU_k2=float(grid[2,i]),largest_certified_k=largest))
        pd.DataFrame(rows).to_csv(out/'certificate_examples.csv',index=False)
        CHECKS.append(dict(check='Table_S9',genes=['NAT1','CHRNA6'],certified_budgets=[2,0]))
    if breast and full:
        rotate(base,allps['WRS'],archives['WRS'],out)
        # Exact intersection-defined WRS RDEG at k=2, both significance levels.
        p=allps['WRS'][focus];obj=LinearIntervals(x,g,'WRS');_,hi=obj.bounds(2)
        rob={m:np.zeros(len(v)) for m,v in vectors(p).items()}
        count=0
        for ix,pp in WRS(x,g).relabelings(2):
            for m,pv in vectors(pp).items():rob[m]=np.maximum(rob[m],bh_adjusted(pv))
            count+=1
        assert count==5461
        for mode in MODES:
            for level in [.001,.01]:
                columns,masks=selections(p,hi,rob,genes,mode,level)
                ref=family_reference(allps['WRS'],focus,mode,level)
                path=(base/'WRS/own_reference'/f'Micma_{mode}_k2_features.csv.gz' if level==.001 else ROOT/'results/rdeg-breast-tau-0p01/WRS'/f'{mode}_k2_features.csv.gz')
                verify_features(path,columns,masks,ref,f'breast_k2/{level}/{mode}',out)
        CHECKS.append(dict(check='exhaustive_k2',case='breast/WRS',labelings=count))
        print('verified breast WRS k=2 at both thresholds',flush=True)

def rotate(base,ps,a,out):
    for i,c in enumerate(a['cohorts']):
        d=load('breast',str(c));obj=LinearIntervals(d['x'],d['group'],'WRS');_,pu=obj.bounds(1)
        for mode in MODES:
            path=base/'rotating_focus'/f'{c}_{mode}_features.csv.gz'
            if not path.exists():raise FileNotFoundError(path)
            e=pd.read_csv(path);pv=vectors(ps[i])[mode];ids=identifiers(a['genes'],mode)
            equal(e.hypothesis,ids,'rotation ids');mask=q(vectors(pu)[mode])<=.001
            ref=family_reference(ps,i,mode,.001);order=np.lexsort((ids,pv));matched=np.zeros(len(pv),bool);matched[order[:mask.sum()]]=True
            equal(e.ERDEG,mask,'rotating ERDEG');equal(e.BH,q(pv)<=.001,'rotating BH');equal(e.matched_ERDEG,matched,'rotating matched')
            equal(e.no_TCGA_reference,ref,'rotating reference')
            for method,m in [('BH',q(pv)<=.001),('ERDEG',mask),('matched_ERDEG',matched)]:
                ROWS.append(dict(case='rotation/'+str(c)+'/'+mode,method=method,selected=int(m.sum()),hits=int((m&ref).sum()),reference_size=int(ref.sum())))
    CHECKS.append(dict(check='rotation',cohorts=len(a['cohorts']),directional_cases=3*len(a['cohorts'])))
    print('verified all 11 breast WRS rotations',flush=True)

def filters():
    d=load('lung_prefilter','GSE19188');x,genes=d['x'],d['genes']
    chosen=[]
    for kind,score in [('lung',np.var(x,axis=0,ddof=1)),('lung_iqr',np.diff(np.quantile(x,[.25,.75],axis=0),axis=0)[0])]:
        selected=genes[np.lexsort((genes,-score))[:5000]]
        expected=load(kind,'GSE19188')['genes']
        equal(np.sort(selected),np.sort(expected),kind+' label-free filter')
        chosen.append(set(selected))
    CHECKS.append(dict(check='filters',candidate_count=len(genes),selected_each=5000,overlap=len(chosen[0]&chosen[1])))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--all',action='store_true',help='include IQR, threshold, common-family and rotating sensitivities');ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if out==ROOT or ROOT/'results'==out or ROOT/'data'==out:raise ValueError('Choose a separate output directory')
    start=time.time();filters()
    run('breast',out,args.all);run('lung',out,args.all)
    if args.all:run('lung_iqr',out,True)
    pd.DataFrame(ROWS).to_csv(out/'recomputed_summary.csv',index=False)
    report=dict(status='PASS',elapsed_seconds=time.time()-start,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,checks=CHECKS,scope='all' if args.all else 'primary')
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS:',len(CHECKS),'checks; output:',out,flush=True)
if __name__=='__main__':main()
