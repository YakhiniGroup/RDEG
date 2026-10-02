"""Linear-budget stability intervals for fixed one-sided WRS and pooled t.

The sweep has O(k) arithmetic per feature after sorting/prefix preprocessing.
Both directions use the same statistic extrema. This module computes raw-p
endpoints; it does not compute intersection-defined RDEG or modify BH.

Baseline rank/tie and centered-moment preparation are versioned in reference/.
Numerical conditioning qualifications are the same as the centered-moment
baseline; the mathematical complexity statement assumes unit-cost arithmetic.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr, stdtr
from reference.analysis_core import WRS
from reference.efficient_ttest import Moments


@dataclass
class SweepAudit:
    net_changes: int
    decrements: np.ndarray
    comparisons: np.ndarray


def net_extremes(remove, add, k, *, maximize=False, audit=None):
    """Yield (d, extreme sum change, removal count) for d=-k,...,k.

    For a minimum, remove is descending and add ascending in each column.
    For a maximum, reverse both orders. Only the first k rows are accessed.
    Exact ties choose the smallest removal count (fewest total flips).
    Inputs must be finite and correctly sorted. Callers enforce admissible k.
    Inner iterations touch only still-moving feature indices, which preserves
    O(N*k) work even when feature pointers stop at different iterations.
    """
    remove, add = np.asarray(remove), np.asarray(add)
    if remove.ndim != 2 or add.ndim != 2 or remove.shape[1] != add.shape[1]:
        raise ValueError('Expected two matrices with the same feature count')
    if not isinstance(k, (int, np.integer)) or k < 0 or k > min(len(remove),len(add)):
        raise ValueError('Budget must be an integer in [0, min(group sizes)]')
    N=remove.shape[1]
    zero=np.zeros((1,N),dtype=float)
    pr=np.concatenate((zero,np.cumsum(remove[:k],axis=0)))
    pa=np.concatenate((zero,np.cumsum(add[:k],axis=0)))
    q=np.full(N,k,dtype=np.int64)
    cols=np.arange(N)
    for d in range(-k,k+1):
        lower=max(0,-d)
        upper=(k-d)//2
        if audit is not None:
            audit.net_changes+=1
            audit.decrements += q-np.minimum(q,upper)
        q=np.minimum(q,upper)
        active=np.flatnonzero(q>lower)
        while active.size:
            # Previous increment is add[q+d-1] - remove[q-1].
            if audit is not None: audit.comparisons[active]+=1
            ii=q[active]
            rv=remove[ii-1,active]
            av=add[ii+d-1,active]
            moving=active[av<=rv if maximize else av>=rv]
            q[moving]-=1
            if audit is not None: audit.decrements[moving]+=1
            active=moving[q[moving]>lower]
        yield d, pa[q+d,cols]-pr[q,cols], q.copy()


class LinearIntervals:
    """Prepare an n-by-N matrix once and compute intervals at specified k.

    WRS: tie-corrected normal approximation, no continuity correction.
    Pooled: equal-variance Student tails, df=n-2. Constant/unavailable features
    get p=1; nonconstant zero-within-variance cases use signed infinite t.
    """
    def __init__(self, x, group, test='WRS'):
        x=np.asarray(x,dtype=float)
        group=np.asarray(group)
        if x.ndim!=2 or group.ndim!=1 or len(group)!=len(x):
            raise ValueError('Expected n-by-N data and n labels')
        if not np.isin(group,[0,1,False,True]).all():
            raise ValueError('Labels must be binary')
        group=group.astype(bool)
        n,N=x.shape
        if n<2 or not 0<group.sum()<n: raise ValueError('Both groups must be nonempty')
        if test not in ('WRS','Pooled'): raise ValueError('Use WRS or Pooled')
        self.test=test; self.n=n; self.N=N; self.m=int(group.sum())
        self.limit=min(self.m,n-self.m)-(1 if test=='WRS' else 2)
        if self.limit<0: raise ValueError('Pooled tests need at least two samples per group')
        self.base=WRS(x,group) if test=='WRS' else Moments(x,group)
        values=self.base.R if test=='WRS' else self.base.x
        self.a=np.sort(values[group],axis=0)
        self.b=np.sort(values[~group],axis=0)
        self.original_sum=self.base.W if test=='WRS' else self.base.sa
        self.valid=self.base.valid

    def _stat(self,s,m):
        if self.test=='WRS': return self.base.z(s,m)
        b=self.n-m
        sb=self.base.total-s
        within=self.base.total_q-s*s/m-sb*sb/b
        tol=2e-12*np.maximum(self.base.total_q,np.finfo(float).tiny)
        if np.any(within < -tol):
            raise FloatingPointError('Centered moments lost excessive precision')
        within=np.maximum(within,0)
        difference=s/m-sb/b
        se=np.sqrt(within/(self.n-2)*(1/m+1/b))
        t=np.divide(difference,se,out=np.zeros_like(difference),where=se>0)
        t[(se==0)&(difference>0)]=np.inf
        t[(se==0)&(difference<0)]=-np.inf
        return t

    def bounds(self,k,*,audit=False):
        if not isinstance(k,(int,np.integer)) or not 0<=k<=self.limit:
            raise ValueError(f'Budget must be an integer in [0, {self.limit}]')
        def record():return SweepAudit(0,np.zeros(self.N,dtype=np.int64),np.zeros(self.N,dtype=np.int64))
        a_min,a_max=(record(),record()) if audit else (None,None)
        minima=net_extremes(self.a[::-1],self.b,k,audit=a_min)
        maxima=net_extremes(self.a,self.b[::-1],k,maximize=True,audit=a_max)
        zmin=np.full(self.N,np.inf); zmax=np.full(self.N,-np.inf)
        for (d,change_min,_),(e,change_max,_) in zip(minima,maxima):
            assert d==e
            m=self.m+d
            # Centered sums use the same sufficient statistics as the baseline;
            # the changed addition order can differ by floating-point rounding.
            zmin=np.minimum(zmin,self._stat(self.original_sum+change_min,m))
            zmax=np.maximum(zmax,self._stat(self.original_sum+change_max,m))
        cdf=ndtr if self.test=='WRS' else lambda t:stdtr(self.n-2,t)
        p_L=np.stack((cdf(-zmax),cdf(zmin)))
        p_U=np.stack((cdf(-zmin),cdf(zmax)))
        p_L[:,~self.valid]=1; p_U[:,~self.valid]=1
        if not np.isfinite(p_L).all() or not np.isfinite(p_U).all():
            raise FloatingPointError('Nonfinite endpoint')
        if audit:
            for a in (a_min,a_max):
                assert a.net_changes==2*k+1
                assert np.all(a.decrements==k)  # starts at k, finishes at 0
                assert np.all(a.comparisons<=3*k+1)
            return p_L,p_U,{'min':a_min,'max':a_max}
        return p_L,p_U


def linear_bounds(x,group,k,test='WRS'):
    """Convenience API; includes fresh preparation on every call."""
    return LinearIntervals(x,group,test).bounds(k)
