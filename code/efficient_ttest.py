"""Exact labeling enumeration with moment updates; pooled-t extreme candidates."""
from itertools import combinations
import warnings
import numpy as np
from scipy.special import stdtr
from scipy.stats import ttest_ind


class Moments:
    def __init__(self, x, group):
        x = np.asarray(x, float)
        self.g = np.asarray(group, bool)
        self.n, self.features = x.shape
        self.m = int(self.g.sum())
        self.valid = np.isfinite(x).all(axis=0)
        clean = np.where(self.valid[None, :], x, 0)
        self.raw = clean
        self.valid &= np.ptp(clean, axis=0) > 0
        self.x = clean - clean.mean(axis=0)
        self.x[:, ~self.valid] = 0
        self.sa = self.x[self.g].sum(axis=0)
        self.sb = self.x[~self.g].sum(axis=0)
        self.qa = (self.x[self.g]**2).sum(axis=0)
        self.qb = (self.x[~self.g]**2).sum(axis=0)
        self.total = self.sa + self.sb
        self.total_q = self.qa + self.qb
        self.delta = 1 - 2*self.g.astype(int)

    def p(self, sa, sb, qa, qb, m, equal_var=False, flipped=()):
        b = self.n - m
        assert min(m, b) >= 2
        ss_a, ss_b = qa - sa**2/m, qb - sb**2/b
        tol = 2e-12*np.maximum(self.total_q, np.finfo(float).tiny)
        if np.any(ss_a < -tol) or np.any(ss_b < -tol):
            raise FloatingPointError('Moment subtraction lost excessive precision')
        ss_a, ss_b = np.maximum(ss_a, 0), np.maximum(ss_b, 0)
        difference = sa/m - sb/b
        if equal_var:
            se2 = (ss_a + ss_b)/(self.n - 2)*(1/m + 1/b)
            df = np.full(self.features, self.n - 2.)
        else:
            va, vb = ss_a/(m*(m-1)), ss_b/(b*(b-1))
            se2 = va + vb
            den = va**2/(m-1) + vb**2/(b-1)
            df = np.divide(se2**2, den, out=np.ones_like(den), where=den > 0)
        t = np.divide(difference, np.sqrt(se2), out=np.zeros_like(difference), where=se2 > 0)
        if equal_var:
            # Continuous limits for nonconstant features partitioned into two constant groups.
            t[(se2 == 0) & (difference > 0)] = np.inf
            t[(se2 == 0) & (difference < 0)] = -np.inf
            valid = self.valid
        else:
            # Match the earlier Welch run's conservative unavailable-test policy.
            valid = self.valid & (se2 > 0) & np.isfinite(df) & (df > 0)
        p = np.stack((stdtr(df, -t), stdtr(df, t)))
        p[:, ~valid] = 1
        # Nearly constant groups make Q-S²/n ill-conditioned. Recompute only
        # affected features from observations, preserving the original SciPy
        # statistic and its unavailable-test policy exactly where possible.
        eps = 64*np.finfo(float).eps
        unstable = self.valid & ((ss_a <= eps*(np.abs(qa)+sa**2/m)) |
                                 (ss_b <= eps*(np.abs(qb)+sb**2/b)))
        if unstable.any():
            g = self.g.copy()
            g[list(flipped)] = ~g[list(flipped)]
            xx = self.raw[:, unstable]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                fit = ttest_ind(xx[g], xx[~g], axis=0, equal_var=equal_var, alternative='greater')
            pp = np.stack((stdtr(fit.df, -fit.statistic), stdtr(fit.df, fit.statistic)))
            unavailable = np.isnan(fit.statistic) if equal_var else ~np.isfinite(fit.statistic)
            pp[:, unavailable] = 1
            p[:, unstable] = pp
        assert np.isfinite(p).all()
        return p

    def relabelings(self, k, equal_var=False):
        assert 0 <= k <= min(self.m, self.n-self.m)-2
        yield (), self.p(self.sa, self.sb, self.qa, self.qb, self.m, equal_var)
        for distance in range(1, k+1):
            for ix in combinations(range(self.n), distance):
                ii = np.asarray(ix)
                change = (self.x[ii]*self.delta[ii, None]).sum(axis=0)
                change_q = (self.x[ii]**2*self.delta[ii, None]).sum(axis=0)
                m = self.m + self.delta[ii].sum()
                yield ix, self.p(self.sa+change, self.sb-change,
                                 self.qa+change_q, self.qb-change_q, m, equal_var, ix)

    def pooled_from_sum(self, sa, m):
        b = self.n - m
        sb = self.total - sa
        within = self.total_q - sa**2/m - sb**2/b
        tol = 2e-12*np.maximum(self.total_q, np.finfo(float).tiny)
        if np.any(within < -tol):
            raise FloatingPointError('Negative within-group sum of squares')
        within = np.maximum(within, 0)
        diff = sa/m - sb/b
        se = np.sqrt(within/(self.n-2)*(1/m+1/b))
        t = np.divide(diff, se, out=np.zeros_like(diff), where=se > 0)
        t[(se == 0) & (diff > 0)] = np.inf
        t[(se == 0) & (diff < 0)] = -np.inf
        p = np.stack((stdtr(self.n-2, -t), stdtr(self.n-2, t)))
        p[:, ~self.valid] = 1
        return p

    def pooled_candidates(self, k):
        assert 0 <= k <= min(self.m, self.n-self.m)-2
        a = np.sort(self.x[self.g], axis=0)
        b = np.sort(self.x[~self.g], axis=0)
        zero = np.zeros((1, self.features))
        def prefix(z):
            return np.concatenate((zero, np.cumsum(z[:k], axis=0)))
        amin, amax, bmin, bmax = prefix(a), prefix(a[::-1]), prefix(b), prefix(b[::-1])
        lo = np.ones((2, self.features))
        hi = np.zeros_like(lo)
        for i in range(k+1):
            for j in range(k+1-i):
                m = self.m-i+j
                for sa in [self.sa-amax[i]+bmin[j], self.sa-amin[i]+bmax[j]]:
                    p = self.pooled_from_sum(sa, m)
                    lo, hi = np.minimum(lo, p), np.maximum(hi, p)
        return lo, hi
