"""Transparent one-sided WRS, BH, and exhaustive robustness calculations."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import rankdata

MODES = ('greater', 'less', 'joint')


def bh_adjusted(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('BH requires a complete, finite vector of probabilities')
    n = len(p)
    o = np.argsort(p, kind='stable')
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    ans = np.empty(n)
    ans[o] = np.minimum(q, 1)
    return ans


def vectors(p):
    return {'greater': p[0], 'less': p[1], 'joint': p.reshape(-1)}


def identifiers(genes, mode):
    return np.array([f'{g}|{d}' for d in ('greater', 'less') for g in genes]) if mode == 'joint' else np.array(genes, dtype=str)


class WRS:
    def __init__(self, X, group):
        self.group = np.asarray(group, bool)
        self.n, self.N = X.shape
        self.m = int(self.group.sum())
        assert 0 < self.m < self.n
        self.valid = np.isfinite(X).all(axis=0)
        clean = np.where(self.valid[None, :], X, 0)
        self.R = rankdata(clean, axis=0, method='average')
        self.C = np.sum((self.R - (self.n + 1) / 2)**2, axis=0) / (self.n * (self.n - 1))
        self.valid &= self.C > 0
        self.W = self.R[self.group].sum(axis=0)
        self.p = self.p_from_sum(self.W, self.m)

    def z(self, W, m):
        den = np.sqrt(m * (self.n - m) * self.C)
        return np.divide(W - m * (self.n + 1) / 2, den, out=np.zeros_like(W, dtype=float), where=den > 0)

    def p_from_sum(self, W, m):
        z = self.z(W, m)
        p = np.stack((ndtr(-z), ndtr(z)))
        p[:, ~self.valid] = 1
        return p

    def candidates(self, k):
        assert 0 <= k < min(self.m, self.n - self.m)
        A = np.sort(self.R[self.group], axis=0)
        B = np.sort(self.R[~self.group], axis=0)
        zero = np.zeros((1, self.N))
        amin = np.concatenate((zero, np.cumsum(A[:k], axis=0)))
        amax = np.concatenate((zero, np.cumsum(A[::-1][:k], axis=0)))
        bmin = np.concatenate((zero, np.cumsum(B[:k], axis=0)))
        bmax = np.concatenate((zero, np.cumsum(B[::-1][:k], axis=0)))
        low = self.z(self.W, self.m)
        high = low.copy()
        for i in range(k + 1):
            for j in range(k + 1 - i):
                m = self.m - i + j
                low = np.minimum(low, self.z(self.W - amax[i] + bmin[j], m))
                high = np.maximum(high, self.z(self.W - amin[i] + bmax[j], m))
        pmax = np.stack((ndtr(-low), ndtr(high)))
        pmin = np.stack((ndtr(-high), ndtr(low)))
        pmax[:, ~self.valid] = 1
        pmin[:, ~self.valid] = 1
        return pmin, pmax

    def relabelings(self, k):
        yield (), self.p
        delta = 1 - 2 * self.group.astype(int)
        for d in range(1, k + 1):
            for idx in combinations(range(self.n), d):
                ii = np.asarray(idx)
                W = self.W + (self.R[ii] * delta[ii, None]).sum(axis=0)
                m = self.m + delta[ii].sum()
                yield idx, self.p_from_sum(W, m)


def initialize_robust(p, pmax, genes, tau):
    state = {}
    for mode, pv in vectors(p).items():
        q0 = bh_adjusted(pv)
        qu = bh_adjusted(vectors(pmax)[mode])
        ids = identifiers(genes, mode)
        order = np.lexsort((ids, pv))
        E = qu <= tau
        matched = np.zeros(len(pv), bool)
        matched[order[:E.sum()]] = True
        state[mode] = dict(q0=q0, qu=qu, rob=q0.copy(), E=E, matched=matched,
                           order=order, losses=[], ids=ids)
    return state


def update_robust(state, p, tau, label):
    for mode, pv in vectors(p).items():
        s = state[mode]
        q = bh_adjusted(pv)
        s['rob'] = np.maximum(s['rob'], q)
        selected = q <= tau
        s['losses'].append(dict(relabeling=label, distance=0 if label == 'original' else label.count(',') + 1,
                                selected=int(selected.sum()),
                                lost_standard=int(np.count_nonzero((s['q0'] <= tau) & ~selected)),
                                lost_matched_ERDEG=int(np.count_nonzero(s['matched'] & ~selected)),
                                lost_ERDEG=int(np.count_nonzero(s['E'] & ~selected))))


def describe_sets(state, p, pmax, genes, reference, tau, cohort, k, out):
    rows = []
    for mode, pv in vectors(p).items():
        s = state[mode]
        q0, qu, rob = s['q0'], s['qu'], s['rob']
        assert np.all(q0 <= rob + 2e-14)
        assert np.all(rob <= qu + 2e-14)
        V, E, R = q0 <= tau, qu <= tau, rob <= tau
        assert not np.any(E & ~R) and not np.any(R & ~V)
        M_E = s['matched']
        M_R = np.zeros(len(pv), bool); M_R[s['order'][:R.sum()]] = True
        ref = reference[mode]
        masks = {'BH': V, 'ERDEG': E, 'RDEG': R, 'matched_ERDEG': M_E, 'matched_RDEG': M_R}
        details = pd.DataFrame({'hypothesis': s['ids'], 'p': pv, 'BH_adjusted_p': q0,
                                'p_U': vectors(pmax)[mode], 'BH_adjusted_p_U': qu,
                                'max_BH_adjusted_p': rob, 'consensus_reference': ref})
        for method, selected in masks.items():
            details[method] = selected
            size, overlap = int(selected.sum()), int((selected & ref).sum())
            rows.append(dict(cohort=cohort, direction=mode, k=k, tau=tau, family_size=len(pv),
                             reference_size=int(ref.sum()), method=method, selected=size,
                             reference_overlap=overlap, reference_nonoverlap=size-overlap,
                             precision_proxy=overlap/size if size and ref.sum() else None,
                             recall_proxy=overlap/ref.sum() if ref.sum() else None,
                             overlap_with_matched=int((selected & (M_R if method == 'RDEG' else M_E)).sum())
                             if method in ('RDEG', 'ERDEG') else None))
        details.to_csv(out / f'{cohort}_{mode}_k{k}_features.csv.gz', index=False)
        pd.DataFrame(s['losses']).to_csv(out / f'{cohort}_{mode}_k{k}_relabelings.csv.gz', index=False)
    return rows
