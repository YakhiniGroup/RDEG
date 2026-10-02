"""Measured endpoint scaling; produces a review proposal, never edits the paper."""
from pathlib import Path
from itertools import combinations, islice
import argparse
import hashlib
import json
import math
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr, stdtr
from scipy.stats import mannwhitneyu, ttest_ind

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from analysis_core import WRS
from efficient_ttest import Moments

BATCH = 64
REPEATS = 3
SEED = 20260923


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounds_from_statistics(low, high, valid, test, n):
    cdf = ndtr if test == 'WRS' else lambda t: stdtr(n - 2, t)
    pmin = np.stack((cdf(-high), cdf(low)))
    pmax = np.stack((cdf(-low), cdf(high)))
    pmin[:, ~valid] = 1
    pmax[:, ~valid] = 1
    return pmin, pmax


def candidate(x, g, k, test):
    obj = WRS(x, g) if test == 'WRS' else Moments(x, g)
    return obj.candidates(k) if test == 'WRS' else obj.pooled_candidates(k)


def exhaustive(x, g, k, test):
    """Visit every labeling; avoid computing identical tail maps repeatedly."""
    n, features = x.shape
    obj = WRS(x, g) if test == 'WRS' else Moments(x, g)
    assert 0 <= k <= min(g.sum(), (~g).sum()) - 2
    delta = 1 - 2 * g.astype(int)
    values = obj.R if test == 'WRS' else obj.x
    original_sum = obj.W if test == 'WRS' else obj.sa
    m0 = int(g.sum())

    def stat(s, m):
        m = np.asarray(m)[:, None]
        if test == 'WRS':
            den = np.sqrt(m * (n - m) * obj.C)
            return np.divide(s - m * (n + 1) / 2, den,
                             out=np.zeros_like(s), where=den > 0)
        b = n - m
        sb = obj.total - s
        within = obj.total_q - s * s / m - sb * sb / b
        tolerance = 2e-12 * np.maximum(obj.total_q, np.finfo(float).tiny)
        if np.any(within < -tolerance):
            raise FloatingPointError('Excessive moment cancellation')
        within = np.maximum(within, 0)
        difference = s / m - sb / b
        se = np.sqrt(within / (n - 2) * (1 / m + 1 / b))
        t = np.divide(difference, se, out=np.zeros_like(s), where=se > 0)
        t[(se == 0) & (difference > 0)] = np.inf
        t[(se == 0) & (difference < 0)] = -np.inf
        return t

    low = stat(original_sum[None, :], [m0])[0]
    high = low.copy()
    count = 1
    for d in range(1, k + 1):
        iterator = combinations(range(n), d)
        while True:
            block = list(islice(iterator, BATCH))
            if not block:
                break
            ii = np.asarray(block, dtype=int)
            sums = np.broadcast_to(original_sum, (len(ii), features)).copy()
            sizes = np.full(len(ii), m0, dtype=int)
            for j in range(d):
                change = delta[ii[:, j]]
                sums += values[ii[:, j]] * change[:, None]
                sizes += change
            ts = stat(sums, sizes)
            low = np.minimum(low, ts.min(axis=0))
            high = np.maximum(high, ts.max(axis=0))
            count += len(ii)
    assert count == sum(math.comb(n, d) for d in range(k + 1))
    return bounds_from_statistics(low, high, obj.valid, test, n)


def compare(a, b):
    for aa, bb in zip(a, b):
        assert np.isfinite(aa).all() and np.isfinite(bb).all()
        np.testing.assert_allclose(aa, bb, atol=5e-13, rtol=3e-10)
    return max(float(np.max(np.abs(aa - bb))) for aa, bb in zip(a, b))


def direct_bounds(x, g, k, test):
    n, features = x.shape
    valid = np.isfinite(x).all(axis=0)
    xx = np.where(valid[None, :], x, 0)
    valid &= np.ptp(xx, axis=0) > 0
    low, high = np.ones((2, features)), np.zeros((2, features))
    count = 0
    for d in range(k + 1):
        for ids in combinations(range(n), d):
            gg = g.copy()
            gg[list(ids)] = ~gg[list(ids)]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                if test == 'WRS':
                    p = np.stack([mannwhitneyu(xx[gg], xx[~gg], axis=0,
                        alternative=tail, method='asymptotic', use_continuity=False).pvalue
                        for tail in ('greater', 'less')])
                else:
                    p = np.stack([ttest_ind(xx[gg], xx[~gg], axis=0,
                        equal_var=True, alternative=tail).pvalue
                        for tail in ('greater', 'less')])
            p[:, ~valid] = 1
            assert np.isfinite(p).all()
            low, high = np.minimum(low, p), np.maximum(high, p)
            count += 1
    return (low, high), count
