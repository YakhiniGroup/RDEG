"""Compute fixed one-sided BH and ERDEG selections for a user-supplied matrix.

Input NPZ: x (samples x features), group (binary target-group indicator),
and optional genes (unique feature names). No sample identifiers are output.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_core import WRS, bh_adjusted, identifiers, vectors
from efficient_ttest import Moments
from linear_intervals import LinearIntervals


def analyze(x, group, genes, *, test='WRS', k=1, tau=0.01,
            direction='joint', exact=False):
    if direction not in ('greater', 'less', 'joint'):
        raise ValueError('Direction must be greater, less or joint')
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError('BH level must be between zero and one')
    obj = LinearIntervals(x, group, test)
    genes = np.asarray(genes, dtype=str)
    if genes.ndim != 1 or len(genes) != obj.N or len(set(genes)) != obj.N:
        raise ValueError('Provide one unique feature name per matrix column')
    if obj.N == 0:
        raise ValueError('The matrix must contain at least one feature')
    _, original = obj.bounds(0)
    lower, upper = obj.bounds(k)
    p = vectors(original)[direction]
    q = bh_adjusted(p)
    q_upper = bh_adjusted(vectors(upper)[direction])
    result = pd.DataFrame({
        'hypothesis': identifiers(genes, direction),
        'p': p,
        'p_L': vectors(lower)[direction],
        'p_U': vectors(upper)[direction],
        'BH_adjusted_p': q,
        'BH_adjusted_p_U': q_upper,
        'BH': q <= tau,
        'ERDEG': q_upper <= tau,
    })
    if exact:
        g = np.asarray(group, dtype=bool)
        enumerated = (WRS(np.asarray(x, dtype=float), g).relabelings(k)
                      if test == 'WRS' else
                      Moments(x, g).relabelings(k, equal_var=True))
        worst_q = np.zeros_like(q)
        for _, values in enumerated:
            worst_q = np.maximum(worst_q, bh_adjusted(vectors(values)[direction]))
        result['max_BH_adjusted_p'] = worst_q
        result['RDEG'] = worst_q <= tau
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--test', choices=['WRS', 'Pooled'], default='WRS')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--direction', choices=['greater', 'less', 'joint'], default='joint')
    parser.add_argument('--exact', action='store_true',
                        help='also enumerate all common labelings for exact RDEG (combinatorial cost)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.out.resolve():
        parser.error('Output must differ from input')
    with np.load(args.input, allow_pickle=False) as data:
        x, group = data['x'], data['group']
        if x.ndim != 2:
            parser.error('x must be a samples-by-features matrix')
        genes = (data['genes'] if 'genes' in data else
                 np.array([f'feature_{j}' for j in range(x.shape[1])]))
        result = analyze(x, group, genes, test=args.test, k=args.k, tau=args.tau,
                         direction=args.direction, exact=args.exact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    methods = ['BH', 'ERDEG'] + (['RDEG'] if args.exact else [])
    print(', '.join(f'{method}: {int(result[method].sum())}' for method in methods))
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
