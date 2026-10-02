"""Check the public CLI against direct SciPy tests on synthetic observations."""
from itertools import combinations
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu, ttest_ind

CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE))
from analyze import analyze


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(43).normal(size=(10, 5))
        self.g = np.arange(10) < 5
        self.x[self.g, 0] += 3
        self.genes = np.array([f'g{i}' for i in range(5)])

    def test_direct_enumeration(self):
        for test in ('WRS', 'Pooled'):
            for k in (0, 1, 2):
                raw = []
                for distance in range(k + 1):
                    for flipped in combinations(range(10), distance):
                        g = self.g.copy()
                        g[list(flipped)] = ~g[list(flipped)]
                        with warnings.catch_warnings():
                            warnings.simplefilter('error', RuntimeWarning)
                            if test == 'WRS':
                                p = [mannwhitneyu(self.x[g], self.x[~g], axis=0,
                                     alternative=d, method='asymptotic',
                                     use_continuity=False).pvalue for d in ('greater', 'less')]
                            else:
                                p = [ttest_ind(self.x[g], self.x[~g], axis=0,
                                     alternative=d, equal_var=True).pvalue for d in ('greater', 'less')]
                        raw.append(p)
                raw = np.asarray(raw)
                for direction in ('greater', 'less', 'joint'):
                    v = (raw.reshape(len(raw), -1) if direction == 'joint' else
                         raw[:, 0 if direction == 'greater' else 1])
                    result = analyze(self.x, self.g, self.genes, test=test, k=k,
                                     direction=direction, tau=0.05, exact=True)
                    np.testing.assert_allclose(result.p_U, v.max(axis=0), atol=5e-13)
                    np.testing.assert_allclose(result.p_L, v.min(axis=0), atol=5e-13)
                    worst_q = np.array([false_discovery_control(p) for p in v]).max(axis=0)
                    np.testing.assert_allclose(result.max_BH_adjusted_p, worst_q, atol=5e-13)
                    np.testing.assert_array_equal(result.RDEG, worst_q <= 0.05)
                    np.testing.assert_array_equal(result.ERDEG,
                        false_discovery_control(v.max(axis=0)) <= 0.05)

    def test_unavailable_features_and_invalid_inputs(self):
        x = self.x.copy()
        x[:, 0] = 1
        x[0, 1] = np.nan
        for test in ('WRS', 'Pooled'):
            result = analyze(x, self.g, self.genes, test=test, exact=True)
            rows = result[result.hypothesis.str.startswith(('g0|', 'g1|'))]
            self.assertTrue((rows[['p', 'p_L', 'p_U', 'BH_adjusted_p_U']] == 1).all().all())
        for kwargs in ({'tau': 0}, {'tau': float('nan')}, {'k': 5}, {'direction': 'two-sided'}):
            with self.assertRaises(ValueError):
                analyze(self.x, self.g, self.genes, **kwargs)

    def test_cli_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'input.npz', Path(tmp) / 'output.csv'
            np.savez(src, x=self.x, group=self.g, genes=self.genes,
                     samples=np.array([f'local_sample_{i}' for i in range(10)]))
            subprocess.run([sys.executable, str(CODE / 'analyze.py'), str(src),
                            '--test', 'Pooled', '--exact', '--out', str(dst)],
                           check=True, capture_output=True, text=True)
            result = pd.read_csv(dst)
            self.assertEqual(len(result), 10)
            self.assertNotIn('local_sample', dst.read_text())
            self.assertTrue({'p_U', 'BH_adjusted_p_U', 'ERDEG', 'RDEG'} <= set(result.columns))


if __name__ == '__main__':
    unittest.main()
