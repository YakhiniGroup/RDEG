"""Small synthetic example, independent of the manuscript expression data."""
import numpy as np
from analyze import analyze


def main():
    rng = np.random.default_rng(20260925)
    group = np.arange(12) < 6
    x = rng.normal(size=(12, 8))
    x[group, :2] += 4
    x[group, 2:4] -= 4
    genes = np.array([f'synthetic_{i}' for i in range(x.shape[1])])
    for test in ('WRS', 'Pooled'):
        result = analyze(x, group, genes, test=test, k=1, tau=0.05, exact=True)
        assert not (result.ERDEG & ~result.RDEG).any()
        assert not (result.RDEG & ~result.BH).any()
        print(f'\n{test}: fixed greater and less directions, joint BH, k=1')
        print(result[['hypothesis', 'BH_adjusted_p', 'BH_adjusted_p_U',
                      'BH', 'ERDEG', 'RDEG']].to_string(index=False))


if __name__ == '__main__':
    main()
