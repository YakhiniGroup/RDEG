# Current one-sided RDEG implementation

Companion code for *Efficient Differential Expression Analysis under Label
Uncertainty*, Ben Galili and Zohar Yakhini. This directory contains code and
documentation only. The independent synthetic checks run without research data.

## Installation and quick start

From the repository root, use Python 3.11 (tested with 3.11.4):

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r current/requirements.txt
python current/code/verify.py
python current/code/demo.py
python -m unittest discover -s current/tests -v
python current/code/validate_linear.py --small-only --out /tmp/rdeg-validation
```

The demo constructs synthetic observations in memory and checks
ERDEG ⊆ RDEG ⊆ ordinary BH. It is a usage example, not a manuscript experiment.
Validation checks the sum sweep exhaustively, compares endpoints with direct
SciPy enumeration, checks exact WRS permutation candidates on small tied and
untied examples, and checks batching and invariance properties.

## Analyze your own data

Prepare a non-pickled NumPy NPZ file with `x` (numeric samples × features),
`group` (one binary label per row; 1/True is the target group), and optionally
`genes` (unique string feature names). Do not include identifier columns in `x`.
The code does not require or output sample identifiers. Hold preprocessing and
the feature family fixed across label perturbations.

```sh
python current/code/analyze.py /path/to/input.npz --test WRS --k 1 --tau 0.01 --direction joint --out /tmp/wrs.csv
python current/code/analyze.py /path/to/input.npz --test Pooled --k 1 --tau 0.01 --direction joint --exact --out /tmp/pooled.csv
```

`joint` adjusts the fixed greater and less hypotheses together as 2N tests;
`greater` and `less` adjust N tests separately. Greater means higher values in the
target group. Directions remain fixed under every label change; do not choose a
direction from the observed effect and treat it as prespecified. The CLI default
0.01 is a usage default; the primary breast analysis uses 0.001 and lung uses 0.01.

Outputs contain raw original p-values, lower/upper stability bounds, original
BH-adjusted p-values, BH-adjusted upper bounds, and BH/ERDEG membership. `--exact`
adds the maximum labeling-specific BH-adjusted p-value and RDEG membership.
Its cost grows combinatorially with the number of allowed labelings.

## Python interface

From the repository root, run Python with `PYTHONPATH=current/code`:

```python
import numpy as np
from linear_intervals import LinearIntervals
from analysis_core import bh_adjusted

rng = np.random.default_rng(1)
x = rng.normal(size=(12, 20))
group = np.arange(12) < 6
intervals = LinearIntervals(x, group, test="WRS")
p_L, p_U = intervals.bounds(k=1)
q_U = bh_adjusted(p_U.reshape(-1))  # Joint directional family
selected = q_U <= 0.01
```

Bounds have shape `(2, N)`: row 0 is greater, row 1 is less. Flattening uses all
greater hypotheses followed by all less hypotheses. `bounds(0)` gives original
p-values. Reuse the prepared object for multiple budgets.

## Statistical and computational scope

- WRS uses pooled average ranks and the tie-corrected normal approximation,
  without continuity correction. Exact optimization of this approximation is
  distinct from an exact permutation p-value.
- Pooled Student uses equal variances and n−2 degrees of freedom. Welch is an
  exhaustive supplementary comparator, not a linear candidate algorithm.
- WRS requires k < min(group sizes); pooled Student requires
  k ≤ min(group sizes)−2. The ball of at most k flips includes the original
  labeling. The label-flip budget is supplied by the user.
- Constant features and features containing nonfinite observations remain in
  the full family with p=1. Nonconstant zero-within-group-variance partitions use
  signed infinite-t limits for pooled Student.
- ERDEG applies BH once to featurewise maximum raw p-values. Exact RDEG
  intersects BH selections across common labelings. Applying BH after exhaustive
  raw-p maximization still gives ERDEG, not exact RDEG.
- The linear search takes O(k) arithmetic per feature after preparation;
  ordinary sorting gives total O(n log n+k) per feature. This does not make
  exact RDEG linear or guarantee faster wall time at every budget.
- FDR control also requires test calibration and suitable dependence assumptions.
  Label-free filtering alone does not prove these conditions. Cross-cohort
  reference agreement is a reproducibility proxy, not biological ground truth.

See [`code/THEORY.md`](code/THEORY.md) and
[`code/PROTOCOL.md`](code/PROTOCOL.md) for the derivation and validation protocol.

## Manuscript reproduction with separately supplied inputs

The research matrices, frozen comparison outputs and manuscript source are not
included. Complete manuscript checks require the private archive layout in
[DATA.md](DATA.md); fresh downloads may differ in processing or release.
No script silently downloads datasets.

```sh
python current/code/verify.py --full --inputs /path/to/private/archive --out /tmp/rdeg-full
```

Individual scripts accept the input root through `RDEG_ANALYSIS_ROOT`:

```sh
export RDEG_ANALYSIS_ROOT=/path/to/private/archive
python current/code/reproduce.py --out /tmp/rdeg-primary
python current/code/reproduce.py --all --out /tmp/rdeg-all
python current/code/validate_linear.py --out /tmp/rdeg-algorithm
python current/code/plot_results.py --out /tmp/rdeg-figures
python current/code/plot_timings.py --out /tmp/rdeg-timing-figures
```

`--all` includes breast threshold and common-family sensitivities, rotating WRS
working cohorts, cross-test references, lung IQR filtering, and exact breast WRS
RDEG at k=2. Excluded DESeq2 and kidney explorations are not part of this runner.

The paper's timing grid can be rerun with the private inputs:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python current/code/benchmark.py --out /tmp/rdeg-timings
```

Times depend on the machine. This measures interval endpoints, excluding BH and
exact RDEG. Plotting archived observations and running a new timing experiment
are separate operations.

## Contents

| File | Purpose |
|---|---|
| `code/linear_intervals.py` | Linear-budget endpoint search |
| `code/analysis_core.py` | WRS, BH and selection utilities |
| `code/efficient_ttest.py` | Pooled candidates and pooled/Welch enumeration |
| `code/reference/` | Frozen baselines for independent comparisons |
| `code/analyze.py`, `code/demo.py` | User-data CLI and synthetic example |
| `code/validate_linear.py` | Independent algorithm checks |
| `code/reproduce.py` | Manuscript analysis reproduction with private inputs |
| `code/benchmark.py` | Endpoint timing benchmark |
| `code/plot_results.py`, `code/plot_timings.py` | Figure reproduction |
| `code/verify.py` | Integrity and optional full reproduction |

The historical notebook and utility file are separate implementations and are
not dependencies of this directory. No new license is assigned by this update;
existing applicable rights remain unchanged.
