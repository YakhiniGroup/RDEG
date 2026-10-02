# RDEG

Code for **Efficient Differential Expression Analysis under Label Uncertainty**,
by Ben Galili and Zohar Yakhini.

The current implementation is in [`current/`](current/README.md). It provides:

- Stability intervals for fixed one-sided Wilcoxon rank-sum (WRS) and pooled
  Student tests, using a linear search in the label-flip budget after sorting.
- Ordinary Benjamini–Hochberg (BH) selection, ERDEG from adjusted worst-case
  raw p-values, and exhaustive exact RDEG calculation.
- Independent validation, timing code, manuscript-analysis scripts, a runnable
  synthetic example, and a command-line interface for user-supplied data.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r current/requirements.txt
python current/code/demo.py
python current/code/validate_linear.py --small-only --out /tmp/rdeg-validation
```

This update does not distribute expression matrices, sample-level inputs, the
manuscript, or the private reproducibility archive. See
[`current/DATA.md`](current/DATA.md) for source datasets and input requirements.
The repository contains the code for the current manuscript version. Earlier
implementations and datasets remain available in Git history.

The robustness criterion determines which original BH discoveries to retain and
how many. A robustness certificate is not, by itself, an unconditional FDR
guarantee. See the [method and usage notes](current/README.md).
