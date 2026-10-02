# Data sources and private input layout

Expression matrices and sample-level records are not included in this code update.
The CLI in `code/analyze.py` accepts your own NPZ file as documented in the README.

Breast inputs derive from the processed Tekpli et al. collection:
*An independent poor-prognosis subtype of breast cancer defined by a distinct
tumor immune microenvironment*, Nature Communications 10, 5499 (2019),
[doi:10.1038/s41467-019-13329-5](https://doi.org/10.1038/s41467-019-13329-5).

| Retained cohort | Source accession |
|---|---|
| MicMa | GSE19536 |
| STK | GSE1456 |
| NeoAva | E-MTAB-4439 |
| UPP | GSE3494 |
| TAI | GSE20685 |
| OSLO2-EMIT0 | GSE135298 |
| METABRIC | EGAD00010000210 |
| Oslo2 | GSE58215 |
| UPSA | GSE4922 |
| MAINZ | GSE11121 |
| STAM | SUPERTAM aggregate |

STAM includes MDA5, TAM, VDX and VDX3 components. The standalone VDX input is
excluded because its identifiers all occur in STAM. TCGA is excluded. Six
METABRIC samples without PAM50 annotations are excluded. The contrast is Luminal A
versus the other annotated PAM50 subtypes.

Lung inputs derive from GEO GSE19188 (working cohort), GSE14814, GSE30219,
GSE37745, GSE50081 and GSE8894. The contrast is adenocarcinoma versus the remaining
samples, which are heterogeneous. The main family is the 5,000 highest-variance
probes among 22,215 common probes; the IQR sensitivity uses a separately frozen
5,000-probe family. Neither filter uses labels.

The code preserves the supplied processing; it does not independently recreate
normalization, gene mapping or PAM50 classification. Source accessions alone do
not guarantee reconstruction of the exact processed matrices used in the paper.
Observe each source's access and reuse terms.

## Optional frozen manuscript archive

Set `RDEG_ANALYSIS_ROOT` to an existing local archive containing:

- `data/breast/`: `cohort_data.npz`, reference `gt_*.npz` files and
  `frozen_common_mask.npy`.
- `data/lung/`, `data/lung_iqr/`, `data/lung_prefilter/`: processed cohort NPZ files.
- `results/`: frozen numeric outputs used by comparison and plotting scripts.
- `manuscript/`: source files read by the figure reproduction script.

The archive NPZ schema is `x` (float samples × features), `group` (Boolean labels),
`genes` (feature names), and `samples` (source sample identifiers for local audits).
All arrays must load with `allow_pickle=False`. The general CLI requires only `x`
and `group`; optional `genes` must contain one unique name per column.

The private runner expects the original archive's filenames and feature order.
It verifies calculations against frozen outputs; it is not a data downloader.
Keep this archive outside the Git checkout. Do not upload its expression
matrices, sample-level inputs or embedded historical copies with a code update.
