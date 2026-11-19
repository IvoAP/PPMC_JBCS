# Clinical Privacy and Explainability Framework

Research infrastructure for the **Diabetes 130-US Hospitals for Years
1999-2008** dataset. The preprocessing stage builds patient-disjoint partitions
for predicting readmission within 30 days. The anonymization stage exports
empirical PerturbPCA-alpha comparisons. Other research stages remain placeholders.

## Install and run

From the repository root, using Python 3.12 or newer and PowerShell:

```powershell
cd clinical_privacy_study
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Place the original diabetic_data.csv in data\raw before running preprocessing.
python main.py --experiment preprocessing
# Replace the placeholder with the ID printed by preprocessing.
python main.py --experiment anonymization --source-run "run_<32 hex digits>" --clusters 3
python main.py --experiment classification
python main.py --experiment all
```

On Unix-like systems activate the environment with `source .venv/bin/activate`.
Run `python main.py --help` to list
the other stages: `feature_selection`, `shap_analysis`, `llm_evaluation`,
`privacy_attacks`, `statistical_analysis`, and `reporting`.

A named stage runs alone. `all` runs stages in the order declared in
`main.py`. Implemented stages report `completed`; placeholders report
`not_implemented`.

## Preprocessing policy and artifacts

The default `--prediction-time admission` uses age band, gender, race, admission
type and source, and historical inpatient, outpatient and emergency visit counts.
Numeric-looking clinical codes are categories. `--prediction-time discharge`
also permits encounter duration, procedure and medication counts, and two lab
result categories. Both policies exclude discharge disposition (an outcome
proxy), identifiers, and any unreviewed columns including final diagnoses.
These are availability assumptions, not timestamp-verified observations.
The evaluation holds out patients, not future calendar periods or hospitals.

Blank and `?` values and column-specific unknown codes are treated as missing.
The literal lab category `None` means no test was recorded and is retained.
Exact duplicate records collapse; conflicting encounter IDs and invalid labels
fail validation. After the patient split, train-only medians and standard
scaling handle numeric features (all-missing training columns impute zero).
Categorical nulls use a missing category; categories unseen in training
produce an all-zero block at transform time.

Artifacts share the run ID printed by the CLI:

| Output | Contents |
| --- | --- |
| `data/processed/<run_id>/` | Per-partition CSR feature matrices (`scipy.sparse.load_npz`), binary target vectors (`numpy.load`), ordered feature provenance, preprocessing config, fitted column transformer |
| `data/splits/<run_id>/` | Encounter and patient identifiers in the exact row order of each matrix and target vector; read CSV identifiers as strings |
| `results/tables/<run_id>/data_quality_report.json` | Missingness before imputation, record and patient counts, target counts, feature eligibility decisions and pairwise patient overlap |

Keep row-level artifacts in a protected workspace; encoding is not
anonymization. Load joblib files only from trusted sources.

## PerturbPCA-alpha integration

Standalone anonymization requires `--source-run` so it never guesses which
split or prediction-time policy to use. `--experiment all` consumes the
preprocessing artifacts generated in the same run. Both export the original
comparator and global transformations at alpha 0, 0.25, 0.50, 0.75 and 1.
Pass `--clusters K` to additionally evaluate each alpha with KMeans; `3`
above is an example, not a recommended clinical parameter.

The author integration point is in [perturbpca.py](./anonymization/perturbpca.py).
The supplied calculations are already wired in: `_fit_basis` stores the
training mean, sorted eigenvectors and interpolated reconstruction basis;
`_apply_basis` applies the original projection and reconstruction. To insert
the canonical implementation, put its fit-only and apply-only operations at
these hooks. Do not call a whole-dataset fitting routine on a holdout.
Clustered mode uses frozen training centroids to route new rows and scatters
results back into input order (instead of reordering labels by cluster).
A cluster with fewer than two training rows fails rather than falling back.

`results/anonymized/<run_id>/` contains `original/`, `global/alpha_025/`,
etc., and optionally `clustered/alpha_025/`, etc. Each variant contains
three dense encoded `.npy` feature matrices, runtime and parameter metadata,
and per-partition reconstruction diagnostics. Metadata references the
original immutable-in-use schema, targets, and encounter mappings: retain the
source preprocessing run. The private `.joblib` object contains exact fitted
parameters and may contain training-derived information. Do not release it,
identifiers, or the original comparator as anonymized data.

No categorical repair is part of the supplied method. Diagnostics report
one-hot violations (with a tolerance for floating-point error), accepting an
all-zero block only where the input block represented an unknown category.
Nonzero-alpha outputs can be continuous embeddings rather than valid clinical
categories; do not decode them as patient records. Alpha zero is a numerical
identity control. Alpha is **not** a privacy budget: no epsilon, delta or
formal differential privacy guarantee is asserted.

## Configuration and reproducibility

Edit the typed dataclasses in [config.py](./config.py). `StudyConfig` groups
paths, dataset column roles and split proportions, seed, and settings for each
research area. Test and validation fractions refer to **unique patients**;
record proportions can differ. The target is `1` for `<30` and `0` for
`>30` or `NO`. Labels are not used in splitting, imputation, scaling or encoding.

By default, paths resolve relative to the project, not the launching shell.
Use `--workspace PATH` to put the entire data and result tree elsewhere.
Each run writes `results/logs/run_<id>/run.log` and `metadata.json`.
Inspect the manifest for the stage order and statuses, UTC start time,
configuration snapshot, seed, and Python and NumPy versions. The unique ID
keeps repeated runs separate. No clinical rows are written to the manifest.

Python and NumPy are reseeded before each stage so stage order does not
change its starting random state. When adding an estimator, pass
`config.seed` as its `random_state`. For optional ML frameworks,
`seed_generators` accepts seed hooks such as `torch.manual_seed` from the
calling stage; it also returns a seeded NumPy `Generator`. Seeds alone do not
guarantee identical results across hardware, library versions, or kernels.

## Project map

| Location | Role |
| --- | --- |
| `main.py`, `config.py` | CLI, ordered stage registry, and study settings |
| `experiments/` | Shared stage contract and independently selectable research stages |
| `anonymization/` | Abstract method contract, PerturbPCA-alpha and its experiment adapter |
| `models/`, `llm/` | Extension points for models, prompts, clients, and explanation evaluation |
| `visualization/` | Plotting and report generation extension points |
| `utils/` | Logging, RNG setup, JSON/CSV and manifest helpers, future metrics |
| `data/raw/`, `data/processed/`, `data/splits/` | Local input, intermediate data, and split artifacts |
| `results/tables/`, `results/figures/`, `results/explanations/`, `results/models/` | Reserved scientific outputs |
| `results/logs/` | Per-run logs and manifests |
| `tests/` | Standard-library infrastructure tests using temporary workspaces |

The input path is `data/raw/diabetic_data.csv`. No automatic download is
performed. Data and generated results are ignored by Git;
this is not access control. Use an appropriately protected workspace for
clinical data and do not put credentials in configuration or manifests.

## Development checks

From this directory, with the environment active:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q main.py config.py anonymization experiments models llm visualization utils tests
```

To implement a stage later, specialize `Experiment.run` in its module and
replace its `STAGE` instance. Keep execution behind the CLI rather than at
import time, read settings from `StudyConfig`, and use temporary or synthetic
inputs in tests.
