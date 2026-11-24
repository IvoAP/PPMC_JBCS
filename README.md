# RPAR: privacy and utility in clinical data

This project evaluates **Randomized Principal Axis Rotation (RPAR)**, a PCA-based orthogonal transformation, on a heart disease dataset. The pipeline compares original and transformed data, adds different noise levels, and analyzes predictive utility, SHAP explanations, Wasserstein distances, and privacy risks through linkage and membership inference attacks. Results are presented in figures and tables.

## Project structure

```text
.
├── src/ppmc_jbcs/        # Pipeline and experiment code
│   ├── main.py           # Experiment configuration and execution
│   ├── load_data.py      # Dataset loading via KaggleHub
│   ├── anon.py           # RPAR transformation
│   ├── classifiers.py    # Classifiers, optimization, and evaluation
│   ├── noise.py          # Noise experiments
│   ├── linkage.py        # Linkage attacks
│   ├── mia.py            # Membership inference
│   ├── shap.py           # SHAP importance analysis
│   ├── wasserstein.py    # Wasserstein distances
│   ├── statics_metrics.py # Transformation properties
│   ├── plot.py           # Figure generation
│   └── prints.py         # Terminal output
├── tests/                # Automated tests
├── figures/              # Generated figures
├── tables/               # Generated CSV tables
├── pyproject.toml        # Dependencies and project command
└── uv.lock               # Dependency versions
```

The `figures/` and `tables/` directories are created during execution, and their generated files are ignored by Git.

## How to run

Install [uv](https://docs.astral.sh/uv/) and use a Python version compatible with the project (3.12 or later).

From the project root, install the dependencies and run the pipeline:

```bash
uv sync
uv run ppmc-jbcs
```

The [`johnsmith88/heart-disease-dataset`](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) dataset is downloaded and cached by KaggleHub. The first download requires internet access. The full pipeline may take some time, especially due to model optimization and SHAP analysis.

Figures are saved in `figures/` and classifier tables in `tables/`. The pipeline's noise levels are defined in `src/ppmc_jbcs/main.py`.

To run the tests:

```bash
uv run python -m unittest discover -s tests
```

## Membership inference

The loss-based attack assumes access to predicted probabilities and the true label. It uses ROC AUC to distinguish training members from non-members, keeping identical feature vectors in the same split. Across five splits, RPAR and the noise scale are fitted using training data only; the original-data baseline uses the same splits. Classifiers use default parameters rather than parameters tuned on the full dataset. Figures show the mean and standard deviation across splits, not confidence intervals; the experiment does not provide a general privacy guarantee.
