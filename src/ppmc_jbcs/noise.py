"""Gaussian noise added on top of the RPAR output, to study the privacy-utility trade-off.

The noise is not part of RPAR. It is an extra step applied to the published data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from .classifiers import SCORING, build_models
from .linkage import DEFAULT_KS, run_linkage
from .shap import EXPLAINED_MODELS, explain_global_importance, rotated_axis_names
from .wasserstein import marginal_wasserstein, sliced_wasserstein

NOISE_STREAM = 1  # separates the noise generator from the permutation seed


def add_feature_noise(X_anon, X, sigma, random_state):
    """Add Gaussian noise with standard deviation ``sigma * sd`` to each column of ``X_anon``.

    ``sd`` is the standard deviation of the original feature in the same column position.
    The columns of ``X_anon`` are rotated axes, so this is a simple scale choice, not a per-feature guarantee.
    """
    rng = np.random.default_rng([random_state, NOISE_STREAM])
    feature_sd = X.std(axis=0, ddof=1)
    return X_anon + rng.normal(0.0, 1.0, size=X_anon.shape) * (sigma * feature_sd)


def _perturbation(sigma):
    def perturb(X_anon, X, random_state):
        return add_feature_noise(X_anon, X, sigma, random_state)

    return perturb


def evaluate_noisy_utility(X_noisy, y, best_params, random_state):
    """Cross-validate every model on the noisy data, with the hyperparameters tuned on clean data."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows = []
    for model_name, model in build_models(random_state).items():
        model.set_params(**best_params.get(("anonymized", model_name), {}))
        scores = cross_validate(model, X_noisy, y, cv=cv, scoring=SCORING)
        rows.append({
            "model": model_name,
            "accuracy": scores["test_accuracy"].mean(),
            "f1_score": scores["test_f1_score"].mean(),
        })
    return pd.DataFrame(rows)


def run_noise_experiment(X, X_anon, y, best_params, sigmas, random_states, ks=DEFAULT_KS):
    """Measure linkage and classifier utility for each noise level ``sigma``.

    Linkage uses every seed in ``random_states`` for the anonymization. Utility uses the
    anonymization ``X_anon`` with noise drawn from the first seed.
    Returns the linkage results and the utility results as DataFrames.
    """
    linkage_frames, utility_frames = [], []
    for sigma in sigmas:
        linkage_results, _ = run_linkage(X, random_states, ks, perturb=_perturbation(sigma))
        linkage_frames.append(linkage_results.assign(sigma=sigma))

        X_noisy = add_feature_noise(X_anon, X, sigma, random_states[0])
        utility = evaluate_noisy_utility(X_noisy, y, best_params, random_states[0])
        utility_frames.append(utility.assign(sigma=sigma))

    return pd.concat(linkage_frames, ignore_index=True), pd.concat(utility_frames, ignore_index=True)


def run_noise_wasserstein(X, X_anon, sigmas, random_states):
    """Measure how far the noisy data moves for each noise level ``sigma`` and seed.

    Distances are in units of each original feature's sd. Two comparisons are returned:
    "isolated" compares the noisy data with the clean RPAR output, so it isolates the noise;
    "total" compares it with the original data, so it also includes the distortion of RPAR.
    """
    scale = X.std(axis=0, ddof=1)
    rows = []
    for sigma in sigmas:
        for seed in random_states:
            X_noisy = add_feature_noise(X_anon, X, sigma, seed) / scale
            for comparison, reference in (("isolated", X_anon), ("total", X)):
                reference_scaled = reference / scale
                rows.append({
                    "sigma": sigma,
                    "seed": seed,
                    "comparison": comparison,
                    "marginal": marginal_wasserstein(reference_scaled, X_noisy).mean(),
                    "sliced": sliced_wasserstein(reference_scaled, X_noisy, random_state=seed),
                })
    return pd.DataFrame(rows)


def run_noise_shap(X, X_anon, y, best_params, sigmas, random_state=42):
    """Compute the SHAP importance of each model trained on the noisy anonymized data, for each ``sigma``.

    Each model is refit on the noisy data with its tuned hyperparameters, so the explanation reflects
    what an analyst learns from the published data. The same seed keeps the kernel subsamples fixed
    across sigma levels, so differences come from the noise and not from the sampling.
    Returns a long DataFrame with the columns sigma, model, feature and importance.
    """
    names = rotated_axis_names(X_anon.shape[1])
    rows = []
    for sigma in sigmas:
        X_noisy = add_feature_noise(X_anon, X, sigma, random_state)
        for model_name in EXPLAINED_MODELS:
            params = best_params.get(("anonymized", model_name), {})
            importance = explain_global_importance(model_name, X_noisy, y, params, random_state)
            rows.extend(
                {"sigma": sigma, "model": model_name, "feature": name, "importance": value}
                for name, value in zip(names, importance)
            )
    return pd.DataFrame(rows)
