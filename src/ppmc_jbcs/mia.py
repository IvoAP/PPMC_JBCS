"""Loss-based membership inference against classifiers trained on RPAR data."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .anon import RPAR
from .classifiers import build_models
from .noise import add_feature_noise


def _membership_auc(model, X_train, y_train, X_test, y_test):
    """Measure whether the true-label loss distinguishes members from non-members."""
    def scores(X, y):
        probabilities = model.predict_proba(X)
        class_indices = np.searchsorted(model.classes_, y)
        true_probabilities = probabilities[np.arange(len(y)), class_indices]
        return np.log(np.clip(true_probabilities, np.finfo(float).eps, 1.0))

    member_scores = scores(X_train, y_train)
    nonmember_scores = scores(X_test, y_test)
    membership = np.concatenate((np.ones(len(y_train)), np.zeros(len(y_test))))
    return roc_auc_score(membership, np.concatenate((member_scores, nonmember_scores)))


def run_membership_inference(X, y, sigmas, random_states=range(5)):
    """Evaluate loss-based membership AUC for each model, noise level and split.

    Identical feature vectors stay in the same split. RPAR and noise scales are fitted
    on members only; the same RPAR transformation is applied to non-members. Models
    use default parameters to avoid tuning on non-member records. The attacker is
    assumed to know each record's label and obtain its prediction probabilities.

    Returns a DataFrame with dataset, sigma, seed, model and auc columns. Original
    data is evaluated once per split and has no sigma.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be a 2D matrix and y a 1D array of matching length.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")
    sigmas = tuple(sigmas)
    if not sigmas or any(not np.isfinite(sigma) or sigma < 0 for sigma in sigmas):
        raise ValueError("sigmas must contain finite, non-negative noise levels.")

    _, groups = np.unique(X, axis=0, return_inverse=True)
    rows = []
    for seed in random_states:
        splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if not np.isin(np.unique(y_test), np.unique(y_train)).all():
            raise ValueError(f"Split {seed} contains a non-member class absent from training.")

        anonymizer = RPAR(random_state=seed)
        anon_train = anonymizer.fit_transform(X_train)
        anon_test = anonymizer.transform(X_test)
        combined = np.concatenate((anon_train, anon_test))
        datasets = [("original", None, X_train, X_test)]
        for sigma in sigmas:
            noisy = add_feature_noise(combined, X_train, sigma, random_state=seed)
            datasets.append(("anonymized", sigma, noisy[:len(X_train)], noisy[len(X_train):]))

        for dataset, sigma, train_data, test_data in datasets:
            for model_name, model in build_models(seed).items():
                model.fit(train_data, y_train)
                rows.append({
                    "dataset": dataset,
                    "sigma": sigma,
                    "seed": seed,
                    "model": model_name,
                    "auc": _membership_auc(model, train_data, y_train, test_data, y_test),
                })
    if not rows:
        raise ValueError("random_states must contain at least one seed.")
    return pd.DataFrame(rows)
