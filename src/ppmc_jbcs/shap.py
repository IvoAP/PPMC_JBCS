import numpy as np
import pandas as pd
import shap

from .classifiers import build_models

EXPLAINED_MODELS = ("Logistic Regression", "Random Forest", "KNN", "Gaussian Naive Bayes", "MLP", "AdaBoost")

KERNEL_BACKGROUND_SIZE = 20
KERNEL_EXPLAIN_SIZE = 50
KERNEL_NSAMPLES = 512


def rotated_axis_names(n_axes):
    """Return generic names for the anonymized axes, which are not the original variables."""
    return [f"rotated_axis_{i}" for i in range(1, n_axes + 1)]


def _positive_class_values(values):
    """Return the SHAP values of class 1 for a binary classifier.

    shap returns a list of arrays (older versions) or a 3-D array (newer versions) for classifiers.
    """
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def _subsample(X, size, random_state):
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=min(size, len(X)), replace=False)
    return X[indices]


def _shap_values(model_name, classifier, X_scaled, random_state):
    """Return SHAP values for ``X_scaled``.

    Kernel-based explainers work on subsamples to keep runtime manageable, so their importance
    is an estimate from a subset of records, not an exact value over the whole dataset.
    """
    if model_name == "Logistic Regression":
        return shap.LinearExplainer(classifier, X_scaled).shap_values(X_scaled)
    if model_name == "Random Forest":
        return shap.TreeExplainer(classifier).shap_values(X_scaled)
    background = _subsample(X_scaled, KERNEL_BACKGROUND_SIZE, random_state)
    explained = _subsample(X_scaled, KERNEL_EXPLAIN_SIZE, random_state)
    explainer = shap.KernelExplainer(classifier.predict_proba, background)
    return explainer.shap_values(explained, nsamples=KERNEL_NSAMPLES)


def explain_global_importance(model_name, X, y, params, random_state=42):
    """Fit the tuned model on ``X`` and return the mean absolute SHAP value of each feature.

    The explanation is computed on the standardized features, which is what the classifier sees.
    """
    pipeline = build_models(random_state)[model_name].set_params(**params).fit(X, y)
    X_scaled = pipeline[0].transform(X)
    values = _positive_class_values(_shap_values(model_name, pipeline[-1], X_scaled, random_state))
    return np.abs(values).mean(axis=0)


def compute_shap_importance(X, X_anon, y, feature_columns, best_params, random_state=42):
    """Compute the global SHAP importance on the original and anonymized datasets.

    The anonymized axes are a rotation of the original features, so they are named generically
    and must not be read as the original variables.
    Returns a long DataFrame with the columns dataset, model, feature and importance.
    """
    datasets = {
        "original": (X, list(feature_columns)),
        "anonymized": (X_anon, rotated_axis_names(X_anon.shape[1])),
    }
    rows = []
    for dataset_name, (data, names) in datasets.items():
        for model_name in EXPLAINED_MODELS:
            params = best_params.get((dataset_name, model_name), {})
            importance = explain_global_importance(model_name, data, y, params, random_state)
            rows.extend(
                {"dataset": dataset_name, "model": model_name, "feature": name, "importance": value}
                for name, value in zip(names, importance)
            )
    return pd.DataFrame(rows)
