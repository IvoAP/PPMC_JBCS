import numpy as np
from scipy.stats import wasserstein_distance


def _check_same_shape(X, X_anon):
    """Raise an error if the original and anonymized matrices differ in shape."""
    if X.ndim != 2 or X.shape != X_anon.shape:
        raise ValueError(
            f"X and X_anon must be 2D arrays with the same shape; "
            f"got {X.shape} and {X_anon.shape}."
        )


def marginal_wasserstein(X, X_anon):
    """Return the 1D Wasserstein distance of each feature between X and X_anon.

    The distances are expressed in the same unit as each feature. High values are
    expected, since RPAR does not preserve individual marginal distributions.
    """
    _check_same_shape(X, X_anon)
    return np.array(
        [wasserstein_distance(X[:, j], X_anon[:, j]) for j in range(X.shape[1])]
    )


def sliced_wasserstein(X, X_anon, n_projections=100, random_state=None):
    """Return the sliced Wasserstein distance between the joint distributions.

    Each record is projected onto ``n_projections`` random unit directions, and the
    result is the mean of the 1D Wasserstein distances of the projections.
    The random directions come from ``random_state`` and are not part of the
    anonymization algorithm, so this metric belongs to the analysis only.
    """
    _check_same_shape(X, X_anon)
    rng = np.random.default_rng(random_state)
    directions = rng.normal(size=(X.shape[1], n_projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    distances = [
        wasserstein_distance(X @ direction, X_anon @ direction)
        for direction in directions.T
    ]
    return float(np.mean(distances))
