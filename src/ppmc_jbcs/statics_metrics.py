import numpy as np


def pairwise_distances(X):
    """Return the condensed vector of Euclidean distances between all row pairs."""
    differences = X[:, None, :] - X[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    return distances[np.triu_indices(X.shape[0], k=1)]


def compute_properties(X):
    """Compute the geometric properties that RPAR is expected to preserve."""
    mean_vector = X.mean(axis=0)
    centered_X = X - mean_vector
    return {
        "Mean": mean_vector,
        "Euclidean Norm (centered records)": np.linalg.norm(centered_X, axis=1),
        "Pairwise Distances": pairwise_distances(X),
        "Total Variance": X.var(axis=0, ddof=1).sum(),
        "Eigenvalue Spectrum": np.linalg.eigvalsh(np.cov(X, rowvar=False)),
    }


def format_values(values, max_items):
    """Format a 1D array, showing at most ``max_items`` entries (all if None)."""
    shown = values if max_items is None else values[:max_items]
    suffix = ""
    if max_items is not None and len(values) > max_items:
        suffix = f" ... ({len(values)} values)"
    return np.array2string(shown, precision=6, max_line_width=120) + suffix


def print_comparison(name, original, anonymized, max_items=5):
    """Print a property for the original and anonymized data side by side."""
    original = np.atleast_1d(original)
    anonymized = np.atleast_1d(anonymized)
    print(f"\n{name}")
    print(f"  original:   {format_values(original, max_items)}")
    print(f"  anonymized: {format_values(anonymized, max_items)}")
    print(f"  max |difference|: {np.max(np.abs(original - anonymized)):.3e}")
