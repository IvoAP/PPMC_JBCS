"""Randomized Principal Axis Rotation anonymization (RPAR).

The transformation is a PCA-based orthogonal map ``R = E P E^T``, where ``E``
holds the covariance eigenvectors and ``P`` is a random permutation matrix.
Because ``R`` is orthogonal, the geometry of the centered data is preserved.

``inverse_transform`` exists only for debugging and mathematical validation.
It must not be exposed to an attacker or included in the anonymization workflow,
since it recovers the original data exactly.
"""

import numpy as np


class RPAR:
    """Randomized Principal Axis Rotation anonymizer with a scikit-learn-style interface.

    Parameters
    ----------
    random_state : int, numpy.random.Generator or None, default=None
        Seed used only to generate the random permutation matrix.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Mean vector of the fitting data.
    eigenvalues_ : ndarray of shape (n_features,)
        Eigenvalues of the covariance matrix, in ascending order.
    eigenvectors_ : ndarray of shape (n_features, n_features)
        Orthonormal eigenvectors ``E`` of the covariance matrix (as columns).
    permutation_matrix_ : ndarray of shape (n_features, n_features)
        Random permutation matrix ``P``.
    rotation_ : ndarray of shape (n_features, n_features)
        Orthogonal transformation ``R = E P E^T``.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X):
        """Estimate the mean, covariance eigendecomposition and transformation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numerical training data with at least two rows.

        Returns
        -------
        self : RPAR
            The fitted anonymizer.
        """
        X = self._validate_input(X)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"fit requires at least 2 samples to compute the covariance; got {n_samples}."
            )

        mean_vector = X.mean(axis=0)
        centered_X = X - mean_vector
        covariance_matrix = centered_X.T @ centered_X / (n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        rng = np.random.default_rng(self.random_state)
        permutation_matrix = np.eye(n_features)[rng.permutation(n_features)]

        rotation = eigenvectors @ permutation_matrix @ eigenvectors.T

        self.mean_ = mean_vector
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        self.permutation_matrix_ = permutation_matrix
        self.rotation_ = rotation
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        """Anonymize data with the already-fitted transformation.

        The transformation is not recalculated here.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with the same number of features used during fitting.

        Returns
        -------
        X_anon : ndarray of shape (n_samples, n_features)
            Anonymized data, ``(X - mean) @ R + mean``.
        """
        self._check_is_fitted()
        X = self._validate_input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but RPAR was fitted with "
                f"{self.n_features_in_} features."
            )
        return (X - self.mean_) @ self.rotation_ + self.mean_

    def fit_transform(self, X):
        """Fit the transformation on ``X`` and anonymize it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numerical training data with at least two rows.

        Returns
        -------
        X_anon : ndarray of shape (n_samples, n_features)
            Anonymized data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_anon):
        """Invert the anonymization (debugging and validation only).

        Since ``R`` is orthogonal, ``R^-1 = R^T``. This method must not be
        exposed to an attacker or used in the anonymization workflow.

        Parameters
        ----------
        X_anon : array-like of shape (n_samples, n_features)
            Anonymized data produced by this fitted instance.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Reconstructed data, ``(X_anon - mean) @ R^T + mean``.
        """
        self._check_is_fitted()
        X_anon = self._validate_input(X_anon)
        if X_anon.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_anon has {X_anon.shape[1]} features, but RPAR was fitted with "
                f"{self.n_features_in_} features."
            )
        return (X_anon - self.mean_) @ self.rotation_.T + self.mean_

    def _check_is_fitted(self):
        """Raise an error if the anonymizer has not been fitted yet."""
        if not hasattr(self, "rotation_"):
            raise RuntimeError("This RPAR instance is not fitted yet. Call 'fit' first.")

    @staticmethod
    def _validate_input(X):
        """Convert ``X`` to a 2D finite float array without modifying the input."""
        try:
            X_array = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Input must be a numerical matrix convertible to float.") from error

        if X_array.ndim != 2:
            raise ValueError(f"Input must be two-dimensional; got {X_array.ndim} dimension(s).")
        if X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError("Input must contain at least one row and one column.")
        if not np.all(np.isfinite(X_array)):
            raise ValueError("Input contains NaN or infinite values.")
        return X_array
