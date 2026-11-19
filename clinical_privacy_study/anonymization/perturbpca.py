"""Inductive integration of the author's supplied PerturbPCA-alpha code."""

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans

from anonymization.base import AnonymizationMethod


class PerturbPCAAlpha(AnonymizationMethod):
    """Apply one training basis globally, or one per training KMeans cluster."""

    def __init__(self, alpha: float, seed: int, clusters: int | None = None) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be in [0, 1].")
        if clusters is not None and clusters < 2:
            raise ValueError("Clustered mode requires at least two clusters.")
        self.alpha = alpha
        self.seed = seed
        self.clusters = clusters
        self.router: KMeans | None = None
        self.bases: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # Author integration point: keep training-only calculations in _fit_basis
    # and the original projection/reconstruction in _apply_basis. Do not call
    # the original whole-dataset function independently on held-out records.
    def _fit_basis(
        self, training: np.ndarray, rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(training) < 2:
            raise ValueError("Each fitted covariance requires at least two encounters.")
        mean = training.mean(axis=0)
        covariance = np.atleast_2d(np.cov(training - mean, rowvar=False))
        values, vectors = eigh(covariance)
        vectors = vectors[:, np.argsort(values)[::-1]]
        shuffled = vectors.copy().T
        for vector in shuffled:
            rng.shuffle(vector)
        reconstruction = (1 - self.alpha) * vectors + self.alpha * shuffled.T
        return mean, vectors, reconstruction

    @staticmethod
    def _apply_basis(
        features: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        mean, vectors, reconstruction = basis
        projected = np.dot(vectors.T, (features - mean).T).T
        return np.dot(projected, reconstruction.T) + mean

    @staticmethod
    def _validate(features: np.ndarray) -> None:
        if features.ndim != 2 or not all(features.shape) or not np.isfinite(features).all():
            raise ValueError("Expected a nonempty finite feature matrix.")

    def fit(self, training_features: np.ndarray) -> None:
        """Fit routing, means, eigenvectors and shuffled bases using only train."""
        self._validate(training_features)
        rng = np.random.RandomState(self.seed)
        router = None
        if self.clusters is None:
            bases = {0: self._fit_basis(training_features, rng)}
        else:
            router = KMeans(n_clusters=self.clusters, random_state=rng)
            labels = router.fit_predict(training_features)
            if len(np.unique(labels)) != self.clusters:
                raise ValueError("KMeans produced an empty training cluster.")
            # Match the supplied function's first-seen cluster shuffle order.
            bases = {int(k): self._fit_basis(training_features[labels == k], rng)
                     for k in dict.fromkeys(labels)}
        self.router, self.bases = router, bases

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Route with frozen centroids and scatter results to original row positions."""
        self._validate(features)
        if not self.bases or features.shape[1] != len(next(iter(self.bases.values()))[0]):
            raise ValueError("Transformation is not fitted or feature width differs.")
        if self.router is None:
            return self._apply_basis(features, self.bases[0])
        labels = self.router.predict(features)
        output = np.empty(features.shape, dtype=float)
        for k in np.unique(labels):
            mask = labels == k
            output[mask] = self._apply_basis(features[mask], self.bases[int(k)])
        return output

    def get_metadata(self) -> dict[str, object]:
        """Summarize parameters; exact fitted arrays are saved in the private model."""
        routing = None
        if self.router is not None:
            routing = {**self.router.get_params(), "random_state": self.seed}
        return {"method": "PerturbPCA-alpha", "alpha": self.alpha, "seed": self.seed,
                "clusters": self.clusters, "kmeans_parameters": routing,
                "components": "all", "rng": "NumPy RandomState (MT19937)"}
