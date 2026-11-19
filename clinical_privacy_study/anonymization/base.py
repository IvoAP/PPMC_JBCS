"""Contract for train-fitted, row-order-preserving empirical transformations."""

from abc import ABC, abstractmethod

import numpy as np


class AnonymizationMethod(ABC):
    """Operate only on finite, encoded feature matrices, never labels or IDs."""

    @abstractmethod
    def fit(self, training_features: np.ndarray) -> None:
        """Learn all data-dependent parameters from training features."""

    @abstractmethod
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Return the same shape, column order and encounter order without fitting."""

    def fit_transform(self, training_features: np.ndarray) -> np.ndarray:
        """Fit once and apply the fitted method to the training set."""
        self.fit(training_features)
        return self.transform(training_features)

    @abstractmethod
    def get_metadata(self) -> dict[str, object]:
        """Describe the method and parameters needed to interpret this run."""
