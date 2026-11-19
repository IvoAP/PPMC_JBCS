"""Check reproducibility utilities used by future anonymizers."""

import random
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans

from anonymization.anonymization_adapter import reconstruction_diagnostics
from anonymization.perturbpca import PerturbPCAAlpha

from utils.reproducibility import seed_generators


def reference(data: np.ndarray, alpha: float, rng: np.random.RandomState) -> np.ndarray:
    """Direct whole-matrix reference from the supplied function, for parity tests."""
    mean = np.mean(data, axis=0)
    centered = data - mean
    values, vectors = eigh(np.cov(centered, rowvar=False))
    vectors = vectors[:, np.argsort(values)[::-1]]
    projected = np.dot(vectors.T, centered.T).T
    shuffled = vectors.copy().T
    for row in shuffled:
        rng.shuffle(row)
    return projected @ ((1 - alpha) * vectors + alpha * shuffled.T).T + mean


class PerturbPCATests(unittest.TestCase):
    def test_training_matches_supplied_method_and_holdouts_use_frozen_state(self) -> None:
        train = np.random.default_rng(42).normal(size=(80, 5))
        for clusters in (None, 2):
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                with self.subTest(clusters=clusters, alpha=alpha):
                    rng = np.random.RandomState(321)
                    model = PerturbPCAAlpha(alpha, 321, clusters)
                    if clusters is None:
                        expected = reference(train, alpha, rng)
                    else:
                        labels = KMeans(n_clusters=clusters, random_state=rng).fit_predict(train)
                        expected = np.empty_like(train)
                        for k in dict.fromkeys(labels):
                            expected[labels == k] = reference(train[labels == k], alpha, rng)
                    np.testing.assert_allclose(model.fit_transform(train), expected, atol=1e-12)
                    probe = train[:1] + 3
                    batch = np.vstack([train * 100, probe])
                    np.testing.assert_allclose(model.transform(batch)[-1],
                                               model.transform(probe)[0], atol=1e-12)

    def test_diagnostics_detect_invalid_categories_without_repair(self) -> None:
        inputs = np.array([[0, 1], [1, 0], [0, 0]])
        outputs = np.array([[0.3, 0.7], [0, 0], [0, 0]])
        result = reconstruction_diagnostics(inputs, outputs, {"category": [0, 1]}, 1e-8)
        self.assertEqual(result["invalid_categorical_rows"], 2)
        self.assertEqual(outputs[0, 0], 0.3)
        outputs[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            reconstruction_diagnostics(inputs, outputs, {}, 1e-8)


class ReproducibilityTests(unittest.TestCase):
    def test_seed_replays_streams_and_calls_framework_hook(self) -> None:
        hook = Mock()
        rng = seed_generators(123, (hook,))
        first = (random.random(), np.random.random(), rng.random())
        rng = seed_generators(123)
        self.assertEqual(first, (random.random(), np.random.random(), rng.random()))
        hook.assert_called_once_with(123)
