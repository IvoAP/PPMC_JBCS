import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from ppmc_jbcs.anon import RPAR
from ppmc_jbcs.mia import _membership_auc, run_membership_inference
from ppmc_jbcs.plot import plot_membership_inference


class MembershipInferenceTests(unittest.TestCase):
    def test_loss_scores_use_true_class_and_identify_members(self):
        class Predictions:
            classes_ = np.array([0, 1])

            def predict_proba(self, X):
                return X

        model = Predictions()
        train = np.array([[0.9, 0.1], [0.2, 0.8]])
        test = np.array([[0.3, 0.7], [0.6, 0.4]])
        self.assertEqual(_membership_auc(model, train, np.array([0, 1]),
                                         test, np.array([0, 1])), 1.0)
        self.assertEqual(_membership_auc(model, test, np.array([0, 1]),
                                         train, np.array([0, 1])), 0.0)

    def test_repeated_splits_keep_duplicates_together_and_plot_results(self):
        X, y = make_classification(n_samples=80, n_features=4, random_state=12)
        X = np.repeat(X, 2, axis=0)
        y = np.repeat(y, 2)
        fitted, transformed = [], []
        original_fit, original_transform = RPAR.fit, RPAR.transform

        def capture_fit(instance, values):
            fitted.append(values.copy())
            return original_fit(instance, values)

        def capture_transform(instance, values):
            transformed.append(values.copy())
            return original_transform(instance, values)

        def models(seed):
            return {"Tree": DecisionTreeClassifier(random_state=seed)}

        with patch("ppmc_jbcs.mia.build_models", models), \
             patch.object(RPAR, "fit", capture_fit), \
             patch.object(RPAR, "transform", capture_transform):
            results = run_membership_inference(X, y, (0, 0.5), random_states=(1, 2))
            repeated = run_membership_inference(X, y, (0, 0.5), random_states=(1, 2))

        self.assertTrue(results.equals(repeated))
        self.assertEqual(len(results), 6)
        self.assertEqual(results["dataset"].value_counts().to_dict(), {"anonymized": 4, "original": 2})
        self.assertTrue(results["auc"].between(0, 1).all())
        for train, test in zip(fitted, transformed[1::2]):
            self.assertFalse(set(map(tuple, train)) & set(map(tuple, test)))
            self.assertEqual(len(train) + len(test), len(X))

        with tempfile.TemporaryDirectory() as output:
            plot_membership_inference(results, output)
            self.assertTrue((Path(output) / "mia_noise_tree.png").is_file())


if __name__ == "__main__":
    unittest.main()
