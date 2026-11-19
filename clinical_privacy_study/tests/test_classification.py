"""Check independent stage execution and run isolation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import Paths, StudyConfig
from experiments.shap_analysis import grouped_shap
from main import STAGES, run
from models.model_evaluation import evaluate
from utils.file_manager import load_metadata
from utils.metrics import compare_scores


class ExecutionTests(unittest.TestCase):
    def test_grouped_shap_has_known_additive_contributions(self) -> None:
        weights = np.array([0.1, 0.3, 0.2])
        background = np.array([[1., 0., 0.], [0., 1., 1.]])
        probe = np.array([[0., 1., 0.]])
        base, predictions, local = grouped_shap(
            lambda x: x @ weights, background, probe, [[0, 1], [2]], 4, 17,
        )
        np.testing.assert_allclose(local, [[0.1, -0.1]])
        np.testing.assert_allclose(base + local.sum(axis=1), predictions)

    def test_rank_reversal_and_single_class_auc(self) -> None:
        result = compare_scores({"a": 3, "b": 2, "c": 1}, {"a": 1, "b": 2, "c": 3}, 2)
        self.assertAlmostEqual(result["kendall"], -1)
        self.assertAlmostEqual(result["spearman"], -1)
        self.assertAlmostEqual(result["jaccard"], 1 / 3)
        metrics = evaluate(np.zeros(3), np.zeros(3), np.full(3, 0.1))
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["pr_auc"])
        self.assertEqual(metrics["confusion_matrix"], [[3, 0], [0, 0]])

    def test_each_stage_runs_alone_without_inputs(self) -> None:
        with TemporaryDirectory() as workspace:
            config = StudyConfig(paths=Paths(Path(workspace)))
            runs = set()
            placeholders = STAGES.keys() - {"preprocessing", "anonymization", "classification",
                                             "feature_selection", "shap_analysis"}
            for name in placeholders:
                with self.subTest(stage=name):
                    directory = run(name, config)
                    runs.add(directory)
                    self.assertEqual(load_metadata(directory)["stages"],
                                     {name: "not_implemented"})
                    self.assertIn(name, (directory / "run.log").read_text("utf-8"))
            self.assertEqual(len(runs), len(placeholders))
            self.assertFalse(config.paths.dataset.exists())
