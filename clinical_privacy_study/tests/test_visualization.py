"""Verify full-pipeline CLI behavior without generating scientific outputs."""

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import load_npz

from main import PIPELINE
from utils.file_manager import load_json, load_metadata, save_csv


class ReportingTests(unittest.TestCase):
    def test_full_cli_records_preprocessing_and_skips_later_stages(self) -> None:
        script = Path(__file__).resolve().parents[1] / "main.py"
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            rows = [
                {"encounter_id": str(i), "patient_nbr": str(i // 2),
                 "readmitted": ("NO", ">30", "<30")[i % 3],
                 "age": "[40-50)", "gender": "Female", "race": "?",
                 "admission_type_id": "1", "admission_source_id": "1",
                 "number_outpatient": "0", "number_inpatient": "0",
                 "number_emergency": str(i % 4)}
                for i in range(72)
            ]
            save_csv(root / "data" / "raw" / "diabetic_data.csv", list(rows[0]), rows)
            subprocess.run(
                [sys.executable, str(script), "--experiment", "all",
                 "--workspace", str(root)],
                cwd=root, check=True, capture_output=True, text=True,
            )
            (directory,) = (root / "results" / "logs").iterdir()
            manifest = load_metadata(directory)
            expected = [(stage.name, "not_implemented") for stage in PIPELINE]
            expected[0] = ("preprocessing", "completed")
            expected[1] = ("anonymization", "completed")
            for index in (2, 3, 4):
                expected[index] = (PIPELINE[index].name, "completed")
            outcomes = manifest["stages"]
            assert isinstance(outcomes, dict)
            self.assertEqual(list(outcomes.items()), expected)
            report = root / "results" / "tables" / directory.name / "data_quality_report.json"
            self.assertTrue(report.is_file())
            output = root / "results" / "anonymized" / directory.name
            for part in ("train", "validation", "test"):
                source = root / "data" / "processed" / directory.name / f"{part}_features.npz"
                actual = np.load(output / "global" / "alpha_000" / f"{part}_features.npy")
                np.testing.assert_allclose(actual, load_npz(source).toarray(), atol=1e-12)
            tables = report.parent
            metrics = load_json(tables / "classification" / "evaluation.json")
            scenarios = [(row["training_variant"], row["scenario"]) for row in metrics["results"]]
            self.assertIn((str(Path("global") / "alpha_000"), "C"), scenarios)
            self.assertNotIn((str(Path("global") / "alpha_025"), "C"), scenarios)
            self.assertTrue((tables / "feature_selection" / "ranking_comparisons.json").exists())
            shap = root / "results" / "explanations" / directory.name / "shap_analysis"
            for file in shap.rglob("local_shap.csv"):
                local = np.genfromtxt(file, delimiter=",", skip_header=1, ndmin=2)
                np.testing.assert_allclose(local[:, 1] + local[:, 3:].sum(1), local[:, 2])
            self.assertEqual(len(list(shap.rglob("local_shap.csv"))), 18)
