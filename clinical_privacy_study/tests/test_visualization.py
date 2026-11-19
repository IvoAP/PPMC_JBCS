"""Verify full-pipeline CLI behavior without generating scientific outputs."""

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from main import PIPELINE
from utils.file_manager import load_metadata, save_csv


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
                 "number_emergency": "1"}
                for i in range(12)
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
            outcomes = manifest["stages"]
            assert isinstance(outcomes, dict)
            self.assertEqual(list(outcomes.items()), expected)
            report = root / "results" / "tables" / directory.name / "data_quality_report.json"
            self.assertTrue(report.is_file())
            for name in ("figures", "models", "explanations"):
                self.assertEqual(list((root / "results" / name).iterdir()), [])
