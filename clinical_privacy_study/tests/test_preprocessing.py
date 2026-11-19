"""Verify split safeguards and artifact handling without clinical records."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from config import Dataset, Preprocessing, StudyConfig
from experiments.preprocessing import PreprocessingPipeline
from utils.file_manager import load_csv, save_csv


class PreprocessingInfrastructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = PreprocessingPipeline(StudyConfig(preprocessing=Preprocessing(
            admission_numeric=("number_inpatient",), admission_categorical=("race",),
        )))
        self.features = pd.DataFrame({
            "number_inpatient": ["1", "3", "?"], "race": ["A", "B", "?"],
            "encounter_id": ["01", "02", "03"], "patient_nbr": ["1", "1", "2"],
            "readmitted": ["<30", ">30", "NO"], "time_in_hospital": [2, 1, 4],
        })

    def test_train_only_statistics_and_feature_provenance(self) -> None:
        train = self.pipeline.fit_transform(self.features)
        metadata = self.pipeline.get_feature_metadata()
        self.assertEqual({entry["source"] for entry in metadata}, {"number_inpatient", "race"})
        self.assertEqual(len(metadata), train.shape[1])
        self.assertAlmostEqual(train.toarray()[2, 0], 0.0)
        holdout = pd.DataFrame({"number_inpatient": [5], "race": ["unseen"]})
        transformed = self.pipeline.transform(holdout).toarray()[0]
        self.assertAlmostEqual(transformed[0], 3 / np.sqrt(2 / 3))
        self.assertFalse(transformed[1:].any())
        np.testing.assert_array_equal(
            self.pipeline.transform(self.features).toarray(), train.toarray()
        )

    def test_schema_deduplication_conflicts_and_target(self) -> None:
        frame = self.features
        clean = self.pipeline.validate_dataset(pd.concat([frame, frame]))
        self.assertEqual(len(clean), 3)
        self.assertEqual(self.pipeline.create_target(clean.readmitted).tolist(), [1, 0, 0])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.pipeline.validate_dataset(pd.concat([frame, frame.assign(race="C")]))
        with self.assertRaisesRegex(ValueError, "Missing required"):
            self.pipeline.validate_dataset(frame.drop(columns="patient_nbr"))
        with self.assertRaisesRegex(ValueError, "label"):
            self.pipeline.create_target(pd.Series(["?"]))

    def test_group_split_disjoint_exhaustive_and_repeatable(self) -> None:
        groups = pd.Series(["a", "b", "c", "a", "d", "e", "b", "f", "g"])
        partitions = self.pipeline.split_by_patient(groups)
        assigned: set[str] = set()
        for name, indices in partitions.items():
            members = set(groups.iloc[indices])
            self.assertTrue(members)
            self.assertFalse(assigned.intersection(members))
            assigned.update(members)
            np.testing.assert_array_equal(indices, self.pipeline.split_by_patient(groups)[name])
        self.assertEqual(sorted(np.concatenate(list(partitions.values()))), list(groups.index))

    def test_discharge_opt_in_and_forbidden_identifiers(self) -> None:
        policy = Preprocessing(prediction_time="discharge")
        discharge = PreprocessingPipeline(StudyConfig(preprocessing=policy))
        numeric, categorical = discharge.identify_feature_types()
        self.assertIn("time_in_hospital", numeric)
        self.assertIn("A1Cresult", categorical)
        self.assertNotIn("discharge_disposition_id", numeric + categorical)
        unsafe = Preprocessing(admission_categorical=("patient_nbr",))
        with self.assertRaisesRegex(ValueError, "identifier"):
            PreprocessingPipeline(StudyConfig(preprocessing=unsafe)).identify_feature_types()

    def test_splits_must_leave_training_data(self) -> None:
        with self.assertRaises(ValueError):
            Dataset(test_fraction=0.8, validation_fraction=0.2)

    def test_csv_preserves_text_and_quoted_values(self) -> None:
        with TemporaryDirectory() as workspace:
            path = Path(workspace) / "processed" / "sample.csv"
            rows = [{"code": "001", "label": "example, quoted"}]
            save_csv(path, ("code", "label"), rows)
            self.assertEqual(load_csv(path), rows)
