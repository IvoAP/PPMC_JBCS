"""Patient-disjoint, prediction-time-aware preprocessing for diabetes encounters."""

import csv
import logging
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import StudyConfig
from experiments.experiment import Experiment
from utils.file_manager import save_json


class PreprocessingPipeline:
    """Learn a feature representation only from the training patients."""

    LABELS = {"<30": 1, ">30": 0, "NO": 0}
    # "None" in the two lab fields means not measured, not a null string.
    COLUMN_NULLS = {
        "gender": ("Unknown/Invalid",),
        "admission_type_id": ("5", "6", "8"),
        "admission_source_id": ("9", "15", "17", "20", "21"),
        "discharge_disposition_id": ("18", "25", "26"),
    }

    def __init__(self, config: StudyConfig) -> None:
        self.config = config
        self.policy = config.preprocessing
        self.schema = config.dataset
        self.encoder: ColumnTransformer | None = None

    def _normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize null codes without discarding meaningful categorical values."""
        result = frame.copy()
        for name in frame:
            values = frame[name].astype("string").str.strip()
            tokens = self.policy.missing_tokens + self.COLUMN_NULLS.get(name, ())
            result[name] = values.mask(values.isin(tokens))
        return result

    def load_dataset(self) -> pd.DataFrame:
        """Read identifiers and coded categories as strings, including leading zeros."""
        path = self.config.paths.dataset
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
            if not header or len(header) != len(set(header)):
                raise ValueError("CSV header is empty or contains duplicate names.")
            stream.seek(0)
            frame = pd.read_csv(stream, dtype="string", keep_default_na=False)
        if not isinstance(frame.index, pd.RangeIndex):
            raise ValueError("CSV rows do not match the header width.")
        return self._normalize(frame)

    def identify_feature_types(self) -> tuple[list[str], list[str]]:
        """Return ordered numeric and categorical allowlists, independent of values."""
        numeric = list(self.policy.admission_numeric)
        categorical = list(self.policy.admission_categorical)
        if self.policy.prediction_time == "discharge":
            numeric.extend(self.policy.discharge_numeric)
            categorical.extend(self.policy.discharge_categorical)
        names = numeric + categorical
        forbidden = {
            "readmitted", "patient_nbr", "encounter_id", "discharge_disposition_id",
            self.schema.target, *self.schema.identifiers,
        }
        if forbidden.intersection(names):
            raise ValueError("Policy includes a target, identifier or outcome proxy.")
        if not names or len(names) != len(set(names)):
            raise ValueError("Feature allowlists must be nonempty and disjoint.")
        return numeric, categorical

    def validate_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicates; reject ambiguous encounters or missing keys."""
        numeric, categorical = self.identify_feature_types()
        required = {
            *numeric, *categorical, *self.schema.identifiers, self.schema.target,
            "encounter_id", "patient_nbr",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        clean = frame.drop_duplicates().reset_index(drop=True)
        if clean.empty or clean[["encounter_id", "patient_nbr"]].isna().any().any():
            raise ValueError("Dataset is empty or contains missing patient/encounter IDs.")
        if clean["encounter_id"].duplicated().any():
            raise ValueError("Conflicting records share an encounter ID.")
        return clean

    def create_target(self, labels: pd.Series) -> pd.Series:
        """Map the three documented labels; never infer or impute outcomes."""
        target = labels.map(self.LABELS)
        if target.isna().any():
            raise ValueError("Missing or unsupported readmission label.")
        return target.astype("int8").rename("target")

    def split_by_patient(self, groups: pd.Series) -> dict[str, np.ndarray]:
        """Randomize unique patients, then return aligned positional row indices."""
        if groups.isna().any():
            raise ValueError("Cannot split records with missing patient IDs.")
        patients = np.random.default_rng(self.config.seed).permutation(
            sorted(groups.unique())
        )
        test = max(1, round(len(patients) * self.schema.test_fraction))
        validation = max(1, round(len(patients) * self.schema.validation_fraction))
        if test + validation >= len(patients):
            raise ValueError("Too few patients for three nonempty partitions.")
        test_ids, val_ids, train_ids = np.split(patients, [test, test + validation])
        return {
            name: np.flatnonzero(groups.isin(ids))
            for name, ids in (
                ("train", train_ids), ("validation", val_ids), ("test", test_ids),
            )
        }

    def _feature_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        numeric, categorical = self.identify_feature_types()
        names = numeric + categorical
        if not set(names).issubset(features.columns):
            raise ValueError("Feature table lacks one or more eligible columns.")
        frame = self._normalize(features[names])
        for name in numeric:
            values = pd.to_numeric(frame[name], errors="coerce").astype(float)
            if (~np.isfinite(values) & frame[name].notna()).any():
                raise ValueError(f"Nonfinite or invalid numeric value in {name}.")
            frame[name] = values
        for name in categorical:
            frame[name] = frame[name].to_numpy(dtype=object, na_value=np.nan)
        return frame

    def fit_transform(self, training_features: pd.DataFrame) -> sparse.csr_matrix:
        """Fit median imputation, optional scaling and one-hot encoding on train."""
        numeric, categorical = self.identify_feature_types()
        numeric_steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
        numeric_pipe = make_pipeline(
            *numeric_steps, *([StandardScaler()] if self.policy.scale_numeric else [])
        )
        category_pipe = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="__missing__",
                          keep_empty_features=True),
            OneHotEncoder(handle_unknown="ignore"),
        )
        encoder = ColumnTransformer([
            ("numeric", numeric_pipe, numeric),
            ("categorical", category_pipe, categorical),
        ], sparse_threshold=1.0)
        matrix = encoder.fit_transform(self._feature_frame(training_features))
        self.encoder = encoder
        return sparse.csr_matrix(matrix)

    def transform(self, features: pd.DataFrame) -> sparse.csr_matrix:
        """Transform unseen encounters without updating learned statistics."""
        if self.encoder is None:
            raise ValueError("Fit the preprocessing pipeline before transforming.")
        return sparse.csr_matrix(self.encoder.transform(self._feature_frame(features)))

    def get_feature_metadata(self) -> list[dict[str, object]]:
        """Map every output position back to its source and optional category."""
        if self.encoder is None:
            raise ValueError("Feature metadata requires a fitted encoder.")
        numeric, categorical = self.identify_feature_types()
        sources = [(name, None) for name in numeric]
        if categorical:
            encoder = self.encoder.named_transformers_["categorical"][-1]
            sources.extend((name, str(level)) for name, levels in zip(
                categorical, encoder.categories_, strict=True
            ) for level in levels)
        return [
            {"index": i, "encoded_name": str(encoded), "source": name, "category": level}
            for i, (encoded, (name, level)) in enumerate(zip(
                self.encoder.get_feature_names_out(), sources, strict=True
            ))
        ]

    def _eligibility_audit(self, columns: pd.Index) -> dict[str, str]:
        numeric, categorical = self.identify_feature_types()
        audit = {}
        for name in columns:
            if name in numeric + categorical:
                reason = "Included: approved for configured prediction time."
            elif name == self.schema.target:
                reason = "Excluded: outcome."
            elif name in self.schema.identifiers:
                reason = "Excluded: identifier used only for grouping or audit."
            elif name == "discharge_disposition_id":
                reason = "Excluded: discharge destination or death can proxy the outcome."
            elif name in self.policy.discharge_numeric + self.policy.discharge_categorical:
                reason = "Excluded: current encounter information not known at admission."
            else:
                reason = "Excluded: availability at prediction time not established."
            audit[name] = reason
        return audit

    def save_artifacts(
        self, run_id: str, partitions: dict[str, np.ndarray],
        matrices: dict[str, sparse.csr_matrix], identifiers: pd.DataFrame,
        target: pd.Series, report: dict[str, object],
    ) -> Path:
        """Store aligned row-level artifacts separately from aggregate reporting."""
        root = self.config.paths.root
        processed = root / "data" / "processed" / run_id
        splits = root / "data" / "splits" / run_id
        processed.mkdir(parents=True, exist_ok=False)
        splits.mkdir(parents=True, exist_ok=False)
        for name, indices in partitions.items():
            sparse.save_npz(processed / f"{name}_features.npz", matrices[name])
            np.save(processed / f"{name}_target.npy", target.iloc[indices].to_numpy())
            identifiers.iloc[indices].to_csv(splits / f"{name}_identifiers.csv", index=False)
        joblib.dump(self.encoder, processed / "column_transformer.joblib")
        save_json(processed / "feature_metadata.json", {"features": self.get_feature_metadata()})
        save_json(processed / "preprocessing_config.json", {
            "policy": asdict(self.policy), "seed": self.config.seed,
            "dataset": asdict(self.schema), "label_mapping": self.LABELS,
            "column_missing_codes": self.COLUMN_NULLS,
        })
        path = self.config.paths.results / "tables" / run_id / "data_quality_report.json"
        save_json(path, report)
        return path

    def execute(self, run_id: str) -> Path:
        """Run validation, splitting, train-only fitting and aggregate auditing."""
        raw = self.load_dataset()
        frame = self.validate_dataset(raw)
        target = self.create_target(frame[self.schema.target])
        groups = frame["patient_nbr"]
        partitions = self.split_by_patient(groups)
        patients = {name: set(groups.iloc[idx]) for name, idx in partitions.items()}
        overlaps = {
            f"{left}_{right}": len(patients[left] & patients[right])
            for left, right in combinations(partitions, 2)
        }
        if any(overlaps.values()):
            raise ValueError("Patient overlap detected between partitions.")
        numeric, categorical = self.identify_feature_types()
        features = frame[numeric + categorical]
        matrices = {"train": self.fit_transform(features.iloc[partitions["train"]])}
        for name in ("validation", "test"):
            matrices[name] = self.transform(features.iloc[partitions[name]])

        def distribution(labels: pd.Series) -> dict[str, int]:
            return {str(k): int((labels == k).sum()) for k in (0, 1)}

        report = {
            "raw_records": len(raw), "records": len(frame),
            "duplicates_removed": len(raw) - len(frame),
            "unique_patients": int(groups.nunique()),
            "numeric_features": len(numeric), "categorical_features": len(categorical),
            "encoded_features": matrices["train"].shape[1],
            "missing_percent": (frame.isna().mean() * 100).round(4).to_dict(),
            "target_counts": distribution(target),
            "patient_overlap_counts": overlaps,
            "partitions": {
                name: {"records": len(idx), "patients": len(patients[name]),
                       "target_counts": distribution(target.iloc[idx])}
                for name, idx in partitions.items()
            },
            "prediction_time": self.policy.prediction_time,
            "feature_eligibility": self._eligibility_audit(frame.columns),
            "limitations": [
                "Feature availability is assumed from definitions, not verified by timestamps.",
                "Patient separation is not chronological or hospital-level validation.",
                "No discharge-based cohort exclusions; deaths may be competing events.",
            ],
        }
        return self.save_artifacts(run_id, partitions, matrices,
                                   frame[list(self.schema.identifiers)], target, report)


class PreprocessingExperiment(Experiment):
    """Execute real preprocessing while leaving other stages independent."""

    def run(
        self, config: StudyConfig, run_directory: Path, logger: logging.Logger,
    ) -> str:
        """Link the quality report and data artifacts to the current run ID."""
        path = PreprocessingPipeline(config).execute(run_directory.name)
        logger.info("Preprocessing complete. Data quality report: %s", path)
        return "completed"


STAGE = PreprocessingExperiment("preprocessing")
