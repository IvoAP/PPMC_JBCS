"""Run matched empirical transformations on a single preprocessing artifact set."""

import csv
import logging
import re
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy import sparse

from anonymization.base import AnonymizationMethod
from anonymization.perturbpca import PerturbPCAAlpha
from config import StudyConfig
from experiments.experiment import Experiment
from utils.file_manager import load_json, save_json

PARTITIONS = ("train", "validation", "test")


def reconstruction_diagnostics(
    original: np.ndarray, result: np.ndarray, blocks: dict[str, list[int]],
    tolerance: float,
) -> dict[str, object]:
    """Audit the raw reconstruction; never round, clip or decode its values."""
    if result.shape != original.shape or not np.isfinite(result).all():
        raise ValueError("Invalid transformation output: shape mismatch or nonfinite values.")
    invalid_counts = {}
    invalid_rows = np.zeros(len(original), dtype=bool)
    for source, indices in blocks.items():
        values = result[:, indices]
        zero = np.abs(values) <= tolerance
        one = np.abs(values - 1) <= tolerance
        valid = (zero | one).all(axis=1) & (one.sum(axis=1) == 1)
        # OneHotEncoder(handle_unknown="ignore") emits all zeros for unseen input.
        original_unknown = (original[:, indices] == 0).all(axis=1)
        invalid = ~(valid | (zero.all(axis=1) & original_unknown))
        invalid_counts[source] = int(invalid.sum())
        invalid_rows |= invalid
    error = result - original
    return {
        "shape": list(result.shape), "rmse": float(np.sqrt(np.mean(error**2))),
        "max_absolute_error": float(np.abs(error).max()),
        "invalid_categorical_rows": int(invalid_rows.sum()),
        "invalid_counts_by_source": invalid_counts, "categorical_tolerance": tolerance,
    }


class AnonymizationAdapter:
    """Keep sidecars outside the method interface; use the preprocessing schema."""

    def __init__(self, config: StudyConfig, source_run: str) -> None:
        if not re.fullmatch(r"run_[a-f0-9]{32}", source_run):
            raise ValueError("Expected a preprocessing run ID of the form run_<32 hex digits>.")
        self.config = config
        self.source_run = source_run
        root = config.paths.root
        processed = root / "data" / "processed" / source_run
        splits = root / "data" / "splits" / source_run
        schema_path = processed / "feature_metadata.json"
        features = load_json(schema_path)["features"]
        settings = load_json(processed / "preprocessing_config.json")["dataset"]
        if not isinstance(features, list) or not features or not isinstance(settings, dict):
            raise ValueError("Invalid preprocessing feature metadata.")
        forbidden = {settings["target"], *settings["identifiers"],
                     "patient_nbr", "encounter_id", "readmitted"}
        self.blocks: dict[str, list[int]] = {}
        for i, feature in enumerate(features):
            if feature["index"] != i or feature["source"] in forbidden:
                raise ValueError("Unexpected feature order, target or identifier in schema.")
            if feature["category"] is not None:
                self.blocks.setdefault(feature["source"], []).append(i)
        self.inputs = {}
        self.sidecars = {}
        for name in PARTITIONS:
            target_path = processed / f"{name}_target.npy"
            ids_path = splits / f"{name}_identifiers.csv"
            matrix = sparse.load_npz(processed / f"{name}_features.npz").toarray()
            with ids_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                if next(reader, []) != settings["identifiers"]:
                    raise ValueError(f"Unexpected identifier schema for {name}.")
                rows = sum(1 for _ in reader)
            if (not rows or matrix.shape != (rows, len(features))
                    or np.load(target_path, mmap_mode="r").shape != (rows,)
                    or not np.isfinite(matrix).all()):
                raise ValueError(f"Inconsistent or nonfinite preprocessing artifacts for {name}.")
            matrix.setflags(write=False)
            self.inputs[name] = matrix
            self.sidecars[name] = {
                "target": str(target_path.relative_to(root)),
                "identifiers": str(ids_path.relative_to(root)),
            }
        self.schema_location = str(schema_path.relative_to(root))

    def export(
        self, directory: Path, method: AnonymizationMethod | None,
    ) -> int:
        """Fit once, write aligned outputs, and return rows with invalid categories."""
        started = perf_counter()
        directory.mkdir(parents=True, exist_ok=False)
        if method is not None:
            method.fit(self.inputs["train"])
            joblib.dump(method, directory / "fitted_method.joblib")
        diagnostics = {}
        invalid = 0
        for name, original in self.inputs.items():
            result = original if method is None else method.transform(original)
            report = reconstruction_diagnostics(
                original, result, self.blocks, self.config.anonymization.categorical_tolerance
            )
            diagnostics[name] = report
            invalid += int(report["invalid_categorical_rows"])
            np.save(directory / f"{name}_features.npy", result, allow_pickle=False)
        save_json(directory / "reconstruction_diagnostics.json", diagnostics)
        save_json(directory / "transformation_metadata.json", {
            "parameters": method.get_metadata() if method else {"method": "original"},
            "fit_partition": "train" if method else None,
            "fitted_state": "fitted_method.joblib" if method else None,
            "source_run": self.source_run, "schema": self.schema_location,
            "sidecars_relative_to_workspace": self.sidecars,
            "row_order": "same as source partition",
            "categorical_handling": "raw encoded reconstruction; no repair",
            "privacy_guarantee": "No formal differential privacy guarantee.",
            "execution_seconds": perf_counter() - started,
        })
        return invalid


class AnonymizationExperiment(Experiment):
    """Export the unmodified comparator and the configured alpha sweeps."""

    def run(
        self, config: StudyConfig, run_directory: Path, logger: logging.Logger,
    ) -> str:
        settings = config.anonymization
        source = settings.source_run or run_directory.name
        adapter = AnonymizationAdapter(config, source)
        output = config.paths.results / "anonymized" / run_directory.name
        adapter.export(output / "original", None)
        modes = {"global": None}
        if settings.clusters is not None:
            modes["clustered"] = settings.clusters
        for mode, clusters in modes.items():
            for alpha in settings.alphas:
                suffix = f"{alpha * 100:03g}"
                destination = output / mode / f"alpha_{suffix}"
                invalid = adapter.export(
                    destination, PerturbPCAAlpha(alpha, config.seed, clusters)
                )
                if invalid:
                    logger.warning("%s alpha=%s: %d rows with invalid categorical blocks; "
                                   "saved unchanged.", mode, alpha, invalid)
        logger.info("Empirical transformation datasets: %s", output)
        return "completed"


STAGE = AnonymizationExperiment("anonymization")
