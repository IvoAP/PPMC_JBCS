"""Verify and share aligned preprocessing and anonymization artifacts."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import StudyConfig
from utils.file_manager import load_json

PARTS = ("train", "validation", "test")


def run_id(value: str) -> str:
    """Accept an existing framework run identifier, never an arbitrary path."""
    if not re.fullmatch(r"run_[a-f0-9]{32}", value):
        raise ValueError("Invalid run ID.")
    return value


class StudyInputs:
    """Load labels and provenance once; memory-map feature variants on demand."""

    def __init__(self, config: StudyConfig, identifier: str) -> None:
        root = config.paths.root
        self.run = run_id(identifier)
        self.folder = config.paths.results / "anonymized" / self.run
        original = load_json(self.folder / "original" / "transformation_metadata.json")
        self.source = run_id(str(original["source_run"]))
        processed = root / "data" / "processed" / self.source
        schema = processed / "feature_metadata.json"
        features = load_json(schema)["features"]
        if not isinstance(features, list) or not features:
            raise ValueError("Missing encoded feature schema.")
        self.features = features
        self.groups: dict[str, list[int]] = {}
        for i, item in enumerate(features):
            if item["index"] != i or item["source"] in ("readmitted", "patient_nbr", "encounter_id"):
                raise ValueError("Invalid predictor schema.")
            self.groups.setdefault(item["source"], []).append(i)
        self.targets = {}
        self.sidecars = {}
        patients: set[str] = set()
        for part in PARTS:
            labels = processed / f"{part}_target.npy"
            keys = root / "data" / "splits" / self.source / f"{part}_identifiers.csv"
            ids = pd.read_csv(keys, dtype=str, keep_default_na=False)
            members = set(ids["patient_nbr"])
            y = np.load(labels)
            if (not members or "" in members or patients & members
                    or y.shape != (len(ids),) or not np.isin(y, [0, 1]).all()):
                raise ValueError(f"Invalid labels or overlapping patient partition: {part}")
            patients.update(members)
            self.targets[part] = y
            self.sidecars[part] = {"identifiers": str(keys.relative_to(root)),
                                   "target": str(labels.relative_to(root))}
        if len(np.unique(self.targets["train"])) != 2:
            raise ValueError("Both classes are required in training.")
        self.variants = [Path("original")]
        self.variants.extend(sorted(p.parent.relative_to(self.folder)
                                    for p in self.folder.glob("*/*/transformation_metadata.json")))
        if len(self.variants) < 2:
            raise ValueError("No anonymized comparisons were found.")
        self.params = {}
        for variant in self.variants:
            meta = load_json(self.folder / variant / "transformation_metadata.json")
            if (meta["source_run"] != self.source or root / meta["schema"] != schema
                    or meta["sidecars_relative_to_workspace"] != self.sidecars
                    or meta["row_order"] != "same as source partition"):
                raise ValueError(f"Incompatible schema or partition provenance: {variant}")
            self.params[variant] = meta["parameters"]

    def matrix(self, variant: Path, part: str) -> np.ndarray:
        """Check each feature matrix against the shared label and column layout."""
        x = np.load(self.folder / variant / f"{part}_features.npy", mmap_mode="r")
        if x.shape != (len(self.targets[part]), len(self.features)) or not np.isfinite(x).all():
            raise ValueError(f"Invalid {part} feature matrix for {variant}.")
        return x

    def identity_control(self, variant: Path) -> bool:
        """Equal dimensionality alone never establishes semantic compatibility."""
        params = self.params[variant]
        return params["method"] == "PerturbPCA-alpha" and params["alpha"] == 0
