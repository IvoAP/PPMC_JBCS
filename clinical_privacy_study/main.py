"""Command-line entry point for the Clinical Privacy and Explainability Framework."""

import argparse
import logging
import platform
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from anonymization import anonymization_adapter
from config import Paths, Preprocessing, StudyConfig
from experiments import (
    classification,
    feature_selection,
    llm_evaluation,
    preprocessing,
    privacy_attacks,
    shap_analysis,
    statistical_analysis,
)
from utils.file_manager import create_directories, save_metadata
from utils.logger import run_logger
from utils.reproducibility import seed_generators
from visualization import report_generator

PIPELINE = (
    preprocessing.STAGE,
    anonymization_adapter.STAGE,
    classification.STAGE,
    feature_selection.STAGE,
    shap_analysis.STAGE,
    llm_evaluation.STAGE,
    privacy_attacks.STAGE,
    statistical_analysis.STAGE,
    report_generator.STAGE,
)
STAGES = {stage.name: stage for stage in PIPELINE}


def run(selection: str, config: StudyConfig) -> Path:
    """Dispatch only the selected stages and return their manifest directory."""
    if selection != "all" and selection not in STAGES:
        raise ValueError(f"Unknown experiment: {selection}")
    stages = PIPELINE if selection == "all" else (STAGES[selection],)
    create_directories(config.paths.directories)
    directory = config.paths.results / "logs" / f"run_{uuid4().hex}"
    directory.mkdir()
    snapshot = asdict(config)
    snapshot["paths"]["root"] = str(config.paths.root.resolve())
    with run_logger(directory / "run.log") as logger:
        logger.info("Run directory: %s", directory)
        outcomes = {}
        started = datetime.now(timezone.utc)
        for stage in stages:
            seed_generators(config.seed)
            outcomes[stage.name] = stage.run(config, directory, logger)
        save_metadata(directory, {
            "started_utc": started.isoformat(),
            "selection": selection,
            "config": snapshot,
            "versions": {
                "python": platform.python_version(),
                **{name: version(name) for name in (
                    "numpy", "pandas", "scikit-learn", "scipy", "joblib",
                )},
            },
            "stages": outcomes,
        })
    return directory


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and report configuration or filesystem errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=(*STAGES, "all"))
    parser.add_argument("--prediction-time", choices=("admission", "discharge"),
                        default=Preprocessing().prediction_time)
    parser.add_argument("--workspace", type=Path, default=Paths().root,
                        help="Base directory for data and results.")
    args = parser.parse_args(argv)
    try:
        run(args.experiment, StudyConfig(
            paths=Paths(args.workspace.resolve()),
            preprocessing=Preprocessing(prediction_time=args.prediction_time),
        ))
    except (OSError, ValueError) as error:
        logging.error("Unable to run %s: %s", args.experiment, error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
