"""Train once per representation and evaluate prespecified held-out scenarios."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import StudyConfig
from experiments.experiment import Experiment
from experiments.study_inputs import PARTS, StudyInputs
from models.model_evaluation import evaluate
from models.model_factory import build_model
from utils.file_manager import save_json


class ClassificationExperiment(Experiment):
    def run(self, config: StudyConfig, run_directory: Path, logger: logging.Logger) -> str:
        data = StudyInputs(config, config.variants_run or run_directory.name)
        output = config.paths.results / "models" / run_directory.name
        tables = config.paths.results / "tables" / run_directory.name / self.name
        original = Path("original")
        records, skipped = [], []
        for variant in data.variants:
            matrices = {part: data.matrix(variant, part) for part in PARTS}
            scenarios = {"A" if variant == original else "B": matrices}
            if data.identity_control(variant):
                baseline = {part: data.matrix(original, part) for part in PARTS}
                if not all(np.allclose(matrices[p], baseline[p], atol=1e-8, rtol=0) for p in PARTS):
                    raise ValueError(f"Identity control differs from original: {variant}")
                scenarios["C"] = baseline
            elif variant != original:
                skipped.append({"variant": str(variant), "scenario": "C",
                                "reason": "Perturbation changes coordinate semantics."})
            for name in config.classification.models:
                model = build_model(name, config)
                model.fit(matrices["train"], data.targets["train"])
                folder = output / variant / name
                folder.mkdir(parents=True)
                joblib.dump(model, folder / "model.joblib")
                save_json(folder / "parameters.json", model.get_params())
                for scenario, inputs in scenarios.items():
                    for part in ("validation", "test"):
                        x, y = inputs[part], data.targets[part]
                        probability = model.predict_proba(x)[:, 1]
                        predicted = model.predict(x)
                        rows = pd.DataFrame({"row_index": np.arange(len(y)), "target": y,
                                             "prediction": predicted, "probability_1": probability})
                        rows.to_csv(folder / f"{scenario}_{part}_predictions.csv", index=False)
                        records.append({"model": name, "training_variant": str(variant),
                                        "scenario": scenario, "partition": part,
                                        **evaluate(y, predicted, probability)})
        save_json(tables / "evaluation.json", {"results": records, "skipped": skipped})
        save_json(output / "model_manifest.json", {
            "variants_run": data.run, "preprocessing_run": data.source,
            "models": config.classification.models, "row_mapping": data.sidecars,
        })
        logger.info("Classification results: %s", tables)
        return "completed"


STAGE = ClassificationExperiment("classification")
