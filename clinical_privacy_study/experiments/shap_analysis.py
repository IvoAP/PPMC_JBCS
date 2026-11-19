"""Grouped interventional permutation SHAP on positive-class probabilities."""

import logging
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import StudyConfig
from experiments.experiment import Experiment
from experiments.study_inputs import StudyInputs, run_id
from utils.file_manager import load_json, save_json
from utils.metrics import compare_scores, ranking


def grouped_shap(
    predict: Callable[[np.ndarray], np.ndarray], background: np.ndarray,
    evaluation: np.ndarray, groups: list[list[int]], permutations: int, seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Average marginal contributions over group orderings and training references."""
    if min(len(background), len(evaluation), permutations) < 1:
        raise ValueError("SHAP budgets must be positive.")
    rng = np.random.default_rng(seed)
    start = predict(background)
    probabilities = predict(evaluation)
    count = len(background)
    donor = np.tile(background, (len(evaluation), 1))
    observed = np.repeat(evaluation, count, axis=0)
    contributions = np.zeros((len(evaluation), len(groups)))
    for _ in range(permutations):
        hybrid = donor.copy()
        previous = np.tile(start, len(evaluation))
        for index in rng.permutation(len(groups)):
            columns = groups[index]
            hybrid[:, columns] = observed[:, columns]
            current = predict(hybrid)
            contributions[:, index] += (current - previous).reshape(-1, count).mean(axis=1)
            previous = current
    contributions /= permutations
    base = float(start.mean())
    if not np.allclose(base + contributions.sum(axis=1), probabilities, atol=1e-8, rtol=0):
        raise ValueError("SHAP additivity failed in positive-class probability space.")
    return base, probabilities, contributions


class ShapExperiment(Experiment):
    def run(self, config: StudyConfig, run_directory: Path, logger: logging.Logger) -> str:
        model_run = run_id(config.models_run or run_directory.name)
        models = config.paths.results / "models" / model_run
        manifest = load_json(models / "model_manifest.json")
        data = StudyInputs(config, str(manifest["variants_run"]))
        if manifest["preprocessing_run"] != data.source or manifest["row_mapping"] != data.sidecars:
            raise ValueError("Model and explanation partitions do not match.")
        output = config.paths.results / "explanations" / run_directory.name / self.name
        original = Path("original")
        rng = np.random.default_rng(config.seed)
        def sample(part: str, size: int) -> np.ndarray:
            n = len(data.targets[part])
            return np.sort(rng.choice(n, min(size, n), replace=False))

        bg = sample("train", config.shap.background_size)
        selected = sample("test", config.shap.explanation_size)
        background = data.matrix(original, "train")[bg]
        evaluation = data.matrix(original, "test")[selected]
        names = list(data.groups)
        save_json(output / "protocol.json", {
            "model_run": model_run, "variants_run": data.run, "preprocessing_run": data.source,
            "space": "original clinical source groups before anonymization",
            "estimand": "interventional grouped Shapley values of class-1 probability",
            "permutations": config.shap.permutations, "seed": config.seed,
            "background_train_rows": bg.tolist(), "explained_test_rows": selected.tolist(),
            "row_mapping": data.sidecars, "groups": data.groups,
        })
        reference, comparisons = {}, []
        for variant in data.variants:
            transform = None
            if variant != original:
                transform = joblib.load(data.folder / variant / "fitted_method.joblib").transform
                if not np.allclose(transform(evaluation), data.matrix(variant, "test")[selected],
                                   atol=1e-8, rtol=0):
                    raise ValueError("Frozen anonymizer does not reproduce saved model inputs.")
            for name in manifest["models"]:
                model = joblib.load(models / variant / name / "model.joblib")

                def predict(x: np.ndarray) -> np.ndarray:
                    encoded = x if transform is None else transform(x)
                    return model.predict_proba(encoded)[:, 1]

                base, probabilities, local = grouped_shap(
                    predict, background, evaluation, list(data.groups.values()),
                    config.shap.permutations, config.seed,
                )
                folder = output / variant / name
                folder.mkdir(parents=True)
                records = pd.DataFrame(local, columns=names)
                records.insert(0, "probability_1", probabilities)
                records.insert(0, "base_probability", base)
                records.insert(0, "test_row_index", selected)
                records.to_csv(folder / "local_shap.csv", index=False)
                scores = dict(zip(names, np.abs(local).mean(axis=0), strict=True))
                save_json(folder / "global_shap.json", {
                    "mean_absolute_shap": scores, "ranks": ranking(scores),
                    "mean_signed_shap": dict(zip(names, local.mean(axis=0), strict=True)),
                })
                if variant == original:
                    reference[name] = scores
                else:
                    comparisons.append({"model": name, "variant": str(variant),
                                        **compare_scores(reference[name], scores,
                                                         config.feature_selection.top_k)})
        save_json(output / "ranking_comparisons.json", {"comparisons": comparisons})
        logger.info("SHAP summaries and local explanations: %s", output)
        return "completed"


STAGE = ShapExperiment("shap_analysis")
