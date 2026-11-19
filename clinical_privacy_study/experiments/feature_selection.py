"""Training-only feature selection on original and reconstructed coordinates."""

import logging
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from config import StudyConfig
from experiments.experiment import Experiment
from experiments.study_inputs import StudyInputs
from models.model_factory import build_model
from utils.file_manager import save_json
from utils.metrics import compare_scores, ranking


class FeatureSelectionExperiment(Experiment):
    def run(self, config: StudyConfig, run_directory: Path, logger: logging.Logger) -> str:
        data = StudyInputs(config, config.variants_run or run_directory.name)
        output = config.paths.results / "tables" / run_directory.name / self.name
        reference, comparisons = {}, []
        y = data.targets["train"]
        for variant in data.variants:
            x = data.matrix(variant, "train")
            original_space = variant == Path("original") or data.identity_control(variant)
            mi = np.zeros(x.shape[1])
            discrete = np.array([original_space and item["category"] is not None
                                 for item in data.features])
            values = np.array(x)
            values[:, discrete] = np.rint(values[:, discrete])
            variable = np.ptp(values, axis=0) > 0
            if variable.any():
                mi[variable] = mutual_info_classif(
                    values[:, variable], y, discrete_features=discrete[variable],
                    n_neighbors=config.feature_selection.mi_neighbors, random_state=config.seed,
                )
            forest = build_model("random_forest", config)
            forest.fit(x, y)
            for method, weights in (("mutual_information", mi),
                                    ("forest_impurity", forest.feature_importances_)):
                encoded = {item["encoded_name"]: float(weights[i])
                           for i, item in enumerate(data.features)}
                groups = {name: float(weights[idx].sum()) for name, idx in data.groups.items()}
                group_ranks = ranking(groups)
                k = config.feature_selection.top_k
                save_json(output / variant / f"{method}.json", {
                    "partition": "train", "variants_run": data.run,
                    "space": "clinical_source_groups" if original_space else
                             "reconstructed_coordinate_groups_by_source_position",
                    "encoded_scores": encoded, "encoded_ranks": ranking(encoded),
                    "group_scores": groups, "group_ranks": group_ranks,
                    "selected_sources": sorted(groups, key=lambda key: (-groups[key], key))[:k],
                })
                if variant == Path("original"):
                    reference[method] = groups
                else:
                    comparisons.append({"variant": str(variant), "method": method,
                                        **compare_scores(reference[method], groups, k)})
        save_json(output / "ranking_comparisons.json", {"comparisons": comparisons})
        logger.info("Feature rankings: %s", output)
        return "completed"


STAGE = FeatureSelectionExperiment("feature_selection")
