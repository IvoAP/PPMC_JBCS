"""Versioned, variant-blind prompt with a small, auditable claim vocabulary."""

import json
import math
import re
from collections.abc import Mapping

LIMITATION = "The available information is insufficient to establish clinical causes."
TEMPLATE = """Explain the main factors influencing the model's 30-day readmission
probability, relative to its SHAP baseline. These are associations, not causes.
Use only the supplied JSON. Do not infer measurements, diagnoses or treatment.
Write one English sentence per line. For each main factor use exactly:
FEATURE increased the predicted probability (SHAP VALUE).
Use decreased for negative and did not change for zero contributions.
Copy feature names and signed numeric values. Do not add a diagnosis, advice,
or other claims. End with: """ + LIMITATION


def minimized_input(values: Mapping[str, float], probability: float, k: int) -> dict:
    """Disclose only top signed contributions and rounded model outputs."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid class probability.")
    features = []
    for name, value in values.items():
        if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name)
                or name in {"patient_nbr", "encounter_id", "readmitted", "test_row_index"}
                or not math.isfinite(value)):
            raise ValueError("Invalid clinical feature attribution.")
    for rank, (name, value) in enumerate(
        sorted(values.items(), key=lambda item: (-abs(item[1]), item[0]))[:k], 1
    ):
        score = round(float(value), 4)
        direction = "increased" if score > 0 else "decreased" if score < 0 else "did not change"
        features.append({"feature": name, "shap": score, "direction": direction, "rank": rank})
    return {"predicted_class": int(probability > 0.5), "probability_1": round(probability, 4),
            "confidence": round(max(probability, 1 - probability), 4), "factors": features}


def generation_prompt(evidence: dict) -> str:
    return TEMPLATE + "\nEvidence:\n" + json.dumps(evidence, sort_keys=True, allow_nan=False)
