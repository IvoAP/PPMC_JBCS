"""Binary encounter-level metrics for the prespecified 30-day outcome."""

import numpy as np
from sklearn import metrics


def evaluate(y: np.ndarray, predicted: np.ndarray, probability: np.ndarray) -> dict:
    """PR-AUC is trapezoidal area; report AP separately and null undefined areas."""
    both = len(np.unique(y)) == 2
    precision, recall, _ = metrics.precision_recall_curve(y, probability) if both else ([], [], [])
    return {
        "accuracy": metrics.accuracy_score(y, predicted),
        "balanced_accuracy": metrics.balanced_accuracy_score(y, predicted),
        "precision": metrics.precision_score(y, predicted, zero_division=0),
        "recall": metrics.recall_score(y, predicted, zero_division=0),
        "macro_f1": metrics.f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0),
        "weighted_f1": metrics.f1_score(y, predicted, average="weighted", zero_division=0),
        "mcc": metrics.matthews_corrcoef(y, predicted),
        "roc_auc": metrics.roc_auc_score(y, probability) if both else None,
        "pr_auc": metrics.auc(recall, precision) if both else None,
        "average_precision": metrics.average_precision_score(y, probability) if both else None,
        "confusion_matrix": metrics.confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }
