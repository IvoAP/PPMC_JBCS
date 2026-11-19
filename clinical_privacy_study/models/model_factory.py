"""Construct fresh seeded binary classifiers for each representation."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from config import StudyConfig

Classifier = LogisticRegression | RandomForestClassifier | MLPClassifier


def build_model(name: str, config: StudyConfig) -> Classifier:
    """Use fixed hyperparameters without row-level internal validation splits."""
    settings = config.classification
    if name == "logistic_regression":
        return LogisticRegression(C=settings.logistic_c, max_iter=settings.logistic_iterations,
                                  random_state=config.seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=settings.forest_trees,
                                      min_samples_leaf=settings.forest_min_leaf,
                                      random_state=config.seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=settings.mlp_hidden,
                             max_iter=settings.mlp_iterations, early_stopping=False,
                             random_state=config.seed)
    raise ValueError(f"Unknown classifier: {name}")
