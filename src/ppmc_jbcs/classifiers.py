import optuna
import pandas as pd
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from optuna.trial import FixedTrial

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1_score": "f1_macro",
}

TUNING_METRIC = "f1_macro"

TABLES_DIR = Path(__file__).resolve().parents[2] / "tables"


def build_models(random_state=42):
    """Return the classifiers to evaluate, each preceded by feature standardization."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=1000, random_state=random_state),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
    }
    return {name: make_pipeline(StandardScaler(), model) for name, model in models.items()}


SEARCH_SPACES = {
    "Logistic Regression": lambda trial: {
        "logisticregression__C": trial.suggest_float("C", 1e-3, 1e2, log=True),
    },
    "Random Forest": lambda trial: {
        "randomforestclassifier__n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "randomforestclassifier__max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 10, 20]),
        "randomforestclassifier__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    },
    "KNN": lambda trial: {
        "kneighborsclassifier__n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
        "kneighborsclassifier__weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
    },
    "Gaussian Naive Bayes": lambda trial: {
        "gaussiannb__var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-3, log=True),
    },
    "MLP": lambda trial: {
        "mlpclassifier__hidden_layer_sizes": (trial.suggest_int("hidden_width", 16, 128),),
        "mlpclassifier__alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "mlpclassifier__learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    },
}


def tune_model(model_name, X, y, n_trials, random_state=42, n_splits=5):
    """Search hyperparameters with Optuna using an inner stratified K-fold on ``X``.

    Returns the pipeline parameters of the best trial. Only the data passed in is used,
    so the outer evaluation folds are never seen during tuning.
    """
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search_space = SEARCH_SPACES[model_name]

    def objective(trial):
        model = build_models(random_state)[model_name].set_params(**search_space(trial))
        return cross_val_score(model, X, y, cv=inner_cv, scoring=TUNING_METRIC).mean()

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return search_space(FixedTrial(study.best_params))


def _metrics_row(dataset_name, model_name, scores):
    """Summarize cross-validation scores as one row with mean and std per metric."""
    row = {"dataset": dataset_name, "model": model_name}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()
    return row


def evaluate_models(datasets, y, tuned_models=(), n_trials=10, n_splits=5, random_state=42, verbose=2):
    """Cross-validate every model on every dataset with stratified K-fold.

    ``datasets`` maps a dataset name (e.g. "original", "anonymized") to its feature matrix.
    Models listed in ``tuned_models`` are tuned with Optuna separately on each dataset.
    Returns the metrics DataFrame (mean and standard deviation per model) and the best
    parameters found for each (dataset, model) pair.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    best_params = {}
    for dataset_name, X in datasets.items():
        for model_name, model in build_models(random_state).items():
            if model_name in tuned_models:
                params = tune_model(model_name, X, y, n_trials, random_state, n_splits)
                model.set_params(**params)
                best_params[(dataset_name, model_name)] = params
            scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, verbose=verbose)
            rows.append(_metrics_row(dataset_name, model_name, scores))
    return pd.DataFrame(rows), best_params


def evaluate_noisy_models(X_noisy, y, best_params, random_state=42, n_splits=5):
    """Cross-validate every model on noisy anonymized data, with the hyperparameters tuned on clean data.

    Returns one row per model, in the same format as ``evaluate_models``.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for model_name, model in build_models(random_state).items():
        model.set_params(**best_params.get(("anonymized", model_name), {}))
        scores = cross_validate(model, X_noisy, y, cv=cv, scoring=SCORING)
        rows.append(_metrics_row("noisy", model_name, scores))
    return pd.DataFrame(rows)


def save_classifier_table(results, sigma, output_dir=TABLES_DIR):
    """Save the classifier results of one noise level to ``output_dir/result_classifier_<sigma>.csv``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"result_classifier_{sigma:g}.csv"
    results.to_csv(file_path, index=False)
    print(f"Classifier table saved to {file_path}")
    return file_path


def save_original_table(results, output_dir=TABLES_DIR):
    """Save the non-anonymized (original) classifier results to ``output_dir/result_non_anon.csv``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "result_non_anon.csv"
    results[results["dataset"] == "original"].to_csv(file_path, index=False)
    print(f"Non-anonymized table saved to {file_path}")
    return file_path


def format_results(results):
    """Pivot the evaluation results into a table: models as rows, metric and dataset as columns."""
    formatted = results[["dataset", "model"]].copy()
    for metric in SCORING:
        formatted[metric] = [
            f"{mean:.3f} ± {std:.3f}"
            for mean, std in zip(results[f"{metric}_mean"], results[f"{metric}_std"])
        ]
    table = formatted.pivot(index="model", columns="dataset", values=list(SCORING))
    table = table.reindex(results["model"].unique())
    return table.rename_axis(index="Model", columns=["Metric", "Dataset"])
