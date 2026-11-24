from .anon import RPAR
from .classifiers import evaluate_models, evaluate_noisy_models, format_results, save_classifier_table, save_original_table

from .classifiers import evaluate_models, evaluate_noisy_models, format_results, save_classifier_table
from .linkage import run_linkage
from .load_data import load_heart_disease
from .mia import run_membership_inference
from .noise import add_feature_noise, run_noise_experiment, run_noise_shap, run_noise_wasserstein
from .plot import plot_membership_inference, plot_noise_shap, plot_noise_tradeoff, plot_noise_wasserstein
from .prints import print_datasets, print_preserved_properties
from .shap import compute_shap_importance

TARGET_COLUMN = "target"
NOISE_SIGMAS = (0, 0.1, 0.5, 1, 5)


def main() -> None:
    df = load_heart_disease()

    feature_columns = df.columns.drop(TARGET_COLUMN, errors="ignore")
    X = df[feature_columns].to_numpy(dtype=float)

    anonymizer = RPAR(random_state=42)
    X_anon = anonymizer.fit_transform(X)

    print_datasets(df, X_anon, feature_columns)
    print_preserved_properties(X, X_anon)

    y = df[TARGET_COLUMN].to_numpy()
    tuned_models = ("Logistic Regression", "Random Forest", "KNN", "Gaussian Naive Bayes", "MLP")
    results, best_params = evaluate_models(
        {"original": X, "anonymized": X_anon},
        y,
        tuned_models=tuned_models,
        n_trials=5,
        n_splits=5,
        random_state=42,
        verbose=2,
    )
    for (dataset_name, model_name), params in best_params.items():
        print(f"Best params [{dataset_name} / {model_name}]: {params}")
    print(format_results(results).to_string())

    mia_results = run_membership_inference(X, y, NOISE_SIGMAS)
    print("Membership inference AUC (default model parameters):")
    print(mia_results.groupby(["dataset", "model", "sigma"], dropna=False)["auc"]
          .agg(["mean", "std"]).round(4).to_string())
    plot_membership_inference(mia_results)

    shap_importance = compute_shap_importance(X, X_anon, y, feature_columns, best_params, random_state=42)
    noise_shap = run_noise_shap(X, X_anon, y, best_params, NOISE_SIGMAS, random_state=42)
    plot_noise_shap(noise_shap, shap_importance)

    linkage_results, _ = run_linkage(X, random_states=range(10))
    summary = linkage_results.groupby(["attack", "k"])["match_rate"].agg(["mean", "std"])
    print(summary.round(4).to_string())

    noise_linkage, noise_utility = run_noise_experiment(
        X, X_anon, y, best_params, NOISE_SIGMAS, random_states=range(10)
    )
    save_original_table(results)
    for sigma in NOISE_SIGMAS:
        X_noisy = add_feature_noise(X_anon, X, sigma, random_state=42)
        noisy_results = evaluate_noisy_models(X_noisy, y, best_params, random_state=42)
        save_classifier_table(noisy_results, sigma)
    original_accuracy = results.loc[results["dataset"] == "original", "accuracy_mean"].mean()
    plot_noise_tradeoff(noise_linkage, noise_utility, n_records=X.shape[0], original_accuracy=original_accuracy)
    noise_wasserstein = run_noise_wasserstein(X, X_anon, NOISE_SIGMAS, random_states=range(10))
    plot_noise_wasserstein(noise_wasserstein)
    noise_summary = noise_utility.groupby("sigma")["accuracy"].agg(["mean", "std"])
    print(noise_summary.round(4).to_string())


if __name__ == "__main__":
    main()