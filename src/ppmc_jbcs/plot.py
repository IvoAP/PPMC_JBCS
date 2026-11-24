from pathlib import Path

import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 19,
})


def _save_tradeoff_figure(fig, file_path):
    fig.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Noise trade-off figure saved to {file_path}")


def plot_noise_tradeoff(linkage_results, utility_results, n_records, original_accuracy, output_dir=FIGURES_DIR):
    """Save the privacy and utility curves against the noise scale sigma, as two separate figures.

    Privacy is the Top-1 match rate of the distance-profile attack; utility is the mean accuracy
    over the models. Sigma levels are evenly spaced, so the x-axis is categorical, not proportional.
    Legends are placed above each plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top1 = linkage_results[
        (linkage_results["attack"] == "Distance-profile matching") & (linkage_results["k"] == 1)
    ]
    privacy = top1.groupby("sigma")["match_rate"].agg(["mean", "std"])
    utility = utility_results.groupby("sigma")["accuracy"].agg(["mean", "std"])
    positions = list(range(len(privacy)))
    labels = [f"{sigma:g}" for sigma in privacy.index]

    privacy_fig, privacy_ax = plt.subplots(figsize=(7, 6))
    privacy_ax.errorbar(
        positions, privacy["mean"], yerr=privacy["std"], marker="o", color="tab:red", capsize=4,
        label="Top-1 match rate (mean ± std over seeds)",
    )
    privacy_ax.axhline(1 / n_records, linestyle="--", color="gray", label=f"random guess (1/{n_records})")
    for x, value in zip(positions, privacy["mean"]):
        privacy_ax.annotate(f"{value:.2f}", (x, value), textcoords="offset points", xytext=(0, 9), ha="center")
    privacy_ax.set_ylabel("Top-1 match rate")
    privacy_ax.set_ylim(-0.05, 1.15)
    privacy_ax.set_xticks(positions, labels)
    privacy_ax.set_xlabel("sigma (noise scale, relative to feature sd)")
    privacy_ax.grid(alpha=0.3)
    privacy_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1)
    privacy_fig.suptitle("Privacy: distance-profile attack (lower is safer)", y=1.0)
    privacy_fig.tight_layout()
    _save_tradeoff_figure(privacy_fig, output_dir / "noise_privacy.png")

    utility_fig, utility_ax = plt.subplots(figsize=(7, 6))
    utility_ax.errorbar(
        positions, utility["mean"], yerr=utility["std"], marker="s", color="tab:blue", capsize=4,
        label="mean accuracy over 6 models (± std across models)",
    )
    utility_ax.axhline(original_accuracy, linestyle=":", color="black", label=f"original data ({original_accuracy:.2f})")
    for x, value in zip(positions, utility["mean"]):
        utility_ax.annotate(f"{value:.2f}", (x, value), textcoords="offset points", xytext=(0, 9), ha="center")
    utility_ax.set_ylabel("accuracy")
    utility_ax.set_xticks(positions, labels)
    utility_ax.set_xlabel("sigma (noise scale, relative to feature sd)")
    utility_ax.grid(alpha=0.3)
    utility_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1)
    utility_fig.suptitle("Utility: classifier accuracy (higher is better)", y=1.0)
    utility_fig.tight_layout()
    _save_tradeoff_figure(utility_fig, output_dir / "noise_utility.png")


COMPARISON_TITLES = {
    "isolated": "Noise alone: clean RPAR vs noisy",
    "total": "Total distortion: original vs noisy",
}


def plot_noise_wasserstein(results, output_dir=FIGURES_DIR):
    """Save one figure per comparison, showing the Wasserstein distances against sigma.

    Distances are in units of each original feature's sd, so both metrics are comparable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for comparison, group in results.groupby("comparison", sort=False):
        summary = group.groupby("sigma")[["marginal", "sliced"]].agg(["mean", "std"])
        positions = list(range(len(summary)))
        labels = [f"{sigma:g}" for sigma in summary.index]

        fig, ax = plt.subplots(figsize=(8, 5))
        for metric, label, marker, color in (
            ("marginal", "mean per-feature Wasserstein", "o", "tab:green"),
            ("sliced", "sliced Wasserstein (joint)", "s", "tab:purple"),
        ):
            values = summary[metric]
            ax.errorbar(
                positions, values["mean"], yerr=values["std"], marker=marker, color=color, capsize=4,
                label=f"{label} (mean ± std over seeds)",
            )
            for x, value in zip(positions, values["mean"]):
                ax.annotate(f"{value:.2f}", (x, value), textcoords="offset points", xytext=(0, 9), ha="center")

        ax.set_xticks(positions, labels)
        ax.set_xlabel("sigma (noise scale, relative to feature sd)")
        ax.set_ylabel("distance (in original feature sd units)")
        ax.set_title(COMPARISON_TITLES[comparison])
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()

        file_path = output_dir / f"noise_wasserstein_{comparison}.png"
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        print(f"Noise Wasserstein figure saved to {file_path}")


def plot_membership_inference(results, output_dir=FIGURES_DIR):
    """Save membership AUC by noise level for each model, with the original-data baseline.

    Error bars and the baseline band are standard deviations across repeated splits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, group in results.groupby("model", sort=False):
        noisy = group[group["dataset"] == "anonymized"].groupby("sigma")["auc"]
        means = noisy.mean()
        deviations = noisy.std(ddof=0)
        original = group.loc[group["dataset"] == "original", "auc"]
        original_mean = original.mean()
        original_std = original.std(ddof=0)
        positions = list(range(len(means)))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(positions, means, yerr=deviations, marker="o", capsize=4,
                    label="RPAR + noise (mean ± std)")
        ax.axhline(original_mean, color="tab:orange", linestyle=":",
                   label=f"Original data (mean = {original_mean:.2f})")
        ax.axhspan(original_mean - original_std, original_mean + original_std,
                   color="tab:orange", alpha=0.15, label="Original data (± std)")
        ax.axhline(0.5, color="gray", linestyle="--", label="Random guess (0.5)")
        ax.set_xticks(positions, [f"{sigma:g}" for sigma in means.index])
        ax.set_xlabel("sigma (noise scale, relative to training feature sd)")
        ax.set_ylabel("Membership inference AUC")
        ax.set_ylim(0, 1)
        fig.suptitle(f"{model_name}: membership inference")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01))
        fig.tight_layout()
        file_path = output_dir / f"mia_noise_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Membership inference figure saved to {file_path}")


def plot_noise_shap(noise_importance, original_importance, output_dir=FIGURES_DIR):
    """Save separate original-feature and anonymized-axis SHAP figures per model.

    Cells show each axis's share of the total mean |SHAP| at that sigma. The original features are
    shown as a bar chart, because they are not the same variables as the anonymized axes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_importance = noise_importance.copy()
    noise_importance["share"] = 100 * noise_importance["importance"] / noise_importance.groupby(
        ["model", "sigma"]
    )["importance"].transform("sum")

    for model_name, model_noise in noise_importance.groupby("model", sort=False):
        heatmap = model_noise.pivot(index="feature", columns="sigma", values="share")
        heatmap = heatmap.sort_values(heatmap.columns[0], ascending=False)

        original = original_importance[
            (original_importance["dataset"] == "original") & (original_importance["model"] == model_name)
        ]
        original = original.assign(share=100 * original["importance"] / original["importance"].sum())
        original = original.sort_values("share")

        model_slug = model_name.lower().replace(" ", "_")
        original_fig, original_ax = plt.subplots(figsize=(8, 7))
        original_ax.barh(original["feature"], original["share"], color="steelblue")
        original_ax.set_title(f"{model_name}: original features (share %)")
        original_ax.set_xlabel("share of total mean |SHAP| (%)")
        original_fig.tight_layout()
        original_path = output_dir / f"shap_noise_{model_slug}_original.png"
        original_fig.savefig(original_path, dpi=150, bbox_inches="tight")
        plt.close(original_fig)
        print(f"Original SHAP figure saved to {original_path}")

        fig, heatmap_ax = plt.subplots(figsize=(10, 7))
        image = heatmap_ax.imshow(heatmap.to_numpy(), cmap="viridis", aspect="auto")
        threshold = heatmap.to_numpy().max() / 2
        for row in range(heatmap.shape[0]):
            for col in range(heatmap.shape[1]):
                value = heatmap.iloc[row, col]
                color = "black" if value > threshold else "white"
                heatmap_ax.text(col, row, f"{value:.1f}", ha="center", va="center", color=color)
        heatmap_ax.set_xticks(range(heatmap.shape[1]), [f"{sigma:g}" for sigma in heatmap.columns])
        heatmap_ax.set_yticks(range(heatmap.shape[0]), heatmap.index)
        heatmap_ax.set_xlabel("sigma (noise scale, relative to feature sd)")
        heatmap_ax.set_title(f"{model_name}: anonymized axes (share %)")
        fig.colorbar(image, ax=heatmap_ax, label="share of total mean |SHAP| (%)")

        fig.tight_layout()
        file_path = output_dir / f"shap_noise_{model_slug}_anonymized.png"
        fig.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"SHAP noise figure saved to {file_path}")

