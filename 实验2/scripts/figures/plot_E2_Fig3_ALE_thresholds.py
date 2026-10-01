#!/usr/bin/env python3
"""Figure E2-3: Key variable ALE curves with model-supported thresholds."""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\人参种在哪\实验2"
ALE_DIR = os.path.join(ROOT, "05_ale")
TH_DIR = os.path.join(ROOT, "06_thresholds")
IMP_DIR = os.path.join(ROOT, "03_global_importance")
FIG_DIR = os.path.join(ROOT, "11_figures")
DATA_DIR = os.path.join(ROOT, "12_figure_data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def main():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
    })

    # Get Tier 1 drivers
    tier_df = pd.read_csv(os.path.join(IMP_DIR, "driver_tier_summary.csv"))
    tier1 = tier_df[tier_df["tier"] == "Tier 1"]["variable"].tolist()
    if len(tier1) > 6:
        tier1 = tier1[:6]
    print(f"Plotting ALE for Tier 1 drivers: {tier1}")

    # Load ALE data
    ale_all = pd.read_csv(os.path.join(ALE_DIR, "ale_1d_all_models.csv"))
    consensus_ale = pd.read_csv(os.path.join(ALE_DIR, "ensemble_consensus_ALE.csv"))

    # Load thresholds
    th_path = os.path.join(TH_DIR, "threshold_consensus_summary.csv")
    thresholds = pd.read_csv(th_path) if os.path.exists(th_path) else None

    # Load data for rug plots
    sample_path = os.path.join(ROOT, "02_analysis_dataset", "explainability_sample.csv")
    data_sample = pd.read_csv(sample_path) if os.path.exists(sample_path) else None

    n_vars = len(tier1)
    ncols = min(3, n_vars)
    nrows = int(np.ceil(n_vars / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    model_colors = {
        "maxent": "#E41A1C",
        "random_forest": "#377EB8",
        "xgboost": "#4DAF4A",
        "brt": "#984EA3",
    }

    plot_data_all = []

    for idx, var in enumerate(tier1):
        ax = axes[idx]
        var_ale = ale_all[ale_all["variable"] == var]

        # Per-model ALE (P5-P95 only)
        for model in var_ale["model"].unique():
            m_data = var_ale[var_ale["model"] == model]
            p5, p95 = m_data["P5"].values[0], m_data["P95"].values[0]
            central = m_data[(m_data["bin_center"] >= p5) & (m_data["bin_center"] <= p95)]
            if len(central) > 2:
                ax.plot(central["bin_center"], central["ale_mean"],
                       color=model_colors.get(model, "gray"), alpha=0.4,
                       linewidth=1, label=model.replace("_", " ").title())

        # Consensus ALE (thick line)
        c_ale = consensus_ale[consensus_ale["variable"] == var]
        if len(c_ale) > 2:
            ax.plot(c_ale["bin_center"], c_ale["ensemble_ALE_z"],
                   color="black", linewidth=2.5, label="Ensemble consensus")
            plot_data_all.append(pd.DataFrame({
                "variable": var,
                "bin_center": c_ale["bin_center"].values,
                "ensemble_ALE_z": c_ale["ensemble_ALE_z"].values,
            }))

        # Zero line
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # Threshold annotations
        if thresholds is not None:
            var_th = thresholds[thresholds["variable"] == var]
            for _, vt in var_th.iterrows():
                if vt["status"] in ["robust", "moderate"] and not np.isnan(vt["median_threshold"]):
                    ax.axvline(x=vt["median_threshold"], color="#D73027", linestyle="--",
                             linewidth=1.2, alpha=0.7)
                    if not np.isnan(vt["transition_interval_low"]):
                        ax.axvspan(vt["transition_interval_low"], vt["transition_interval_high"],
                                  alpha=0.1, color="#D73027")

        # Rug plot for data support
        if data_sample is not None and var in data_sample.columns:
            var_vals = data_sample[var].dropna().values
            # Sample for rug
            rug_n = min(500, len(var_vals))
            rug_vals = np.random.choice(var_vals, rug_n, replace=False)
            ylim = ax.get_ylim()
            ax.plot(rug_vals, [ylim[0]] * rug_n, "|", color="black", alpha=0.1, markersize=3)

        ax.set_title(f"{var}")
        ax.set_xlabel(var)
        ax.set_ylabel("ALE (centered)")
        if idx == 0:
            ax.legend(fontsize=6, loc="best")

    # Hide unused axes
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Figure E2-3: Key Variable ALE Curves with Model-Supported Thresholds",
                fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "E2_Fig3_ALE_thresholds.png")
    fig.savefig(fig_path, dpi=600)
    fig.savefig(fig_path.replace(".png", ".pdf"))
    plt.close()
    print(f"Figure E2-3 saved to {fig_path}")

    # Save plot data
    if plot_data_all:
        pd.concat(plot_data_all, ignore_index=True).to_csv(
            os.path.join(DATA_DIR, "E2_Fig3_ALE_curves.csv"), index=False
        )
    if thresholds is not None:
        thresholds.to_csv(os.path.join(DATA_DIR, "E2_Fig3_thresholds.csv"), index=False)
    print("Figure data saved")

if __name__ == "__main__":
    main()
