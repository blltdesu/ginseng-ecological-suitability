#!/usr/bin/env python3
"""Figure E2-2: SHAP summary (adapted for raster-based analysis)."""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = r"E:\人参种在哪\实验2"
IMP_DIR = os.path.join(ROOT, "03_global_importance")
DATA_DIR = os.path.join(ROOT, "02_analysis_dataset")
FIG_DIR = os.path.join(ROOT, "11_figures")
os.makedirs(FIG_DIR, exist_ok=True)

def main():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10,
        "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Per-model variable importance comparison
    imp_path = os.path.join(IMP_DIR, "permutation_importance_summary.csv")
    imp_df = pd.read_csv(imp_path) if os.path.exists(imp_path) else None

    if imp_df is not None and "model" in imp_df.columns:
        rho_col = "abs_rho" if "abs_rho" in imp_df.columns else "spearman_rho"
        pivoted = imp_df.pivot(index="variable", columns="model", values=rho_col).fillna(0)
        pivoted = pivoted.loc[pivoted.sum(axis=1).sort_values(ascending=True).index]

        models = sorted(imp_df["model"].unique())
        x_pos = np.arange(len(pivoted))
        width = 0.8 / len(models) if len(models) > 0 else 0.2
        colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"]

        for i, model in enumerate(models):
            vals = pivoted[model].values if model in pivoted.columns else np.zeros(len(pivoted))
            axes[0].barh(x_pos + i * width, vals, width,
                        label=model.replace("_", " ").title(),
                        color=colors[i % len(colors)], edgecolor="white", linewidth=0.3)

        axes[0].set_yticks(x_pos + width * (len(models) - 1) / 2)
        axes[0].set_yticklabels(pivoted.index)
        axes[0].set_xlabel("|Spearman ρ| with Suitability")
        axes[0].set_title("A. Per-Model Variable Importance")
        axes[0].legend(fontsize=7)
        axes[0].invert_yaxis()
    else:
        axes[0].text(0.5, 0.5, "SHAP not available\n(raster-based analysis)", ha="center",
                    va="center", transform=axes[0].transAxes)
        axes[0].set_title("A. SHAP (Not Available)")

    # Panel B: Response direction analysis
    consensus = pd.read_csv(os.path.join(IMP_DIR, "ensemble_consensus_importance.csv"))
    ale_path = os.path.join(ROOT, "05_ale", "ensemble_consensus_ALE.csv")
    ale = pd.read_csv(ale_path) if os.path.exists(ale_path) else None

    if ale is not None:
        variables = consensus["variable"].tolist()[::-1]
        directions = []
        for var in variables:
            av = ale[ale["variable"] == var]
            if len(av) > 2:
                vals = av["ensemble_ALE_z"].values
                # Determine direction: monotonic or peaked
                if vals[-1] > vals[0] + 0.001:
                    directions.append(("↑ increasing", "#D73027"))
                elif vals[-1] < vals[0] - 0.001:
                    directions.append(("↓ decreasing", "#4575B4"))
                else:
                    directions.append(("~ nonlinear", "#FEE090"))

        y_pos = range(len(variables))
        for i, (label, color) in enumerate(directions):
            axes[1].barh(i, 1, color=color, edgecolor="white")
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(variables)
        axes[1].set_xticks([])
        axes[1].set_title("B. Response Direction\n(Ensemble Consensus)")

        legend_patches = [
            Patch(facecolor="#D73027", label="Increasing (high to suitable)"),
            Patch(facecolor="#4575B4", label="Decreasing (low to suitable)"),
            Patch(facecolor="#FEE090", label="Nonlinear/Peaked"),
        ]
        axes[1].legend(handles=legend_patches, fontsize=7, loc="lower right")
    else:
        axes[1].text(0.5, 0.5, "Response data unavailable", ha="center",
                    va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"E2_Fig2_SHAP_summary.{ext}"), dpi=600)
    plt.close()
    print("Figure E2-2 saved")

    # Data
    if imp_df is not None:
        imp_df.to_csv(os.path.join(DATA_DIR, "E2_Fig2_SHAP_plotdata.csv"), index=False)
    print("Figure data saved")

if __name__ == "__main__":
    main()
