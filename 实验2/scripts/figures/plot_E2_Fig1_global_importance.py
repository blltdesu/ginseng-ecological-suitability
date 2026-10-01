#!/usr/bin/env python3
"""Figure E2-1: Global variable importance."""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = r"E:\人参种在哪\实验2"
IMP_DIR = os.path.join(ROOT, "03_global_importance")
FIG_DIR = os.path.join(ROOT, "11_figures")
DATA_DIR = os.path.join(ROOT, "12_figure_data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def main():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
        "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
    })

    consensus = pd.read_csv(os.path.join(IMP_DIR, "ensemble_consensus_importance.csv"))
    pct_col = "consensus_importance_pct" if "consensus_importance_pct" in consensus.columns else "importance_pct"
    tier_col = "tier" if "tier" in consensus.columns else None
    consensus = consensus.sort_values(pct_col, ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Consensus importance
    if tier_col and tier_col in consensus.columns:
        colors_a = [("#2166AC" if r[tier_col] == "Tier 1" else
                     "#92C5DE" if r[tier_col] == "Tier 2" else "#D1E5F0")
                    for _, r in consensus.iterrows()]
    else:
        colors_a = ["#2166AC"] * len(consensus)

    ax1.barh(range(len(consensus)), consensus[pct_col], color=colors_a, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(len(consensus)))
    ax1.set_yticklabels(consensus["variable"])
    ax1.set_xlabel("Importance (%)")
    ax1.set_title("A. Consensus Variable Importance\n(Spatial Spearman Correlation)")
    ax1.invert_yaxis()

    legend_elements = [
        Patch(facecolor="#2166AC", label="Tier 1 (top 60%)"),
        Patch(facecolor="#92C5DE", label="Tier 2 (60-85%)"),
        Patch(facecolor="#D1E5F0", label="Tier 3 (remaining)"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=8)
    for i, (_, r) in enumerate(consensus.iterrows()):
        ax1.text(r[pct_col] + 0.5, i, f'{r[pct_col]:.1f}%', va="center", fontsize=8)

    # Panel B: Per-model importance
    imp_summary_path = os.path.join(IMP_DIR, "permutation_importance_summary.csv")
    summary = pd.read_csv(imp_summary_path) if os.path.exists(imp_summary_path) else None

    if summary is not None and "model" in summary.columns and "variable" in summary.columns:
        models = sorted(summary["model"].unique())
        variables = list(consensus["variable"])[::-1]
        rho_col = "abs_rho" if "abs_rho" in summary.columns else "spearman_rho"

        heatmap_data = np.zeros((len(variables), len(models)))
        for i, var in enumerate(variables):
            for j, model in enumerate(models):
                vd = summary[(summary["variable"] == var) & (summary["model"] == model)]
                if len(vd) > 0:
                    val = abs(vd[rho_col].values[0])
                    model_max = summary[summary["model"] == model][rho_col].abs().max()
                    heatmap_data[i, j] = val / model_max if model_max > 0 else 0

        im = ax2.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=45, ha="right")
        ax2.set_yticks(range(len(variables)))
        ax2.set_yticklabels(variables)
        ax2.set_title("B. Per-Model Relative\nImportance (|Spearman ρ|)")
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label("Relative Importance", fontsize=9)
        for i in range(len(variables)):
            for j in range(len(models)):
                if heatmap_data[i, j] > 0.01:
                    ax2.text(j, i, f"{heatmap_data[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if heatmap_data[i, j] > 0.5 else "black")
    else:
        ax2.text(0.5, 0.5, "Per-model breakdown\nnot available\n(raster-based analysis)", ha="center",
                va="center", transform=ax2.transAxes)
        ax2.set_title("B. Per-Model Importance")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"E2_Fig1_global_importance.{ext}"), dpi=600)
    plt.close()
    print("Figure E2-1 saved")

    # Save plot data
    consensus.to_csv(os.path.join(DATA_DIR, "E2_Fig1_consensus_importance.csv"), index=False)
    if summary is not None:
        summary.to_csv(os.path.join(DATA_DIR, "E2_Fig1_model_importance_matrix.csv"), index=False)
    print("Figure data saved")

if __name__ == "__main__":
    main()
