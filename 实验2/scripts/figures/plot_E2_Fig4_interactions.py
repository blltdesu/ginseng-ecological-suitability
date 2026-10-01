#!/usr/bin/env python3
"""Figure E2-4: Key variable interactions ranking."""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\人参种在哪\实验2"
INT_DIR = os.path.join(ROOT, "07_interactions")
FIG_DIR = os.path.join(ROOT, "11_figures")
DATA_DIR = os.path.join(ROOT, "12_figure_data")
os.makedirs(FIG_DIR, exist_ok=True)

def main():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10,
        "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
    })

    ranking_path = os.path.join(INT_DIR, "interaction_consensus_ranking.csv")
    if not os.path.exists(ranking_path):
        print("No interaction data - skipping Fig E2-4")
        return

    ranking = pd.read_csv(ranking_path)
    ranking = ranking.sort_values("consensus_H", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Interaction ranking
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(ranking)))
    ax1.barh(range(len(ranking)), ranking["consensus_H"], color=colors, edgecolor="white")
    ax1.set_yticks(range(len(ranking)))
    ax1.set_yticklabels([f"{r['variable_1']} × {r['variable_2']}" for _, r in ranking.iterrows()])
    ax1.set_xlabel("Interaction Proxy (H*: conditional correlation variation)")
    ax1.set_title("A. Pairwise Interaction Ranking")
    ax1.invert_yaxis()
    for i, (_, r) in enumerate(ranking.iterrows()):
        ax1.text(r["consensus_H"] + 0.001, i, f'{r["consensus_H"]:.4f}', va="center", fontsize=8)

    # Panel B: Top pair interaction pattern
    if len(ranking) > 0:
        top = ranking.iloc[-1]  # Highest H
        ax2.text(0.5, 0.7, f"Top Interaction:", ha="center", fontsize=12, fontweight="bold",
                transform=ax2.transAxes)
        ax2.text(0.5, 0.55, f"{top['variable_1']} × {top['variable_2']}", ha="center",
                fontsize=14, transform=ax2.transAxes)
        ax2.text(0.5, 0.4, f"Consensus H*: {top['consensus_H']:.4f}", ha="center",
                fontsize=11, transform=ax2.transAxes)
        ax2.text(0.5, 0.25, f"Models supporting: {int(top.get('models_positive_H', top.get('n_models', 4)))}/4",
                ha="center", fontsize=10, transform=ax2.transAxes)
        ax2.text(0.5, 0.15, "Note: H* is a conditional-correlation\ninteraction proxy (raster-based analysis)",
                ha="center", fontsize=8, color="gray", transform=ax2.transAxes)
        ax2.set_title("B. Top Interaction Pair")
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"E2_Fig4_interactions.{ext}"), dpi=600)
    plt.close()
    print("Figure E2-4 saved")

    # Data
    ranking.to_csv(os.path.join(DATA_DIR, "E2_Fig4_interaction_ranking.csv"), index=False)
    print("Figure data saved")

if __name__ == "__main__":
    main()
