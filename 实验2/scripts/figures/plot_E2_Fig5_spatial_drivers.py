#!/usr/bin/env python3
"""Figure E2-5: Spatial dominant driver maps."""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from matplotlib.colors import ListedColormap

ROOT = r"E:\人参种在哪\实验2"
SPATIAL_DIR = os.path.join(ROOT, "08_spatial_driver_map")
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
FIG_DIR = os.path.join(ROOT, "11_figures")
DATA_DIR = os.path.join(ROOT, "12_figure_data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def main():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 11,
        "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
    })

    point_data_path = os.path.join(SPATIAL_DIR, "dominant_driver_pointdata.csv")
    if not os.path.exists(point_data_path):
        print("Spatial driver data not found - skipping Figure E2-5")
        return

    points = pd.read_csv(point_data_path)
    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    predictors = pred_df["variable"].tolist()

    # Load boundaries
    adm0_path = os.path.join(INPUT_DIR, "boundaries", "study_context_adm0.gpkg")
    adm1_path = os.path.join(INPUT_DIR, "boundaries", "study_context_adm1.gpkg")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Color maps
    var_colors = plt.cm.tab10(np.linspace(0, 1, len(predictors)))
    group_colors = {"climate": "#E41A1C", "soil": "#4DAF4A", "terrain": "#377EB8"}

    # Panel A: Dominant variable
    if os.path.exists(adm1_path):
        adm1 = gpd.read_file(adm1_path)
        adm1.boundary.plot(ax=ax1, color="gray", linewidth=0.3, alpha=0.5)
    if os.path.exists(adm0_path):
        adm0 = gpd.read_file(adm0_path)
        adm0.boundary.plot(ax=ax1, color="black", linewidth=0.8)

    for i, var in enumerate(predictors):
        sub = points[points["dominant_variable"] == var]
        if len(sub) > 0:
            ax1.scatter(sub["longitude"], sub["latitude"], c=[var_colors[i]], s=0.5,
                       alpha=0.6, label=var, rasterized=True)

    ax1.set_title("A. Dominant Environmental Variable")
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
    ax1.legend(markerscale=8, fontsize=7, loc="lower right", ncol=2)
    ax1.set_aspect("equal")

    # Panel B: Dominant group
    if os.path.exists(adm1_path):
        adm1.boundary.plot(ax=ax2, color="gray", linewidth=0.3, alpha=0.5)
    if os.path.exists(adm0_path):
        adm0.boundary.plot(ax=ax2, color="black", linewidth=0.8)

    group_order = ["climate", "soil", "terrain"]
    for grp in group_order:
        sub = points[points["dominant_group"] == grp]
        if len(sub) > 0:
            ax2.scatter(sub["longitude"], sub["latitude"], c=[group_colors[grp]], s=0.5,
                       alpha=0.6, label=grp.capitalize(), rasterized=True)

    ax2.set_title("B. Dominant Environmental Group")
    ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")
    ax2.legend(markerscale=8, fontsize=9, loc="lower right")
    ax2.set_aspect("equal")

    fig.suptitle("Figure E2-5: Spatial Distribution of Dominant Environmental Drivers",
                fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "E2_Fig5_spatial_dominant_drivers.png")
    fig.savefig(fig_path, dpi=600)
    fig.savefig(fig_path.replace(".png", ".pdf"))
    plt.close()
    print(f"Figure E2-5 saved to {fig_path}")

    # Copy plot data
    points.to_csv(os.path.join(DATA_DIR, "E2_Fig5_dominant_driver_variable.csv"), index=False)
    print("Figure data saved")

if __name__ == "__main__":
    main()
