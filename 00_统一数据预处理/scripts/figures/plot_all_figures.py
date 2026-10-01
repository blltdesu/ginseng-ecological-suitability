"""Generate all required QC figures (P1-P7)."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from matplotlib.ticker import MaxNLocator

FIG_DIR = r"E:\人参种在哪\00_统一数据预处理\12_figures"
FIG_DATA_DIR = r"E:\人参种在哪\00_统一数据预处理\13_figure_data"
PREPROCESS_DIR = r"E:\人参种在哪\00_统一数据预处理"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(FIG_DATA_DIR, exist_ok=True)

# Matplotlib settings for CJK compatibility
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False
RANDOM_SEED = 20260807
np.random.seed(RANDOM_SEED)


def plot_P1_occurrence_flow():
    """Figure P1: Occurrence filtering flowchart (sankey-like horizontal bar)."""
    print("Plotting P1: Occurrence filtering flow...")

    data = [
        ("Raw GBIF", 922, 0, "#4472C4"),
        ("Taxonomy\nretained", 922, 0, "#5B9BD5"),
        ("Valid\ncoordinates", 564, 358, "#70AD47"),
        ("Non-cultivated\ncandidates", 413, 149, "#FFC000"),
        ("10 km\nthinned", 252, 161, "#ED7D31"),
    ]

    fig, ax = plt.subplots(figsize=(12, 4))

    stages = [d[0] for d in data]
    counts = [d[1] for d in data]
    colors = [d[3] for d in data]

    bars = ax.bar(stages, counts, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                str(count), ha="center", va="bottom", fontweight="bold", fontsize=14)

    ax.set_ylabel("Number of Records", fontsize=12)
    ax.set_title("P1: Occurrence Record Filtering Flow\nPanax ginseng C.A.Mey.",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.25)

    # Add arrows between bars
    for i in range(len(bars) - 1):
        mid_y = (bars[i].get_height() + bars[i+1].get_height()) / 2 + 30
        mid_x = (bars[i].get_x() + bars[i].get_width()/2 + bars[i+1].get_x() + bars[i+1].get_width()/2) / 2
        ax.annotate("", xy=(bars[i+1].get_x() + bars[i+1].get_width()/2, bars[i+1].get_height() + 5),
                    xytext=(bars[i].get_x() + bars[i].get_width()/2, bars[i].get_height() + 5),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "P1_occurrence_filtering_flow.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "P1_occurrence_filtering_flow.pdf"), bbox_inches="tight")
    plt.close()

    # Save figure data
    pd.DataFrame(data, columns=["stage", "n_records", "n_removed", "color"]).to_csv(
        os.path.join(FIG_DATA_DIR, "P1_occurrence_filtering_flow.csv"), index=False
    )


def plot_P2_occurrence_map():
    """Figure P2: Raw vs cleaned occurrence map."""
    print("Plotting P2: Occurrence distribution map...")

    occ_path = os.path.join(PREPROCESS_DIR, "14_handoff", "to_experiment_1", "occurrence_noncultivated_candidate.csv")
    cult_path = os.path.join(PREPROCESS_DIR, "02_occurrence", "occurrence_cultivated_or_uncertain.csv")
    boundary_path = os.path.join(PREPROCESS_DIR, "03_boundaries", "study_context_adm0.gpkg")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot boundaries
    try:
        boundaries = gpd.read_file(boundary_path)
        boundaries.boundary.plot(ax=ax, color="gray", linewidth=0.5, alpha=0.7)
        boundaries.plot(ax=ax, color="lightyellow", edgecolor="gray", linewidth=0.5, alpha=0.5)
    except Exception as e:
        print(f"  WARNING: Could not plot boundaries: {e}")

    # Plot cleaned occurrence points
    if os.path.exists(occ_path):
        df = pd.read_csv(occ_path)
        ax.scatter(df["decimalLongitude"], df["decimalLatitude"],
                   c="steelblue", s=15, alpha=0.7, edgecolors="darkblue",
                   linewidth=0.3, label=f"Non-cultivated candidates (n={len(df)})", zorder=5)

    # Plot cultivated/uncertain
    if os.path.exists(cult_path):
        df_c = pd.read_csv(cult_path)
        if len(df_c) > 0:
            ax.scatter(df_c["decimalLongitude"], df_c["decimalLatitude"],
                       c="red", s=30, alpha=0.9, marker="x",
                       label=f"Cultivated/uncertain (n={len(df_c)})", zorder=10)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("P2: Panax ginseng Occurrence Records\nCleaned vs. Cultivated/Uncertain",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "P2_occurrence_map.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "P2_occurrence_map.pdf"), bbox_inches="tight")
    plt.close()


def plot_P3_thinning_counts():
    """Figure P3: Thinning sensitivity bar chart."""
    print("Plotting P3: Thinning comparison...")

    data = pd.DataFrame([
        {"distance_km": "5 km", "n_records": 290},
        {"distance_km": "10 km", "n_records": 252},
        {"distance_km": "20 km", "n_records": 212},
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#5B9BD5", "#ED7D31", "#A5A5A5"]
    bars = ax.bar(data["distance_km"], data["n_records"], color=colors,
                  edgecolor="white", linewidth=2, width=0.5)

    for bar, count in zip(bars, data["n_records"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha="center", fontweight="bold", fontsize=14)

    ax.set_ylabel("Number of Records", fontsize=12)
    ax.set_xlabel("Spatial Thinning Distance", fontsize=12)
    ax.set_title("P3: Spatial Thinning Sensitivity\nPanax ginseng Occurrence Records",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(data["n_records"]) * 1.2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "P3_thinning_counts.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "P3_thinning_counts.pdf"), bbox_inches="tight")
    plt.close()

    data.to_csv(os.path.join(FIG_DATA_DIR, "P3_thinning_counts.csv"), index=False)


def plot_P4_future_climate_matrix():
    """Figure P4: Future climate data completeness (note: no data)."""
    print("Plotting P4: Future climate completeness matrix...")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, "Future Climate Data Not Yet Downloaded\n\n"
            "Run 02_download_worldclim_future.py to download\n"
            "CMIP6 downscaled future climate data before Experiment 4.",
            ha="center", va="center", fontsize=14,
            transform=ax.transAxes, color="darkred",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"))
    ax.set_title("P4: Future Climate Data Completeness Matrix", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "P4_future_climate_completeness.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "P4_future_climate_completeness.pdf"), bbox_inches="tight")
    plt.close()

    pd.DataFrame({"status": ["No future climate data available"]}).to_csv(
        os.path.join(FIG_DATA_DIR, "P4_future_climate_completeness.csv"), index=False
    )


def plot_P5_soil_distribution():
    """Figure P5: Soil variable value distributions (sampled)."""
    print("Plotting P5: Soil distribution...")

    soil_dir = os.path.join(PREPROCESS_DIR, "06_soil", "aligned")
    import rasterio

    samples = []
    for f in sorted(os.listdir(soil_dir)):
        if not f.endswith(".tif"):
            continue
        var_name = f.replace("_0_15cm.tif", "")
        with rasterio.open(os.path.join(soil_dir, f)) as src:
            data = src.read(1, masked=True)
            valid = data.compressed()
            # Sample 5000 points
            if len(valid) > 5000:
                sampled = np.random.choice(valid, 5000, replace=False)
            else:
                sampled = valid
            for v in sampled:
                samples.append({"variable": var_name, "value": float(v)})

    df = pd.DataFrame(samples)
    df.to_csv(os.path.join(FIG_DATA_DIR, "P5_soil_distribution.csv"), index=False)

    soil_vars = sorted(df["variable"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    var_labels = {
        "bdod": "BDOD (kg/dm³)", "cec": "CEC (cmol(c)/kg)",
        "clay": "Clay (%)", "nitrogen": "Nitrogen (g/kg)",
        "phh2o": "pH (H2O)", "sand": "Sand (%)", "soc": "SOC (g/kg)",
    }

    for i, var in enumerate(soil_vars):
        ax = axes[i]
        vals = df[df["variable"] == var]["value"]
        vals = vals[(vals > vals.quantile(0.01)) & (vals < vals.quantile(0.99))]  # trim extremes
        ax.hist(vals, bins=50, color="#5B9BD5", edgecolor="white", alpha=0.8)
        ax.set_title(var_labels.get(var, var), fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")

    # Hide extra subplot
    if len(soil_vars) < len(axes):
        axes[-1].set_visible(False)

    fig.suptitle("P5: Soil Variable Distribution (0-15 cm, Converted Units)\nSampled Pixels",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "P5_soil_distribution.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "P5_soil_distribution.pdf"), bbox_inches="tight")
    plt.close()


def plot_P6_predictor_coverage():
    """Figure P6: Predictor spatial coverage map."""
    print("Plotting P6: Predictor coverage...")

    mask_path = os.path.join(PREPROCESS_DIR, "09_reference_grid", "common_valid_mask.tif")
    if not os.path.exists(mask_path):
        print("  WARNING: Common valid mask not found")
        return

    import rasterio
    with rasterio.open(mask_path) as src:
        data = src.read(1)
        # Subsample for display
        sample_factor = 10
        data_sub = data[::sample_factor, ::sample_factor]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data_sub, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
                   extent=[-180, 180, -90, 90], interpolation="none")

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("P6: Predictor Spatial Coverage\nCommon Valid Mask (all predictors available)",
                 fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("1 = All predictors available\n0 = At least one missing", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "P6_predictor_coverage.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "P6_predictor_coverage.pdf"), bbox_inches="tight")
    plt.close()


def plot_P7_spatial_extent_qc():
    """Figure P7: Spatial extent comparison across data sources."""
    print("Plotting P7: Spatial extent comparison...")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot boundaries
    boundary_path = os.path.join(PREPROCESS_DIR, "03_boundaries", "study_context_adm0.gpkg")
    try:
        boundaries = gpd.read_file(boundary_path)
        boundaries.boundary.plot(ax=ax, color="black", linewidth=0.8, label="Country Boundaries")
    except Exception as e:
        print(f"  WARNING: {e}")

    # Plot occurrence points
    occ_path = os.path.join(PREPROCESS_DIR, "14_handoff", "to_experiment_1", "occurrence_noncultivated_candidate.csv")
    if os.path.exists(occ_path):
        dfo = pd.read_csv(occ_path)
        ax.scatter(dfo["decimalLongitude"], dfo["decimalLatitude"],
                   c="red", s=8, alpha=0.6, label=f"Occurrences (n={len(dfo)})", zorder=10)

    # Add rectangles for different data extents
    from matplotlib.patches import Rectangle

    # WorldClim extent (global)
    rect_wc = Rectangle((-180, -90), 360, 180, linewidth=2, edgecolor="blue",
                         facecolor="none", linestyle="--", label="WorldClim (Global)")
    ax.add_patch(rect_wc)

    # East Asia extent (approximate)
    rect_ea = Rectangle((73, 18), 72, 36, linewidth=2, edgecolor="green",
                         facecolor="none", linestyle="-", label="Original Soil/Terrain/LC (East Asia)")
    ax.add_patch(rect_ea)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("P7: Spatial Extent Comparison\nData Source Overlaps",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "P7_spatial_extent_qc.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "P7_spatial_extent_qc.pdf"), bbox_inches="tight")
    plt.close()


def main():
    print("=" * 50)
    print("GENERATING QC FIGURES (P1-P7)")

    plot_P1_occurrence_flow()
    plot_P2_occurrence_map()
    plot_P3_thinning_counts()
    plot_P4_future_climate_matrix()
    plot_P5_soil_distribution()
    plot_P6_predictor_coverage()
    plot_P7_spatial_extent_qc()

    print(f"\nAll figures saved to: {FIG_DIR}")
    print(f"Figure data saved to: {FIG_DATA_DIR}")

    figs = [
        "P1_occurrence_filtering_flow",
        "P2_occurrence_map",
        "P3_thinning_counts",
        "P4_future_climate_completeness",
        "P5_soil_distribution",
        "P6_predictor_coverage",
        "P7_spatial_extent_qc",
    ]
    for f in figs:
        png_path = os.path.join(FIG_DIR, f"{f}.png")
        pdf_path = os.path.join(FIG_DIR, f"{f}.pdf")
        print(f"  {f}: PNG={'OK' if os.path.exists(png_path) else 'MISSING'}, "
              f"PDF={'OK' if os.path.exists(pdf_path) else 'MISSING'}")


if __name__ == "__main__":
    main()
