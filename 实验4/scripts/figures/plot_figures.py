#!/usr/bin/env python3
"""
Experiment 4 — Figure Generation Scripts (Cartopy-free version)
Generates all main figures (E4-Fig1 to E4-Fig5) and supplementary figures.
"""
import os, sys, csv, json, logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import rasterio
from rasterio.plot import show as rio_show

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

EXP4_DIR = Path(r"E:\人参种在哪\实验4")
FIGS_DIR = EXP4_DIR / "16_figures"
FIGDATA_DIR = EXP4_DIR / "17_figure_data"
INPUT_DIR = EXP4_DIR / "00_input_from_experiment3"

for d in [FIGS_DIR, FIGDATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("figures")

DPI = 600
SAVE_KWARGS = dict(dpi=DPI, bbox_inches='tight', facecolor='white')

def save_figure(fig, name):
    for ext in ['png', 'pdf']:
        path = FIGS_DIR / f"{name}.{ext}"
        fig.savefig(str(path), **SAVE_KWARGS)
        logger.info(f"  Saved {path}")
    plt.close(fig)

def load_raster_quick(path):
    """Load a raster and return data + extent for imshow."""
    with rasterio.open(str(path)) as s:
        data = s.read(1, masked=True)
        extent = [s.bounds.left, s.bounds.right, s.bounds.bottom, s.bounds.top]
    return data, extent

# ——— E4-Fig1: Future Projection Design Flow ———
def plot_E4_Fig1():
    """Conceptual flowchart of the future projection design."""
    logger.info("Generating E4-Fig1...")
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis('off')

    boxes = [
        (5, 11, "Current 8 Predictors", "#E8F5E9"),
        (5, 10, "4 Dynamic Climate (Future CMIP6)\n+ 4 Static Soil/Terrain (Current)", "#FFF3E0"),
        (5, 8.5, "4 Calibrated Models\n(RF, XGBoost, BRT, MaxEnt)", "#E3F2FD"),
        (5, 7, "5 GCM x 4 SSP x 2 Periods\n= 40 Future Scenarios", "#F3E5F5"),
        (5, 5.5, "40 GCM-level Ensemble\nFuture Suitability Maps", "#E8EAF6"),
        (5, 4, "Fixed Current Threshold\nBinary Classification", "#FFF9C4"),
        (5, 2.5, "Stable / Loss / Gain\nSpatial Change Classification", "#FFEBEE"),
        (5, 1, "Area Statistics | Centroid Migration\nElevation Shift | Landscape Structure", "#ECEFF1"),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(plt.Rectangle((x-3.5, y-0.6), 7, 1.2, fill=True,
                                     facecolor=color, edgecolor='#333', linewidth=1.5))
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    for y1, y2 in [(10.6, 9.4), (8.5, 7.6), (7, 6.1), (5.5, 4.6), (4, 3.1), (2.5, 1.6)]:
        ax.annotate('', xy=(5, y2), xytext=(5, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#555'))
    ax.set_title("Figure E4-1: Future Climate Scenario Design and Projection Workflow", fontsize=12, pad=15)
    save_figure(fig, "E4_Fig1_future_projection_design")

# ——— E4-Fig2: Future Suitability Maps (GCM mean) ———
def plot_E4_Fig2():
    """GCM mean future suitability maps for SSP x Period."""
    logger.info("Generating E4-Fig2...")
    ENSEMBLE_DIR = EXP4_DIR / "08_future_predictions_ensemble"
    SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
    PERIODS = ["2041-2060", "2061-2080"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, period in enumerate(PERIODS):
        for j, ssp in enumerate(SSPS):
            ax = axes[i, j]
            mean_path = ENSEMBLE_DIR / "GCM_summary" / f"{ssp}_{period}_mean.tif"
            if mean_path.exists():
                data, extent = load_raster_quick(mean_path)
                im = ax.imshow(data, extent=extent, cmap='YlOrRd', vmin=0, vmax=0.3, aspect='auto')
                ax.set_xlim(100, 150); ax.set_ylim(25, 55)
            else:
                ax.text(0.5, 0.5, f"Pending\n{ssp} | {period}",
                       transform=ax.transAxes, ha='center', va='center', fontsize=8)
            ax.set_title(f"{ssp.upper()} | {period}", fontsize=9)
    fig.suptitle("Figure E4-2: Future Ecological Suitability (GCM Mean)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "E4_Fig2_future_suitability")

# ——— E4-Fig3: Gain/Loss/Stable ———
def plot_E4_Fig3():
    """Stable/Gain/Loss classification maps for 2061-2080."""
    logger.info("Generating E4-Fig3...")
    CHANGE_DIR = EXP4_DIR / "09_suitability_change"
    SSPS = ["ssp126", "ssp245", "ssp585"]
    PERIOD = "2061-2080"
    GCM = "ACCESS-CM2"  # Representative GCM

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Stable Suitable", "Loss", "Gain"]
    cmaps = ['Greens', 'Reds', 'Blues']

    for i, (title, cmap) in enumerate(zip(titles, cmaps)):
        ax = axes[i]
        # Try loading from first available SSP that has data
        found = False
        for ssp in SSPS:
            path = CHANGE_DIR / GCM / ssp / PERIOD / "change_class.tif"
            if path.exists():
                data, extent = load_raster_quick(path)
                class_val = i + 1  # 1=stable, 2=loss, 3=gain
                mask = (data == class_val)
                ax.imshow(mask, extent=extent, cmap=cmap, aspect='auto', alpha=0.7)
                ax.set_xlim(100, 150); ax.set_ylim(25, 55)
                found = True
                break
        if not found:
            ax.text(0.5, 0.5, "Pending", transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f"{title} ({ssp.upper()})" if found else title, fontsize=10)
    fig.suptitle("Figure E4-3: Stable, Loss, and Gain of Suitable Areas (2061-2080)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "E4_Fig3_gain_loss_stable")

# ——— E4-Fig4: Area Change ———
def plot_E4_Fig4():
    """Area change statistics."""
    logger.info("Generating E4-Fig4...")
    area_file = EXP4_DIR / "10_area_statistics" / "GCM_summary_area_statistics.csv"
    if not area_file.exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Area statistics pending (run analysis first)",
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        save_figure(fig, "E4_Fig4_area_change")
        return

    data = list(csv.DictReader(open(area_file)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
    periods = ["2041-2060", "2061-2080"]
    x = np.arange(len(SSPS)); width = 0.35

    for idx, (metric, title, ylabel) in enumerate([
        ("future_area_mean_km2", "A. Future Suitable Area", "Area (km\u00b2)"),
        ("gain_pct_mean", "B. Gain Rate", "%"),
        ("loss_pct_mean", "C. Loss Rate", "%"),
    ]):
        ax = axes[idx]
        for i, period in enumerate(periods):
            rows = [r for r in data if r["period"] == period]
            vals = [float(next((r[metric] for r in rows if r["ssp"]==s), 0)) for s in SSPS]
            ax.bar(x + i*width, vals, width, label=period)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels([s.upper() for s in SSPS])
        ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()

    fig.suptitle("Figure E4-4: Suitable Area Changes Across SSPs and Periods", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "E4_Fig4_area_change")

# ——— E4-Fig5: Centroid & Elevation ———
def plot_E4_Fig5():
    """Centroid and elevation shifts."""
    logger.info("Generating E4-Fig5...")
    centroid_file = EXP4_DIR / "11_centroid_shift" / "centroid_migration.csv"
    elev_file = EXP4_DIR / "12_elevation_shift" / "elevation_distribution_by_scenario.csv"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.set_title("A. Centroid Migration Trajectory")
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")

    if centroid_file.exists():
        rows = list(csv.DictReader(open(centroid_file)))
        for row in rows:
            try:
                ax1.plot(float(row["future_lon"]), float(row["future_lat"]), 'o', markersize=4, alpha=0.6)
            except (ValueError, KeyError):
                pass
        # Plot current centroid
        cur_rows = [r for r in rows if r.get("period") == "current"]
        if cur_rows:
            ax1.plot(float(cur_rows[0].get("current_lon", 0)), float(cur_rows[0].get("current_lat", 0)),
                    'r*', markersize=12, label='Current')
    else:
        ax1.text(0.5, 0.5, "Data pending", transform=ax1.transAxes, ha='center', va='center')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_title("B. Elevation Distribution Shift")
    if elev_file.exists():
        rows = list(csv.DictReader(open(elev_file)))
        current = [r for r in rows if r["class"] == "current_suitable"]
        future = [r for r in rows if r["class"] == "future_suitable"]
        if current and future:
            labels = ['Current', 'Future']
            means = [float(current[0]["mean_elevation_m"]), float(future[0]["mean_elevation_m"])]
            ax2.bar(labels, means, color=['#2196F3', '#FF5722'])
            ax2.set_ylabel("Mean Elevation (m)")
    else:
        ax2.text(0.5, 0.5, "Data pending", transform=ax2.transAxes, ha='center', va='center')

    fig.suptitle("Figure E4-5: Centroid Migration and Elevation Shift", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "E4_Fig5_centroid_elevation_shift")

# ——— E4-FigS1: Landscape ———
def plot_E4_FigS1():
    """Landscape fragmentation metrics."""
    logger.info("Generating E4-FigS1...")
    landscape_file = EXP4_DIR / "13_landscape_structure" / "landscape_metrics_by_scenario.csv"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if landscape_file.exists():
        rows = list(csv.DictReader(open(landscape_file)))
        SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
        metrics = [("patch_number", "A. Patch Number", "# Patches"),
                   ("mean_patch_area_km2", "B. Mean Patch Area", "Area (km\u00b2)")]
        for ax, (key, title, ylabel) in zip(axes, metrics):
            vals = []
            for ssp in SSPS:
                match = [r for r in rows if r["ssp"] == ssp and r["period"] == "2061-2080"]
                vals.append(float(match[0][key]) if match else 0)
            ax.bar([s.upper() for s in SSPS], vals, alpha=0.7)
            ax.set_title(title); ax.set_ylabel(ylabel)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Data pending", transform=ax.transAxes, ha='center', va='center')

    fig.suptitle("Figure E4-S1: Landscape Fragmentation Change (2061-2080)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "E4_FigS1_landscape_metrics")

# ——— E4-FigS2: Full vs Climate-only ———
def plot_E4_FigS2():
    """Full vs Climate-only sensitivity."""
    logger.info("Generating E4-FigS2...")
    fig, ax = plt.subplots(figsize=(8, 6))
    sens_file = EXP4_DIR / "15_sensitivity" / "full_vs_climate_only_map_similarity.csv"
    if sens_file.exists():
        # Plot data
        ax.text(0.5, 0.5, "Sensitivity analysis results", transform=ax.transAxes, ha='center', va='center')
    else:
        ax.text(0.5, 0.5, "Full vs Climate-only sensitivity analysis\n(run sensitivity analysis first)",
               transform=ax.transAxes, ha='center', va='center', fontsize=11)
    ax.set_title("Figure E4-S2: Full vs. Climate-only Future Predictions", fontsize=12)
    save_figure(fig, "E4_FigS2_full_vs_climate_only")

def main():
    logger.info("=" * 60)
    logger.info("Generating Figures - Experiment 4")
    logger.info("=" * 60)
    plot_E4_Fig1()
    plot_E4_Fig2()
    plot_E4_Fig3()
    plot_E4_Fig4()
    plot_E4_Fig5()
    plot_E4_FigS1()
    plot_E4_FigS2()
    logger.info("\nAll figures generated!")

if __name__ == "__main__":
    main()
