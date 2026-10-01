#!/usr/bin/env python3
"""
Experiment 5 - Script 05: Figure Generation
=============================================
Generates:
  E5-Fig1: Future suitability agreement and uncertainty
  E5-Fig2: Uncertainty sources (Algorithm, GCM, SSP, Time, Dominant)
  E5-Fig3: Environmental novelty
  E5-Fig4: Stability and Loss
  E5-Fig5: Vulnerability and Robust Core
  E5-FigS1: Variance partition
  E5-FigS2: Missing scenario sensitivity
"""

import os
import sys
import json
from pathlib import Path
import rasterio
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore")

# === Configuration ===
EXP5_DIR = Path(r"E:\人参种在哪\实验5")
INPUT_DIR = EXP5_DIR / "00_input_from_experiment4"
OUT_FIGS = EXP5_DIR / "15_figures"
OUT_FIGDATA = EXP5_DIR / "16_figure_data"

# Global plotting settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})


def load_raster(path, band=1):
    """Load raster with NaN handling."""
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        data[data == src.nodata] = np.nan
        profile = src.profile.copy()
    return data, profile


def mask_ocean(data, valid_mask_path=None):
    """Mask ocean areas for better visualization."""
    if valid_mask_path is None:
        valid_mask_path = INPUT_DIR / "current_baseline" / "common_valid_mask.tif"
    if valid_mask_path.exists():
        with rasterio.open(valid_mask_path) as src:
            vm = src.read(1)
        masked = data.copy()
        masked[vm == 0] = np.nan
        return masked
    return data


def create_figure_1():
    """E5-Fig1: Future suitability frequency and Scenario SD."""
    print("\n[E5-Fig1] Future suitability agreement and uncertainty...")

    freq, profile = load_raster(EXP5_DIR / "04_agreement" / "future_suitability_frequency.tif")
    sd, _ = load_raster(EXP5_DIR / "05_uncertainty_components" / "scenario_sd.tif")

    freq = mask_ocean(freq)
    sd = mask_ocean(sd)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Suitability Frequency
    im1 = ax1.imshow(freq, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto",
                      extent=[-180, 180, -90, 90])
    ax1.set_title("A. Future Suitability Frequency")
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7)
    cbar1.set_label("Fraction of scenarios suitable")

    # Panel B: Scenario SD
    im2 = ax2.imshow(sd, cmap="viridis", aspect="auto",
                      extent=[-180, 180, -90, 90])
    ax2.set_title("B. Scenario SD")
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7)
    cbar2.set_label("SD of ensemble suitability")

    for ax in [ax1, ax2]:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.suptitle("E5-Fig1: Future Suitability Agreement and Uncertainty", fontsize=14, y=1.01)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(OUT_FIGS / f"E5_Fig1_future_agreement_uncertainty.{fmt}",
                    dpi=600 if fmt == "png" else 300)
    plt.close(fig)

    # Copy figure data
    # Data already in place; just record paths
    print("  E5-Fig1 saved.")


def create_figure_2():
    """E5-Fig2: Uncertainty sources (5 panels)."""
    print("\n[E5-Fig2] Uncertainty sources...")

    components = {
        "Algorithm": EXP5_DIR / "05_uncertainty_components" / "algorithm_uncertainty_norm.tif",
        "GCM": EXP5_DIR / "05_uncertainty_components" / "gcm_uncertainty_norm.tif",
        "SSP": EXP5_DIR / "05_uncertainty_components" / "ssp_uncertainty_norm.tif",
        "Time": EXP5_DIR / "05_uncertainty_components" / "time_uncertainty_norm.tif",
    }
    dominant_path = EXP5_DIR / "05_uncertainty_components" / "dominant_uncertainty_source.tif"

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    # Plot 4 normalized components
    for idx, (name, path) in enumerate(components.items()):
        if path.exists():
            data, _ = load_raster(path)
            data = mask_ocean(data)
            ax = axes[idx]
            im = ax.imshow(data, cmap="Reds", vmin=0, vmax=1, aspect="auto",
                           extent=[-180, 180, -90, 90])
            ax.set_title(f"{name} Uncertainty\n(normalized 0-1)")
            plt.colorbar(im, ax=ax, shrink=0.6)

    # Panel 5: Dominant source
    if dominant_path.exists():
        dom_data, _ = load_raster(dominant_path)
        dom_data = mask_ocean(dom_data)
        ax = axes[4]
        cmap = mcolors.ListedColormap(["#d73027", "#fc8d59", "#91bfdb", "#4575b4"])
        im = ax.imshow(dom_data, cmap=cmap, vmin=0.5, vmax=4.5, aspect="auto",
                       extent=[-180, 180, -90, 90])
        ax.set_title("Dominant Uncertainty Source")
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, ticks=[1, 2, 3, 4])
        cbar.ax.set_yticklabels(["Algorithm", "GCM", "SSP", "Time"])

    # Hide extra subplot
    axes[5].set_visible(False)

    for ax in axes[:5]:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.suptitle("E5-Fig2: Sources of Prediction Uncertainty", fontsize=14, y=1.01)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(OUT_FIGS / f"E5_Fig2_uncertainty_sources.{fmt}",
                    dpi=600 if fmt == "png" else 300)
    plt.close(fig)
    print("  E5-Fig2 saved.")


def create_figure_3():
    """E5-Fig3: Environmental novelty."""
    print("\n[E5-Fig3] Environmental novelty...")

    novelty_freq_path = EXP5_DIR / "07_environmental_novelty" / "novelty_frequency.tif"
    mess_path = EXP5_DIR / "07_environmental_novelty" / "MESS_summary.tif"
    extrap_path = EXP5_DIR / "07_environmental_novelty" / "univariate_extrapolation_count.tif"

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel A: Novelty Frequency
    ax = axes[0]
    if novelty_freq_path.exists():
        data, _ = load_raster(novelty_freq_path)
        data = mask_ocean(data)
        im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto",
                       extent=[-180, 180, -90, 90])
        ax.set_title("A. Novelty Frequency")
        plt.colorbar(im, ax=ax, shrink=0.7)
    else:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("A. Novelty Frequency")

    # Panel B: MESS Summary
    ax = axes[1]
    if mess_path.exists():
        data, _ = load_raster(mess_path)
        data = mask_ocean(data)
        im = ax.imshow(data, cmap="RdYlBu_r", aspect="auto",
                       extent=[-180, 180, -90, 90])
        ax.set_title("B. MESS Summary")
        plt.colorbar(im, ax=ax, shrink=0.7)
    else:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("B. MESS Summary")

    # Panel C: Extrapolation Count
    ax = axes[2]
    if extrap_path.exists():
        data, _ = load_raster(extrap_path)
        data = mask_ocean(data)
        im = ax.imshow(data, cmap="Oranges", vmin=0, vmax=4, aspect="auto",
                       extent=[-180, 180, -90, 90])
        ax.set_title("C. Univariate Extrapolation Count")
        plt.colorbar(im, ax=ax, shrink=0.7)
    else:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("C. Extrapolation Count")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.suptitle("E5-Fig3: Environmental Novelty Assessment", fontsize=14, y=1.01)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(OUT_FIGS / f"E5_Fig3_environmental_novelty.{fmt}",
                    dpi=600 if fmt == "png" else 300)
    plt.close(fig)
    print("  E5-Fig3 saved.")


def create_figure_4():
    """E5-Fig4: Stability and Loss."""
    print("\n[E5-Fig4] Stability and loss...")

    fsi_path = EXP5_DIR / "08_stability_probability" / "future_stability_index.tif"
    lfi_path = EXP5_DIR / "09_loss_probability" / "loss_frequency_current_suitable.tif"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Future Stability Index
    if fsi_path.exists():
        fsi, _ = load_raster(fsi_path)
        im1 = ax1.imshow(fsi, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
                          extent=[-180, 180, -90, 90])
        ax1.set_title("A. Future Stability Index (FSI)")
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7)
        cbar1.set_label("FSI (stable frequency)")
    else:
        ax1.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("A. Future Stability Index")

    # Panel B: Loss Frequency
    if lfi_path.exists():
        lfi, _ = load_raster(lfi_path)
        im2 = ax2.imshow(lfi, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto",
                          extent=[-180, 180, -90, 90])
        ax2.set_title("B. Loss Frequency (LFI)")
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7)
        cbar2.set_label("LFI (loss frequency)")
    else:
        ax2.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("B. Loss Frequency")

    for ax in [ax1, ax2]:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.suptitle("E5-Fig4: Future Stability and Loss Frequency", fontsize=14, y=1.01)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(OUT_FIGS / f"E5_Fig4_stability_loss.{fmt}",
                    dpi=600 if fmt == "png" else 300)
    plt.close(fig)
    print("  E5-Fig4 saved.")


def create_figure_5():
    """E5-Fig5: Vulnerability and Robust Core."""
    print("\n[E5-Fig5] Vulnerability and robust core...")

    vuln_path = EXP5_DIR / "10_climate_vulnerability" / "vulnerability_confidence_adjusted.tif"
    robust_path = EXP5_DIR / "11_robust_core" / "robust_climatic_core.tif"
    loss_zone_path = EXP5_DIR / "11_robust_core" / "high_confidence_loss_zone.tif"

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel A: Confidence-adjusted vulnerability
    ax = axes[0]
    if vuln_path.exists():
        data, _ = load_raster(vuln_path)
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto",
                       extent=[-180, 180, -90, 90])
        ax.set_title("A. Confidence-Adjusted Vulnerability")
        plt.colorbar(im, ax=ax, shrink=0.7)
    else:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)

    # Panel B: Robust Climatic Core
    ax = axes[1]
    if robust_path.exists():
        data, _ = load_raster(robust_path)
        # Binary: show as discrete
        data_display = np.where(data == 1, 1, np.nan)
        ax.imshow(np.zeros_like(data_display), cmap="Greys", vmin=0, vmax=1,
                  aspect="auto", extent=[-180, 180, -90, 90])
        im = ax.imshow(data_display, cmap="Greens", vmin=0, vmax=1, aspect="auto",
                       extent=[-180, 180, -90, 90])
        ax.set_title("B. Robust Climatic Core")
    else:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)

    # Panel C: High-Confidence Loss Zone
    ax = axes[2]
    if loss_zone_path.exists():
        data, _ = load_raster(loss_zone_path)
        data_display = np.where(data == 1, 1, np.nan)
        ax.imshow(np.zeros_like(data_display), cmap="Greys", vmin=0, vmax=1,
                  aspect="auto", extent=[-180, 180, -90, 90])
        im = ax.imshow(data_display, cmap="Reds", vmin=0, vmax=1, aspect="auto",
                       extent=[-180, 180, -90, 90])
        ax.set_title("C. High-Confidence Loss Zone")
    else:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.suptitle("E5-Fig5: Climate Vulnerability and Robust Core", fontsize=14, y=1.01)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(OUT_FIGS / f"E5_Fig5_vulnerability_core.{fmt}",
                    dpi=600 if fmt == "png" else 300)
    plt.close(fig)
    print("  E5-Fig5 saved.")


def create_figure_s1():
    """E5-FigS1: Variance partition bar chart."""
    print("\n[E5-FigS1] Variance partition...")

    var_path = EXP5_DIR / "06_variance_partition" / "variance_component_summary.csv"
    if not var_path.exists():
        print("  Variance partition data not available, skipping.")
        return

    df = pd.read_csv(var_path)
    df = df[df["Source"] != "Residual"]  # Show main components

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df["Source"], df["MeanContribution"] * 100,
                  color=["#d73027", "#fc8d59", "#91bfdb", "#4575b4"][:len(df)])
    ax.set_ylabel("Variance Contribution (%)")
    ax.set_title("E5-FigS1: Variance Partition of Future Suitability Projections")
    ax.set_ylim(0, max(df["MeanContribution"] * 100) * 1.3)

    # Add value labels
    for bar, val in zip(bars, df["MeanContribution"] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(OUT_FIGS / f"E5_FigS1_variance_partition.{fmt}",
                    dpi=600 if fmt == "png" else 300)
    plt.close(fig)
    print("  E5-FigS1 saved.")


def create_figure_s2():
    """E5-FigS2: Balanced subset vs full comparison."""
    print("\n[E5-FigS2] Missing scenario sensitivity...")

    sens_path = EXP5_DIR / "14_sensitivity" / "balanced_subset_sensitivity.csv"
    if not sens_path.exists():
        print("  Sensitivity data not available, skipping.")
        return

    df = pd.read_csv(sens_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    if "full_scenarios" in df.columns:
        scenarios = [df["full_scenarios"].iloc[0], df["balanced_subset_scenarios"].iloc[0]]
        bars = ax.bar(["Full Set (36)", "Balanced Subset"], scenarios, color=["#4575b4", "#91bfdb"])
        ax.set_ylabel("Number of Scenarios")
        for bar, val in zip(bars, scenarios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(int(val)), ha="center", fontsize=12)
    else:
        ax.text(0.5, 0.5, "Sensitivity data format unexpected", ha="center",
                va="center", transform=ax.transAxes)

    ax.set_title("E5-FigS2: Scenario Count — Full vs Balanced Subset")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(OUT_FIGS / f"E5_FigS2_missing_scenario_sensitivity.{fmt}",
                    dpi=600 if fmt == "png" else 300)
    plt.close(fig)
    print("  E5-FigS2 saved.")


def main():
    print("=" * 60)
    print("Experiment 5 - Figure Generation")
    print("=" * 60)

    # Check that required data exists
    required_files = [
        EXP5_DIR / "04_agreement" / "future_suitability_frequency.tif",
        EXP5_DIR / "05_uncertainty_components" / "scenario_sd.tif",
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print(f"WARNING: {len(missing)} required files missing. Figures may be incomplete.")
        for m in missing:
            print(f"  - {m}")

    create_figure_1()
    create_figure_2()
    create_figure_3()
    create_figure_4()
    create_figure_5()
    create_figure_s1()
    create_figure_s2()

    print("\n" + "=" * 60)
    print("All figures generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
