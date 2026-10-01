#!/usr/bin/env python3
"""
Experiment 5 - Script 03: Derived Products
============================================
Computes (after core analysis and novelty):
  - Future Stability Index (FSI)
  - Loss Frequency Index (LFI) on current suitable
  - Prediction Confidence
  - Climate Vulnerability (base + confidence-adjusted)
  - Vulnerability Classes
  - Robust Climatic Core
  - High-Confidence Loss Zone
  - High-Uncertainty Zone
  - Stability-Confidence Quadrants
"""

import os
import sys
import json
from pathlib import Path
import rasterio
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# === Configuration ===
EXP5_DIR = Path(r"E:\人参种在哪\实验5")
INPUT_DIR = EXP5_DIR / "00_input_from_experiment4"
ENSEMBLE_THRESHOLD = 0.12846759691320192

OUT_STABILITY = EXP5_DIR / "08_stability_probability"
OUT_LOSS = EXP5_DIR / "09_loss_probability"
OUT_VULN = EXP5_DIR / "10_climate_vulnerability"
OUT_ROBUST = EXP5_DIR / "11_robust_core"
OUT_ZONES = EXP5_DIR / "12_spatial_uncertainty_zones"
OUT_AGREEMENT = EXP5_DIR / "04_agreement"
OUT_UNCERTAINTY = EXP5_DIR / "05_uncertainty_components"
OUT_NOVELTY = EXP5_DIR / "07_environmental_novelty"


def main():
    print("=" * 60)
    print("Experiment 5 - Derived Products")
    print("=" * 60)

    # Load current baseline
    print("\n[1] Loading current baseline...")
    cur_suit_path = INPUT_DIR / "current_baseline" / "current_ensemble_suitability.tif"
    cur_bin_path = INPUT_DIR / "current_baseline" / "current_binary_suitability.tif"
    mask_path = INPUT_DIR / "current_baseline" / "common_valid_mask.tif"

    with rasterio.open(cur_suit_path) as src:
        current_suit = src.read(1).astype(np.float32)
        current_suit[current_suit == src.nodata] = np.nan
        profile = src.profile.copy()

    with rasterio.open(cur_bin_path) as src:
        current_binary = src.read(1).astype(np.uint8)

    with rasterio.open(mask_path) as src:
        valid_mask = src.read(1).astype(np.uint8)

    current_suitable = (current_binary == 1) & (valid_mask > 0)
    n_suitable = np.sum(current_suitable)
    print(f"  Current suitable pixels: {n_suitable}")

    profile.update(dtype="float32", nodata=np.nan)

    # Load computed frequencies
    print("\n[2] Loading frequency rasters...")
    with rasterio.open(OUT_STABILITY / "stable_frequency.tif") as src:
        stable_freq = src.read(1).astype(np.float32)
        stable_freq[stable_freq == src.nodata] = np.nan

    with rasterio.open(OUT_LOSS / "loss_frequency.tif") as src:
        loss_freq = src.read(1).astype(np.float32)
        loss_freq[loss_freq == src.nodata] = np.nan

    with rasterio.open(OUT_UNCERTAINTY / "scenario_sd.tif") as src:
        scenario_sd = src.read(1).astype(np.float32)
        scenario_sd[scenario_sd == src.nodata] = np.nan

    # Load novelty (may or may not exist)
    novelty_path = OUT_NOVELTY / "novelty_frequency.tif"
    novelty_available = novelty_path.exists()
    if novelty_available:
        with rasterio.open(novelty_path) as src:
            novelty_freq = src.read(1).astype(np.float32)
            novelty_freq[novelty_freq == src.nodata] = np.nan
        print(f"  Novelty frequency loaded. Range: {np.nanmin(novelty_freq):.4f}-{np.nanmax(novelty_freq):.4f}")
    else:
        novelty_freq = np.zeros_like(stable_freq)
        print("  WARNING: Novelty frequency not found, using zeros for robust core filter.")

    # ============================================================
    # FSI: Future Stability Index
    # ============================================================
    print("\n[3] Computing Future Stability Index...")
    fsi = stable_freq.copy()
    fsi[~current_suitable] = np.nan

    fsi_path = OUT_STABILITY / "future_stability_index.tif"
    with rasterio.open(fsi_path, "w", **profile) as dst:
        dst.write(fsi.astype(np.float32), 1)

    # FSI classes
    fsi_classes = np.full_like(fsi, np.nan, dtype=np.float32)
    fsi_classes[(fsi >= 0.90) & current_suitable] = 1
    fsi_classes[(fsi >= 0.75) & (fsi < 0.90) & current_suitable] = 2
    fsi_classes[(fsi >= 0.50) & (fsi < 0.75) & current_suitable] = 3
    fsi_classes[(fsi >= 0.25) & (fsi < 0.50) & current_suitable] = 4
    fsi_classes[(fsi < 0.25) & current_suitable] = 5

    fsi_class_path = OUT_STABILITY / "future_stability_classes.tif"
    with rasterio.open(fsi_class_path, "w", **profile) as dst:
        dst.write(fsi_classes.astype(np.float32), 1)

    labels_fsi = ["Very stable(>=0.90)", "Stable(0.75-0.90)", "Intermediate(0.50-0.75)",
                  "Unstable(0.25-0.50)", "Highly unstable(<0.25)"]
    print(f"  FSI on current suitable: median={np.nanmedian(fsi[current_suitable]):.4f}")
    for i, label in enumerate(labels_fsi, 1):
        print(f"    {label}: {np.sum(fsi_classes == i)} pixels")

    # ============================================================
    # LFI: Loss Frequency Index
    # ============================================================
    print("\n[4] Computing Loss Frequency Index...")
    lfi = loss_freq.copy()
    lfi[~current_suitable] = np.nan

    lfi_path = OUT_LOSS / "loss_frequency_current_suitable.tif"
    with rasterio.open(lfi_path, "w", **profile) as dst:
        dst.write(lfi.astype(np.float32), 1)
    print(f"  LFI on current suitable: median={np.nanmedian(lfi[current_suitable]):.4f}")

    # Verify: StableFrequency + LossFrequency ≈ 1 for current suitable
    sum_freq = stable_freq + loss_freq
    deviation = np.abs(sum_freq - 1.0)
    max_dev = np.nanmax(deviation[current_suitable]) if n_suitable > 0 else 0
    print(f"  QC: |Stable+Loss-1| max deviation = {max_dev:.6f}")

    # ============================================================
    # Prediction Confidence
    # ============================================================
    print("\n[5] Computing Prediction Confidence...")
    sd_valid = scenario_sd[current_suitable]
    if len(sd_valid) > 0 and np.sum(~np.isnan(sd_valid)) > 10:
        sd_p1 = np.nanpercentile(sd_valid, 1)
        sd_p99 = np.nanpercentile(sd_valid, 99)
        denom = sd_p99 - sd_p1
        if denom < 1e-10:
            denom = 1e-10
        sd_norm = np.clip((scenario_sd - sd_p1) / denom, 0, 1)
    else:
        sd_norm = np.zeros_like(scenario_sd)

    pred_conf = 1.0 - sd_norm
    pred_conf[~valid_mask.astype(bool)] = np.nan

    conf_path = OUT_VULN / "prediction_confidence.tif"
    with rasterio.open(conf_path, "w", **profile) as dst:
        dst.write(pred_conf.astype(np.float32), 1)
    print(f"  Prediction confidence: median={np.nanmedian(pred_conf[current_suitable]):.4f}")

    # ============================================================
    # Climate Vulnerability
    # ============================================================
    print("\n[6] Computing Climate Vulnerability...")

    # Base vulnerability
    v_base = current_suit * lfi
    v_base[~current_suitable] = np.nan

    vbase_path = OUT_VULN / "vulnerability_base.tif"
    with rasterio.open(vbase_path, "w", **profile) as dst:
        dst.write(v_base.astype(np.float32), 1)

    # Confidence-adjusted vulnerability
    v_confident = current_suit * lfi * pred_conf
    v_confident[~current_suitable] = np.nan

    vconf_path = OUT_VULN / "vulnerability_confidence_adjusted.tif"
    with rasterio.open(vconf_path, "w", **profile) as dst:
        dst.write(v_confident.astype(np.float32), 1)

    # Vulnerability classes
    v_valid = v_confident[current_suitable]
    if len(v_valid) > 0 and np.sum(~np.isnan(v_valid)) > 10:
        p25 = np.nanpercentile(v_valid, 25)
        p50 = np.nanpercentile(v_valid, 50)
        p75 = np.nanpercentile(v_valid, 75)
        p90 = np.nanpercentile(v_valid, 90)

        v_class = np.full_like(v_confident, np.nan, dtype=np.float32)
        v_class[(v_confident <= p25) & current_suitable] = 1
        v_class[(v_confident > p25) & (v_confident <= p50) & current_suitable] = 2
        v_class[(v_confident > p50) & (v_confident <= p75) & current_suitable] = 3
        v_class[(v_confident > p75) & (v_confident <= p90) & current_suitable] = 4
        v_class[(v_confident > p90) & current_suitable] = 5

        vclass_path = OUT_VULN / "vulnerability_classes.tif"
        with rasterio.open(vclass_path, "w", **profile) as dst:
            dst.write(v_class.astype(np.float32), 1)
        print(f"  Vulnerability thresholds: P25={p25:.6f}, P50={p50:.6f}, P75={p75:.6f}, P90={p90:.6f}")
        for i, label in enumerate(["Very low", "Low", "Moderate", "High", "Very high"], 1):
            n = np.sum(v_class == i)
            print(f"    {label}: {n} pixels ({n/n_suitable*100:.1f}% of suitable)" if n_suitable > 0 else f"    {label}: 0")
    else:
        print("  WARNING: Not enough valid vulnerability values for classification")

    # ============================================================
    # Robust Climatic Core
    # ============================================================
    print("\n[7] Computing Robust Climatic Core...")
    # Definition:
    #   CurrentSuitability >= 50th percentile of suitable pixels
    #   AND FSI >= 0.90
    #   AND ScenarioSD <= 25th percentile
    #   AND NoveltyFrequency <= 0.10

    suit_median = np.nanmedian(current_suit[current_suitable]) if n_suitable > 0 else 0
    sd_p25 = np.nanpercentile(scenario_sd[current_suitable], 25) if n_suitable > 0 else 0

    robust = np.zeros_like(current_suit, dtype=np.uint8)
    robust[current_suitable &
           (current_suit >= suit_median) &
           (fsi >= 0.90) &
           (scenario_sd <= sd_p25)] = 1
    if novelty_available:
        robust[novelty_freq > 0.10] = 0

    profile_byte = profile.copy()
    profile_byte.update(dtype="uint8", nodata=255)
    robust_out = robust.copy()
    robust_out[~current_suitable] = 255

    robust_path = OUT_ROBUST / "robust_climatic_core.tif"
    with rasterio.open(robust_path, "w", **profile_byte) as dst:
        dst.write(robust_out, 1)
    print(f"  Criteria: suit>={suit_median:.6f}, FSI>=0.90, SD<={sd_p25:.6f}, novelty<=0.10")
    print(f"  Robust core pixels: {np.sum(robust == 1)}")

    # ============================================================
    # High-Confidence Loss Zone
    # ============================================================
    print("\n[8] Computing High-Confidence Loss Zone...")
    high_loss = np.zeros_like(current_suit, dtype=np.uint8)
    high_loss[current_suitable &
              (lfi >= 0.75) &
              (pred_conf >= 0.75)] = 1

    hloss_out = high_loss.copy()
    hloss_out[~current_suitable] = 255
    hloss_path = OUT_ROBUST / "high_confidence_loss_zone.tif"
    with rasterio.open(hloss_path, "w", **profile_byte) as dst:
        dst.write(hloss_out, 1)
    print(f"  High-confidence loss zone pixels: {np.sum(high_loss == 1)}")

    # ============================================================
    # High-Uncertainty Zone
    # ============================================================
    print("\n[9] Computing High-Uncertainty Zone...")
    sd_p75 = np.nanpercentile(scenario_sd[current_suitable], 75) if n_suitable > 0 else 0

    high_uncertain = np.zeros_like(current_suit, dtype=np.uint8)
    condition = (scenario_sd >= sd_p75) | (novelty_freq >= 0.50)
    high_uncertain[condition & valid_mask.astype(bool)] = 1

    hunc_out = high_uncertain.copy()
    hunc_out[~valid_mask.astype(bool)] = 255
    hunc_path = OUT_ZONES / "high_uncertainty_zone.tif"
    with rasterio.open(hunc_path, "w", **profile_byte) as dst:
        dst.write(hunc_out, 1)
    print(f"  High-uncertainty zone pixels: {np.sum(high_uncertain == 1)}")

    # ============================================================
    # Stability-Confidence Quadrants
    # ============================================================
    print("\n[10] Computing Stability-Confidence Quadrants...")
    fsi_median = np.nanmedian(fsi[current_suitable]) if n_suitable > 0 else 0.5
    conf_median = np.nanmedian(pred_conf[current_suitable]) if n_suitable > 0 else 0.5

    quad = np.full_like(current_suit, np.nan, dtype=np.float32)
    high_stab = fsi >= fsi_median
    high_conf = pred_conf >= conf_median
    quad[high_stab & high_conf & current_suitable] = 1
    quad[high_stab & (~high_conf) & current_suitable] = 2
    quad[(~high_stab) & high_conf & current_suitable] = 3
    quad[(~high_stab) & (~high_conf) & current_suitable] = 4

    quad_path = OUT_ZONES / "stability_confidence_quadrants.tif"
    with rasterio.open(quad_path, "w", **profile) as dst:
        dst.write(quad.astype(np.float32), 1)

    quad_labels = ["I: HighStab+HighConf", "II: HighStab+LowConf",
                   "III: LowStab+HighConf", "IV: LowStab+LowConf"]
    for i, label in enumerate(quad_labels, 1):
        n = np.sum(quad == i)
        print(f"    {label}: {n} pixels")

    # ============================================================
    # Summary
    # ============================================================
    print("\n[11] Saving summary...")
    summary = {
        "n_suitable_pixels": int(n_suitable),
        "fsi": {
            "median": float(np.nanmedian(fsi[current_suitable])) if n_suitable > 0 else None,
            "min": float(np.nanmin(fsi[current_suitable])) if n_suitable > 0 else None,
            "max": float(np.nanmax(fsi[current_suitable])) if n_suitable > 0 else None,
        },
        "lfi": {
            "median": float(np.nanmedian(lfi[current_suitable])) if n_suitable > 0 else None,
        },
        "prediction_confidence": {
            "median": float(np.nanmedian(pred_conf[current_suitable])) if n_suitable > 0 else None,
        },
        "vulnerability_confidence_adjusted": {
            "median": float(np.nanmedian(v_confident[current_suitable])) if n_suitable > 0 else None,
        },
        "robust_core": {
            "n_pixels": int(np.sum(robust == 1)),
            "pct_of_suitable": float(np.sum(robust == 1) / n_suitable * 100) if n_suitable > 0 else 0,
        },
        "high_confidence_loss": {
            "n_pixels": int(np.sum(high_loss == 1)),
            "pct_of_suitable": float(np.sum(high_loss == 1) / n_suitable * 100) if n_suitable > 0 else 0,
        },
        "high_uncertainty": {
            "n_pixels": int(np.sum(high_uncertain == 1)),
        },
        "fsi_class_counts": {labels_fsi[i]: int(np.sum(fsi_classes == i + 1)) for i in range(5)},
        "quadrant_counts": {quad_labels[i]: int(np.sum(quad == i + 1)) for i in range(4)},
    }

    with open(EXP5_DIR / "10_climate_vulnerability" / "vulnerability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("Derived products complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
