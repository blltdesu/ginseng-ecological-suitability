#!/usr/bin/env python3
"""
Experiment 5 - Script 08: Recompute change-frequency products with CORRECTED encoding
======================================================================================
Background: verification on 2026-08-08 showed the Experiment 4 change_class.tif
maps use encoding 1=Stable, 2=Loss, 3=Gain (0=Non-habitat), the reverse of the
classes 1/2 assumed by the original pipeline. Binary predictions were verified
authoritative (100% agreement with ensemble thresholding).

This script recomputes, block-wise over the 36 available change maps:
  - stable_frequency.tif        (class 1 / n_available)
  - loss_frequency.tif          (class 2 / n_available)
  - gain_frequency.tif          (class 3 / n_available)
Then recomputes all downstream products with the same definitions as
03_derived_products.py:
  - future_stability_index.tif / future_stability_classes.tif
  - loss_frequency_current_suitable.tif
  - vulnerability_base.tif / vulnerability_confidence_adjusted.tif / vulnerability_classes.tif
  - robust_climatic_core.tif / high_confidence_loss_zone.tif
  - stability_confidence_quadrants.tif
  - vulnerability_summary.json
Prediction confidence and high-uncertainty zone do NOT depend on change maps
and are left untouched.
"""

import json
from pathlib import Path
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

EXP5_DIR = Path(r"E:\人参种在哪\实验5")
INPUT_DIR = EXP5_DIR / "00_input_from_experiment4"
BLOCK_ROWS = 540

OUT_AGREEMENT = EXP5_DIR / "04_agreement"
OUT_UNCERTAINTY = EXP5_DIR / "05_uncertainty_components"
OUT_STABILITY = EXP5_DIR / "08_stability_probability"
OUT_LOSS = EXP5_DIR / "09_loss_probability"
OUT_VULN = EXP5_DIR / "10_climate_vulnerability"
OUT_ROBUST = EXP5_DIR / "11_robust_core"
OUT_ZONES = EXP5_DIR / "12_spatial_uncertainty_zones"
OUT_NOVELTY = EXP5_DIR / "07_environmental_novelty"
OUT_QC = EXP5_DIR / "18_qc"

# VERIFIED encoding (see notes/DATA_INTEGRITY_NOTE.md)
CLASS_STABLE = 1
CLASS_LOSS = 2
CLASS_GAIN = 3


def recompute_change_frequencies():
    print("=" * 60)
    print("[1] Recomputing stable/loss/gain frequencies (corrected encoding)")
    print("=" * 60)

    reg = pd.read_csv(EXP5_DIR / "02_scenario_registry" / "scenario_availability.csv")
    avail = reg[reg["ensemble_available"] == True]
    change_paths = []
    for _, r in avail.iterrows():
        p = INPUT_DIR / f"change_class_maps/{r['gcm']}/{r['ssp']}/{r['period']}/change_class.tif"
        change_paths.append(p)
    n_scen = len(change_paths)
    print(f"  Scenarios: {n_scen}")

    ref_path = INPUT_DIR / "current_baseline" / "reference_grid_template.tif"
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        n_rows, n_cols = ref.shape
    profile.update(dtype="float32", nodata=np.nan)

    stable_path = OUT_STABILITY / "stable_frequency.tif"
    loss_path = OUT_LOSS / "loss_frequency.tif"
    gain_path = OUT_STABILITY / "gain_frequency.tif"

    w_stable = rasterio.open(stable_path, "w", **profile)
    w_loss = rasterio.open(loss_path, "w", **profile)
    w_gain = rasterio.open(gain_path, "w", **profile)

    for blk_start in range(0, n_rows, BLOCK_ROWS):
        blk_h = min(BLOCK_ROWS, n_rows - blk_start)
        win = Window(0, blk_start, n_cols, blk_h)
        stable_cnt = np.zeros((blk_h, n_cols), dtype=np.float32)
        loss_cnt = np.zeros((blk_h, n_cols), dtype=np.float32)
        gain_cnt = np.zeros((blk_h, n_cols), dtype=np.float32)
        valid_cnt = np.zeros((blk_h, n_cols), dtype=np.float32)

        for p in change_paths:
            with rasterio.open(p) as src:
                data = src.read(1, window=win)
                nd = src.nodata
            valid = np.ones(data.shape, dtype=bool) if nd is None else (data != nd)
            valid &= ~np.isnan(data) if np.issubdtype(data.dtype, np.floating) else valid
            stable_cnt += ((data == CLASS_STABLE) & valid)
            loss_cnt += ((data == CLASS_LOSS) & valid)
            gain_cnt += ((data == CLASS_GAIN) & valid)
            valid_cnt += valid

        with np.errstate(invalid="ignore", divide="ignore"):
            stable_f = np.where(valid_cnt > 0, stable_cnt / valid_cnt, np.nan)
            loss_f = np.where(valid_cnt > 0, loss_cnt / valid_cnt, np.nan)
            gain_f = np.where(valid_cnt > 0, gain_cnt / valid_cnt, np.nan)

        w_stable.write(stable_f.astype(np.float32), 1, window=win)
        w_loss.write(loss_f.astype(np.float32), 1, window=win)
        w_gain.write(gain_f.astype(np.float32), 1, window=win)
        print(f"  Block rows {blk_start}-{blk_start+blk_h} done", flush=True)

    w_stable.close(); w_loss.close(); w_gain.close()
    return profile


def recompute_derived_products():
    print("\n" + "=" * 60)
    print("[2] Recomputing derived products (same definitions as script 03)")
    print("=" * 60)

    def load(path):
        with rasterio.open(path) as src:
            d = src.read(1).astype(np.float32)
            if src.nodata is not None:
                d[d == src.nodata] = np.nan
        return d

    cur_suit = load(INPUT_DIR / "current_baseline/current_ensemble_suitability.tif")
    cur_bin = load(INPUT_DIR / "current_baseline/current_binary_suitability.tif")
    valid_mask = load(INPUT_DIR / "current_baseline/common_valid_mask.tif")
    stable_freq = load(OUT_STABILITY / "stable_frequency.tif")
    loss_freq = load(OUT_LOSS / "loss_frequency.tif")
    scenario_sd = load(OUT_UNCERTAINTY / "scenario_sd.tif")
    pred_conf = load(OUT_VULN / "prediction_confidence.tif")

    novelty_path = OUT_NOVELTY / "novelty_frequency.tif"
    novelty_available = novelty_path.exists()
    novelty_freq = load(novelty_path) if novelty_available else np.zeros_like(stable_freq)

    with rasterio.open(INPUT_DIR / "current_baseline/reference_grid_template.tif") as ref:
        profile = ref.profile.copy()
    profile.update(dtype="float32", nodata=np.nan)
    profile_byte = profile.copy(); profile_byte.update(dtype="uint8", nodata=255)

    current_suitable = (cur_bin == 1) & (valid_mask > 0)
    n_suitable = int(np.sum(current_suitable))
    print(f"  Current suitable pixels: {n_suitable}")

    # --- FSI ---
    print("\n  FSI...")
    fsi = stable_freq.copy(); fsi[~current_suitable] = np.nan
    with rasterio.open(OUT_STABILITY / "future_stability_index.tif", "w", **profile) as dst:
        dst.write(fsi.astype(np.float32), 1)

    fsi_classes = np.full_like(fsi, np.nan, dtype=np.float32)
    fsi_classes[(fsi >= 0.90) & current_suitable] = 1
    fsi_classes[(fsi >= 0.75) & (fsi < 0.90) & current_suitable] = 2
    fsi_classes[(fsi >= 0.50) & (fsi < 0.75) & current_suitable] = 3
    fsi_classes[(fsi >= 0.25) & (fsi < 0.50) & current_suitable] = 4
    fsi_classes[(fsi < 0.25) & current_suitable] = 5
    with rasterio.open(OUT_STABILITY / "future_stability_classes.tif", "w", **profile) as dst:
        dst.write(fsi_classes.astype(np.float32), 1)
    labels_fsi = ["Very stable(>=0.90)", "Stable(0.75-0.90)", "Intermediate(0.50-0.75)",
                  "Unstable(0.25-0.50)", "Highly unstable(<0.25)"]
    print(f"    FSI on suitable: median={np.nanmedian(fsi[current_suitable]):.4f} "
          f"min={np.nanmin(fsi[current_suitable]):.4f} max={np.nanmax(fsi[current_suitable]):.4f}")
    for i, lab in enumerate(labels_fsi, 1):
        print(f"      {lab}: {int(np.sum(fsi_classes == i))} px")

    # --- LFI ---
    print("\n  LFI...")
    lfi = loss_freq.copy(); lfi[~current_suitable] = np.nan
    with rasterio.open(OUT_LOSS / "loss_frequency_current_suitable.tif", "w", **profile) as dst:
        dst.write(lfi.astype(np.float32), 1)
    sum_freq = stable_freq + loss_freq
    max_dev = float(np.nanmax(np.abs(sum_freq - 1.0)[current_suitable])) if n_suitable else 0.0
    print(f"    LFI on suitable: median={np.nanmedian(lfi[current_suitable]):.4f} "
          f"min={np.nanmin(lfi[current_suitable]):.4f} max={np.nanmax(lfi[current_suitable]):.4f}")
    print(f"    QC |Stable+Loss-1| max deviation on suitable = {max_dev:.6f}")

    # --- Vulnerability ---
    print("\n  Vulnerability...")
    v_base = cur_suit * lfi; v_base[~current_suitable] = np.nan
    with rasterio.open(OUT_VULN / "vulnerability_base.tif", "w", **profile) as dst:
        dst.write(v_base.astype(np.float32), 1)
    v_conf = cur_suit * lfi * pred_conf; v_conf[~current_suitable] = np.nan
    with rasterio.open(OUT_VULN / "vulnerability_confidence_adjusted.tif", "w", **profile) as dst:
        dst.write(v_conf.astype(np.float32), 1)

    v_valid = v_conf[current_suitable]
    p25, p50, p75, p90 = [float(np.nanpercentile(v_valid, q)) for q in (25, 50, 75, 90)]
    v_class = np.full_like(v_conf, np.nan, dtype=np.float32)
    v_class[(v_conf <= p25) & current_suitable] = 1
    v_class[(v_conf > p25) & (v_conf <= p50) & current_suitable] = 2
    v_class[(v_conf > p50) & (v_conf <= p75) & current_suitable] = 3
    v_class[(v_conf > p75) & (v_conf <= p90) & current_suitable] = 4
    v_class[(v_conf > p90) & current_suitable] = 5
    with rasterio.open(OUT_VULN / "vulnerability_classes.tif", "w", **profile) as dst:
        dst.write(v_class.astype(np.float32), 1)
    print(f"    Thresholds: P25={p25:.6f} P50={p50:.6f} P75={p75:.6f} P90={p90:.6f}")
    vclass_counts = {}
    for i, lab in enumerate(["Very low", "Low", "Moderate", "High", "Very high"], 1):
        n = int(np.sum(v_class == i)); vclass_counts[lab] = n
        print(f"      {lab}: {n} px ({n/n_suitable*100:.1f}%)")

    # --- Robust core ---
    print("\n  Robust climatic core...")
    suit_median = float(np.nanmedian(cur_suit[current_suitable]))
    sd_p25 = float(np.nanpercentile(scenario_sd[current_suitable], 25))
    robust = np.zeros_like(cur_suit, dtype=np.uint8)
    robust[current_suitable & (cur_suit >= suit_median) & (fsi >= 0.90) & (scenario_sd <= sd_p25)] = 1
    if novelty_available:
        robust[novelty_freq > 0.10] = 0
    robust_out = robust.copy(); robust_out[~current_suitable] = 255
    with rasterio.open(OUT_ROBUST / "robust_climatic_core.tif", "w", **profile_byte) as dst:
        dst.write(robust_out, 1)
    n_robust = int(np.sum(robust == 1))
    print(f"    Criteria: suit>={suit_median:.6f}, FSI>=0.90, SD<={sd_p25:.6f}, novelty<=0.10")
    print(f"    Robust core pixels: {n_robust}")

    # --- High-confidence loss zone ---
    print("\n  High-confidence loss zone...")
    high_loss = np.zeros_like(cur_suit, dtype=np.uint8)
    high_loss[current_suitable & (lfi >= 0.75) & (pred_conf >= 0.75)] = 1
    hloss_out = high_loss.copy(); hloss_out[~current_suitable] = 255
    with rasterio.open(OUT_ROBUST / "high_confidence_loss_zone.tif", "w", **profile_byte) as dst:
        dst.write(hloss_out, 1)
    n_hloss = int(np.sum(high_loss == 1))
    print(f"    Pixels: {n_hloss}")

    # --- Quadrants ---
    print("\n  Stability-confidence quadrants...")
    fsi_median = float(np.nanmedian(fsi[current_suitable]))
    conf_median = float(np.nanmedian(pred_conf[current_suitable]))
    quad = np.full_like(cur_suit, np.nan, dtype=np.float32)
    high_stab = fsi >= fsi_median; high_conf = pred_conf >= conf_median
    quad[high_stab & high_conf & current_suitable] = 1
    quad[high_stab & ~high_conf & current_suitable] = 2
    quad[~high_stab & high_conf & current_suitable] = 3
    quad[~high_stab & ~high_conf & current_suitable] = 4
    with rasterio.open(OUT_ZONES / "stability_confidence_quadrants.tif", "w", **profile) as dst:
        dst.write(quad.astype(np.float32), 1)
    quad_labels = ["I: HighStab+HighConf", "II: HighStab+LowConf",
                   "III: LowStab+HighConf", "IV: LowStab+LowConf"]
    quad_counts = {}
    for i, lab in enumerate(quad_labels, 1):
        n = int(np.sum(quad == i)); quad_counts[lab] = n
        print(f"      {lab}: {n} px")

    # --- Summary ---
    summary = {
        "correction": "change-class encoding corrected: 1=Stable, 2=Loss, 3=Gain (verified 2026-08-08)",
        "n_suitable_pixels": n_suitable,
        "fsi": {"median": float(np.nanmedian(fsi[current_suitable])),
                "min": float(np.nanmin(fsi[current_suitable])),
                "max": float(np.nanmax(fsi[current_suitable]))},
        "lfi": {"median": float(np.nanmedian(lfi[current_suitable])),
                "min": float(np.nanmin(lfi[current_suitable])),
                "max": float(np.nanmax(lfi[current_suitable]))},
        "prediction_confidence": {"median": float(np.nanmedian(pred_conf[current_suitable]))},
        "vulnerability_confidence_adjusted": {"median": float(np.nanmedian(v_conf[current_suitable]))},
        "vulnerability_class_counts": vclass_counts,
        "robust_core": {"n_pixels": n_robust,
                        "pct_of_suitable": n_robust / n_suitable * 100 if n_suitable else 0},
        "high_confidence_loss": {"n_pixels": n_hloss,
                                 "pct_of_suitable": n_hloss / n_suitable * 100 if n_suitable else 0},
        "fsi_class_counts": {labels_fsi[i]: int(np.sum(fsi_classes == i + 1)) for i in range(5)},
        "quadrant_counts": quad_counts,
    }
    with open(OUT_VULN / "vulnerability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n  vulnerability_summary.json updated")
    return summary


def main():
    recompute_change_frequencies()
    summary = recompute_derived_products()
    print("\n" + "=" * 60)
    print("CORRECTED RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("\nRecompute complete.")


if __name__ == "__main__":
    main()
