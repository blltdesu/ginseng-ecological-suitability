#!/usr/bin/env python3
"""
实验6 - Step 16-22: 敏感性分析 (权重/FSI/Confidence/Novelty/Land)
"""
import os, sys, json, csv
import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage

ROOT = Path(r"E:\人参种在哪\实验6")
SENS_DIR = ROOT / "11_sensitivity"
os.makedirs(SENS_DIR, exist_ok=True)

# Load base data
with rasterio.open(ROOT / "04_standardized_layers/current_suitability_S.tif") as src:
    S = src.read(1).astype(np.float32); ref_meta = src.meta.copy()
with rasterio.open(ROOT / "04_standardized_layers/future_stability_F.tif") as src:
    F = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "04_standardized_layers/prediction_confidence_C.tif") as src:
    C = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "04_standardized_layers/novelty_reliability_N.tif") as src:
    N = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "04_standardized_layers/land_availability_L.tif") as src:
    L = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "05_candidate_masks/candidate_pool_mask.tif") as src:
    candidate_mask = src.read(1).astype(np.uint8)
with rasterio.open(ROOT / "07_final_zoning/final_cultivation_zoning.tif") as src:
    main_zoning = src.read(1)
with rasterio.open(ROOT / "00_input_from_experiment5/current/current_binary_suitability.tif") as src:
    cb = src.read(1).astype(np.float32); cb[np.isnan(cb)] = 0.0
with rasterio.open(ROOT / "00_input_from_experiment5/vulnerability/high_confidence_loss_zone.tif") as src:
    hcl = src.read(1).astype(np.float32); hcl[np.isnan(hcl)] = 0.0
with rasterio.open(ROOT / "00_input_from_experiment5/uncertainty/high_uncertainty_zone.tif") as src:
    huz = src.read(1).astype(np.float32); huz[np.isnan(huz)] = 0.0

with open(ROOT / "00_input_from_experiment5/current/ensemble_threshold.json") as f:
    threshold_data = json.load(f)
ensemble_threshold = threshold_data["ensemble_threshold"]

cand_bool = candidate_mask.astype(bool)
eps = 1e-10
S_safe = np.maximum(S, eps)
F_safe = np.maximum(F, eps)
C_safe = np.maximum(C, eps)
N_safe = np.maximum(N, eps)
L_safe = np.maximum(L, eps)

# Get thresholds for zones
with open(ROOT / "06_priority_index/zoning_thresholds.json") as f:
    thr = json.load(f)
s_high = thr["s_median_on_suitable"]
gpi_p50 = thr["gpi_p50"]
gpi_p75 = thr["gpi_p75"]

print("=== Sensitivity Analyses ===")

def compute_gpi(s_pow, f_pow, c_pow, n_pow, l_pow):
    return (S_safe**s_pow) * (F_safe**f_pow) * (C_safe**c_pow) * (N_safe**n_pow) * (L_safe**l_pow)

def compute_zone_i(gpi_arr, fsi_thr=0.90, conf_thr=0.75, nov_thr=0.10):
    """Compute Zone I pixels based on thresholds"""
    z_i = (
        (candidate_mask == 1)
        & (S >= s_high)
        & (F >= fsi_thr)
        & (C >= conf_thr)
        & ((1 - N) <= nov_thr)
        & (L == 1.0)
        & (gpi_arr >= np.percentile(gpi_arr[cand_bool], 75) if cand_bool.sum() > 0 else 0)
    )
    return int(z_i.sum())

def compute_zone_ii(gpi_arr, fsi_thr=0.75, conf_thr=0.50, nov_thr=0.50):
    z_ii = (
        (candidate_mask == 1)
        & (F >= fsi_thr)
        & (C >= conf_thr)
        & ((1 - N) < nov_thr)
        & (gpi_arr >= np.percentile(gpi_arr[cand_bool], 50) if cand_bool.sum() > 0 else 0)
    )
    return int(z_ii.sum())

def compute_zones(gpi_arr, fsi_thr=0.90, conf_thr=0.75, nov_thr=0.10):
    """Compute full zone statistics"""
    gpi_p75_val = np.percentile(gpi_arr[cand_bool], 75) if cand_bool.sum() > 0 else 0
    gpi_p50_val = np.percentile(gpi_arr[cand_bool], 50) if cand_bool.sum() > 0 else 0

    zoning = np.zeros(S.shape, dtype=np.int32)

    # IV: Climate-vulnerable
    zone_iv = ((cb == 1) & (hcl == 1))
    zoning[zone_iv] = 4

    # V: High-uncertainty
    zone_v = (huz == 1) & (zoning == 0)
    zoning[zone_v] = 5

    # I: Core
    zone_i = ((candidate_mask==1) & (S>=s_high) & (F>=fsi_thr) & (C>=conf_thr) &
              ((1-N)<=nov_thr) & (L==1.0) & (gpi_arr>=gpi_p75_val) & (zoning==0))
    zoning[zone_i] = 1

    # II: General
    zone_ii = ((candidate_mask==1) & (F>=0.75) & (C>=0.50) & ((1-N)<0.50) &
               (gpi_arr>=gpi_p50_val) & (zoning==0))
    zoning[zone_ii] = 2

    # III: Conditional
    zone_iii = ((candidate_mask==1) & (zoning==0)) | ((cb==1) & (L==0.5) & (F>=0.50) & (zoning==0))
    zoning[zone_iii] = 3

    # VI: Non-priority
    zoning[zoning==0] = 6

    return {
        "ZoneI": int((zoning==1).sum()),
        "ZoneII": int((zoning==2).sum()),
        "ZoneIII": int((zoning==3).sum()),
        "ZoneIV": int((zoning==4).sum()),
        "ZoneV": int((zoning==5).sum()),
        "ZoneVI": int((zoning==6).sum()),
    }

# --- 1. Weight sensitivity ---
print("\n1. Weight scheme sensitivity")
weight_schemes = {
    "main":              {"wS": 0.25, "wF": 0.35, "wC": 0.15, "wN": 0.10, "wL": 0.15},
    "stability_first":   {"wS": 0.20, "wF": 0.45, "wC": 0.15, "wN": 0.10, "wL": 0.10},
    "balanced":          {"wS": 0.20, "wF": 0.20, "wC": 0.20, "wN": 0.20, "wL": 0.20},
    "suitability_first": {"wS": 0.40, "wF": 0.25, "wC": 0.10, "wN": 0.10, "wL": 0.15},
}

weight_results = {}
gpi_by_scheme = {}
for name, w in weight_schemes.items():
    gpi = compute_gpi(w["wS"], w["wF"], w["wC"], w["wN"], w["wL"])
    gpi_by_scheme[name] = gpi
    zones = compute_zones(gpi)
    weight_results[name] = zones
    print(f"  {name}: I={zones['ZoneI']}, II={zones['ZoneII']}, III={zones['ZoneIII']}")

# Save weight sensitivity
weight_csv = SENS_DIR / "weight_scheme_sensitivity.csv"
with open(weight_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["scheme", "ZoneI", "ZoneII", "ZoneIII", "ZoneIV", "ZoneV", "ZoneVI"])
    for name, r in weight_results.items():
        writer.writerow([name, r["ZoneI"], r["ZoneII"], r["ZoneIII"], r["ZoneIV"], r["ZoneV"], r["ZoneVI"]])

# --- 2. FSI threshold sensitivity ---
print("\n2. FSI threshold sensitivity")
fsi_thresholds = [0.80, 0.90, 0.95]
fsi_results = {}
main_gpi = gpi_by_scheme["main"]
for fsi_t in fsi_thresholds:
    zones = compute_zones(main_gpi, fsi_thr=fsi_t)
    fsi_results[fsi_t] = zones
    print(f"  FSI >= {fsi_t}: I={zones['ZoneI']}, II={zones['ZoneII']}")

fsi_csv = SENS_DIR / "FSI_threshold_sensitivity.csv"
with open(fsi_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["FSI_threshold", "ZoneI", "ZoneII", "ZoneIII", "ZoneIV", "ZoneV", "ZoneVI"])
    for t, r in fsi_results.items():
        writer.writerow([t, r["ZoneI"], r["ZoneII"], r["ZoneIII"], r["ZoneIV"], r["ZoneV"], r["ZoneVI"]])

# --- 3. Confidence threshold sensitivity ---
print("\n3. Confidence threshold sensitivity")
conf_thresholds = [0.60, 0.75, 0.85]
conf_results = {}
for conf_t in conf_thresholds:
    zones = compute_zones(main_gpi, conf_thr=conf_t)
    conf_results[conf_t] = zones
    print(f"  Confidence >= {conf_t}: I={zones['ZoneI']}, II={zones['ZoneII']}")

conf_csv = SENS_DIR / "confidence_threshold_sensitivity.csv"
with open(conf_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["confidence_threshold", "ZoneI", "ZoneII", "ZoneIII", "ZoneIV", "ZoneV", "ZoneVI"])
    for t, r in conf_results.items():
        writer.writerow([t, r["ZoneI"], r["ZoneII"], r["ZoneIII"], r["ZoneIV"], r["ZoneV"], r["ZoneVI"]])

# --- 4. Novelty threshold sensitivity ---
print("\n4. Novelty threshold sensitivity")
nov_thresholds = [0.05, 0.10, 0.25]
nov_results = {}
for nov_t in nov_thresholds:
    zones = compute_zones(main_gpi, nov_thr=nov_t)
    nov_results[nov_t] = zones
    print(f"  NoveltyFrequency <= {nov_t}: I={zones['ZoneI']}, II={zones['ZoneII']}")

nov_csv = SENS_DIR / "novelty_threshold_sensitivity.csv"
with open(nov_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["novelty_threshold", "ZoneI", "ZoneII", "ZoneIII", "ZoneIV", "ZoneV", "ZoneVI"])
    for t, r in nov_results.items():
        writer.writerow([t, r["ZoneI"], r["ZoneII"], r["ZoneIII"], r["ZoneIV"], r["ZoneV"], r["ZoneVI"]])

# --- 5. Land factor sensitivity ---
print("\n5. Land factor sensitivity")
# Test conditional land values: 0.25, 0.50, 0.75
land_factors = [0.25, 0.50, 0.75]
land_results = {}
land_maps = {}
for lf in land_factors:
    L_alt = L.copy()
    # Adjust: wherever L was 0.5 (conditional), change to lf
    L_alt[(L >= 0.499) & (L <= 0.501)] = lf
    L_alt_safe = np.maximum(L_alt, eps)
    gpi_lf = (S_safe**0.25) * (F_safe**0.35) * (C_safe**0.15) * (N_safe**0.10) * (L_alt_safe**0.15)
    land_maps[lf] = L_alt
    zones = compute_zones(gpi_lf)
    land_results[lf] = zones
    print(f"  Land factor = {lf}: I={zones['ZoneI']}, II={zones['ZoneII']}, III={zones['ZoneIII']}")

# Actually need to adjust the zone I criteria too (L==1.0), so let's redo properly
# zone I requires LandAvailability = 1 (not just conditional)
# The land_factor sensitivity is mainly about the GPI formula
# For zone I: still require L == 1.0 (full availability)
# The conditional land_factor only affects GPI values but not zone I mask

land_csv = SENS_DIR / "land_factor_sensitivity.csv"
with open(land_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["land_factor", "ZoneI", "ZoneII", "ZoneIII", "ZoneIV", "ZoneV", "ZoneVI"])
    for t, r in land_results.items():
        writer.writerow([t, r["ZoneI"], r["ZoneII"], r["ZoneIII"], r["ZoneIV"], r["ZoneV"], r["ZoneVI"]])

print("\nDone. All sensitivity analyses complete.")
