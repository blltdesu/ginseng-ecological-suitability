#!/usr/bin/env python3
"""
实验6 - Step 9-11: 最终六区分区 + 空间后处理 + 斑块识别
"""
import os, sys, json, csv
import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage

ROOT = Path(r"E:\人参种在哪\实验6")
INPUT_DIR = ROOT / "00_input_from_experiment5"
STD_DIR = ROOT / "04_standardized_layers"
CAND_DIR = ROOT / "05_candidate_masks"
GPI_DIR = ROOT / "06_priority_index"
ZONE_DIR = ROOT / "07_final_zoning"
SPAT_DIR = ROOT / "08_spatial_postprocessing"
os.makedirs(ZONE_DIR, exist_ok=True)
os.makedirs(SPAT_DIR, exist_ok=True)

# Load thresholds
with open(GPI_DIR / "zoning_thresholds.json") as f:
    thr = json.load(f)

# Load standardized layers
with rasterio.open(STD_DIR / "current_suitability_S.tif") as src:
    S = src.read(1).astype(np.float32)
    ref_meta = src.meta.copy()
with rasterio.open(STD_DIR / "future_stability_F.tif") as src:
    F = src.read(1).astype(np.float32)
with rasterio.open(STD_DIR / "prediction_confidence_C.tif") as src:
    C = src.read(1).astype(np.float32)
with rasterio.open(STD_DIR / "novelty_reliability_N.tif") as src:
    N = src.read(1).astype(np.float32)
with rasterio.open(STD_DIR / "land_availability_L.tif") as src:
    L = src.read(1).astype(np.float32)
with rasterio.open(CAND_DIR / "candidate_pool_mask.tif") as src:
    candidate_mask = src.read(1).astype(np.uint8)
with rasterio.open(GPI_DIR / "GPI_main.tif") as src:
    GPI = src.read(1).astype(np.float32)

# Load auxiliary layers
with rasterio.open(INPUT_DIR / "current/current_binary_suitability.tif") as src:
    cb = src.read(1).astype(np.float32)
with rasterio.open(INPUT_DIR / "vulnerability/high_confidence_loss_zone.tif") as src:
    hcl = src.read(1).astype(np.float32)
with rasterio.open(INPUT_DIR / "uncertainty/high_uncertainty_zone.tif") as src:
    huz = src.read(1).astype(np.float32)
with rasterio.open(INPUT_DIR / "novelty/novelty_frequency.tif") as src:
    nf = src.read(1).astype(np.float32)
with rasterio.open(INPUT_DIR / "future_stability/loss_frequency_current_suitable.tif") as src:
    lfi = src.read(1).astype(np.float32)
with rasterio.open(INPUT_DIR / "future_stability/future_stability_index.tif") as src:
    fsi_raw = src.read(1).astype(np.float32)

# Fill NaNs
for arr in [cb, hcl, huz, nf, lfi, fsi_raw]:
    arr[np.isnan(arr)] = 0.0

# Load prediction confidence for zone IV
with rasterio.open(INPUT_DIR / "uncertainty/prediction_confidence.tif") as src:
    pconf_raw = src.read(1).astype(np.float32)
pconf_raw[np.isnan(pconf_raw)] = 0.0

print("=== Building Final 6-Zone Classification ===")

# Zone thresholds
s_high_threshold = thr["s_median_on_suitable"]
gpi_p50 = thr["gpi_p50"]
gpi_p75 = thr["gpi_p75"]
ensemble_threshold = thr["ensemble_threshold"]

print(f"S high threshold (median on suitable): {s_high_threshold:.6f}")
print(f"GPI P50: {gpi_p50:.6f}, P75: {gpi_p75:.6f}")
print(f"FSI >= 0.90 count: {(fsi_raw >= 0.90).sum()}")
print(f"FSI >= 0.75 count: {(fsi_raw >= 0.75).sum()}")
print(f"Confidence >= 0.75: {(C >= 0.75).sum()}")
print(f"Novelty freq <= 0.10: {((1-N) <= 0.10).sum()}")

# Initialize zoning array (0 = NoData, 1-6 = zones)
zoning = np.zeros(S.shape, dtype=np.int32)

# Zone definitions following manual Section 14:

# Zone IV: Climate-vulnerable (highest priority in conflict rule)
# Condition 1: CurrentBinarySuitable AND HighConfidenceLossZone = 1
zone_iv_a = (cb == 1) & (hcl == 1)
# Condition 2: LossFrequency >= 0.75 AND PredictionConfidence >= 0.75
zone_iv_b = (lfi >= 0.75) & (pconf_raw >= 0.75)
zone_iv = zone_iv_a | zone_iv_b
zoning[zone_iv] = 4
print(f"Zone IV (Climate-vulnerable): {int(zone_iv.sum())} pixels")

# Zone V: High-uncertainty (not Zone IV)
zone_v = (huz == 1) & (zoning == 0)
zoning[zone_v] = 5
print(f"Zone V (High-uncertainty): {int(zone_v.sum())} pixels")

# Zone I: Core stable candidate
# CandidateMask AND CurrentSuitability >= high-suitability threshold
# AND FSI >= 0.90 AND PredictionConfidence >= 0.75
# AND NoveltyFrequency <= 0.10 AND LandAvailability = 1 AND GPI >= P75
zone_i = (
    (candidate_mask == 1)
    & (S >= s_high_threshold)
    & (F >= 0.90)
    & (C >= 0.75)
    & ((1 - N) <= 0.10)  # NoveltyFrequency = 1-N <= 0.10
    & (L == 1.0)
    & (GPI >= gpi_p75)
    & (zoning == 0)
)
zoning[zone_i] = 1
print(f"Zone I (Core stable): {int(zone_i.sum())} pixels")

# Zone II: General stable candidate
zone_ii = (
    (candidate_mask == 1)
    & (F >= 0.75)
    & (C >= 0.50)
    & ((1 - N) < 0.50)
    & (GPI >= gpi_p50)
    & (zoning == 0)
)
zoning[zone_ii] = 2
print(f"Zone II (General stable): {int(zone_ii.sum())} pixels")

# Zone III: Conditional candidate
# Remaining candidate_mask pixels OR pixels meeting basic criteria but with caveats
zone_iii = (
    (candidate_mask == 1)
    & (zoning == 0)
)
# Also include pixels with LandAvailability=0.5 or confidence issues
zone_iii_conditional = (
    (cb == 1)
    & (L == 0.5)
    & (F >= 0.50)
    & (zoning == 0)
)
zone_iii = zone_iii | zone_iii_conditional
zoning[zone_iii] = 3
print(f"Zone III (Conditional): {int(zone_iii.sum())} pixels")

# Zone VI: Non-priority (everything else)
zone_vi = (zoning == 0)
zoning[zone_vi] = 6
print(f"Zone VI (Non-priority): {int(zone_vi.sum())} pixels")

# --- Logic conflict checks ---
print("\n=== Logic Conflict Checks ===")
zone_i_mask = (zoning == 1)
conflict_hcl = (zone_i_mask & (hcl == 1)).sum()
conflict_huz = (zone_i_mask & (huz == 1)).sum()
print(f"Zone I ∩ HighConfidenceLossZone: {conflict_hcl} (must be 0)")
print(f"Zone I ∩ HighUncertaintyZone: {conflict_huz} (must be 0)")

if conflict_hcl > 0 or conflict_huz > 0:
    print("ERROR: LOGIC CONFLICT DETECTED!")
    conflict_path = ROOT / "STOP_C_ZONING_LOGIC_CONFLICT.md"
    with open(conflict_path, 'w') as f:
        f.write("# STOP_C_ZONING_LOGIC_CONFLICT\n\n")
        f.write(f"Zone I ∩ HighConfidenceLossZone = {conflict_hcl}\n")
        f.write(f"Zone I ∩ HighUncertaintyZone = {conflict_huz}\n")

# --- Write raw zoning ---
out_meta_int = ref_meta.copy()
out_meta_int.update(dtype='int32', nodata=0, compress='lzw')
zoning_raw_path = SPAT_DIR / "zoning_raw.tif"
with rasterio.open(zoning_raw_path, 'w', **out_meta_int) as dst:
    dst.write(zoning, 1)

# --- Zoning class dictionary ---
zone_dict = [
    (1, "Core stable candidate", "核心稳定候选区"),
    (2, "General stable candidate", "一般稳定候选区"),
    (3, "Conditional candidate", "条件性候选区(field verification required)"),
    (4, "Climate-vulnerable", "气候脆弱区"),
    (5, "High-uncertainty", "高不确定区"),
    (6, "Non-priority", "非候选区"),
]
zoning_csv = ZONE_DIR / "zoning_class_dictionary.csv"
with open(zoning_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["zone_code", "zone_name_en", "zone_name_cn", "hex_color"])
    colors = ["#1a9641", "#a6d96a", "#fdae61", "#d7191c", "#ffffbf", "#d9d9d9"]
    for (code, en, cn), color in zip(zone_dict, colors):
        writer.writerow([code, en, cn, color])

# --- Spatial postprocessing: management-ready zoning ---
print("\n=== Spatial Postprocessing ===")

# For management-ready: Zone I requires at least 3 connected pixels (8-neighbor)
zoning_mgmt = zoning.copy()

# Find connected components in Zone I
zone_i_binary = (zoning == 1).astype(np.uint8)
labeled, n_labels = ndimage.label(zone_i_binary, structure=np.ones((3, 3)))
print(f"Zone I connected components: {n_labels}")

# Find component sizes
for label_id in range(1, n_labels + 1):
    comp_mask = (labeled == label_id)
    comp_size = int(comp_mask.sum())
    if comp_size < 3:
        # Demote small patches to Zone II
        zoning_mgmt[comp_mask] = 2
        print(f"  Label {label_id}: {comp_size} pixels → demoted to Zone II")

# Recount after demotion
print(f"Zone I raw: {(zoning == 1).sum()} pixels")
print(f"Zone I management-ready: {(zoning_mgmt == 1).sum()} pixels")

zoning_mgmt_path = SPAT_DIR / "zoning_management_ready.tif"
with rasterio.open(zoning_mgmt_path, 'w', **out_meta_int) as dst:
    dst.write(zoning_mgmt, 1)

# Also write final zoning to the correct output directory
final_zoning_path = ZONE_DIR / "final_cultivation_zoning.tif"
with rasterio.open(final_zoning_path, 'w', **out_meta_int) as dst:
    dst.write(zoning, 1)

# --- Priority patches (Zone I and II) ---
print("\n=== Extracting Priority Patches ===")
# Combine Zone I and II
zone_i_ii = ((zoning_mgmt == 1) | (zoning_mgmt == 2)).astype(np.uint8)
labeled_pri, n_pri = ndimage.label(zone_i_ii, structure=np.ones((3, 3)))
print(f"Priority patches (Zone I+II): {n_pri}")

# Build patch attributes
patch_data = []
for label_id in range(1, n_pri + 1):
    comp_mask = (labeled_pri == label_id)
    comp_size = int(comp_mask.sum())
    # Determine zone (1 or 2) based on majority
    zone_vals = zoning_mgmt[comp_mask]
    dominant_zone = int(np.bincount(zone_vals.astype(int))[1:].argmax() + 1)

    patch_data.append({
        "patch_id": label_id,
        "zone": dominant_zone,
        "area_pixels": comp_size,
        "area_km2_approx": comp_size * (0.04 * 111.32) ** 2,
        "mean_GPI": float(GPI[comp_mask].mean()),
        "mean_S": float(S[comp_mask].mean()),
        "mean_F": float(F[comp_mask].mean()),
        "mean_C": float(C[comp_mask].mean()),
        "mean_N": float(N[comp_mask].mean()),
        "min_row": int(np.where(comp_mask)[0].min()),
        "max_row": int(np.where(comp_mask)[0].max()),
        "min_col": int(np.where(comp_mask)[1].min()),
        "max_col": int(np.where(comp_mask)[1].max()),
    })

# Save patch CSV
patch_csv = SPAT_DIR / "priority_patches.csv"
with open(patch_csv, 'w', newline='') as f:
    fields = ["patch_id", "zone", "area_pixels", "area_km2_approx", "mean_GPI", "mean_S", "mean_F", "mean_C", "mean_N",
              "min_row", "max_row", "min_col", "max_col"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for p in patch_data:
        writer.writerow(p)

# Save patch raster
labeled_path = SPAT_DIR / "priority_patches_labeled.tif"
out_meta_int.update(dtype='int32', nodata=0, compress='lzw')
with rasterio.open(labeled_path, 'w', **out_meta_int) as dst:
    dst.write(labeled_pri, 1)

# --- Zone stats summary ---
print("\n=== Zone Statistics ===")
for code, en, cn in zone_dict:
    n = int((zoning == code).sum())
    print(f"  Zone {code} ({en}): {n} pixels")

print("\nDone. Zoning, spatial postprocessing, and patch extraction complete.")
