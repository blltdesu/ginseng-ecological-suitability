#!/usr/bin/env python3
"""
实验6 - Step 2-4: 构建最终有效空间掩膜 + 土地覆盖政策 + 标准化五层
"""
import os, sys, json, csv
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

ROOT = Path(r"E:\人参种在哪\实验6")
INPUT_DIR = ROOT / "00_input_from_experiment5"
MASK_DIR = ROOT / "04_standardized_layers"
POLICY_DIR = ROOT / "03_landcover_policy"
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(POLICY_DIR, exist_ok=True)

# Load reference
with rasterio.open(INPUT_DIR / "reference/reference_grid_template.tif") as ref:
    ref_meta = ref.meta.copy()
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_shape = (ref.height, ref.width)

# --- Load all inputs ---
with rasterio.open(INPUT_DIR / "current/current_ensemble_suitability.tif") as src:
    current_suit = src.read(1).astype(np.float32)
with rasterio.open(INPUT_DIR / "current/current_binary_suitability.tif") as src:
    current_binary = src.read(1).astype(np.float32)
with rasterio.open(INPUT_DIR / "future_stability/future_stability_index.tif") as src:
    fsi = src.read(1).astype(np.float32)
    fsi_nodata = src.nodata
with rasterio.open(INPUT_DIR / "uncertainty/prediction_confidence.tif") as src:
    confidence = src.read(1).astype(np.float32)
    conf_nodata = src.nodata
with rasterio.open(INPUT_DIR / "novelty/novelty_frequency.tif") as src:
    novelty = src.read(1).astype(np.float32)
    nov_nodata = src.nodata
with rasterio.open(INPUT_DIR / "landcover/landcover_aligned.tif") as src:
    landcover = src.read(1).astype(np.float32)
    lc_nodata = src.nodata
with rasterio.open(INPUT_DIR / "reference/common_valid_mask.tif") as src:
    common_mask = src.read(1).astype(np.float32)

# --- Step 2: Final Valid Mask ---
print("Building final valid mask...")
final_valid = (
    (common_mask == 1)
    & (current_suit > 0)  # valid current suitability (not nodata)
    & (fsi != fsi_nodata)
    & (confidence != conf_nodata)
    & (novelty != nov_nodata)
    & (landcover != lc_nodata)
).astype(np.uint8)

n_valid = int(final_valid.sum())
# Compute area: each pixel is ~0.04° x 0.04°
# At equator: 0.04° ~ 4.45 km → ~19.8 km^2, varies by latitude
# Use WGS84 with approximate area: pixel_area_km2 = (0.04*111.32)^2 * cos(lat*π/180)
# But manual says "等面积投影" - we'll compute precisely later
# For now use approximate:
n_pixels_total = ref_shape[0] * ref_shape[1]
valid_area_approx = n_valid * (0.04 * 111.32) ** 2  # rough estimate at equator
print(f"  Final valid pixels: {n_valid}/{n_pixels_total} = {100*n_valid/n_pixels_total:.2f}%")
print(f"  Excluded: {n_pixels_total - n_valid} pixels")

# Write final valid mask
out_meta = ref_meta.copy()
out_meta.update(dtype='uint8', nodata=0, compress='lzw')
with rasterio.open(MASK_DIR / "final_valid_mask.tif", 'w', **out_meta) as dst:
    dst.write(final_valid, 1)

# Record
with open(MASK_DIR / "valid_mask_stats.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerow(["valid_pixel_count", n_valid])
    writer.writerow(["excluded_pixel_count", n_pixels_total - n_valid])

# --- Step 3: Landcover Policy ---
print("\nBuilding landcover policy...")
# Read class dictionary
lc_classes = {}
with open(INPUT_DIR / "landcover/landcover_class_dictionary.csv", encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        code = int(row["class_code"])
        name = row["class_name"]
        lc_classes[code] = name

# Policy definition following manual Section 7:
# A. Potentially available: croplands, cropland/natural vegetation mosaics → land_factor=1.0
# B. Conditional: forest types, shrublands, savannas, grasslands → land_factor=0.5
# C. Excluded: urban, water, snow/ice, barren → land_factor=0.0
policy = {
    0:  ("Water", "Excluded", 0.0, "Water bodies cannot be cultivated", ""),
    1:  ("Evergreen Needleleaf Forest", "Conditional", 0.5, "Forest - understory planting may be possible but requires field verification; no land tenure data", "conditional candidate only"),
    2:  ("Evergreen Broadleaf Forest", "Conditional", 0.5, "Forest - field verification required", "conditional candidate only"),
    3:  ("Deciduous Needleleaf Forest", "Conditional", 0.5, "Forest - field verification required", "conditional candidate only"),
    4:  ("Deciduous Broadleaf Forest", "Conditional", 0.5, "Forest - field verification required", "conditional candidate only"),
    5:  ("Mixed Forest", "Conditional", 0.5, "Forest - field verification required", "conditional candidate only"),
    6:  ("Closed Shrublands", "Conditional", 0.5, "Shrubland - field verification required", "conditional candidate only"),
    7:  ("Open Shrublands", "Conditional", 0.5, "Shrubland - field verification required", "conditional candidate only"),
    8:  ("Woody Savannas", "Conditional", 0.5, "Savanna mosaic - field verification required", "conditional candidate only"),
    9:  ("Savannas", "Conditional", 0.5, "Savanna - field verification required", "conditional candidate only"),
    10: ("Grasslands", "Conditional", 0.5, "Grassland - field verification required; may be convertible", "conditional candidate only"),
    11: ("Permanent Wetlands", "Excluded", 0.0, "Wetlands - unsuitable for cultivation", ""),
    12: ("Croplands", "Potentially available", 1.0, "Agricultural land - potentially convertible for ginseng", ""),
    13: ("Urban and Built-up", "Excluded", 0.0, "Urban areas cannot be cultivated", ""),
    14: ("Cropland/Natural Vegetation Mosaic", "Potentially available", 1.0, "Mosaic agricultural land", ""),
    15: ("Snow and Ice", "Excluded", 0.0, "Permanent snow/ice - unsuitable", ""),
    16: ("Barren or Sparsely Vegetated", "Excluded", 0.0, "Barren land - unsuitable", ""),
    17: ("Water Bodies", "Excluded", 0.0, "Water bodies cannot be cultivated", ""),
    255: ("Unclassified", "Excluded", 0.0, "Unclassified - excluded for safety", ""),
}

# Write policy CSV
policy_path = POLICY_DIR / "landcover_suitability_policy.csv"
with open(policy_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["class_code", "class_name", "cultivation_policy", "land_factor", "decision_basis", "notes"])
    for code in sorted(policy.keys()):
        name, pol, factor, basis, notes = policy[code]
        writer.writerow([code, name, pol, factor, basis, notes])
print(f"  Landcover policy saved to {policy_path}")

# --- Build land availability raster ---
print("\nBuilding land availability factor raster...")
land_avail = np.zeros(ref_shape, dtype=np.float32)
lc_int = landcover.astype(np.int32)

for code, (name, pol, factor, basis, notes) in policy.items():
    mask = lc_int == code
    land_avail[mask] = factor

# Set nodata where landcover is nodata
land_avail[lc_int == lc_nodata] = 0.0

# Apply final_valid_mask: invalid pixels get 0
land_avail[~final_valid.astype(bool)] = 0.0

out_meta.update(dtype='float32', nodata=0.0)
with rasterio.open(POLICY_DIR / "land_availability_factor.tif", 'w', **out_meta) as dst:
    dst.write(land_avail, 1)

# --- Step 4: Standardize Five Core Layers ---
print("\nStandardizing five core layers...")
# All five layers are already 0-1 in valid domain

# S = Current Suitability (already 0-1)
S = current_suit.copy().astype(np.float32)
S[np.isnan(S)] = 0.0
S[~final_valid.astype(bool)] = 0.0
with rasterio.open(MASK_DIR / "current_suitability_S.tif", 'w', **out_meta) as dst:
    dst.write(S, 1)

# F = Future Stability Index (already 0-1, defined only on current suitable)
# Set NaN/NoData to 0 for non-suitable pixels
F = fsi.copy().astype(np.float32)
F[np.isnan(F)] = 0.0
F[~final_valid.astype(bool)] = 0.0
with rasterio.open(MASK_DIR / "future_stability_F.tif", 'w', **out_meta) as dst:
    dst.write(F, 1)

# C = Prediction Confidence (already 0-1, normalized)
C = confidence.copy().astype(np.float32)
C[np.isnan(C)] = 0.0
C[~final_valid.astype(bool)] = 0.0
with rasterio.open(MASK_DIR / "prediction_confidence_C.tif", 'w', **out_meta) as dst:
    dst.write(C, 1)

# N = Novelty Reliability = 1 - NoveltyFrequency (already 0-1 novelty frequency)
N = 1.0 - novelty.copy().astype(np.float32)
N[np.isnan(N)] = 0.0
N[~final_valid.astype(bool)] = 0.0
with rasterio.open(MASK_DIR / "novelty_reliability_N.tif", 'w', **out_meta) as dst:
    dst.write(N, 1)

# L = Land Availability (already 0-1)
L = land_avail.copy()
L[np.isnan(L)] = 0.0
with rasterio.open(MASK_DIR / "land_availability_L.tif", 'w', **out_meta) as dst:
    dst.write(L, 1)

print("  S, F, C, N, L layers standardized and saved.")

# Quick stats
for name, arr in [("S", S), ("F", F), ("C", C), ("N", N), ("L", L)]:
    vals = arr[final_valid.astype(bool)]
    if len(vals) > 0:
        print(f"  {name}: mean={vals.mean():.4f}, median={np.median(vals):.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

print("\nDone. Final valid mask, landcover policy, and standardized layers ready.")
