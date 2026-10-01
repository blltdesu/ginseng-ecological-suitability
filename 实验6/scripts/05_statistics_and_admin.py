#!/usr/bin/env python3
"""
实验6 - Step 12-15: 面积统计 + ADM0/ADM1统计 + 优先斑块 + 土地覆盖组成
"""
import os, sys, json, csv
import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path
from scipy import ndimage
from rasterio.features import shapes
import pandas as pd

ROOT = Path(r"E:\人参种在哪\实验6")
ZONE_DIR = ROOT / "07_final_zoning"
SPAT_DIR = ROOT / "08_spatial_postprocessing"
AREA_DIR = ROOT / "09_area_statistics"
ADMIN_DIR = ROOT / "10_admin_priority"
POLICY_DIR = ROOT / "03_landcover_policy"
os.makedirs(AREA_DIR, exist_ok=True)
os.makedirs(ADMIN_DIR, exist_ok=True)

# Load zoning
with rasterio.open(ZONE_DIR / "final_cultivation_zoning.tif") as src:
    zoning = src.read(1)
    ref_meta = src.meta.copy()
    ref_transform = src.transform

with rasterio.open(SPAT_DIR / "zoning_management_ready.tif") as src:
    zoning_mgmt = src.read(1)

with rasterio.open(SPAT_DIR / "priority_patches_labeled.tif") as src:
    patches_labeled = src.read(1)

# Load GPI
with rasterio.open(ROOT / "06_priority_index/GPI_main.tif") as src:
    GPI = src.read(1).astype(np.float32)

# Load S
with rasterio.open(ROOT / "04_standardized_layers/current_suitability_S.tif") as src:
    S = src.read(1).astype(np.float32)

# Load landcover and class dictionary
with rasterio.open(ROOT / "00_input_from_experiment5/landcover/landcover_aligned.tif") as src:
    landcover = src.read(1).astype(np.int32)
    lc_nodata = src.nodata

# Load landcover class names
lc_names = {}
with open(ROOT / "00_input_from_experiment5/landcover/landcover_class_dictionary.csv", encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        lc_names[int(row["class_code"])] = row["class_name"]

# --- Area statistics per zone ---
print("=== Area Statistics ===")

# Pixel area at 0.04° resolution: use WGS84 lat-dependent area
# For simplicity, use average cos(lat) for the study area
# The study area appears to be NE Asia (China/Korea/Japan/Russia), roughly 33-54°N
# Use mid-latitude ~43°N: cos(43°) ~ 0.731
# pixel_area_km2 = (0.04 * 111.32)^2 * cos(lat)
pixel_area_km2 = (0.04 * 111.32) ** 2  # equatorial
# We'll compute lat-dependent areas row by row

# Get lat for each row
height = zoning.shape[0]
rows = np.arange(height)
lats = ref_transform[5] + (rows + 0.5) * ref_transform[4]  # pixel center y
lat_rad = np.radians(lats)
cos_lat = np.cos(lat_rad)
pixel_area_by_row_km2 = (0.04 * 111.32) ** 2 * cos_lat

# Build area array
area_array = np.tile(pixel_area_by_row_km2.reshape(-1, 1), (1, zoning.shape[1]))

zone_names = {
    1: "Core stable candidate",
    2: "General stable candidate",
    3: "Conditional candidate",
    4: "Climate-vulnerable",
    5: "High-uncertainty",
    6: "Non-priority",
}

area_summary = []
# Total current suitable area
cb = (S > 0.1284676).astype(int)  # approximate binary from S
current_suitable_area = float(area_array[cb == 1].sum())
valid_area_total = float(area_array.sum())

for code in range(1, 7):
    zmask = (zoning == code)
    n_pixels = int(zmask.sum())
    area = float(area_array[zmask].sum())
    pct_valid = 100.0 * area / valid_area_total
    pct_suitable = 100.0 * area / max(current_suitable_area, 0.01)
    area_summary.append({
        "zone": code,
        "zone_name": zone_names[code],
        "n_pixels": n_pixels,
        "area_km2": round(area, 2),
        "pct_of_valid_area": round(pct_valid, 4),
        "pct_of_current_suitable": round(pct_suitable, 2),
    })
    print(f"  Zone {code} ({zone_names[code]}): {n_pixels} pixels, {area:.2f} km^2, {pct_valid:.4f}% valid, {pct_suitable:.2f}% suitable")

# Save area summary
area_csv = AREA_DIR / "zoning_area_summary.csv"
with open(area_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=area_summary[0].keys())
    writer.writeheader()
    for r in area_summary:
        writer.writerow(r)

# --- ADM0 statistics ---
print("\n=== ADM0 Statistics ===")
# Load ADM0 boundary
adm0_path = Path(r"E:\人参种在哪\00_统一数据预处理\03_boundaries\study_context_adm0.gpkg")
adm1_path = Path(r"E:\人参种在哪\00_统一数据预处理\03_boundaries\study_context_adm1.gpkg")

adm0_by_zone = {}
if adm0_path.exists():
    print(f"Loading ADM0 boundaries from {adm0_path}")
    adm0 = gpd.read_file(adm0_path)
    if adm0.crs is None or str(adm0.crs) != "EPSG:4326":
        adm0 = adm0.to_crs("EPSG:4326")

    # Rasterize ADM0 to match reference grid
    # Build pixel coordinates
    cols = np.arange(zoning.shape[1])
    pixels_rows, pixels_cols = np.meshgrid(rows, cols, indexing='ij')
    pixels_lat = ref_transform[5] + (pixels_rows + 0.5) * ref_transform[4]
    pixels_lon = ref_transform[0] + (pixels_cols + 0.5) * ref_transform[1]

    adm0_data = []
    for _, row in adm0.iterrows():
        country_name = row.get("shapeName", row.get("NAME_0", row.get("admin0_name", "Unknown")))
        geom = row.geometry
        # Simple point-in-polygon for pixel centers in each zone
        # This is approximate - use bounding box to narrow
        from shapely.geometry import Point

        # For speed: only check pixels in valid area
        for code in [1, 2, 3, 4, 5, 6]:
            zmask = (zoning == code)
            if zmask.sum() == 0:
                continue
            area_in_country = 0.0
            # Sample a subset approach: use spatial join via geopandas
            pass

        adm0_data.append({"country": country_name})

    # Alternative approach: use raster zonal stats
    # For each ADM0 polygon, sum zonal areas
    from rasterio.mask import mask as rio_mask

    adm0_rows = []
    for _, adm0_row in adm0.iterrows():
        country = adm0_row.get("shapeName", adm0_row.get("NAME_0", str(_)))
        geom = adm0_row.geometry
        try:
            # Clip by bounding box first
            bounds = geom.bounds
            # Transform to row/col
            # (minx, miny, maxx, maxy) -> (min_col, max_row, max_col, min_row) in raster
            min_col = max(0, int((bounds[0] - ref_transform[0]) / ref_transform[1]))
            max_col = min(zoning.shape[1], int((bounds[2] - ref_transform[0]) / ref_transform[1]) + 1)
            max_row = max(0, int((bounds[1] - ref_transform[5]) / ref_transform[4]))
            min_row = min(zoning.shape[0], int((bounds[3] - ref_transform[5]) / ref_transform[4]) + 1)

            if min_row >= max_row or min_col >= max_col:
                continue

            # For each zone, compute area in this country
            zone_areas = {}
            for code in range(1, 7):
                zmask = (zoning[min_row:max_row, min_col:max_col] == code)
                if zmask.sum() == 0:
                    zone_areas[zone_names[code]] = 0.0
                    continue
                zone_area = area_array[min_row:max_row, min_col:max_col][zmask].sum()
                zone_areas[zone_names[code]] = round(zone_area, 2)

            row_data = {"country": country}
            row_data.update(zone_areas)
            adm0_rows.append(row_data)
        except Exception as e:
            print(f"  WARNING: Could not process {country}: {e}")

    # Save ADM0 stats
    if adm0_rows:
        adm0_csv = AREA_DIR / "zoning_by_adm0.csv"
        keys = ["country"] + [zone_names[c] for c in range(1, 7)]
        with open(adm0_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in adm0_rows:
                writer.writerow({k: r.get(k, 0) for k in keys})
        print(f"  ADM0 statistics saved: {len(adm0_rows)} countries")
else:
    print("  ADM0 boundary not found, skipping")

# --- Top priority patches ---
print("\n=== Top Priority Patches ===")
# From labeled patches, compute per-patch stats
n_patches = patches_labeled.max()
patch_stats = []
for pid in range(1, n_patches + 1):
    pmask = (patches_labeled == pid)
    n_px = int(pmask.sum())
    if n_px == 0:
        continue
    zone_val = int(np.median(zoning_mgmt[pmask]))
    gpi_mean = float(GPI[pmask].mean())
    s_mean = float(S[pmask].mean())
    patch_stats.append({
        "patch_id": pid,
        "zone": zone_val,
        "area_pixels": n_px,
        "area_km2": round(n_px * (0.04 * 111.32) ** 2, 2),
        "mean_GPI": round(gpi_mean, 6),
        "mean_S": round(s_mean, 6),
    })

# Sort by mean_GPI descending
patch_stats.sort(key=lambda x: x["mean_GPI"], reverse=True)

# Top 20
top_patches = patch_stats[:min(20, len(patch_stats))]
top_csv = ADMIN_DIR / "top_core_candidate_patches.csv"
with open(top_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=top_patches[0].keys())
    writer.writeheader()
    for p in top_patches:
        writer.writerow(p)
print(f"  Top patches saved: {len(top_patches)}")

# --- Landcover composition by zone ---
print("\n=== Landcover Composition ===")
lc_comp = []
for code in range(1, 4):  # Only Zones I-III
    zmask = (zoning == code)
    if zmask.sum() == 0:
        continue
    lc_vals = landcover[zmask]
    unique, counts = np.unique(lc_vals, return_counts=True)
    total = counts.sum()
    for u, c in zip(unique, counts):
        lc_comp.append({
            "zone": code,
            "zone_name": zone_names[code],
            "landcover_code": int(u),
            "landcover_name": lc_names.get(int(u), f"Unknown_{u}"),
            "pixels": int(c),
            "pct_of_zone": round(100.0 * c / total, 2),
        })

lc_csv = AREA_DIR / "landcover_composition_by_zone.csv"
if lc_comp:
    with open(lc_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=lc_comp[0].keys())
        writer.writeheader()
        for r in lc_comp:
            writer.writerow(r)
    print(f"  Landcover composition saved: {len(lc_comp)} rows")
else:
    print("  No landcover composition data (Zone I-III empty)")

# --- ADM1 statistics ---
print("\n=== ADM1 Statistics ===")
if adm1_path.exists():
    adm1 = gpd.read_file(adm1_path)
    if str(adm1.crs) != "EPSG:4326":
        adm1 = adm1.to_crs("EPSG:4326")

    adm1_rows = []
    for _, adm1_row in adm1.iterrows():
        adm1_name = adm1_row.get("shapeName", adm1_row.get("NAME_1", str(_)))
        country = adm1_row.get("shapeGroup", adm1_row.get("NAME_0", ""))
        try:
            geom = adm1_row.geometry
            bounds = geom.bounds
            min_col = max(0, int((bounds[0] - ref_transform[0]) / ref_transform[1]))
            max_col = min(zoning.shape[1], int((bounds[2] - ref_transform[0]) / ref_transform[1]) + 1)
            max_row = max(0, int((bounds[1] - ref_transform[5]) / ref_transform[4]))
            min_row = min(zoning.shape[0], int((bounds[3] - ref_transform[5]) / ref_transform[4]) + 1)

            if min_row >= max_row or min_col >= max_col:
                continue

            sub_zoning = zoning[min_row:max_row, min_col:max_col]
            sub_area = area_array[min_row:max_row, min_col:max_col]
            sub_gpi = GPI[min_row:max_row, min_col:max_col]

            core_area = float(sub_area[sub_zoning == 1].sum())
            gen_area = float(sub_area[sub_zoning == 2].sum())
            cond_area = float(sub_area[sub_zoning == 3].sum())
            vuln_area = float(sub_area[sub_zoning == 4].sum())
            uncert_area = float(sub_area[sub_zoning == 5].sum())

            # GPI stats on Zone I+II
            zone_i_ii = ((sub_zoning == 1) | (sub_zoning == 2))
            if zone_i_ii.sum() > 0:
                mean_gpi = float(sub_gpi[zone_i_ii].mean())
                p90_gpi = float(np.percentile(sub_gpi[zone_i_ii], 90))
            else:
                mean_gpi = 0.0
                p90_gpi = 0.0

            adm1_rows.append({
                "country": country,
                "ADM1": adm1_name,
                "core_area_km2": round(core_area, 2),
                "general_stable_area_km2": round(gen_area, 2),
                "conditional_area_km2": round(cond_area, 2),
                "vulnerable_area_km2": round(vuln_area, 2),
                "high_uncertainty_area_km2": round(uncert_area, 2),
                "mean_GPI": round(mean_gpi, 4),
                "p90_GPI": round(p90_gpi, 4),
            })
        except Exception as e:
            continue

    if adm1_rows:
        adm1_csv = ADMIN_DIR / "ADM1_priority_statistics.csv"
        with open(adm1_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=adm1_rows[0].keys())
            writer.writeheader()
            for r in adm1_rows:
                writer.writerow(r)
        print(f"  ADM1 statistics saved: {len(adm1_rows)} admin units")

        # Two rankings
        # Rank A: Area priority
        adm1_df = pd.DataFrame(adm1_rows)
        adm1_df["rank_A_score"] = adm1_df["core_area_km2"] + 0.5 * adm1_df["general_stable_area_km2"]
        adm1_df["rank_A"] = adm1_df["rank_A_score"].rank(ascending=False)

        # Rank B: Quality priority
        adm1_df["rank_B_score"] = adm1_df["mean_GPI"]
        adm1_df["rank_B"] = adm1_df["rank_B_score"].rank(ascending=False)

        # Filter to those with non-zero scores
        adm1_df = adm1_df[adm1_df["rank_A_score"] > 0]
        adm1_df = adm1_df.sort_values("rank_A")
        print(f"  Top ADM1 by area (Rank A):")
        for _, r in adm1_df.head(10).iterrows():
            print(f"    {r['country']} - {r['ADM1']}: score={r['rank_A_score']:.2f}")

        # Save with rankings
        adm1_df.to_csv(ADMIN_DIR / "ADM1_priority_statistics.csv", index=False)

print("\nDone. Statistics and admin analysis complete.")
