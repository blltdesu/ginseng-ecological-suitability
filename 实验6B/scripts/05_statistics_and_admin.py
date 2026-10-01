#!/usr/bin/env python3
"""
实验6B - Step 12-15: 面积统计 + ADM0/ADM1统计 + 优先斑块排名 + 土地覆盖组成
ADM统计使用精确的栅格化(rasterio.features.rasterize), 非包围盒近似。
"""
import os, json, csv
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from pathlib import Path
from rasterio.features import rasterize

ROOT = Path(r"E:\人参种在哪\实验6B")
INPUT_DIR = ROOT / "00_input_from_experiment5"
STD_DIR = ROOT / "04_standardized_layers"
ZONE_DIR = ROOT / "07_final_zoning"
SPAT_DIR = ROOT / "08_spatial_postprocessing"
AREA_DIR = ROOT / "09_area_statistics"
ADMIN_DIR = ROOT / "10_admin_priority"
for d in [AREA_DIR, ADMIN_DIR]:
    os.makedirs(d, exist_ok=True)

def load(p):
    with rasterio.open(p) as src:
        return src.read(1), src.meta.copy()

zoning, meta = load(ZONE_DIR / "final_cultivation_zoning.tif")
zoning_mgmt, _ = load(SPAT_DIR / "zoning_management_ready.tif")
fv, _ = load(STD_DIR / "final_valid_mask.tif")
final_valid = fv == 1
GPI, _ = load(ROOT / "06_priority_index/GPI_main.tif")
S, _ = load(STD_DIR / "current_suitability_S.tif")
cb, _ = load(INPUT_DIR / "current/current_binary_suitability.tif")
lc, _ = load(INPUT_DIR / "landcover/landcover_aligned.tif")
area_array = np.load(STD_DIR / "pixel_area_km2.npy").astype(np.float64)

lc_names = {}
with open(INPUT_DIR / "landcover/landcover_class_dictionary.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        lc_names[int(row["class_code"])] = row["class_name"]

zone_names = {1: "Core stable candidate", 2: "General stable candidate", 3: "Conditional candidate",
              4: "Climate-vulnerable", 5: "High-uncertainty", 6: "Non-priority"}

# --- Step 12: 面积统计 (纬度加权真实像元面积) ---
print("=== Step 12: 面积统计 ===")
valid_area = float(area_array[final_valid].sum())
suit_area = float(area_array[(cb == 1) & final_valid].sum())
rows_out = []
for code in range(1, 7):
    zm = zoning == code
    a = float(area_array[zm].sum())
    rows_out.append({
        "zone": code, "zone_name": zone_names[code], "n_pixels": int(zm.sum()),
        "area_km2": round(a, 3),
        "pct_of_final_valid_area": round(100 * a / valid_area, 3) if valid_area > 0 else 0,
        "pct_of_current_suitable_area": round(100 * a / suit_area, 3) if suit_area > 0 else 0,
    })
    print(f"  Zone {code} ({zone_names[code]}): {int(zm.sum())} px, {a:.2f} km2")

with open(AREA_DIR / "zoning_area_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
    w.writeheader(); w.writerows(rows_out)
    nw = csv.writer(f)
    nw.writerow([])
    nw.writerow(["note", "分区统计域=final_valid_mask(167验证适生像元); 面积采用纬度加权像元面积(等面积方法)"])

# --- 边界栅格化 (ADM0/ADM1) ---
print("\n=== ADM边界栅格化 ===")
adm0_path = Path(r"E:\人参种在哪\00_统一数据预处理\03_boundaries\study_context_adm0.gpkg")
adm1_path = Path(r"E:\人参种在哪\00_统一数据预处理\03_boundaries\study_context_adm1.gpkg")
H, W = zoning.shape
transform = meta["transform"]

adm0 = gpd.read_file(adm0_path).to_crs("EPSG:4326")
adm1 = gpd.read_file(adm1_path).to_crs("EPSG:4326")
print(f"  ADM0: {len(adm0)} 个要素, 列: {list(adm0.columns)[:8]}")
print(f"  ADM1: {len(adm1)} 个要素, 列: {list(adm1.columns)[:8]}")

def name_col(gdf, prefs):
    for c in prefs:
        if c in gdf.columns:
            return c
    return gdf.columns[0]

adm0_name = name_col(adm0, ["shapeName", "NAME_0", "admin", "name"])
adm1_name = name_col(adm1, ["shapeName", "NAME_1", "name"])
adm1_group = name_col(adm1, ["shapeGroup", "NAME_0", "adm0_a3", "country"]) if any(
    c in adm1.columns for c in ["shapeGroup", "NAME_0", "adm0_a3", "country"]) else None

# all_touched=True: 像元中心在多边形内即归属(0.0417°像元 vs 国家/省界)
adm0_ras = rasterize(((g, i + 1) for i, g in enumerate(adm0.geometry)),
                     out_shape=(H, W), transform=transform, fill=0, dtype="int32", all_touched=True)
adm1_ras = rasterize(((g, i + 1) for i, g in enumerate(adm1.geometry)),
                     out_shape=(H, W), transform=transform, fill=0, dtype="int32", all_touched=True)

# --- Step 13a: ADM0统计 ---
print("\n=== Step 13: ADM0统计 ===")
adm0_rows = []
for i in range(1, len(adm0) + 1):
    m = (adm0_ras == i) & final_valid
    if m.sum() == 0:
        continue
    row = {"country": str(adm0.iloc[i - 1][adm0_name]), "n_valid_pixels": int(m.sum())}
    for code in range(1, 7):
        row[f"zone{code}_area_km2"] = round(float(area_array[m & (zoning == code)].sum()), 3)
    row["mean_GPI_valid"] = round(float(GPI[m].mean()), 6)
    adm0_rows.append(row)
    print(f"  {row['country']}: {row['n_valid_pixels']} px, ZoneIV={row['zone4_area_km2']} km2")

if adm0_rows:
    keys = ["country", "n_valid_pixels"] + [f"zone{c}_area_km2" for c in range(1, 7)] + ["mean_GPI_valid"]
    with open(AREA_DIR / "zoning_by_adm0.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(adm0_rows)

# --- Step 13b: ADM1统计 ---
print("\n=== ADM1统计 ===")
adm1_rows = []
for i in range(1, len(adm1) + 1):
    m = (adm1_ras == i) & final_valid
    if m.sum() == 0:
        continue
    rec = adm1.iloc[i - 1]
    zi_ii = m & ((zoning == 1) | (zoning == 2))
    gpi_vals = GPI[zi_ii]
    row = {
        "country": str(rec[adm1_group]) if adm1_group else "",
        "ADM1": str(rec[adm1_name]),
        "n_valid_pixels": int(m.sum()),
        "core_area_km2": round(float(area_array[m & (zoning == 1)].sum()), 3),
        "general_stable_area_km2": round(float(area_array[m & (zoning == 2)].sum()), 3),
        "conditional_area_km2": round(float(area_array[m & (zoning == 3)].sum()), 3),
        "vulnerable_area_km2": round(float(area_array[m & (zoning == 4)].sum()), 3),
        "high_uncertainty_area_km2": round(float(area_array[m & (zoning == 5)].sum()), 3),
        "non_priority_area_km2": round(float(area_array[m & (zoning == 6)].sum()), 3),
        "mean_GPI_zoneI_II": round(float(gpi_vals.mean()), 6) if len(gpi_vals) else "",
        "p90_GPI_zoneI_II": round(float(np.percentile(gpi_vals, 90)), 6) if len(gpi_vals) else "",
        "largest_core_patch_km2": 0.0,  # Zone I为空
    }
    adm1_rows.append(row)

adm1_df = pd.DataFrame(adm1_rows)
# 双排名: Rank A面积优先(core+0.5*general), Rank B质量优先(mean GPI of Zone I/II)
adm1_df["rank_A_score"] = adm1_df["core_area_km2"] + 0.5 * adm1_df["general_stable_area_km2"]
adm1_df["rank_A"] = adm1_df["rank_A_score"].rank(ascending=False, method="min").astype(int)
adm1_df["rank_B_score"] = pd.to_numeric(adm1_df["mean_GPI_zoneI_II"], errors="coerce").fillna(0)
adm1_df["rank_B"] = adm1_df["rank_B_score"].rank(ascending=False, method="min").astype(int)
adm1_df = adm1_df.sort_values(["rank_A", "vulnerable_area_km2"], ascending=[True, False])
adm1_df.to_csv(ADMIN_DIR / "ADM1_priority_statistics.csv", index=False, encoding="utf-8-sig")
print(f"  ADM1统计: {len(adm1_df)} 个行政单元 (注: Zone I/II全为空, 排名退化, 仅供参考)")
print(adm1_df[["country", "ADM1", "vulnerable_area_km2", "high_uncertainty_area_km2"]].to_string(index=False))

# --- Step 14: Top核心候选斑块 ---
print("\n=== Step 14: Top核心候选斑块 ===")
patch_csv = SPAT_DIR / "priority_patches.csv"
patches = pd.read_csv(patch_csv) if os.path.getsize(patch_csv) > 60 else pd.DataFrame()
core_patches = patches[patches["zone"] == 1].sort_values("mean_GPI", ascending=False).head(20) if len(patches) else patches
top_fields = ["rank", "patch_id", "country", "ADM1", "area_km2", "mean_GPI", "mean_S",
              "mean_FSI", "mean_C", "mean_N", "landcover"]
with open(ADMIN_DIR / "top_core_candidate_patches.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(top_fields)
    if len(core_patches):
        for rank, (_, p) in enumerate(core_patches.iterrows(), 1):
            w.writerow([rank, p["patch_id"], p.get("country", ""), p.get("ADM1", ""), p["area_km2"],
                        p["mean_GPI"], p["mean_current_suitability"], p["mean_FSI"],
                        p["mean_confidence"], p["mean_novelty"], p.get("dominant_landcover", "")])
    w.writerow([])
    w.writerow(["note", "修正版数据下无Zone I核心候选斑块(候选池为空)", "", "", "", "", "", "", "", "", ""])
print(f"  Top核心斑块: {len(core_patches)} (空)")

# --- Step 15: 土地覆盖组成 (Zone I/II/III; 同时报告IV-VI供参考) ---
print("\n=== Step 15: 土地覆盖组成 ===")
lc_rows = []
for code in range(1, 7):
    zm = zoning == code
    if zm.sum() == 0:
        lc_rows.append({"zone": code, "zone_name": zone_names[code], "landcover_code": "",
                        "landcover_name": "(zone empty)", "pixels": 0, "pct_of_zone": 0})
        continue
    vals, counts = np.unique(lc[zm].astype(int), return_counts=True)
    for v, c in zip(vals, counts):
        lc_rows.append({"zone": code, "zone_name": zone_names[code], "landcover_code": int(v),
                        "landcover_name": lc_names.get(int(v), "?"), "pixels": int(c),
                        "pct_of_zone": round(100 * c / counts.sum(), 2)})

with open(AREA_DIR / "landcover_composition_by_zone.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=lc_rows[0].keys())
    w.writeheader(); w.writerows(lc_rows)
print("  完成 (Zone I/II/III为空; IV-VI组成见CSV)")

print("\nDone.")
