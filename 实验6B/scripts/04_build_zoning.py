#!/usr/bin/env python3
"""
实验6B - Step 9-11: 最终六区分区 + 空间后处理 + 斑块识别
分区域 = final_valid_mask (167个验证适生像元); 域外=0(NoData)。
修正版数据下候选池为空 → Zone I/II/III 预期为0像元(数据驱动的真实结果)。
"""
import os, json, csv
import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage

ROOT = Path(r"E:\人参种在哪\实验6B")
INPUT_DIR = ROOT / "00_input_from_experiment5"
STD_DIR = ROOT / "04_standardized_layers"
GPI_DIR = ROOT / "06_priority_index"
ZONE_DIR = ROOT / "07_final_zoning"
SPAT_DIR = ROOT / "08_spatial_postprocessing"
for d in [ZONE_DIR, SPAT_DIR]:
    os.makedirs(d, exist_ok=True)

def load(p):
    with rasterio.open(p) as src:
        return src.read(1).astype(np.float32), src.meta.copy()

S, meta = load(STD_DIR / "current_suitability_S.tif")
F, _ = load(STD_DIR / "future_stability_F.tif")
C, _ = load(STD_DIR / "prediction_confidence_C.tif")
N, _ = load(STD_DIR / "novelty_reliability_N.tif")
L, _ = load(STD_DIR / "land_availability_L.tif")
fv, _ = load(STD_DIR / "final_valid_mask.tif")
final_valid = fv == 1
cand, _ = load(ROOT / "05_candidate_masks/candidate_pool_mask.tif")
candidate = cand == 1
GPI, _ = load(GPI_DIR / "GPI_main.tif")

cb, _ = load(INPUT_DIR / "current/current_binary_suitability.tif")
hcl, _ = load(INPUT_DIR / "vulnerability/high_confidence_loss_zone.tif")
huz, _ = load(INPUT_DIR / "uncertainty/high_uncertainty_zone.tif")
lfi, _ = load(INPUT_DIR / "future_stability/loss_frequency_current_suitable.tif")
pconf, _ = load(INPUT_DIR / "uncertainty/prediction_confidence.tif")
for a in [cb, hcl, huz, lfi, pconf]:
    np.nan_to_num(a, copy=False)

with open(GPI_DIR / "zoning_thresholds.json") as f:
    thr = json.load(f)
s_high = thr["s_high_threshold"]
gpi_p50, gpi_p75 = thr["gpi_p50"], thr["gpi_p75"]
n_cand = thr["n_candidate_pixels"]

print("=== Step 9: 六区分区 ===")
print(f"  候选池: {n_cand} px; S高适生阈值: {s_high:.6f}; GPI P50/P75: {gpi_p50}/{gpi_p75}")

zoning = np.zeros(S.shape, dtype=np.int32)

# Zone IV: 气候脆弱 (最高优先级)
zone_iv = ((cb == 1) & (hcl == 1)) | ((lfi >= 0.75) & (pconf >= 0.75) & (cb == 1))
zone_iv &= final_valid
zoning[zone_iv] = 4
print(f"  Zone IV 气候脆弱: {int(zone_iv.sum())} px")

# Zone V: 高不确定 (未进入IV)
zone_v = (huz == 1) & final_valid & (zoning == 0)
zoning[zone_v] = 5
print(f"  Zone V 高不确定: {int(zone_v.sum())} px")

# Zone I: 核心稳定候选 (候选池为空时自然为空; NaN阈值比较=False)
zone_i = (
    candidate & (S >= s_high) & (F >= 0.90) & (C >= 0.75)
    & ((1 - N) <= 0.10) & (L == 1.0)
    & (zoning == 0)
)
if n_cand > 0 and np.isfinite(gpi_p75):
    zone_i &= (GPI >= gpi_p75)
zoning[zone_i] = 1
print(f"  Zone I 核心稳定: {int(zone_i.sum())} px")

# Zone II: 一般稳定候选
zone_ii = candidate & (F >= 0.75) & (C >= 0.50) & ((1 - N) < 0.50) & (zoning == 0)
if n_cand > 0 and np.isfinite(gpi_p50):
    zone_ii &= (GPI >= gpi_p50)
zoning[zone_ii] = 2
print(f"  Zone II 一般稳定: {int(zone_ii.sum())} px")

# Zone III: 条件性候选 (剩余候选 OR 适生+条件土地+FSI达标, 未进I/II)
zone_iii = (candidate & (zoning == 0)) | ((cb == 1) & (L == 0.5) & (F >= 0.50) & (zoning == 0))
zone_iii &= final_valid
zoning[zone_iii] = 3
print(f"  Zone III 条件性: {int(zone_iii.sum())} px")

# Zone VI: 非候选 (final_valid域内其余像元)
zone_vi = final_valid & (zoning == 0)
zoning[zone_vi] = 6
print(f"  Zone VI 非候选: {int(zone_vi.sum())} px")
print(f"  合计: {int((zoning>0).sum())} = final_valid {int(final_valid.sum())} → {(int((zoning>0).sum())==int(final_valid.sum()))}")

# --- 逻辑冲突检查 (Step 17) ---
print("\n=== 逻辑冲突检查 ===")
zi_mask = zoning == 1
conflict_hcl = int((zi_mask & (hcl == 1)).sum())
conflict_huz = int((zi_mask & (huz == 1)).sum())
print(f"  Zone I ∩ HighConfidenceLoss = {conflict_hcl} (须为0)")
print(f"  Zone I ∩ HighUncertainty    = {conflict_huz} (须为0)")
if conflict_hcl > 0 or conflict_huz > 0:
    with open(ROOT / "STOP_C_ZONING_LOGIC_CONFLICT.md", "w") as f:
        f.write(f"# STOP_C\n\nZoneI∩HCL={conflict_hcl}, ZoneI∩HUZ={conflict_huz}\n")
    raise SystemExit("STOP_C: 分区逻辑冲突")

out_int = meta.copy(); out_int.update(dtype="int32", nodata=0, compress="lzw")
with rasterio.open(ZONE_DIR / "final_cultivation_zoning.tif", "w", **out_int) as dst:
    dst.write(zoning, 1)

zone_dict = [
    (1, "Core stable candidate", "核心稳定候选区", "#1a9641"),
    (2, "General stable candidate", "一般稳定候选区", "#a6d96a"),
    (3, "Conditional candidate", "条件性候选区(需实地核验)", "#fdae61"),
    (4, "Climate-vulnerable", "气候脆弱区", "#d7191c"),
    (5, "High-uncertainty", "高不确定区", "#984ea3"),
    (6, "Non-priority", "非候选区", "#d9d9d9"),
]
with open(ZONE_DIR / "zoning_class_dictionary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["zone_code", "zone_name_en", "zone_name_cn", "hex_color"])
    for code, en, cn, color in zone_dict:
        w.writerow([code, en, cn, color])

# --- Step 10: 空间后处理 (raw + management-ready) ---
print("\n=== Step 10: 空间后处理 ===")
with rasterio.open(SPAT_DIR / "zoning_raw.tif", "w", **out_int) as dst:
    dst.write(zoning, 1)

zoning_mgmt = zoning.copy()
labeled, n_lab = ndimage.label((zoning == 1).astype(np.uint8), structure=np.ones((3, 3)))
n_demoted = 0
for lab in range(1, n_lab + 1):
    comp = labeled == lab
    if int(comp.sum()) < 3:  # 主管理版: Zone I至少3个8-neighbor连续像元
        zoning_mgmt[comp] = 2
        n_demoted += 1
print(f"  Zone I连通斑块: {n_lab}; 降级(<3px): {n_demoted}")
print(f"  Zone I raw: {int((zoning==1).sum())} → management-ready: {int((zoning_mgmt==1).sum())}")
with rasterio.open(SPAT_DIR / "zoning_management_ready.tif", "w", **out_int) as dst:
    dst.write(zoning_mgmt, 1)

# --- Step 11: 优先斑块识别 (Zone I+II, 8-neighbor) ---
print("\n=== Step 11: 优先斑块 ===")
pri = ((zoning_mgmt == 1) | (zoning_mgmt == 2)).astype(np.uint8)
labeled_pri, n_pri = ndimage.label(pri, structure=np.ones((3, 3)))
print(f"  优先斑块数(Zone I+II): {n_pri}")

area_array = np.load(STD_DIR / "pixel_area_km2.npy")
patch_rows = []
for lab in range(1, n_pri + 1):
    comp = labeled_pri == lab
    zvals = zoning_mgmt[comp]
    dom_zone = int(np.bincount(zvals[zvals > 0])[1:].argmax() + 1)
    patch_rows.append({
        "patch_id": lab, "zone": dom_zone,
        "area_km2": round(float(area_array[comp].sum()), 3),
        "mean_GPI": float(GPI[comp].mean()),
        "mean_current_suitability": float(S[comp].mean()),
        "mean_FSI": float(F[comp].mean()),
        "mean_confidence": float(C[comp].mean()),
        "mean_novelty": float((1 - N)[comp].mean()),
    })

# 始终输出(可能为空, 带完整表头)
patch_fields = ["patch_id", "zone", "area_km2", "mean_GPI", "mean_current_suitability",
                "mean_FSI", "mean_confidence", "mean_novelty", "dominant_landcover", "country", "ADM1"]
with open(SPAT_DIR / "priority_patches.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=patch_fields)
    w.writeheader()
    for p in patch_rows:
        p.setdefault("dominant_landcover", "")
        p.setdefault("country", "")
        p.setdefault("ADM1", "")
        w.writerow(p)

out_i32 = meta.copy(); out_i32.update(dtype="int32", nodata=0, compress="lzw")
with rasterio.open(SPAT_DIR / "priority_patches_labeled.tif", "w", **out_i32) as dst:
    dst.write(labeled_pri.astype(np.int32), 1)

# GPKG (可能为空矢量, 保留schema)
import geopandas as gpd
from shapely.geometry import box
geoms = []
if n_pri > 0:
    for p in patch_rows:
        rows_w, cols_w = np.where(labeled_pri == p["patch_id"])
        tr = meta["transform"]
        minx = tr[2] + cols_w.min() * tr[0]; maxx = tr[2] + (cols_w.max() + 1) * tr[0]
        maxy = tr[5] + rows_w.min() * tr[4]; miny = tr[5] + (rows_w.max() + 1) * tr[4]
        geoms.append(box(minx, miny, maxx, maxy))
gdf = gpd.GeoDataFrame(patch_rows, geometry=geoms, crs="EPSG:4326")
for col in ["dominant_landcover", "country", "ADM1"]:
    if col not in gdf.columns:
        gdf[col] = ""
gdf.to_file(SPAT_DIR / "priority_patches.gpkg", driver="GPKG", layer="priority_patches")
print(f"  priority_patches.gpkg: {len(gdf)} 个斑块" + ("(空——修正版数据下无Zone I/II像元)" if n_pri == 0 else ""))

with open(ZONE_DIR / "zoning_run_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "domain": "final_valid_mask (167 validated suitable pixels)",
        "zone_pixel_counts": {str(c): int((zoning == c).sum()) for c in range(1, 7)},
        "candidate_pool_pixels": n_cand,
        "logic_conflicts": {"zoneI_x_highLoss": conflict_hcl, "zoneI_x_highUncertainty": conflict_huz},
        "note": "修正版数据下FSI≤1/36, 候选池为空, Zone I/II/III=0; "
                "全部验证适生栖息地被划入气候脆弱(IV)/高不确定(V)/非候选(VI)。",
    }, f, ensure_ascii=False, indent=2)

print("\nDone.")
