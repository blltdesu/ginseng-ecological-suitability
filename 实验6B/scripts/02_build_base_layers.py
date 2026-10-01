#!/usr/bin/env python3
"""
实验6B - Step 2-4: 最终有效掩膜 + 土地覆盖政策 + S/F/C/N/L标准化
"""
import os, json, csv
import numpy as np
import rasterio
from pathlib import Path

ROOT = Path(r"E:\人参种在哪\实验6B")
INPUT_DIR = ROOT / "00_input_from_experiment5"
STD_DIR = ROOT / "04_standardized_layers"
POLICY_DIR = ROOT / "03_landcover_policy"
os.makedirs(STD_DIR, exist_ok=True)
os.makedirs(POLICY_DIR, exist_ok=True)

def load(p):
    with rasterio.open(p) as src:
        return src.read(1).astype(np.float32), src.nodata, src.meta.copy()

with rasterio.open(INPUT_DIR / "reference/reference_grid_template.tif") as ref:
    ref_meta = ref.meta.copy()
    ref_shape = (ref.height, ref.width)

current_suit, _, _ = load(INPUT_DIR / "current/current_ensemble_suitability.tif")
current_binary, _, _ = load(INPUT_DIR / "current/current_binary_suitability.tif")
fsi, fsi_nd, _ = load(INPUT_DIR / "future_stability/future_stability_index.tif")
confidence, conf_nd, _ = load(INPUT_DIR / "uncertainty/prediction_confidence.tif")
novelty, nov_nd, _ = load(INPUT_DIR / "novelty/novelty_frequency.tif")
landcover, lc_nd, _ = load(INPUT_DIR / "landcover/landcover_aligned.tif")
common_mask, _, _ = load(INPUT_DIR / "reference/common_valid_mask.tif")

# --- Step 2: Final Valid Mask ---
print("=== Step 2: Final Valid Mask ===")
final_valid = (
    (common_mask == 1)
    & np.isfinite(current_suit)
    & np.isfinite(fsi)
    & np.isfinite(confidence)
    & np.isfinite(novelty)
    & np.isfinite(landcover)
    & (landcover != 255)
)
n_valid = int(final_valid.sum())
n_total = ref_shape[0] * ref_shape[1]
n_common = int((common_mask == 1).sum())

# 纬度加权像元面积 (km2)
tr = ref_meta["transform"]
rows = np.arange(ref_shape[0])
cos_lat = np.cos(np.radians(tr[5] + (rows + 0.5) * tr[4]))
pixel_area_rows = (abs(tr[0]) * 111.32) ** 2 * cos_lat  # km2 per pixel per row
area_array = np.tile(pixel_area_rows.reshape(-1, 1), (1, ref_shape[1])).astype(np.float64)
valid_area_km2 = float(area_array[final_valid].sum())

print(f"  common_valid_mask: {n_common} px")
print(f"  final_valid_mask: {n_valid} px (约束来自FSI定义域=当前适生域)")
print(f"  valid area: {valid_area_km2:.2f} km2")
print(f"  excluded: {n_total - n_valid} px")

out_u8 = ref_meta.copy(); out_u8.update(dtype="uint8", nodata=0, compress="lzw")
out_f32 = ref_meta.copy(); out_f32.update(dtype="float32", nodata=0.0, compress="lzw")

with rasterio.open(STD_DIR / "final_valid_mask.tif", "w", **out_u8) as dst:
    dst.write(final_valid.astype(np.uint8), 1)

with open(STD_DIR / "valid_mask_stats.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["valid_pixel_count", n_valid])
    w.writerow(["valid_area_km2", round(valid_area_km2, 2)])
    w.writerow(["excluded_pixel_count", n_total - n_valid])
    w.writerow(["common_valid_mask_pixels", n_common])
    w.writerow(["note", "final_valid = common_mask AND valid S AND valid FSI AND valid C AND valid N AND valid landcover; FSI定义域(=当前适生167像元)为约束条件"])

np.save(STD_DIR / "pixel_area_km2.npy", area_array.astype(np.float32))

# --- Step 3: Landcover Policy ---
print("\n=== Step 3: Landcover Policy ===")
lc_classes = {}
with open(INPUT_DIR / "landcover/landcover_class_dictionary.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        lc_classes[int(row["class_code"])] = row["class_name"]

# A. Potentially available (1.0): Croplands, Cropland/Natural Vegetation Mosaic
# B. Conditional (0.5): 森林/灌丛/稀树草原/草地 —— 林下种植可能但需实地核验; 禁止无条件=1
# C. Excluded (0.0): 水体/湿地/城市/冰雪/裸地/未分类
policy = {
    0:  ("Excluded", 0.0, "Water - cannot be cultivated", ""),
    1:  ("Conditional", 0.5, "Forest - understory cultivation possible in principle, but no protected-area/tenure/legal data; field verification required", "conditional candidate only"),
    2:  ("Conditional", 0.5, "Forest - field verification required", "conditional candidate only"),
    3:  ("Conditional", 0.5, "Forest - field verification required (ginseng's native understory habitat)", "conditional candidate only"),
    4:  ("Conditional", 0.5, "Deciduous broadleaf forest - typical ginseng understory habitat, but tenure/conservation unknown; field verification required", "conditional candidate only"),
    5:  ("Conditional", 0.5, "Mixed forest - understory potential; field verification required", "conditional candidate only"),
    6:  ("Conditional", 0.5, "Closed shrublands - field verification required", "conditional candidate only"),
    7:  ("Conditional", 0.5, "Open shrublands - field verification required", "conditional candidate only"),
    8:  ("Conditional", 0.5, "Woody savannas - field verification required", "conditional candidate only"),
    9:  ("Conditional", 0.5, "Savannas - field verification required", "conditional candidate only"),
    10: ("Conditional", 0.5, "Grasslands - convertible in principle; field verification required", "conditional candidate only"),
    11: ("Excluded", 0.0, "Permanent wetlands - unsuitable for cultivation", ""),
    12: ("Potentially available", 1.0, "Croplands - existing agricultural land, potentially convertible", ""),
    13: ("Excluded", 0.0, "Urban and built-up - cannot be cultivated", ""),
    14: ("Potentially available", 1.0, "Cropland/natural vegetation mosaic - agricultural mosaic, potentially convertible", ""),
    15: ("Excluded", 0.0, "Snow and ice - unsuitable", ""),
    16: ("Excluded", 0.0, "Barren or sparsely vegetated - unsuitable", ""),
    17: ("Excluded", 0.0, "Water bodies - cannot be cultivated", ""),
    255: ("Excluded", 0.0, "Unclassified - excluded for safety", ""),
}

with open(POLICY_DIR / "landcover_suitability_policy.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["class_code", "class_name", "cultivation_policy", "land_factor", "decision_basis", "notes"])
    for code in sorted(policy):
        pol, factor, basis, notes = policy[code]
        w.writerow([code, lc_classes.get(code, f"Class_{code}"), pol, factor, basis, notes])

lc_int = np.nan_to_num(landcover, nan=255).astype(np.int32)
L = np.zeros(ref_shape, dtype=np.float32)
for code, (pol, factor, basis, notes) in policy.items():
    L[lc_int == code] = factor
L[~final_valid] = 0.0

with rasterio.open(POLICY_DIR / "land_availability_factor.tif", "w", **out_f32) as dst:
    dst.write(L, 1)

# 适生像元上的土地覆盖构成(供报告)
cb = current_binary == 1
print("  当前适生像元的土地覆盖构成:")
for code in sorted(np.unique(lc_int[cb])):
    n = int((lc_int[cb] == code).sum())
    pol, factor = policy[int(code)][0], policy[int(code)][1]
    print(f"    {int(code):3d} {lc_classes.get(int(code),'?'):<38s} {n:4d} px → {pol} (L={factor})")

# --- Step 4: 标准化五层 S/F/C/N/L (0-1, 掩膜外=0) ---
print("\n=== Step 4: Standardize S/F/C/N/L ===")
def standardize(arr, name, transform=None):
    out = arr.copy().astype(np.float32)
    if transform:
        out = transform(out)
    out[~np.isfinite(out)] = 0.0
    out[~final_valid] = 0.0
    out = np.clip(out, 0.0, 1.0)
    with rasterio.open(STD_DIR / name, "w", **out_f32) as dst:
        dst.write(out, 1)
    vals = out[final_valid]
    print(f"  {name}: mean={vals.mean():.4f}, median={np.median(vals):.4f}, range=[{vals.min():.4f}, {vals.max():.4f}]")
    return out

S = standardize(current_suit, "current_suitability_S.tif")
F = standardize(fsi, "future_stability_F.tif")
C = standardize(confidence, "prediction_confidence_C.tif")
N = standardize(novelty, "novelty_reliability_N.tif", transform=lambda x: 1.0 - x)
L_out = standardize(L, "land_availability_L.tif")

print("\nDone.")
