#!/usr/bin/env python3
"""
实验6B - Step 0: 输入QC
检查SHA256、栅格对齐、值域范围
基于修正后的实验5交接包（change-class编码修正：1=Stable, 2=Loss）
"""
import os, sys, json, hashlib, csv
import numpy as np
import rasterio
from pathlib import Path

ROOT = Path(r"E:\人参种在哪\实验6B")
INPUT_DIR = ROOT / "00_input_from_experiment5"
QC_DIR = ROOT / "01_input_qc"
LOG_DIR = ROOT / "logs"
os.makedirs(QC_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("=" * 60)
print("实验6B 输入QC（修正版handoff）")
print("=" * 60)

# --- 1. SHA256 verification ---
manifest_path = INPUT_DIR / "sha256_manifest.csv"
sha_results = []
with open(manifest_path, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        file_rel = row["file"].replace("\\", "/")
        file_path = INPUT_DIR / file_rel
        expected = row["sha256"]
        if not file_path.exists():
            sha_results.append({"file": file_rel, "expected": expected, "actual": "FILE_NOT_FOUND", "match": False})
            continue
        with open(file_path, "rb") as bf:
            actual = hashlib.sha256(bf.read()).hexdigest()
        sha_results.append({"file": file_rel, "expected": expected, "actual": actual, "match": actual == expected})

sha_all_ok = all(r["match"] for r in sha_results)
n_ok = sum(1 for r in sha_results if r["match"])
print(f"\n1. SHA256: {n_ok}/{len(sha_results)} 匹配")
for r in sha_results:
    if not r["match"]:
        print(f"   MISMATCH: {r['file']}")

# --- 2. Raster alignment check ---
tif_files = sorted(INPUT_DIR.glob("**/*.tif"))
alignment_results = []
ref_info = None
ref_name = None
all_aligned = True

for tif_path in tif_files:
    with rasterio.open(tif_path) as src:
        info = {
            "file": str(tif_path.relative_to(INPUT_DIR)).replace("\\", "/"),
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "res_x": round(abs(src.transform[0]), 10),
            "res_y": round(abs(src.transform[4]), 10),
            "origin_x": round(src.transform[2], 10),
            "origin_y": round(src.transform[5], 10),
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
        }
        if ref_info is None:
            ref_info, ref_name = info, info["file"]
        aligned = (
            info["crs"] == ref_info["crs"]
            and info["width"] == ref_info["width"]
            and info["height"] == ref_info["height"]
            and info["res_x"] == ref_info["res_x"]
            and info["res_y"] == ref_info["res_y"]
            and info["origin_x"] == ref_info["origin_x"]
            and info["origin_y"] == ref_info["origin_y"]
        )
        info["aligned_to_reference"] = aligned
        if not aligned:
            all_aligned = False
        alignment_results.append(info)

print(f"\n2. 栅格对齐: {sum(1 for a in alignment_results if a['aligned_to_reference'])}/{len(alignment_results)} 对齐 (参考: {ref_name})")
print(f"   参考网格: {ref_info['crs']}, {ref_info['width']}x{ref_info['height']}, res={ref_info['res_x']}")

# --- 3. Value range checks ---
range_checks = []
def check_range(name, path, lo=None, hi=None, discrete=False):
    with rasterio.open(path) as src:
        d = src.read(1).astype(np.float64)
        valid = np.isfinite(d)
        if src.nodata is not None and not np.isnan(src.nodata):
            valid &= (d != src.nodata)
        vals = d[valid]
        if len(vals) == 0:
            vmin = vmax = None
            ok = False
        else:
            vmin, vmax = float(vals.min()), float(vals.max())
            ok = True
            if lo is not None and vmin < lo - 1e-9:
                ok = False
            if hi is not None and vmax > hi + 1e-9:
                ok = False
            if discrete:
                ok = ok and np.all(np.abs(vals - np.round(vals)) < 1e-9)
        range_checks.append({
            "layer": name, "n_valid": int(valid.sum()),
            "min": vmin, "max": vmax, "expected": f"[{lo},{hi}]" if lo is not None else "discrete",
            "ok": ok,
        })
        print(f"   {name}: n_valid={int(valid.sum())}, range=[{vmin}, {vmax}], ok={ok}")

print("\n3. 值域检查:")
check_range("current_ensemble_suitability", INPUT_DIR / "current/current_ensemble_suitability.tif", 0, 1)
check_range("current_binary_suitability", INPUT_DIR / "current/current_binary_suitability.tif", discrete=True)
check_range("future_stability_index", INPUT_DIR / "future_stability/future_stability_index.tif", 0, 1)
check_range("future_stability_classes", INPUT_DIR / "future_stability/future_stability_classes.tif", discrete=True)
check_range("loss_frequency_current_suitable", INPUT_DIR / "future_stability/loss_frequency_current_suitable.tif", 0, 1)
check_range("scenario_sd", INPUT_DIR / "uncertainty/scenario_sd.tif", 0, None)
check_range("prediction_confidence", INPUT_DIR / "uncertainty/prediction_confidence.tif", 0, 1)
check_range("high_uncertainty_zone", INPUT_DIR / "uncertainty/high_uncertainty_zone.tif", discrete=True)
check_range("novelty_frequency", INPUT_DIR / "novelty/novelty_frequency.tif", 0, 1)
check_range("MESS_summary", INPUT_DIR / "novelty/MESS_summary.tif")
check_range("vulnerability_confidence_adjusted", INPUT_DIR / "vulnerability/vulnerability_confidence_adjusted.tif", 0, 1)
check_range("robust_climatic_core", INPUT_DIR / "vulnerability/robust_climatic_core.tif", discrete=True)
check_range("high_confidence_loss_zone", INPUT_DIR / "vulnerability/high_confidence_loss_zone.tif", discrete=True)
check_range("landcover_aligned", INPUT_DIR / "landcover/landcover_aligned.tif", discrete=True)
check_range("common_valid_mask", INPUT_DIR / "reference/common_valid_mask.tif", discrete=True)

ranges_ok = all(r["ok"] for r in range_checks)

# --- 4. Corrected-data sanity checks (实验6B特有) ---
print("\n4. 修正数据健全性检查:")
with rasterio.open(INPUT_DIR / "current/current_binary_suitability.tif") as src:
    cb = src.read(1) == 1
with rasterio.open(INPUT_DIR / "future_stability/future_stability_index.tif") as src:
    fsi = src.read(1)
with rasterio.open(INPUT_DIR / "future_stability/loss_frequency_current_suitable.tif") as src:
    lfi = src.read(1)
fsi_v = fsi[cb]
lfi_v = lfi[cb]
print(f"   当前适生像元: {int(cb.sum())}")
print(f"   FSI中位数: {np.median(fsi_v):.4f} (修正后应为0)")
print(f"   LFI中位数: {np.median(lfi_v):.4f} (修正后应为1)")
print(f"   FSI+LFI=1 最大偏差: {np.max(np.abs(fsi_v + lfi_v - 1)):.2e}")
corrected_ok = (np.median(fsi_v) == 0.0) and (np.median(lfi_v) == 1.0) and int(cb.sum()) == 167

# --- Outputs ---
with open(QC_DIR / "raster_alignment_check.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(alignment_results[0].keys()))
    writer.writeheader()
    for r in alignment_results:
        writer.writerow(r)

with open(QC_DIR / "sha256_verification.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "expected", "actual", "match"])
    writer.writeheader()
    for r in sha_results:
        writer.writerow(r)

with open(QC_DIR / "value_range_check.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["layer", "n_valid", "min", "max", "expected", "ok"])
    writer.writeheader()
    for r in range_checks:
        writer.writerow(r)

overall = "PASS" if (sha_all_ok and all_aligned and ranges_ok and corrected_ok) else "FAIL"
with open(QC_DIR / "EXPERIMENT6_INPUT_QC.md", "w", encoding="utf-8") as f:
    f.write("# 实验6B 输入QC报告\n\n")
    f.write("**输入包**: 实验5修正版交接（CORRECTED HANDOFF, 2026-08-14）\n")
    f.write("**关键修正**: change-class编码 1=Stable/2=Loss（旧包中FSI/LFI互换）\n\n")
    f.write(f"**总体状态**: {overall}\n\n")
    f.write("## 检查结果\n\n")
    f.write(f"- SHA256: {n_ok}/{len(sha_results)} 匹配 → {'PASS' if sha_all_ok else 'FAIL'}\n")
    f.write(f"- 栅格对齐 (CRS/尺寸/分辨率/原点): {'全部对齐 EPSG:4326, 8640x4320, 0.0417°' if all_aligned else 'FAIL — 见raster_alignment_check.csv'}\n")
    f.write(f"- 值域检查: {'全部通过' if ranges_ok else 'FAIL — 见value_range_check.csv'}\n")
    f.write(f"- 修正数据健全性: FSI中位数={np.median(fsi_v):.4f}, LFI中位数={np.median(lfi_v):.4f}, "
            f"FSI+LFI≈1, 当前适生={int(cb.sum())}像元 → {'PASS' if corrected_ok else 'FAIL'}\n\n")
    f.write("## 修正版handoff的关键数值（与旧版相反）\n\n")
    f.write("| 指标 | 修正后（本包） | 旧版（错误） |\n|---|---|---|\n")
    f.write("| FSI中位数 | 0.0000 | 1.0000 |\n")
    f.write("| LFI中位数 | 1.0000 | 0.0000 |\n")
    f.write("| 稳健气候核心 | 0 像元 | 7 像元 |\n")
    f.write("| 高置信丧失区 | 124 像元 (74.3%) | 0 像元 |\n\n")
    f.write("## 缺失项（不阻塞）\n\n")
    f.write("- dominant_uncertainty_source.tif（实验5遗留警告项，操作手册第3节允许缺失）\n")
    f.write("- 边界文件不在handoff中；ADM统计使用 `00_统一数据预处理/03_boundaries/` 外部边界\n")

if not all_aligned:
    with open(ROOT / "STOP_A_INPUT_ALIGNMENT_FAILED.md", "w") as f:
        f.write("# STOP_A: 输入栅格错位\n\n见 01_input_qc/raster_alignment_check.csv\n")
    print("\n!!! STOP_A: 栅格错位，停止 !!!")
    sys.exit(1)

print(f"\n输入QC总体: {overall}")
print("Done.")
