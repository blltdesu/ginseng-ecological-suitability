#!/usr/bin/env python3
"""
实验6B - Step 25: 表格 (E6-Table1~5) + 图件数据准备 (14_figure_data)
"""
import os, json, csv, shutil
import numpy as np
import rasterio
from pathlib import Path

ROOT = Path(r"E:\人参种在哪\实验6B")
INPUT_DIR = ROOT / "00_input_from_experiment5"
TABLE_DIR = ROOT / "15_tables"
FIG_DATA = ROOT / "14_figure_data"
for d in [TABLE_DIR, FIG_DATA]:
    os.makedirs(d, exist_ok=True)

# --- E6-Table1: 指标定义 ---
with open(TABLE_DIR / "E6_Table1_indicator_definition.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["indicator", "symbol", "range", "source", "description"])
    w.writerow(["Current Suitability", "S", "0-1", "Experiment 1 (via Exp5 handoff)", "当前4模型ensemble适宜性"])
    w.writerow(["Future Stability Index", "F", "0-1", "Experiment 5 (CORRECTED)", "跨情景稳定频率; 修正后中位数=0"])
    w.writerow(["Prediction Confidence", "C", "0-1", "Experiment 5", "1 - 归一化情景SD"])
    w.writerow(["Novelty Reliability", "N", "0-1", "Experiment 5", "1 - NoveltyFrequency"])
    w.writerow(["Land Availability", "L", "0/0.5/1", "Experiment 6B policy", "土地覆盖政策因子(林地=0.5条件性)"])

# --- E6-Table2: 分区规则 ---
with open(TABLE_DIR / "E6_Table2_zoning_rules.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["zone", "zone_name", "rule", "priority"])
    w.writerow([1, "Core stable candidate",
                "CandidateMask=1 AND S>=high AND FSI>=0.90 AND Conf>=0.75 AND NovFreq<=0.10 AND L=1 AND GPI>=P75(candidate)", 3])
    w.writerow([2, "General stable candidate",
                "CandidateMask=1 AND FSI>=0.75 AND Conf>=0.50 AND NovFreq<0.50 AND GPI>=P50(candidate)", 4])
    w.writerow([3, "Conditional candidate",
                "剩余候选池像元 OR (当前适生 AND L=0.5 AND FSI>=0.50); 标记field verification required", 5])
    w.writerow([4, "Climate-vulnerable",
                "当前适生 AND (HighConfidenceLoss=1 OR (LFI>=0.75 AND Conf>=0.75))", 1])
    w.writerow([5, "High-uncertainty", "HighUncertaintyZone=1 AND 未进入Zone IV", 2])
    w.writerow([6, "Non-priority", "final_valid域内其余像元(当前不适宜/L=0/FSI<0.50)", 6])
    w.writerow([])
    w.writerow(["note", "分区域=final_valid_mask(167验证适生像元)", "CandidateMask=当前适生 AND L>0 AND NOT高置信丧失 AND FSI>=0.50", ""])

# --- E6-Table3: 分区面积 ---
shutil.copy2(ROOT / "09_area_statistics/zoning_area_summary.csv", TABLE_DIR / "E6_Table3_zoning_area.csv")

# --- E6-Table4: ADM1优先 ---
shutil.copy2(ROOT / "10_admin_priority/ADM1_priority_statistics.csv", TABLE_DIR / "E6_Table4_ADM1_priority.csv")

# --- E6-Table5: Top核心斑块 ---
shutil.copy2(ROOT / "10_admin_priority/top_core_candidate_patches.csv", TABLE_DIR / "E6_Table5_top_core_patches.csv")

print("Tables 1-5 完成")

# --- 图件数据 ---
# Fig1 框架节点 + 权重
with open(FIG_DATA / "E6_Fig1_framework_nodes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["node", "description"])
    w.writerow(["S", "Current Suitability (Exp1)"])
    w.writerow(["F", "Future Stability Index (Exp5, CORRECTED)"])
    w.writerow(["C", "Prediction Confidence (Exp5)"])
    w.writerow(["N", "Novelty Reliability = 1 - NoveltyFrequency (Exp5)"])
    w.writerow(["L", "Land Availability (Exp6B landcover policy)"])
    w.writerow(["CandidateMask", "CurrentBinary AND L>0 AND NOT HighConfLoss AND FSI>=0.50"])
    w.writerow(["GPI", "S^0.25 * F^0.35 * C^0.15 * N^0.10 * L^0.15 (geometric weighted)"])
    w.writerow(["FinalZoning", "6-zone: I Core / II General / III Conditional / IV Vulnerable / V Uncertain / VI Non-priority"])
shutil.copy2(ROOT / "06_priority_index/GPI_weight_scheme_main.csv", FIG_DATA / "E6_Fig1_weight_scheme.csv")

# Fig2 五个输入因子GeoTIFF
for src_name, dst_name in [
    ("current_suitability_S.tif", "E6_Fig2_S_current_suitability.tif"),
    ("future_stability_F.tif", "E6_Fig2_F_future_stability.tif"),
    ("prediction_confidence_C.tif", "E6_Fig2_C_prediction_confidence.tif"),
    ("novelty_reliability_N.tif", "E6_Fig2_N_novelty_reliability.tif"),
    ("land_availability_L.tif", "E6_Fig2_L_land_availability.tif"),
]:
    shutil.copy2(ROOT / "04_standardized_layers" / src_name, FIG_DATA / dst_name)

# Fig3 GPI
shutil.copy2(ROOT / "06_priority_index/GPI_main.tif", FIG_DATA / "E6_Fig3_GPI_main.tif")

# Fig4 分区 + 字典 + 边界
shutil.copy2(ROOT / "07_final_zoning/final_cultivation_zoning.tif", FIG_DATA / "E6_Fig4_final_zoning.tif")
shutil.copy2(ROOT / "07_final_zoning/zoning_class_dictionary.csv", FIG_DATA / "E6_Fig4_zoning_dictionary.csv")
bnd_src = Path(r"E:\人参种在哪\00_统一数据预处理\03_boundaries\study_context_adm0.gpkg")
if bnd_src.exists():
    shutil.copy2(bnd_src, FIG_DATA / "E6_Fig4_boundaries.gpkg")

# Fig5 斑块 + ADM1
shutil.copy2(ROOT / "08_spatial_postprocessing/priority_patches.gpkg", FIG_DATA / "E6_Fig5_priority_patches.gpkg")
shutil.copy2(ROOT / "10_admin_priority/ADM1_priority_statistics.csv", FIG_DATA / "E6_Fig5_ADM1_priority.csv")

# 绘图crop窗口(适生域bbox + 5°缓冲)
with rasterio.open(INPUT_DIR / "current/current_binary_suitability.tif") as src:
    cb = src.read(1) == 1
    tr = src.transform
rows, cols = np.where(cb)
pad = int(5 / abs(tr[0]))  # 5度
r0 = max(0, rows.min() - pad); r1 = min(cb.shape[0], rows.max() + pad + 1)
c0 = max(0, cols.min() - pad); c1 = min(cb.shape[1], cols.max() + pad + 1)
lon0 = tr[2] + c0 * tr[0]; lon1 = tr[2] + c1 * tr[0]
lat1 = tr[5] + r0 * tr[4]; lat0 = tr[5] + r1 * tr[4]
crop = {"row0": int(r0), "row1": int(r1), "col0": int(c0), "col1": int(c1),
        "lon_min": float(lon0), "lon_max": float(lon1), "lat_min": float(lat0), "lat_max": float(lat1)}
with open(FIG_DATA / "crop_window.json", "w") as f:
    json.dump(crop, f, indent=2)
print(f"crop窗口: rows {r0}-{r1}, cols {c0}-{c1}; lon {lon0:.1f}-{lon1:.1f}, lat {lat0:.1f}-{lat1:.1f}")

print("Figure data 完成")
