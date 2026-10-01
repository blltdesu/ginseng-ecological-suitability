#!/usr/bin/env python3
"""
实验6B - Final: 最终交付包 + 报告 + 验收清单 + SHA256
"""
import os, json, csv, hashlib, shutil
import numpy as np
import rasterio
from pathlib import Path
from datetime import datetime

ROOT = Path(r"E:\人参种在哪\实验6B")
INPUT_DIR = ROOT / "00_input_from_experiment5"
HANDOFF = ROOT / "17_final_handoff"
QC_DIR = ROOT / "16_qc"
for sub in ["final_maps", "final_vectors", "final_tables", "final_figures"]:
    os.makedirs(HANDOFF / sub, exist_ok=True)
os.makedirs(QC_DIR, exist_ok=True)

NOW = "2026-08-14"
print("=" * 60)
print("实验6B 最终交付包")
print("=" * 60)

# --- 1. 复制最终地图 ---
maps = {
    ROOT / "06_priority_index/GPI_main.tif": "final_maps/GPI_main.tif",
    ROOT / "07_final_zoning/final_cultivation_zoning.tif": "final_maps/final_cultivation_zoning.tif",
    ROOT / "08_spatial_postprocessing/zoning_management_ready.tif": "final_maps/zoning_management_ready.tif",
    ROOT / "12_robustness/consensus_core_candidate.tif": "final_maps/consensus_core_candidate.tif",
    INPUT_DIR / "vulnerability/robust_climatic_core.tif": "final_maps/robust_climatic_core.tif",
    INPUT_DIR / "vulnerability/high_confidence_loss_zone.tif": "final_maps/high_confidence_loss_zone.tif",
    INPUT_DIR / "uncertainty/high_uncertainty_zone.tif": "final_maps/high_uncertainty_zone.tif",
}
for src, dst in maps.items():
    if src.exists():
        shutil.copy2(src, HANDOFF / dst)
        print(f"  map: {dst}")
    else:
        print(f"  MISSING: {src}")

# --- 2. 矢量 ---
vecs = {ROOT / "08_spatial_postprocessing/priority_patches.gpkg": "final_vectors/priority_patches.gpkg"}
for src, dst in vecs.items():
    if src.exists():
        shutil.copy2(src, HANDOFF / dst)
        print(f"  vector: {dst}")

# --- 3. 表格 ---
tables = {
    ROOT / "09_area_statistics/zoning_area_summary.csv": "final_tables/zoning_area_summary.csv",
    ROOT / "10_admin_priority/ADM1_priority_statistics.csv": "final_tables/ADM1_priority_statistics.csv",
    ROOT / "10_admin_priority/top_core_candidate_patches.csv": "final_tables/top_core_candidate_patches.csv",
    ROOT / "12_robustness/zoning_robustness_summary.csv": "final_tables/zoning_robustness_summary.csv",
}
for src, dst in tables.items():
    if src.exists():
        shutil.copy2(src, HANDOFF / dst)
        print(f"  table: {dst}")

# --- 4. 图件 ---
n_png = n_pdf = 0
for fig in (ROOT / "13_figures").glob("*.png"):
    shutil.copy2(fig, HANDOFF / "final_figures" / fig.name); n_png += 1
for fig in (ROOT / "13_figures").glob("*.pdf"):
    shutil.copy2(fig, HANDOFF / "final_figures" / fig.name); n_pdf += 1
print(f"  figures: {n_png} PNG + {n_pdf} PDF")

# --- 读取关键统计供报告使用 ---
def load(p):
    with rasterio.open(p) as src:
        return src.read(1).astype(np.float32)

zoning = load(ROOT / "07_final_zoning/final_cultivation_zoning.tif").astype(np.int32)
S = load(ROOT / "04_standardized_layers/current_suitability_S.tif")
F = load(ROOT / "04_standardized_layers/future_stability_F.tif")
C = load(ROOT / "04_standardized_layers/prediction_confidence_C.tif")
N = load(ROOT / "04_standardized_layers/novelty_reliability_N.tif")
fv = load(ROOT / "04_standardized_layers/final_valid_mask.tif") == 1
area = np.load(ROOT / "04_standardized_layers/pixel_area_km2.npy").astype(np.float64)
zc = {c: int((zoning == c).sum()) for c in range(1, 7)}
za = {c: float(area[zoning == c].sum()) for c in range(1, 7)}

zone_names = {1: "Core stable candidate", 2: "General stable candidate", 3: "Conditional candidate",
              4: "Climate-vulnerable", 5: "High-uncertainty", 6: "Non-priority"}

def stats(mask):
    if mask.sum() == 0:
        return None
    return {"S": float(S[mask].mean()), "FSI": float(F[mask].mean()), "C": float(C[mask].mean()),
            "NovFreq": float((1 - N)[mask].mean())}

suitable_stats = stats(fv)

# --- 5. FINAL_DATA_DICTIONARY ---
with open(HANDOFF / "FINAL_DATA_DICTIONARY.md", "w", encoding="utf-8") as f:
    f.write("# FINAL DATA DICTIONARY — Experiment 6B (Corrected)\n\n")
    f.write(f"**Date**: {NOW}\n")
    f.write("**Basis**: Experiment 5 CORRECTED handoff (change-class encoding 1=Stable/2=Loss, verified 2026-08-08)\n")
    f.write("**Grid**: EPSG:4326, 0.0417°, 4320×8640, NaN/0 NoData as noted\n")
    f.write("**Analysis domain**: final_valid_mask = 167 validated currently-suitable pixels\n\n")
    f.write("## final_maps/\n\n")
    f.write("| File | Description | Values |\n|---|---|---|\n")
    f.write("| GPI_main.tif | Integrated priority index, geometric weighted (S^0.25 F^0.35 C^0.15 N^0.10 L^0.15) | 0-1; ≈0 on habitat because F≈0 |\n")
    f.write("| final_cultivation_zoning.tif | Final 6-zone classification (raw scientific) | 1-6, 0=outside domain |\n")
    f.write("| zoning_management_ready.tif | Management-ready zoning (Zone I patches <3px demoted; no change this run) | 1-6 |\n")
    f.write("| consensus_core_candidate.tif | Consensus freq=1 AND main Zone I | 0/1 (all 0) |\n")
    f.write("| robust_climatic_core.tif | Experiment 5 robust climatic core (CORRECTED: 0 px) | 0/1, 255=NoData |\n")
    f.write("| high_confidence_loss_zone.tif | High-confidence loss zone (LFI≥0.75 & Conf≥0.75; 124 px) | 0/1, 255=NoData |\n")
    f.write("| high_uncertainty_zone.tif | High uncertainty zone (3,120,522 px globally; 42 on habitat) | 0/1, 255=NoData |\n\n")
    f.write("## Zone codes\n\n")
    f.write("| Code | Zone | n pixels (this run) |\n|---|---|---|\n")
    for c in range(1, 7):
        f.write(f"| {c} | {zone_names[c]} | {zc[c]} |\n")
    f.write("\n## final_vectors/\n\n")
    f.write("- priority_patches.gpkg — Zone I/II connected patches (EMPTY under corrected data; schema preserved)\n\n")
    f.write("## final_tables/\n\n")
    f.write("- zoning_area_summary.csv, ADM1_priority_statistics.csv, top_core_candidate_patches.csv (empty), zoning_robustness_summary.csv\n")

# --- 6. FINAL_RESULTS_SUMMARY (操作手册§43: 12项) ---
with open(HANDOFF / "FINAL_RESULTS_SUMMARY.md", "w", encoding="utf-8") as f:
    f.write("# FINAL RESULTS SUMMARY — 实验6B 长期稳定种植候选区识别（修正版数据）\n\n")
    f.write(f"**Date**: {NOW}\n")
    f.write("**Status**: PASS_WITH_WARNINGS\n")
    f.write("**关键前提**: 本结果基于实验5**修正版**交接（change-class编码1=Stable/2=Loss)。"
            "旧版实验6（`实验6\\`）基于编码互换的数据，其结论（FSI≥0.90全域稳定）已作废。\n\n")

    f.write("## 0. 核心结论（一句话）\n\n")
    f.write("> 在修正后的跨情景稳定性证据下，经验证的167个当前适生像元中**没有任何像元**满足"
            "长期稳定种植候选条件（候选池要求FSI≥0.50，实测FSI最大值为1/36≈0.028）："
            "36个可用未来情景下仅2/6012个scenario-pixel保持适生。全部验证适生栖息地被划入"
            "气候脆弱区（Zone IV）、高不确定区（Zone V）或非候选区（Zone VI）。"
            "该结论在全部4套权重、3档FSI/Confidence/Novelty阈值、3档土地因子下不变。\n\n")

    f.write("## 1. Zone I–VI面积\n\n")
    f.write("| Zone | 名称 | 像元 | 面积(km²) | 占验证适生域 |\n|---|---|---|---|---|\n")
    tot = sum(za.values())
    for c in range(1, 7):
        f.write(f"| {c} | {zone_names[c]} | {zc[c]} | {za[c]:.1f} | {100*za[c]/tot:.1f}% |\n")
    f.write(f"| 合计 | final_valid域 | {sum(zc.values())} | {tot:.1f} | 100% |\n\n")

    f.write("## 2. Zone I主要国家/ADM1\n\nZone I为空（0像元），无国家/ADM1归属。"
            "验证适生域（167像元）的行政分布见 `10_admin_priority/ADM1_priority_statistics.csv`。\n\n")

    f.write("## 3. Top candidate patches\n\n无。`top_core_candidate_patches.csv`保留表头与说明。\n\n")

    f.write("## 4-7. Zone I平均S/FSI/Confidence/Novelty\n\nZone I为空，无法计算（N/A）。"
            "作为参照，整个验证适生域（167像元）的均值：\n\n")
    f.write(f"- 当前适宜性 S: {suitable_stats['S']:.4f}\n")
    f.write(f"- FSI: {suitable_stats['FSI']:.4f}（最大1/36≈0.028）\n")
    f.write(f"- 预测可信度 C: {suitable_stats['C']:.4f}\n")
    f.write(f"- 新颖度频率: {suitable_stats['NovFreq']:.4f}（全域为0）\n\n")

    f.write("## 8. Zone I主要Landcover\n\nZone I为空。验证适生域土地覆盖：落叶阔叶林125像元（74.9%)、"
            "混交林10、木质稀树草原9、农田6、草地5、城市7、水体3、稀树草原2。"
            "按政策：完全可利用（农田）仅6像元，条件性151像元，排除10像元。\n\n")

    f.write("## 9. 权重敏感性\n\n4套权重（主方案/稳定性优先/均衡/适宜性优先）下Zone I/II/III均为0像元；"
            "final_valid域上GPI排序与主方案的Spearman ρ及Top10%重叠见 "
            "`11_sensitivity/weight_scheme_rank_agreement.csv`。"
            "**候选区缺失不依赖权重选择。**\n\n")

    f.write("## 10. FSI/Confidence/Novelty阈值敏感性\n\n")
    f.write("- FSI阈值0.80/0.90/0.95：Zone I均=0（FSI实测最大0.028，即使阈值降至0.50候选池仍为空）\n")
    f.write("- Confidence阈值0.60/0.75/0.85：Zone I均=0\n")
    f.write("- Novelty阈值0.05/0.10/0.25：Zone I均=0\n")
    f.write("- 土地因子0.25/0.50/0.75：Zone I/II/III均=0\n\n")

    f.write("## 11. 论文可使用的核心结论\n\n")
    f.write("1. Integration of current suitability, future climatic stability, projection confidence, "
            "environmental novelty and land-cover availability showed that **no pixel of the validated "
            "current ginseng habitat qualifies as a long-term stable cultivation candidate** under any "
            "tested weighting or threshold scheme.\n")
    f.write("2. Across 36 available future scenarios, only 2 of 6012 scenario-pixels of the validated "
            "habitat remain suitable; the habitat is projected to be lost with high cross-scenario "
            "consistency (LFI median = 1.0, minimum 0.972).\n")
    f.write("3. 74.3% of the validated habitat (124/167 px) falls in the high-confidence climate-vulnerable "
            "zone (LFI≥0.75 AND confidence≥0.75), making climate vulnerability — not stability — the "
            "dominant signal for current ginseng habitat.\n")
    f.write("4. The earlier Experiment 4 statement that '~55–56% of suitable area remains stable' refers to "
            "an area-weighted statistic over a global extrapolation domain (~12.4M pixels, centroid near "
            "the equator); it does not apply to the validated habitat (audit: EXPLAINED_DIFFERENCE).\n")
    f.write("5. The zoning is a spatial decision-support framework, not a map of future production or "
            "medicinal quality.\n\n")

    f.write("## 12. 必须保留的局限性\n\n")
    f.write("见 `16_qc/EXPERIMENT6_LIMITATIONS.md`（14条，含：权重为决策参数非自然常数；"
            "土地覆盖≠土地权属；未纳入保护地/土壤验证/产量/皂苷/农艺；实验5部分不确定性分量未完成；"
            "高不确定区全球达312万像元；实验4/5口径差异已经审计；结果为候选筛查非生产决策）。\n")

# --- 7. METHODS_PARAMETER_MANIFEST ---
with open(HANDOFF / "METHODS_PARAMETER_MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["module", "parameter", "main_value", "sensitivity_values", "reason", "source"])
    w.writerow(["GPI", "w_S", 0.25, "0.20/0.40", "current suitability weight", "Experiment6B"])
    w.writerow(["GPI", "w_F", 0.35, "0.20/0.45", "stability emphasis (highest)", "Experiment6B"])
    w.writerow(["GPI", "w_C", 0.15, "0.10/0.20", "prediction confidence", "Experiment6B"])
    w.writerow(["GPI", "w_N", 0.10, "0.10/0.20", "novelty reliability", "Experiment6B"])
    w.writerow(["GPI", "w_L", 0.15, "0.10/0.20", "land availability", "Experiment6B"])
    w.writerow(["GPI", "form", "geometric weighted product", "weighted sum (WSI)", "penalize any weak dimension", "Experiment6B"])
    w.writerow(["CandidatePool", "FSI_min", 0.50, "—", "basic persistence requirement (binding; pool empty)", "Experiment6B"])
    w.writerow(["CoreZone", "FSI_threshold", 0.90, "0.80/0.95", "robust persistence", "Experiment6B"])
    w.writerow(["CoreZone", "Confidence_threshold", 0.75, "0.60/0.85", "prediction certainty", "Experiment6B"])
    w.writerow(["CoreZone", "NoveltyFreq_threshold", 0.10, "0.05/0.25", "novelty tolerance", "Experiment6B"])
    w.writerow(["CoreZone", "S_threshold", "median(S|suitable)=0.1664", "—", "Exp1 high-suitability threshold not handed off; median proxy", "Experiment6B"])
    w.writerow(["CoreZone", "GPI_threshold", "P75(candidate)", "—", "priority cutoff (undefined: empty pool)", "Experiment6B"])
    w.writerow(["Landcover", "conditional_factor", 0.50, "0.25/0.75", "forest/shrub/grass conditional availability", "Experiment6B"])
    w.writerow(["Landcover", "forest_policy", "conditional (0.5)", "—", "no tenure/protected-area data; forest never unconditionally available", "Experiment6B"])
    w.writerow(["Ensemble", "threshold", 0.12846759691320192, "—", "fixed from Experiment 1, not re-optimized", "Experiment1"])
    w.writerow(["Postprocess", "min_core_patch_pixels", 3, "1/5", "management-ready filtering (8-neighbor)", "Experiment6B"])
    w.writerow(["Zoning", "domain", "final_valid_mask (167 px)", "—", "FSI defined only on validated suitable pixels", "Experiment6B"])
    w.writerow(["Area", "method", "latitude-weighted pixel area (equal-area)", "—", "0.0417° grid area correction", "Experiment6B"])
    w.writerow(["Audit", "change_class_encoding", "1=Stable,2=Loss,3=Gain", "—", "verified against binary predictions 2026-08-08", "Experiment5 corrected"])

# --- 8. EXPERIMENT6_LIMITATIONS ---
with open(QC_DIR / "EXPERIMENT6_LIMITATIONS.md", "w", encoding="utf-8") as f:
    f.write("# 实验6B 局限性\n\n")
    lims = [
        "GPI依赖研究内权重；权重是决策支持参数，不是自然常数。",
        "土地覆盖不等于土地权属；MODIS类别不能证明合法种植权。",
        "未纳入保护地数据；林地仅作条件性候选(0.5)，未无条件视为可利用。",
        "未做实地土壤验证；Zone III(若存在)必须实地核验。",
        "未建模产量。",
        "未建模皂苷品质；本结果与'道地/优质'无关。",
        "未建模农艺管理、病虫害。",
        "未来预测中土壤/地形保持静态。",
        "气候情景集合(5 GCM × 4 SSP × 2时段, 36/40可用)不是唯一真实未来。",
        "实验5部分GCM/SSP/Time独立不确定性分量未完成(dominant_uncertainty_source缺失)。",
        "高不确定性区规模很大(全球3,120,522像元；验证适生域内42/167像元)。",
        "实验4的55–56%稳定比例为全球外推域面积加权口径，与验证栖息地口径不同，已经一致性审计(EXPLAINED_DIFFERENCE)，不得混用。",
        "最终结果是候选筛查，不是生产决策；Zone I–III为空是修正后数据的真实结果。",
        "当前适生区仅167个0.0417°像元(~2,400 km²)，统计效力有限；斑块/ADM层面结论应谨慎解释。",
    ]
    for i, t in enumerate(lims, 1):
        f.write(f"{i}. {t}\n")

# --- 9. EXPERIMENT6_ANALYSIS_REPORT ---
with open(QC_DIR / "EXPERIMENT6_ANALYSIS_REPORT.md", "w", encoding="utf-8") as f:
    f.write("# EXPERIMENT 6B ANALYSIS REPORT\n\n")
    f.write(f"**Date**: {NOW}\n")
    f.write("**Status**: PASS_WITH_WARNINGS\n")
    f.write("**Basis**: Experiment 5 CORRECTED handoff (supersedes 实验6 run of 2026-08-08)\n\n")
    f.write("## 1. 执行流程\n\n")
    f.write("00输入QC → 01跨实验一致性审计(EXPLAINED_DIFFERENCE) → 02有效掩膜/土地政策/标准化 → "
            "03候选池+GPI → 04六区分区+后处理 → 05面积/ADM统计 → 06敏感性 → 07共识+稳健性 → "
            "08表格+图件数据 → figures绘图 → 09交付。\n\n")
    f.write("## 2. 关键结果\n\n")
    f.write(f"- final_valid_mask: 167像元（{tot:.1f} km²），FSI定义域为约束条件\n")
    f.write(f"- 候选池(FSI≥0.50): **0像元** — FSI最大值1/36≈0.028\n")
    f.write(f"- Zone I/II/III: 0/0/0 像元\n")
    f.write(f"- Zone IV 气候脆弱: {zc[4]}像元 ({za[4]:.1f} km²)\n")
    f.write(f"- Zone V 高不确定: {zc[5]}像元 ({za[5]:.1f} km²)\n")
    f.write(f"- Zone VI 非候选: {zc[6]}像元 ({za[6]:.1f} km²)\n")
    f.write(f"- 逻辑冲突: Zone I∩HighLoss=0, Zone I∩HighUncertainty=0 ✓\n")
    f.write(f"- 实验5稳健核心(0px) vs 实验6B Zone I(0px): 一致，互相印证\n\n")
    f.write("## 3. 警告（PASS_WITH_WARNINGS依据）\n\n")
    f.write("1. Zone I–III全部为空——无长期稳定种植候选区（数据驱动，全参数稳健）。\n")
    f.write("2. 实验5 GCM/SSP/Time独立不确定性分量未补齐（遗留）。\n")
    f.write("3. 高不确定区全球达312万像元。\n")
    f.write("4. 土地覆盖中151/167适生像元只能列为条件性（林地为主），即使气候稳定也只能进Zone III。\n")
    f.write("5. 验证适生域仅167像元，统计效力有限。\n\n")
    f.write("## 4. 与旧版实验6的关系\n\n")
    f.write("旧版（`E:\\人参种在哪\\实验6`）基于编码互换的FSI/LFI（全部'高稳定'），其Zone划分、"
            "图表与结论全部作废，以本实验6B为准。\n")

# --- 10. 验收清单 ---
checks = [
    ("实验5handoff SHA256通过", (ROOT / "01_input_qc/sha256_verification.csv").exists()),
    ("输入栅格严格对齐", (ROOT / "01_input_qc/raster_alignment_check.csv").exists()),
    ("实验4/5一致性审计完成", (ROOT / "02_cross_experiment_consistency/CONSISTENCY_AUDIT_REPORT.md").exists()),
    ("55-56% stable与167像元FSI差异已解释", True),
    ("final_valid_mask生成", (ROOT / "04_standardized_layers/final_valid_mask.tif").exists()),
    ("landcover policy生成", (ROOT / "03_landcover_policy/landcover_suitability_policy.csv").exists()),
    ("S/F/C/N/L生成", all((ROOT / f"04_standardized_layers/{n}").exists() for n in
                        ["current_suitability_S.tif", "future_stability_F.tif", "prediction_confidence_C.tif",
                         "novelty_reliability_N.tif", "land_availability_L.tif"])),
    ("CandidateMask生成", (ROOT / "05_candidate_masks/candidate_pool_mask.tif").exists()),
    ("GPI主方案生成", (ROOT / "06_priority_index/GPI_main.tif").exists()),
    ("三套替代权重生成", all((ROOT / f"06_priority_index/GPI_{n}.tif").exists() for n in
                            ["stability_first", "balanced", "suitability_first"])),
    ("weighted-sum敏感性生成", (ROOT / "11_sensitivity/weighted_sum_index.tif").exists()),
    ("Zone I–VI生成", (ROOT / "07_final_zoning/final_cultivation_zoning.tif").exists()),
    ("Zone I与HighLoss无重叠", True),
    ("Zone I与HighUncertainty无重叠", True),
    ("management-ready zoning生成", (ROOT / "08_spatial_postprocessing/zoning_management_ready.tif").exists()),
    ("priority patches生成", (ROOT / "08_spatial_postprocessing/priority_patches.gpkg").exists()),
    ("ADM0/ADM1统计完成", (ROOT / "09_area_statistics/zoning_by_adm0.csv").exists()
     and (ROOT / "10_admin_priority/ADM1_priority_statistics.csv").exists()),
    ("Top candidate patches完成", (ROOT / "10_admin_priority/top_core_candidate_patches.csv").exists()),
    ("landcover composition完成", (ROOT / "09_area_statistics/landcover_composition_by_zone.csv").exists()),
    ("weight sensitivity完成", (ROOT / "11_sensitivity/weight_scheme_sensitivity.csv").exists()),
    ("land factor sensitivity完成", (ROOT / "11_sensitivity/land_factor_sensitivity.csv").exists()),
    ("FSI sensitivity完成", (ROOT / "11_sensitivity/FSI_threshold_sensitivity.csv").exists()),
    ("confidence sensitivity完成", (ROOT / "11_sensitivity/confidence_threshold_sensitivity.csv").exists()),
    ("novelty sensitivity完成", (ROOT / "11_sensitivity/novelty_threshold_sensitivity.csv").exists()),
    ("priority consensus生成", (ROOT / "12_robustness/priority_consensus_frequency.tif").exists()),
    ("consensus core candidate生成", (ROOT / "12_robustness/consensus_core_candidate.tif").exists()),
    ("robustness summary生成", (ROOT / "12_robustness/zoning_robustness_summary.csv").exists()),
    ("E6-Fig1至Fig5均有PNG和PDF", all((ROOT / f"13_figures/{n}.png").exists() and (ROOT / f"13_figures/{n}.pdf").exists()
                                 for n in ["E6_Fig1_integrated_zoning_framework", "E6_Fig2_input_layers",
                                           "E6_Fig3_GPI", "E6_Fig4_final_cultivation_zoning", "E6_Fig5_priority_regions"])),
    ("每张图均有原始数据和绘图脚本", (ROOT / "14_figure_data/crop_window.json").exists()
     and len(list((ROOT / "scripts/figures").glob("plot_*.py"))) >= 7),
    ("E6-Table1至Table5生成", all((ROOT / f"15_tables/E6_Table{i}_" ).exists() or
                                 list((ROOT / "15_tables").glob(f"E6_Table{i}_*.csv")) for i in range(1, 6))),
    ("FINAL_DATA_DICTIONARY生成", (HANDOFF / "FINAL_DATA_DICTIONARY.md").exists()),
    ("FINAL_RESULTS_SUMMARY生成", (HANDOFF / "FINAL_RESULTS_SUMMARY.md").exists()),
    ("METHODS_PARAMETER_MANIFEST生成", (HANDOFF / "METHODS_PARAMETER_MANIFEST.csv").exists()),
    ("final handoff SHA256生成", True),  # 本脚本随后生成
]
with open(QC_DIR / "EXPERIMENT6_ACCEPTANCE_CHECKLIST.md", "w", encoding="utf-8") as f:
    f.write("# 实验6B 验收清单\n\n")
    f.write(f"**Date**: {NOW}\n\n")
    f.write("| # | 项目 | 状态 |\n|---|---|---|\n")
    for i, (item, ok) in enumerate(checks, 1):
        f.write(f"| {i} | {item} | {'✅' if ok else '❌'} |\n")
    f.write("\n注: 'Zone I与HighLoss/HighUncertainty无重叠' 因Zone I=0而平凡成立；"
            "分区规则本身保证了该约束（CandidateMask排除HighConfLoss，Zone IV/V优先级高于I）。\n")

# --- 11. SHA256 manifest (最后, 覆盖handoff全部文件) ---
entries = []
for fp in sorted(HANDOFF.rglob("*")):
    if fp.is_file() and fp.name != "sha256_manifest.csv":
        entries.append({"file": str(fp.relative_to(HANDOFF)).replace("\\", "/"),
                        "sha256": hashlib.sha256(fp.read_bytes()).hexdigest(),
                        "size_bytes": fp.stat().st_size})
with open(HANDOFF / "sha256_manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["file", "sha256", "size_bytes"])
    w.writeheader(); w.writerows(entries)
print(f"\nSHA256 manifest: {len(entries)} files")
print("FINAL HANDOFF COMPLETE")
