#!/usr/bin/env python3
"""
实验6B - Step 5-8: 候选池掩膜 + GPI主指数 + 替代权重 + 加权和(WSI)敏感性
修正版数据下 FSI≤1/36≈0.028 < 0.50 → 候选池预期为空; 脚本对此做了显式处理。
GPI地图仍在final_valid域上计算并保存(作为完整记录), 但候选统计为空时阈值置为NaN。
"""
import os, json, csv
import numpy as np
import rasterio
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(r"E:\人参种在哪\实验6B")
INPUT_DIR = ROOT / "00_input_from_experiment5"
STD_DIR = ROOT / "04_standardized_layers"
CAND_DIR = ROOT / "05_candidate_masks"
GPI_DIR = ROOT / "06_priority_index"
SENS_DIR = ROOT / "11_sensitivity"
for d in [CAND_DIR, GPI_DIR, SENS_DIR]:
    os.makedirs(d, exist_ok=True)

def load(p):
    with rasterio.open(p) as src:
        return src.read(1).astype(np.float32), src.meta.copy()

S, meta = load(STD_DIR / "current_suitability_S.tif")
F, _ = load(STD_DIR / "future_stability_F.tif")
C, _ = load(STD_DIR / "prediction_confidence_C.tif")
N, _ = load(STD_DIR / "novelty_reliability_N.tif")
L, _ = load(STD_DIR / "land_availability_L.tif")
final_valid, _ = load(STD_DIR / "final_valid_mask.tif")
final_valid = final_valid == 1
cb, _ = load(INPUT_DIR / "current/current_binary_suitability.tif")
hcl, _ = load(INPUT_DIR / "vulnerability/high_confidence_loss_zone.tif")

cb = np.nan_to_num(cb)
hcl = np.nan_to_num(hcl)

out_f32 = meta.copy(); out_f32.update(dtype="float32", nodata=0.0, compress="lzw")
out_u8 = meta.copy(); out_u8.update(dtype="uint8", nodata=0, compress="lzw")

with open(INPUT_DIR / "current/ensemble_threshold.json") as f:
    ensemble_threshold = json.load(f)["ensemble_threshold"]

# 高适生阈值(Zone I用): 实验1未单独交接high-suitability阈值, 用适生域S中位数作代理并记录
s_vals = S[cb == 1]
s_high = float(np.median(s_vals))
print(f"高适生阈值(S适生域中位数代理): {s_high:.6f} (ensemble二值阈值={ensemble_threshold:.6f})")

# --- Step 5: 候选池硬性掩膜 ---
print("\n=== Step 5: Candidate Mask ===")
candidate_mask = (
    (cb == 1)
    & (L > 0)
    & (hcl != 1)
    & (F >= 0.50)
).astype(np.uint8)
n_cand = int(candidate_mask.sum())
print(f"  候选池像元: {n_cand}")
print(f"  约束分解(167适生像元中):")
print(f"    L>0:            {int(((cb==1)&(L>0)).sum())}")
print(f"    非高置信丧失:    {int(((cb==1)&(hcl!=1)).sum())}")
print(f"    FSI>=0.50:      {int(((cb==1)&(F>=0.50)).sum())}  ← FSI最大值={F[cb==1].max():.4f}, 约束为空集")

with rasterio.open(CAND_DIR / "candidate_pool_mask.tif", "w", **out_u8) as dst:
    dst.write(candidate_mask, 1)

with open(CAND_DIR / "candidate_pool_report.json", "w", encoding="utf-8") as f:
    json.dump({
        "n_candidate_pixels": n_cand,
        "criteria": "CurrentBinary=1 AND LandAvailability>0 AND NOT HighConfidenceLoss AND FSI>=0.50",
        "binding_constraint": "FSI>=0.50 (max FSI on suitable domain = 1/36 = 0.0278)",
        "note": "修正版数据下候选池为空: 36个可用未来情景下仅2/6012个scenario-pixel保持适生。"
                "这不是管线错误——审计(02)已确认编码修正正确且与二值预测图完全一致。",
    }, f, ensure_ascii=False, indent=2)

# --- Step 6: GPI主方案 (几何加权) ---
print("\n=== Step 6: GPI Main ===")
W_MAIN = {"wS": 0.25, "wF": 0.35, "wC": 0.15, "wN": 0.10, "wL": 0.15}
eps = 1e-10
Ss, Fs, Cs, Ns, Ls = [np.maximum(a, eps) for a in [S, F, C, N, L]]

def compute_gpi(w):
    g = (Ss ** w["wS"]) * (Fs ** w["wF"]) * (Cs ** w["wC"]) * (Ns ** w["wN"]) * (Ls ** w["wL"])
    g[~final_valid] = 0.0
    return g.astype(np.float32)

GPI_main = compute_gpi(W_MAIN)
with rasterio.open(GPI_DIR / "GPI_main.tif", "w", **out_f32) as dst:
    dst.write(GPI_main, 1)

gpi_v = GPI_main[final_valid]
print(f"  GPI(final_valid域): mean={gpi_v.mean():.6f}, max={gpi_v.max():.6f}")
print(f"  注: F≈0 → 几何加权GPI≈0, 符合预期(未来稳定性为最高权重0.35)")

cand_bool = candidate_mask.astype(bool)
if cand_bool.sum() > 0:
    gpi_c = GPI_main[cand_bool]
    gpi_p25, gpi_p50, gpi_p75, gpi_p90 = [float(np.percentile(gpi_c, p)) for p in [25, 50, 75, 90]]
else:
    gpi_p25 = gpi_p50 = gpi_p75 = gpi_p90 = float("nan")
    print("  候选池为空 → GPI分位数阈值不可用(NaN), 分区脚本将相应处理")

with open(GPI_DIR / "GPI_formula.json", "w", encoding="utf-8") as f:
    json.dump({
        "type": "geometric_weighted_index",
        "formula": "GPI = S^wS * F^wF * C^wC * N^wN * L^wL",
        "weights": W_MAIN,
        "weights_sum": sum(W_MAIN.values()),
        "note": "权重为决策支持参数, 非自然常数。Vulnerability不进入GPI(已含S和C, 避免重复计权)。",
        "domain": "GPI在final_valid_mask内计算; 仅对候选池像元有排序意义(本次候选池为空)",
    }, f, ensure_ascii=False, indent=2)

with open(GPI_DIR / "GPI_weight_scheme_main.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["indicator", "symbol", "weight", "rationale"])
    w.writerow(["Current Suitability", "S", 0.25, "当前生态适宜性"])
    w.writerow(["Future Stability", "F", 0.35, "未来气候稳定性(最高权重)"])
    w.writerow(["Prediction Confidence", "C", 0.15, "预测可信度"])
    w.writerow(["Novelty Reliability", "N", 0.10, "环境新颖度可靠性"])
    w.writerow(["Land Availability", "L", 0.15, "土地可利用性"])

# --- Step 7: 替代权重 ---
print("\n=== Step 7: Alternative Weights ===")
ALT = {
    "stability_first": {"wS": 0.20, "wF": 0.45, "wC": 0.15, "wN": 0.10, "wL": 0.10},
    "balanced": {"wS": 0.20, "wF": 0.20, "wC": 0.20, "wN": 0.20, "wL": 0.20},
    "suitability_first": {"wS": 0.40, "wF": 0.25, "wC": 0.10, "wN": 0.10, "wL": 0.15},
}
for name, w in ALT.items():
    assert abs(sum(w.values()) - 1.0) < 1e-9
    g = compute_gpi(w)
    with rasterio.open(GPI_DIR / f"GPI_{name}.tif", "w", **out_f32) as dst:
        dst.write(g, 1)
    print(f"  GPI_{name}: mean(valid)={g[final_valid].mean():.6f}")

# --- Step 8: WSI加权和敏感性 ---
print("\n=== Step 8: Weighted Sum Index (WSI) ===")
WSI = (W_MAIN["wS"] * S + W_MAIN["wF"] * F + W_MAIN["wC"] * C + W_MAIN["wN"] * N + W_MAIN["wL"] * L)
WSI[~final_valid] = 0.0
with rasterio.open(SENS_DIR / "weighted_sum_index.tif", "w", **out_f32) as dst:
    dst.write(WSI.astype(np.float32), 1)

# GPI vs WSI 比较(在final_valid域上; 候选池为空时仍报告域上相关性供参考)
gv, wv = GPI_main[final_valid], WSI[final_valid]
if len(gv) > 10 and gv.std() > 0 and wv.std() > 0:
    rho, pval = spearmanr(gv, wv)
    g_top10 = gv >= np.percentile(gv, 90); w_top10 = wv >= np.percentile(wv, 90)
    g_top25 = gv >= np.percentile(gv, 75); w_top25 = wv >= np.percentile(wv, 75)
    top10 = float((g_top10 & w_top10).sum() / max(g_top10.sum(), 1))
    top25 = float((g_top25 & w_top25).sum() / max(g_top25.sum(), 1))
    domain = "final_valid_mask (n=167; candidate pool empty)"
else:
    rho, pval, top10, top25 = float("nan"), float("nan"), float("nan"), float("nan")
    domain = "undefined"

print(f"  比较域: {domain}")
print(f"  Spearman rho={rho}, Top10% overlap={top10}, Top25% overlap={top25}")

with open(SENS_DIR / "GPI_vs_WSI.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value", "domain"])
    w.writerow(["spearman_rho", rho, domain])
    w.writerow(["spearman_p", pval, domain])
    w.writerow(["top10pct_overlap", top10, domain])
    w.writerow(["top25pct_overlap", top25, domain])
    w.writerow(["note", "候选池为空, 比较在final_valid域(167适生像元)上进行", ""])
    w.writerow(["adm1_ranking_agreement", "N/A", "Zone I/II为空, 无ADM1排序可比较"])

with open(GPI_DIR / "zoning_thresholds.json", "w") as f:
    json.dump({
        "ensemble_threshold": ensemble_threshold,
        "s_high_threshold": s_high,
        "s_high_threshold_note": "实验1高适生阈值未交接, 用适生域S中位数代理",
        "gpi_p25": gpi_p25, "gpi_p50": gpi_p50, "gpi_p75": gpi_p75, "gpi_p90": gpi_p90,
        "n_candidate_pixels": n_cand,
    }, f, indent=2)

print("\nDone.")
