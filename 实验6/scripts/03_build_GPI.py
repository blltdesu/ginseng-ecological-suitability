#!/usr/bin/env python3
"""
实验6 - Step 5-8: 候选池掩膜 + GPI主指数 + 替代权重 + 加权和敏感性
"""
import os, sys, json, csv
import numpy as np
import rasterio
from pathlib import Path

ROOT = Path(r"E:\人参种在哪\实验6")
INPUT_DIR = ROOT / "00_input_from_experiment5"
STD_DIR = ROOT / "04_standardized_layers"
CAND_DIR = ROOT / "05_candidate_masks"
GPI_DIR = ROOT / "06_priority_index"
SENS_DIR = ROOT / "11_sensitivity"
os.makedirs(CAND_DIR, exist_ok=True)
os.makedirs(GPI_DIR, exist_ok=True)
os.makedirs(SENS_DIR, exist_ok=True)

# Load reference
with rasterio.open(STD_DIR / "final_valid_mask.tif") as ref:
    ref_meta = ref.meta.copy()
    ref_meta.update(dtype='float32', nodata=0.0, compress='lzw')
    ref_shape = (ref.height, ref.width)

# Load standardized layers
with rasterio.open(STD_DIR / "current_suitability_S.tif") as src:
    S = src.read(1).astype(np.float32)
with rasterio.open(STD_DIR / "future_stability_F.tif") as src:
    F = src.read(1).astype(np.float32)
with rasterio.open(STD_DIR / "prediction_confidence_C.tif") as src:
    C = src.read(1).astype(np.float32)
with rasterio.open(STD_DIR / "novelty_reliability_N.tif") as src:
    N = src.read(1).astype(np.float32)
with rasterio.open(STD_DIR / "land_availability_L.tif") as src:
    L = src.read(1).astype(np.float32)

# Load additional inputs for masks
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

# Fill NaNs
for arr in [cb, hcl, huz, nf, lfi]:
    arr[np.isnan(arr)] = 0.0

with open(INPUT_DIR / "current/ensemble_threshold.json") as f:
    threshold_data = json.load(f)
ensemble_threshold = threshold_data["ensemble_threshold"]

# Load experiment5 params for high-suitability threshold
with open(INPUT_DIR / "reference/experiment5_params.json") as f:
    exp5_params = json.load(f)

# High suitability threshold from Experiment 1: use ensemble_threshold-based
# Actually, for zone I, we need the "high-suitability threshold" from experiment 1
# Let's use the median of current suitable area as proxy
suitable_mask = (cb == 1)
s_vals_on_suitable = S[suitable_mask]
if len(s_vals_on_suitable) > 0:
    s_median_on_suitable = float(np.median(s_vals_on_suitable))
    s_p75_on_suitable = float(np.percentile(s_vals_on_suitable, 75))
else:
    s_median_on_suitable = ensemble_threshold
    s_p75_on_suitable = ensemble_threshold

print(f"High-suitability threshold (median of suitable): {s_median_on_suitable:.6f}")
print(f"S P75 on suitable: {s_p75_on_suitable:.6f}")

# --- Step 5: Candidate Mask ---
print("\n=== Building Candidate Mask ===")
candidate_mask = (
    (cb == 1)                    # CurrentBinarySuitable
    & (L > 0)                    # LandAvailability > 0
    & (hcl != 1)                 # NOT HighConfidenceLossZone
    & (F >= 0.50)                # FSI >= 0.50
).astype(np.uint8)

n_candidate = int(candidate_mask.sum())
print(f"Candidate pool pixels: {n_candidate}")

out_meta_uint8 = ref_meta.copy()
out_meta_uint8.update(dtype='uint8', nodata=0, compress='lzw')
with rasterio.open(CAND_DIR / "candidate_pool_mask.tif", 'w', **out_meta_uint8) as dst:
    dst.write(candidate_mask, 1)

# --- Step 6: GPI Main (Geometric Weighted) ---
print("\n=== Building GPI Main ===")
# Weights: S=0.25, F=0.35, C=0.15, N=0.10, L=0.15
wS, wF, wC, wN, wL = 0.25, 0.35, 0.15, 0.10, 0.15

# Geometric weighted index: GPI = S^wS * F^wF * C^wC * N^wN * L^wL
# To avoid log(0), add small epsilon where values are 0
eps = 1e-10
S_safe = np.maximum(S, eps)
F_safe = np.maximum(F, eps)
C_safe = np.maximum(C, eps)
N_safe = np.maximum(N, eps)
L_safe = np.maximum(L, eps)

GPI_main = (S_safe ** wS) * (F_safe ** wF) * (C_safe ** wC) * (N_safe ** wN) * (L_safe ** wL)

# Apply candidate mask: GPI only meaningful on candidate pixels
# But we save the full GPI and also report candidate-only stats
out_meta_f32 = ref_meta.copy()
out_meta_f32.update(dtype='float32', nodata=0.0, compress='lzw')
with rasterio.open(GPI_DIR / "GPI_main.tif", 'w', **out_meta_f32) as dst:
    dst.write(GPI_main.astype(np.float32), 1)

# Stats on candidate pixels
gpi_candidate = GPI_main[candidate_mask.astype(bool)]
if len(gpi_candidate) > 0:
    gpi_p25 = float(np.percentile(gpi_candidate, 25))
    gpi_p50 = float(np.percentile(gpi_candidate, 50))
    gpi_p75 = float(np.percentile(gpi_candidate, 75))
    gpi_p90 = float(np.percentile(gpi_candidate, 90))
    print(f"GPI on candidate: mean={gpi_candidate.mean():.6f}, median={gpi_p50:.6f}, P25={gpi_p25:.6f}, P75={gpi_p75:.6f}")
else:
    gpi_p25 = gpi_p50 = gpi_p75 = gpi_p90 = 0.0
    print("WARNING: No candidate pixels!")

# Save GPI formula
gpi_formula = {
    "type": "geometric_weighted_index",
    "formula": "GPI = S^wS * F^wF * C^wC * N^wN * L^wL",
    "weights": {"wS": wS, "wF": wF, "wC": wC, "wN": wN, "wL": wL},
    "note": "GPI只在候选池像元上有意义。权重为决策支持参数，非自然常数。"
}
with open(GPI_DIR / "GPI_formula.json", 'w') as f:
    json.dump(gpi_formula, f, indent=2)

# Save weight scheme
with open(GPI_DIR / "GPI_weight_scheme_main.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["indicator", "symbol", "weight", "rationale"])
    writer.writerow(["Current Suitability", "S", 0.25, "当前生态适宜性"])
    writer.writerow(["Future Stability", "F", 0.35, "未来气候稳定性(最高权重)"])
    writer.writerow(["Prediction Confidence", "C", 0.15, "预测可信度"])
    writer.writerow(["Novelty Reliability", "N", 0.10, "环境新颖度可靠性"])
    writer.writerow(["Land Availability", "L", 0.15, "土地可利用性"])

# --- Step 7: Alternative Weight Schemes ---
print("\n=== Building Alternative Weight Schemes ===")
alt_weights = {
    "stability_first": {"wS": 0.20, "wF": 0.45, "wC": 0.15, "wN": 0.10, "wL": 0.10},
    "balanced":        {"wS": 0.20, "wF": 0.20, "wC": 0.20, "wN": 0.20, "wL": 0.20},
    "suitability_first":{"wS": 0.40, "wF": 0.25, "wC": 0.10, "wN": 0.10, "wL": 0.15},
}

gpi_alts = {}
for name, w in alt_weights.items():
    gpi = (S_safe ** w["wS"]) * (F_safe ** w["wF"]) * (C_safe ** w["wC"]) * (N_safe ** w["wN"]) * (L_safe ** w["wL"])
    gpi_alts[name] = gpi
    out_path = GPI_DIR / f"GPI_{name}.tif"
    with rasterio.open(out_path, 'w', **out_meta_f32) as dst:
        dst.write(gpi.astype(np.float32), 1)
    vals = gpi[candidate_mask.astype(bool)]
    if len(vals) > 0:
        print(f"  {name}: GPI mean={vals.mean():.6f}, median={np.median(vals):.6f}")

# --- Step 8: Weighted Sum Index (WSI) for comparison ---
print("\n=== Building Weighted Sum Index ===")
WSI = wS*S + wF*F + wC*C + wN*N + wL*L
with rasterio.open(SENS_DIR / "weighted_sum_index.tif", 'w', **out_meta_f32) as dst:
    dst.write(WSI.astype(np.float32), 1)

# Compare GPI vs WSI
from scipy.stats import spearmanr
cand_bool = candidate_mask.astype(bool)
gpi_c = GPI_main[cand_bool]
wsi_c = WSI[cand_bool]
if len(gpi_c) > 10:
    rho, pval = spearmanr(gpi_c, wsi_c)
    # Top 10% overlap
    gpi_top10 = gpi_c >= np.percentile(gpi_c, 90)
    wsi_top10 = wsi_c >= np.percentile(wsi_c, 90)
    top10_overlap = np.sum(gpi_top10 & wsi_top10) / max(np.sum(gpi_top10), 1)
    # Top 25% overlap
    gpi_top25 = gpi_c >= np.percentile(gpi_c, 75)
    wsi_top25 = wsi_c >= np.percentile(wsi_c, 75)
    top25_overlap = np.sum(gpi_top25 & wsi_top25) / max(np.sum(gpi_top25), 1)
    print(f"  Spearman rho: {rho:.4f} (p={pval:.6f})")
    print(f"  Top 10% overlap: {top10_overlap:.4f}")
    print(f"  Top 25% overlap: {top25_overlap:.4f}")
else:
    rho, pval = 0, 1
    top10_overlap = 0
    top25_overlap = 0

# Save comparison
with open(SENS_DIR / "GPI_vs_WSI.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerow(["spearman_rho", rho])
    writer.writerow(["spearman_p", pval])
    writer.writerow(["top10%_overlap", top10_overlap])
    writer.writerow(["top25%_overlap", top25_overlap])

# Save key thresholds for zoning
thresholds = {
    "ensemble_threshold": ensemble_threshold,
    "s_median_on_suitable": s_median_on_suitable,
    "s_p75_on_suitable": s_p75_on_suitable,
    "gpi_p25": gpi_p25,
    "gpi_p50": gpi_p50,
    "gpi_p75": gpi_p75,
    "gpi_p90": gpi_p90,
    "n_candidate_pixels": n_candidate,
}
with open(GPI_DIR / "zoning_thresholds.json", 'w') as f:
    json.dump(thresholds, f, indent=2)

print("\nDone. Candidate mask, GPI main, alternative weights, and WSI built.")
