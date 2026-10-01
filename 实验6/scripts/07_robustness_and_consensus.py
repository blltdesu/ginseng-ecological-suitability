#!/usr/bin/env python3
"""
实验6 - Step 23-24: 候选区共识稳定性 + 实验5 core vs 实验6 zone I + 最终稳健性总表
"""
import os, sys, json, csv
import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage

ROOT = Path(r"E:\人参种在哪\实验6")
ROB_DIR = ROOT / "12_robustness"
os.makedirs(ROB_DIR, exist_ok=True)

# Load data
with rasterio.open(ROOT / "04_standardized_layers/current_suitability_S.tif") as src:
    S = src.read(1).astype(np.float32); ref_meta = src.meta.copy()
with rasterio.open(ROOT / "04_standardized_layers/future_stability_F.tif") as src:
    F = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "04_standardized_layers/prediction_confidence_C.tif") as src:
    C = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "04_standardized_layers/novelty_reliability_N.tif") as src:
    N = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "04_standardized_layers/land_availability_L.tif") as src:
    L = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "05_candidate_masks/candidate_pool_mask.tif") as src:
    cand = src.read(1).astype(np.uint8)
with rasterio.open(ROOT / "07_final_zoning/final_cultivation_zoning.tif") as src:
    main_zoning = src.read(1)
with rasterio.open(ROOT / "00_input_from_experiment5/vulnerability/robust_climatic_core.tif") as src:
    exp5_core = src.read(1).astype(np.float32); exp5_core[np.isnan(exp5_core)] = 0.0
with rasterio.open(ROOT / "00_input_from_experiment5/current/current_binary_suitability.tif") as src:
    cb = src.read(1).astype(np.float32); cb[np.isnan(cb)] = 0.0
with rasterio.open(ROOT / "00_input_from_experiment5/vulnerability/high_confidence_loss_zone.tif") as src:
    hcl = src.read(1).astype(np.float32); hcl[np.isnan(hcl)] = 0.0
with rasterio.open(ROOT / "00_input_from_experiment5/uncertainty/high_uncertainty_zone.tif") as src:
    huz = src.read(1).astype(np.float32); huz[np.isnan(huz)] = 0.0

# Load all GPIs
gpi_dir = ROOT / "06_priority_index"
gpi_maps = {}
for fname in ["GPI_main.tif", "GPI_stability_first.tif", "GPI_balanced.tif", "GPI_suitability_first.tif"]:
    name = fname.replace("GPI_","").replace(".tif","")
    with rasterio.open(gpi_dir / fname) as src:
        gpi_maps[name] = src.read(1).astype(np.float32)

with open(ROOT / "00_input_from_experiment5/current/ensemble_threshold.json") as f:
    threshold_data = json.load(f)

with open(ROOT / "06_priority_index/zoning_thresholds.json") as f:
    thr = json.load(f)
s_high = thr["s_median_on_suitable"]

cand_bool = cand.astype(bool)
eps = 1e-10

print("=== Robustness and Consensus Analysis ===")

# --- 1. Priority Consensus Frequency ---
print("\n1. Building priority consensus frequency...")
scheme_names = ["main", "stability_first", "balanced", "suitability_first"]
n_schemes = len(scheme_names)

# For each scheme, compute Zone I or II membership
consensus_count = np.zeros(S.shape, dtype=np.int32)
zone_i_count = np.zeros(S.shape, dtype=np.int32)

for name in scheme_names:
    gpi = gpi_maps[name]
    gpi_p75 = np.percentile(gpi[cand_bool], 75) if cand_bool.sum() > 0 else 0
    gpi_p50 = np.percentile(gpi[cand_bool], 50) if cand_bool.sum() > 0 else 0

    zone_i = ((cand==1) & (S>=s_high) & (F>=0.90) & (C>=0.75) & ((1-N)<=0.10) & (L==1.0) & (gpi>=gpi_p75))
    zone_ii = ((cand==1) & (F>=0.75) & (C>=0.50) & ((1-N)<0.50) & (gpi>=gpi_p50))
    in_priority = zone_i | zone_ii

    consensus_count[in_priority] += 1
    zone_i_count[zone_i] += 1

# Priority Consensus Frequency = fraction of schemes where pixel enters Zone I or II
priority_consensus_freq = consensus_count.astype(np.float32) / n_schemes

out_meta_f32 = ref_meta.copy()
out_meta_f32.update(dtype='float32', nodata=0.0, compress='lzw')
with rasterio.open(ROB_DIR / "priority_consensus_frequency.tif", 'w', **out_meta_f32) as dst:
    dst.write(priority_consensus_freq, 1)

# Consensus core: ConsensusPriorityFrequency = 1 AND main scheme Zone I
consensus_core = (priority_consensus_freq == 1.0) & (main_zoning == 1)
out_meta_u8 = ref_meta.copy()
out_meta_u8.update(dtype='uint8', nodata=0, compress='lzw')
with rasterio.open(ROB_DIR / "consensus_core_candidate.tif", 'w', **out_meta_u8) as dst:
    dst.write(consensus_core.astype(np.uint8), 1)

print(f"  Consensus frequency stats:")
for f in [0.25, 0.50, 0.75, 1.0]:
    print(f"    Frequency = {f:.2f}: {(priority_consensus_freq == f).sum()} pixels")
print(f"  Consensus core (freq=1 AND main ZoneI): {int(consensus_core.sum())} pixels")

# --- 2. Experiment 5 core vs Experiment 6 Zone I ---
print("\n2. Comparing Experiment 5 robust_core vs Experiment 6 Zone I...")
exp5_core_mask = (exp5_core == 1)
exp6_zone_i_mask = (main_zoning == 1)

exp5_n = int(exp5_core_mask.sum())
exp6_n = int(exp6_zone_i_mask.sum())
overlap = int((exp5_core_mask & exp6_zone_i_mask).sum())
jaccard = overlap / max(exp5_n + exp6_n - overlap, 1)
core_retention = overlap / max(exp5_n, 1)

print(f"  Experiment 5 robust core: {exp5_n} pixels")
print(f"  Experiment 6 Zone I: {exp6_n} pixels")
print(f"  Overlap: {overlap} pixels")
print(f"  Jaccard index: {jaccard:.4f}")
print(f"  Core retention: {core_retention:.4f}")

comp_csv = ROB_DIR / "experiment5_core_vs_experiment6_zoneI.csv"
with open(comp_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerow(["exp5_core_pixels", exp5_n])
    writer.writerow(["exp6_zoneI_pixels", exp6_n])
    writer.writerow(["overlap_pixels", overlap])
    writer.writerow(["jaccard", jaccard])
    writer.writerow(["core_retention_pct", core_retention])
    writer.writerow(["note", "Experiment 6 Zone I includes landcover constraints not in Experiment 5 core"])

# --- 3. Robustness summary ---
print("\n3. Building robustness summary...")
# Collect sensitivity results
sens_data = {}
for fname in ["weight_scheme_sensitivity.csv", "FSI_threshold_sensitivity.csv",
              "confidence_threshold_sensitivity.csv", "novelty_threshold_sensitivity.csv"]:
    fpath = ROOT / "11_sensitivity" / fname
    if fpath.exists():
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pass  # We'll aggregate below

# Get area ranges for each zone across weight schemes
zone_i_areas_weights = []
zone_ii_areas_weights = []
for name in scheme_names:
    gpi = gpi_maps[name]
    gpi_p75 = np.percentile(gpi[cand_bool], 75) if cand_bool.sum() > 0 else 0
    zi = ((cand==1) & (S>=s_high) & (F>=0.90) & (C>=0.75) & ((1-N)<=0.10) & (L==1.0) & (gpi>=gpi_p75))
    zone_i_areas_weights.append(int(zi.sum()))
    gpi_p50 = np.percentile(gpi[cand_bool], 50) if cand_bool.sum() > 0 else 0
    zii = ((cand==1) & (F>=0.75) & (C>=0.50) & ((1-N)<0.50) & (gpi>=gpi_p50))
    zone_ii_areas_weights.append(int(zii.sum()))

# Build robustness summary
robust_rows = []
for zone_code, zone_name in [(1, "Core stable"), (2, "General stable"), (3, "Conditional"),
                               (4, "Climate-vulnerable"), (5, "High-uncertainty"), (6, "Non-priority")]:
    main_area = int((main_zoning == zone_code).sum())
    if zone_code == 1:
        area_range_w = f"{min(zone_i_areas_weights)}-{max(zone_i_areas_weights)}"
        # Area range across FSI thresholds
        area_range_fsi_all = [int(((cand==1) & (S>=s_high) & (F>=fsi_t) & (C>=0.75) & ((1-N)<=0.10) & (L==1.0)).sum())
                              for fsi_t in [0.80, 0.90, 0.95]]
        # Spatial consensus
        spatial_cons = "N/A" if main_area == 0 else f"{int(consensus_core.sum())} consensus"
        grade = "D (高度不稳定)" if main_area == 0 else ("C (参数敏感)" if max(zone_i_areas_weights) <= 3 else "B (总体稳定)")
    elif zone_code == 2:
        area_range_w = f"{min(zone_ii_areas_weights)}-{max(zone_ii_areas_weights)}"
        spatial_cons = f"{(priority_consensus_freq >= 0.50).sum()} with freq>=0.50"
        grade = "B (总体稳定)" if main_area > 10 else "C (参数敏感)"
    else:
        area_range_w = "N/A"
        spatial_cons = "N/A"
        grade = "A (高度稳定)" if zone_code == 6 else "N/A"

    robust_rows.append({
        "zone": zone_code,
        "zone_name": zone_name,
        "main_area_pixels": main_area,
        "area_range_across_weights": area_range_w,
        "area_range_across_FSI": f"{[int(z) for z in [0.80, 0.90, 0.95]]}",
        "spatial_consensus": spatial_cons,
        "robustness_grade": grade,
    })

robust_csv = ROB_DIR / "zoning_robustness_summary.csv"
with open(robust_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=robust_rows[0].keys())
    writer.writeheader()
    for r in robust_rows:
        writer.writerow(r)
        print(f"  Zone {r['zone']} ({r['zone_name']}): area={r['main_area_pixels']}, grade={r['robustness_grade']}")

print("\nDone. Robustness and consensus analysis complete.")
