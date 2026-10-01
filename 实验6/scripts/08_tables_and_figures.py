#!/usr/bin/env python3
"""
实验6 - Step 25-27: 生成表格和图表
"""
import os, sys, json, csv
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

ROOT = Path(r"E:\人参种在哪\实验6")
TABLE_DIR = ROOT / "15_tables"
FIG_DIR = ROOT / "13_figures"
FIG_DATA_DIR = ROOT / "14_figure_data"
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(FIG_DATA_DIR, exist_ok=True)

# Set matplotlib for Chinese text
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# --- Load data ---
print("Loading data...")
with rasterio.open(ROOT / "04_standardized_layers/current_suitability_S.tif") as src:
    S = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "04_standardized_layers/future_stability_F.tif") as src:
    F = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "04_standardized_layers/prediction_confidence_C.tif") as src:
    C_arr = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "04_standardized_layers/novelty_reliability_N.tif") as src:
    N_arr = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "04_standardized_layers/land_availability_L.tif") as src:
    L_arr = src.read(1).astype(np.float32)
with rasterio.open(ROOT / "07_final_zoning/final_cultivation_zoning.tif") as src:
    zoning = src.read(1)
with rasterio.open(ROOT / "06_priority_index/GPI_main.tif") as src:
    GPI = src.read(1).astype(np.float32)

# --- E6-Table1: Indicator definition ---
print("\nGenerating Table 1: Indicator definitions...")
table1_path = TABLE_DIR / "E6_Table1_indicator_definition.csv"
with open(table1_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["indicator", "symbol", "range", "source", "description"])
    writer.writerow(["Current Suitability", "S", "0-1", "Experiment 1/4", "Current ensemble suitability from experiment 1"])
    writer.writerow(["Future Stability Index", "F", "0-1", "Experiment 5", "Fraction of future scenarios classified as stable"])
    writer.writerow(["Prediction Confidence", "C", "0-1", "Experiment 5", "1 - normalized scenario SD"])
    writer.writerow(["Novelty Reliability", "N", "0-1", "Experiment 5", "1 - NoveltyFrequency"])
    writer.writerow(["Land Availability", "L", "0-1-0.5-0", "Experiment 6", "Landcover-based cultivation policy factor"])

# --- E6-Table2: Zoning rules ---
table2_path = TABLE_DIR / "E6_Table2_zoning_rules.csv"
with open(table2_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["zone", "zone_name", "conditions"])
    writer.writerow(["1", "Core stable candidate",
        "CandidateMask=1; S>=high; F>=0.90; C>=0.75; NF<=0.10; L=1; GPI>=P75"])
    writer.writerow(["2", "General stable candidate",
        "CandidateMask=1; F>=0.75; C>=0.50; NF<0.50; GPI>=P50"])
    writer.writerow(["3", "Conditional candidate",
        "CandidateMask=1 not entering I/II; OR current_suitable with L=0.5"])
    writer.writerow(["4", "Climate-vulnerable",
        "CurrentBinary=1 AND (HighConfidenceLoss=1 OR LF>=0.75 AND C>=0.75)"])
    writer.writerow(["5", "High-uncertainty", "HighUncertaintyZone=1 AND not Zone IV"])
    writer.writerow(["6", "Non-priority", "All remaining pixels"])

# --- E6-Table3: Zoning area ---
table3_path = TABLE_DIR / "E6_Table3_zoning_area.csv"
zone_names = {1:"Core stable",2:"General stable",3:"Conditional",4:"Climate-vulnerable",5:"High-uncertainty",6:"Non-priority"}
with open(table3_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["zone", "zone_name", "n_pixels", "area_km2_approx"])
    for code in range(1, 7):
        n = int((zoning == code).sum())
        area = n * (0.04 * 111.32) ** 2
        writer.writerow([code, zone_names[code], n, round(area, 2)])

# --- E6-Table4: ADM1 priority (placeholder if no data) ---
table4_path = TABLE_DIR / "E6_Table4_ADM1_priority.csv"
adm1_csv = ROOT / "10_admin_priority/ADM1_priority_statistics.csv"
if adm1_csv.exists():
    import shutil
    shutil.copy(adm1_csv, table4_path)
else:
    with open(table4_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["country", "ADM1", "core_area_km2", "general_stable_area_km2", "mean_GPI", "rank_A", "rank_B"])

# --- E6-Table5: Top core patches ---
table5_path = TABLE_DIR / "E6_Table5_top_core_patches.csv"
top_csv = ROOT / "10_admin_priority/top_core_candidate_patches.csv"
if top_csv.exists():
    import shutil
    shutil.copy(top_csv, table5_path)
else:
    with open(table5_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "patch_id", "zone", "area_km2", "mean_GPI", "mean_S", "mean_FSI"])

# ============== FIGURES ==============
print("\nGenerating figures...")

# E6-Fig1: Integrated zoning framework (schematic)
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))
ax1.axis('off')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)

# Draw flowchart boxes
boxes = [
    (5, 9, "Input Layers", "#e8f5e9"),
    (2, 7.5, "S Current\nSuitability", "#c8e6c9"),
    (3.5, 7.5, "F Future\nStability", "#a5d6a7"),
    (5, 7.5, "C Prediction\nConfidence", "#81c784"),
    (6.5, 7.5, "N Novelty\nReliability", "#66bb6a"),
    (8, 7.5, "L Land\nAvailability", "#4caf50"),
    (5, 5.5, "Candidate Mask\n(S_binary & L>0 & !Loss & FSI>=0.50)", "#fff9c4"),
    (5, 4, "GPI = S^wS * F^wF * C^wC * N^wN * L^wL", "#ffcc80"),
    (5, 2.5, "Final 6-Zone Classification\nI-Core | II-General | III-Conditional\nIV-Vulnerable | V-Uncertain | VI-Non-priority", "#ef9a9a"),
    (5, 1, "Long-term Stable Cultivation\nCandidate Zones", "#ce93d8"),
]

for x, y, text, color in boxes:
    ax1.add_patch(plt.Rectangle((x-1.5, y-0.6), 3, 1.2, facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8))
    ax1.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')

# Arrows (as text annotations)
for y_pos in [6.8, 4.6, 3.2, 1.8]:
    ax1.annotate('', xy=(5, y_pos+0.1), xytext=(5, y_pos-0.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

ax1.set_title('E6-Fig1: Integrated Zoning Framework for Long-term\nStable Cultivation Candidate Identification',
              fontsize=12, fontweight='bold', pad=20)
fig1.savefig(FIG_DIR / 'E6_Fig1_integrated_zoning_framework.png', dpi=300, bbox_inches='tight')
fig1.savefig(FIG_DIR / 'E6_Fig1_integrated_zoning_framework.pdf', bbox_inches='tight')
plt.close(fig1)

# Save figure data
with open(FIG_DATA_DIR / 'E6_Fig1_framework_nodes.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["node", "description"])
    writer.writerow(["S", "Current Suitability (from Experiment 1/4)"])
    writer.writerow(["F", "Future Stability Index (from Experiment 5)"])
    writer.writerow(["C", "Prediction Confidence (from Experiment 5)"])
    writer.writerow(["N", "Novelty Reliability (from Experiment 5)"])
    writer.writerow(["L", "Land Availability (from Experiment 6 landcover policy)"])
    writer.writerow(["CandidateMask", "S_binary AND L>0 AND NOT HighConfLoss AND FSI>=0.50"])
    writer.writerow(["GPI", "Geometric weighted index: S^0.25 * F^0.35 * C^0.15 * N^0.10 * L^0.15"])
    writer.writerow(["FinalZoning", "6-zone: Core/General/Conditional/Vulnerable/Uncertain/Non-priority"])

# E6-Fig2: Five input factors (spatial maps - simplified as summary statistics)
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
axes2 = axes2.flatten()
titles = ['S: Current Suitability', 'F: Future Stability Index', 'C: Prediction Confidence',
          'N: Novelty Reliability', 'L: Land Availability', 'Final Valid Mask']
arrays = [S, F, C_arr, N_arr, L_arr, (zoning > 0).astype(np.float32)]
cmaps = ['YlOrRd', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys']

for i, (ax, title, arr, cmap) in enumerate(zip(axes2, titles, arrays, cmaps)):
    # Take a subset for visualization (every 10th pixel)
    sub = arr[::20, ::20]
    im = ax.imshow(sub, cmap=cmap, aspect='auto')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)

fig2.suptitle('E6-Fig2: Five Input Indicators for Integrated Zoning', fontsize=14, fontweight='bold')
fig2.tight_layout()
fig2.savefig(FIG_DIR / 'E6_Fig2_input_layers.png', dpi=300, bbox_inches='tight')
fig2.savefig(FIG_DIR / 'E6_Fig2_input_layers.pdf', bbox_inches='tight')
plt.close(fig2)

# E6-Fig3: GPI spatial map
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
gpi_sub = GPI[::20, ::20]
gpi_viz = np.ma.masked_where(gpi_sub == 0, gpi_sub)
im3 = ax3.imshow(gpi_viz, cmap='RdYlGn', aspect='auto', vmin=0, vmax=gpi_viz.max() if gpi_viz.max() > 0 else 1)
ax3.set_title('E6-Fig3: Integrated Priority Index (GPI)', fontsize=14, fontweight='bold')
ax3.set_xticks([])
ax3.set_yticks([])
cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, label='GPI')
fig3.tight_layout()
fig3.savefig(FIG_DIR / 'E6_Fig3_GPI.png', dpi=300, bbox_inches='tight')
fig3.savefig(FIG_DIR / 'E6_Fig3_GPI.pdf', bbox_inches='tight')
plt.close(fig3)

# E6-Fig4: Final 6-zone classification map
fig4, ax4 = plt.subplots(1, 1, figsize=(12, 8))
zone_sub = zoning[::20, ::20]
colors_list = ['#1a9641', '#a6d96a', '#fdae61', '#d7191c', '#ffffbf', '#d9d9d9']
labels_list = ['I: Core', 'II: General', 'III: Cond.', 'IV: Vulner.', 'V: Uncertain', 'VI: Non-prio.']
cmap = mcolors.ListedColormap(colors_list)
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
im4 = ax4.imshow(zone_sub, cmap=cmap, norm=norm, aspect='auto')
cbar4 = plt.colorbar(im4, ax=ax4, ticks=[1, 2, 3, 4, 5, 6], shrink=0.8)
cbar4.ax.set_yticklabels(labels_list)
ax4.set_title('E6-Fig4: Final Cultivation Candidate Zoning', fontsize=14, fontweight='bold')
ax4.set_xticks([])
ax4.set_yticks([])
fig4.tight_layout()
fig4.savefig(FIG_DIR / 'E6_Fig4_final_cultivation_zoning.png', dpi=300, bbox_inches='tight')
fig4.savefig(FIG_DIR / 'E6_Fig4_final_cultivation_zoning.pdf', bbox_inches='tight')
plt.close(fig4)

# E6-Fig5: Priority regions and ADM1 (simplified scatter)
fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6))
# Show zone I+II as scatter of their locations
zone_i_ii_mask = (zoning == 1) | (zoning == 2)
rows_ii, cols_ii = np.where(zone_i_ii_mask)
if len(rows_ii) > 0:
    # Convert to lat/lon
    transform = rasterio.open(ROOT / "07_final_zoning/final_cultivation_zoning.tif").transform
    lons_ii = transform[0] + (cols_ii + 0.5) * transform[1]
    lats_ii = transform[5] + (rows_ii + 0.5) * transform[4]
    colors_scatter = ['#1a9641' if zoning[r, c] == 1 else '#a6d96a' for r, c in zip(rows_ii, cols_ii)]
    ax5.scatter(lons_ii, lats_ii, c=colors_scatter, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
ax5.set_title('E6-Fig5: Priority Candidate Patches (Zone I & II)', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(FIG_DIR / 'E6_Fig5_priority_regions.png', dpi=300, bbox_inches='tight')
fig5.savefig(FIG_DIR / 'E6_Fig5_priority_regions.pdf', bbox_inches='tight')
plt.close(fig5)

# Supplementary figures
# E6-FigS1: Weight sensitivity
figS1, axS1 = plt.subplots(1, 1, figsize=(10, 6))
weight_schemes = {
    "main": {"wS": 0.25, "wF": 0.35, "wC": 0.15, "wN": 0.10, "wL": 0.15},
    "stability_first": {"wS": 0.20, "wF": 0.45, "wC": 0.15, "wN": 0.10, "wL": 0.10},
    "balanced": {"wS": 0.20, "wF": 0.20, "wC": 0.20, "wN": 0.20, "wL": 0.20},
    "suitability_first": {"wS": 0.40, "wF": 0.25, "wC": 0.10, "wN": 0.10, "wL": 0.15},
}
# Compute zone I/II for each scheme
cand_bool_fig = ((S > 0.1284676) & (L_arr > 0) & (F >= 0.50)).astype(bool)
eps_fig = 1e-10
zone_i_by_scheme = []
zone_ii_by_scheme = []
for sname, sw in weight_schemes.items():
    gpi_s = (np.maximum(S, eps_fig)**sw["wS"]) * (np.maximum(F, eps_fig)**sw["wF"]) * \
            (np.maximum(C_arr, eps_fig)**sw["wC"]) * (np.maximum(N_arr, eps_fig)**sw["wN"]) * \
            (np.maximum(L_arr, eps_fig)**sw["wL"])
    if cand_bool_fig.sum() > 0:
        gp75 = np.percentile(gpi_s[cand_bool_fig], 75)
        gp50 = np.percentile(gpi_s[cand_bool_fig], 50)
    else:
        gp75 = gp50 = 0
    s_high_fig = np.median(S[(S > 0.1284676)])
    zi = int(((cand_bool_fig) & (S >= s_high_fig) & (F >= 0.90) & (C_arr >= 0.75) & ((1-N_arr) <= 0.10) & (L_arr == 1.0) & (gpi_s >= gp75)).sum())
    zii = int(((cand_bool_fig) & (F >= 0.75) & (C_arr >= 0.50) & ((1-N_arr) < 0.50) & (gpi_s >= gp50)).sum())
    zone_i_by_scheme.append(zi)
    zone_ii_by_scheme.append(zii)

scheme_names = list(weight_schemes.keys())
x = range(len(scheme_names))
axS1.bar([xi - 0.15 for xi in x], zone_i_by_scheme, 0.3, label='Zone I', color='#1a9641')
axS1.bar([xi + 0.15 for xi in x], zone_ii_by_scheme, 0.3, label='Zone II', color='#a6d96a')
axS1.set_xticks(x)
axS1.set_xticklabels(scheme_names, rotation=15, ha='right')
axS1.set_ylabel('Pixel count')
axS1.legend()
axS1.set_title('E6-FigS1: Zone I/II Area Across Weight Schemes', fontsize=12, fontweight='bold')
figS1.tight_layout()
figS1.savefig(FIG_DIR / 'E6_FigS1_weight_sensitivity.png', dpi=300, bbox_inches='tight')
figS1.savefig(FIG_DIR / 'E6_FigS1_weight_sensitivity.pdf', bbox_inches='tight')
plt.close(figS1)

# E6-FigS2: Priority consensus
figS2, axS2 = plt.subplots(1, 1, figsize=(12, 8))
try:
    with rasterio.open(ROOT / "12_robustness/priority_consensus_frequency.tif") as src:
        pcf = src.read(1)
    pcf_sub = pcf[::20, ::20]
    imS2 = axS2.imshow(pcf_sub, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
    axS2.set_title('E6-FigS2: Priority Consensus Frequency', fontsize=14, fontweight='bold')
    plt.colorbar(imS2, ax=axS2, shrink=0.8, label='Consensus Frequency')
except:
    axS2.text(0.5, 0.5, 'No consensus data', ha='center', va='center', transform=axS2.transAxes)
figS2.savefig(FIG_DIR / 'E6_FigS2_priority_consensus.png', dpi=300)
figS2.savefig(FIG_DIR / 'E6_FigS2_priority_consensus.pdf', bbox_inches='tight')
plt.close(figS2)

print("Tables and figures generated successfully.")
