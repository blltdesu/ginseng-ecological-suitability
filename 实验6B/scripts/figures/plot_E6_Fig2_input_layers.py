#!/usr/bin/env python3
"""E6-Fig2: 五个输入因子空间图 (S/F/C/N/L, 167像元以方块符号绘制)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

crop = load_crop()
layers = [
    ("E6_Fig2_S_current_suitability.tif", "S  Current suitability", "YlOrRd", 0.10, 0.25),
    ("E6_Fig2_F_future_stability.tif", "F  Future stability index", "Blues", 0, 0.03),
    ("E6_Fig2_C_prediction_confidence.tif", "C  Prediction confidence", "Greens", 0, 1),
    ("E6_Fig2_N_novelty_reliability.tif", "N  Novelty reliability", "Purples", 0, 1),
    ("E6_Fig2_L_land_availability.tif", "L  Land availability", "Oranges", 0, 1),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (fname, title, cmap, vmin, vmax) in zip(axes.flat, layers):
    d, _ = load_raster(FIG_DATA / fname, crop)
    lons, lats = pixel_centers(d > 0, crop)
    vals = d[d > 0]
    sc = ax.scatter(lons, lats, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                    marker="s", s=55, edgecolors="black", linewidths=0.4, zorder=4)
    setup_map(ax, crop, title)
    plt.colorbar(sc, ax=ax, shrink=0.8)

ax = axes.flat[5]
with rasterio.open(ROOT / "00_input_from_experiment5/current/current_binary_suitability.tif") as src:
    cb = src.read(1)[crop["row0"]:crop["row1"], crop["col0"]:crop["col1"]]
lons, lats = pixel_centers(cb == 1, crop)
ax.scatter(lons, lats, c="#d7191c", marker="s", s=55, edgecolors="black", linewidths=0.4, zorder=4)
setup_map(ax, crop, "Validated suitable domain (167 px)")
ax.text(0.02, 0.02, "Corrected data: FSI=0 on all 167 px;\nN=1 (novelty freq=0) everywhere;\nL: 6 px cropland, 151 px conditional, 10 px excluded",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

fig.suptitle("E6-Fig2  Five standardized input indicators on the validated suitable domain (corrected handoff)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "E6_Fig2_input_layers")
