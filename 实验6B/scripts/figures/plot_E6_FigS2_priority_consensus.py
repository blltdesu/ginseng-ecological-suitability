#!/usr/bin/env python3
"""E6-FigS2: 候选区共识频率 (修正版数据下全域为0)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

crop = load_crop()
pcf, _ = load_raster(ROOT / "12_robustness/priority_consensus_frequency.tif", crop)

fig, ax = plt.subplots(figsize=(10, 7))
# 背景: 适生域位置
with rasterio.open(ROOT / "00_input_from_experiment5/current/current_binary_suitability.tif") as src:
    cb = src.read(1)[crop["row0"]:crop["row1"], crop["col0"]:crop["col1"]]
lons, lats = pixel_centers(cb == 1, crop)
ax.scatter(lons, lats, c="0.75", marker="s", s=55, edgecolors="black", linewidths=0.4, zorder=4,
           label="Validated habitat (consensus=0)")
nz = pcf > 0
if nz.any():
    lons2, lats2 = pixel_centers(nz, crop)
    sc = ax.scatter(lons2, lats2, c=pcf[nz], cmap="RdYlBu", vmin=0, vmax=1,
                    marker="s", s=60, edgecolors="black", linewidths=0.5, zorder=5)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Consensus frequency")
setup_map(ax, crop, "E6-FigS2  Priority consensus frequency (4 weight schemes)")
ax.text(0.5, 0.5, "No pixel reaches Zone I/II in any of the 4 weight schemes\n(consensus frequency = 0 on the entire validated habitat)",
        transform=ax.transAxes, ha="center", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
save(fig, "E6_FigS2_priority_consensus")
