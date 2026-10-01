#!/usr/bin/env python3
"""E6-Fig4: 最终六区分区图 (全文最终核心应用图; 167像元以方块符号绘制)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *
import matplotlib.patches as mpatches

crop = load_crop()
zoning, _ = load_raster(FIG_DATA / "E6_Fig4_final_zoning.tif", crop)
zoning = zoning.astype(np.int32)

colors = {1: "#1a9641", 2: "#a6d96a", 3: "#fdae61", 4: "#d7191c", 5: "#984ea3", 6: "#d9d9d9"}
labels = {1: "I Core stable", 2: "II General stable", 3: "III Conditional",
          4: "IV Climate-vulnerable", 5: "V High-uncertainty", 6: "VI Non-priority"}

fig, ax = plt.subplots(figsize=(10, 7.5))
for code in [6, 5, 4, 3, 2, 1]:  # 低优先级先画
    lons, lats = pixel_centers(zoning == code, crop)
    if len(lons):
        ax.scatter(lons, lats, c=colors[code], marker="s", s=60,
                   edgecolors="black", linewidths=0.5, zorder=4, label=None)
setup_map(ax, crop, "E6-Fig4  Final cultivation candidate zoning (Experiment 6B, corrected data)")

present = [i for i in range(1, 7) if (zoning == i).any()]
handles = [mpatches.Patch(color=colors[i], label=f"{labels[i]}  (n={int((zoning == i).sum())})")
           for i in present]
handles.append(mpatches.Patch(facecolor="white", edgecolor="gray",
                              label="I–III: none (candidate pool empty, FSI≤1/36)"))
ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.9)
fig.tight_layout()
save(fig, "E6_Fig4_final_cultivation_zoning")
