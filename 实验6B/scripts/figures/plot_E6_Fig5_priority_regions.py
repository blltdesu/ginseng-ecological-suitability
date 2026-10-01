#!/usr/bin/env python3
"""E6-Fig5: 优先区域与ADM1统计 (修正版数据下无Zone I/II斑块, 展示ADM1脆弱区分布)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *
import pandas as pd

crop = load_crop()
zoning, _ = load_raster(FIG_DATA / "E6_Fig4_final_zoning.tif", crop)
zoning = zoning.astype(np.int32)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左: Zone IV/V/VI 空间分布 (无优先斑块时的信息承载图)
for code, color, z in [(6, "#d9d9d9", 4), (5, "#984ea3", 5), (4, "#d7191c", 6)]:
    lons, lats = pixel_centers(zoning == code, crop)
    if len(lons):
        ax1.scatter(lons, lats, c=color, marker="s", s=55, edgecolors="black", linewidths=0.4, zorder=z)
setup_map(ax1, crop, "Validated habitat: climate-vulnerable (red) /\nhigh-uncertainty (purple) / non-priority (grey)")
ax1.text(0.02, 0.02, "No Zone I/II priority patches exist under corrected data;\nmap shows where the validated habitat is climate-vulnerable.",
         transform=ax1.transAxes, fontsize=8.5, va="bottom",
         bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

# 右: ADM1 脆弱面积条形图
adm1 = pd.read_csv(FIG_DATA / "E6_Fig5_ADM1_priority.csv")
adm1 = adm1.sort_values("vulnerable_area_km2", ascending=True)
labels_adm = (adm1["ADM1"].astype(str) + " (" + adm1["country"].astype(str) + ")")
ax2.barh(labels_adm, adm1["vulnerable_area_km2"], color="#d7191c", alpha=0.85, label="Climate-vulnerable")
ax2.barh(labels_adm, adm1["high_uncertainty_area_km2"], left=adm1["vulnerable_area_km2"],
         color="#984ea3", alpha=0.85, label="High-uncertainty")
ax2.barh(labels_adm, adm1["non_priority_area_km2"],
         left=adm1["vulnerable_area_km2"] + adm1["high_uncertainty_area_km2"],
         color="#d9d9d9", alpha=0.9, label="Non-priority")
ax2.set_xlabel("Area (km²)")
ax2.set_title("Validated suitable habitat by ADM1", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.tick_params(axis="y", labelsize=8)

fig.suptitle("E6-Fig5  Administrative distribution of the validated habitat (no priority patches identified)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "E6_Fig5_priority_regions")
