#!/usr/bin/env python3
"""E6-Fig3: GPI主图 (167像元方块符号)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

crop = load_crop()
gpi, _ = load_raster(FIG_DATA / "E6_Fig3_GPI_main.tif", crop)

fig, ax = plt.subplots(figsize=(10, 7))
lons, lats = pixel_centers(gpi > 0, crop)
vals = gpi[gpi > 0]
vmax = float(vals.max()) if len(vals) else 1.0
sc = ax.scatter(lons, lats, c=vals, cmap="RdYlGn", vmin=0, vmax=vmax,
                marker="s", s=55, edgecolors="black", linewidths=0.4, zorder=4)
setup_map(ax, crop, "E6-Fig3  Integrated Priority Index (GPI, main weights)")
plt.colorbar(sc, ax=ax, shrink=0.8, label="GPI")
ax.text(0.02, 0.02, f"max GPI = {vmax:.3f} (single pixel with FSI=1/36);\nmedian GPI ≈ {float(np.median(vals)):.2e}\n"
        "F≈0 dominates the geometric index → GPI≈0 on the entire validated habitat",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))
fig.tight_layout()
save(fig, "E6_Fig3_GPI")
