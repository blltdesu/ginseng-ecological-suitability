#!/usr/bin/env python3
"""E6-Fig1: 综合框架图 (示意流程)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *

fig, ax = plt.subplots(figsize=(11, 7.5))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)

boxes = [
    (5, 9.3, "Upstream evidence (Experiments 1-5, CORRECTED handoff)", "#eceff1", 4.2),
    (1.6, 7.6, "S\nCurrent\nSuitability\nw=0.25", "#c8e6c9", 1.55),
    (3.35, 7.6, "F\nFuture\nStability\nw=0.35", "#a5d6a7", 1.55),
    (5.1, 7.6, "C\nPrediction\nConfidence\nw=0.15", "#90caf9", 1.55),
    (6.85, 7.6, "N\nNovelty\nReliability\nw=0.10", "#ce93d8", 1.55),
    (8.6, 7.6, "L\nLand\nAvailability\nw=0.15", "#ffcc80", 1.55),
    (5, 5.6, "Candidate Mask\nCurrent binary AND L>0 AND NOT high-conf loss AND FSI>=0.50\n(corrected data: FSI<=1/36 -> pool EMPTY, n=0)", "#fff9c4", 6.5),
    (5, 4.1, "GPI = S^0.25 F^0.35 C^0.15 N^0.10 L^0.15\n(geometric weighted; 4 weight schemes + WSI sensitivity)", "#ffe0b2", 6.5),
    (5, 2.5, "Final 6-zone classification (priority: IV>V>I>II>III>VI)\nI Core | II General | III Conditional\nIV Climate-vulnerable | V High-uncertainty | VI Non-priority", "#ef9a9a", 6.5),
    (5, 0.9, "Long-term stable cultivation candidate zones\n(corrected data: NO Zone I/II/III identified;\nentire validated habitat = climate-vulnerable / high-uncertainty / non-priority)", "#d1c4e9", 6.5),
]
for x, y, text, color, wdt in boxes:
    ax.add_patch(plt.Rectangle((x - wdt / 2, y - 0.62), wdt, 1.24, facecolor=color,
                               edgecolor="black", linewidth=1.2, alpha=0.9, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=8, zorder=3)
for y0, y1 in [(8.65, 8.25), (6.95, 6.25), (4.95, 4.75), (3.45, 3.15), (1.85, 1.55)]:
    ax.annotate("", xy=(5, y1), xytext=(5, y0), arrowprops=dict(arrowstyle="->", lw=2, color="gray"))
for xi in [1.6, 3.35, 5.1, 6.85, 8.6]:
    ax.annotate("", xy=(5, 6.25), xytext=(xi, 6.95), arrowprops=dict(arrowstyle="->", lw=1, color="gray", alpha=0.6))

ax.set_title("E6-Fig1  Integrated zoning framework for long-term stable\ncultivation candidate identification (Experiment 6B, corrected data)",
             fontsize=12, fontweight="bold")
save(fig, "E6_Fig1_integrated_zoning_framework")
