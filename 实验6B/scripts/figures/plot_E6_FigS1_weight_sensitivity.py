#!/usr/bin/env python3
"""E6-FigS1: 权重方案敏感性 (Zone I/II/III 面积跨方案)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import *
import pandas as pd

ws = pd.read_csv(ROOT / "11_sensitivity/weight_scheme_sensitivity.csv")
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(ws))
w = 0.25
ax.bar(x - w, ws["ZoneI"], w, label="Zone I", color="#1a9641")
ax.bar(x, ws["ZoneII"], w, label="Zone II", color="#a6d96a")
ax.bar(x + w, ws["ZoneIII"], w, label="Zone III", color="#fdae61")
ax.set_xticks(x)
ax.set_xticklabels(ws["weight_scheme"], rotation=15, ha="right")
ax.set_ylabel("Pixels")
ax.set_title("E6-FigS1  Candidate zone area across weight schemes", fontsize=12, fontweight="bold")
ax.legend()
ax.text(0.5, 0.75, "Zone I/II/III = 0 under ALL weight schemes\n"
        "(candidate pool requires FSI>=0.50; max FSI = 1/36 ≈ 0.028)\n"
        "→ absence of candidate zones is robust to weight choice",
        transform=ax.transAxes, ha="center", fontsize=9.5,
        bbox=dict(facecolor="#fff9c4", edgecolor="gray"))
fig.tight_layout()
save(fig, "E6_FigS1_weight_sensitivity")
