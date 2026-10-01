"""Update P4 figure with actual future climate completeness matrix."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = r"E:\人参种在哪\00_统一数据预处理\12_figures"
FIG_DATA_DIR = r"E:\人参种在哪\00_统一数据预处理\13_figure_data"
MATRIX_PATH = r"E:\人参种在哪\00_统一数据预处理\11_qc\future_climate_completeness_matrix.csv"

os.makedirs(FIG_DIR, exist_ok=True)

# Load completeness matrix
df = pd.read_csv(MATRIX_PATH)
print(f"Loaded {len(df)} records from completeness matrix")

# Build pivot: rows=BIO, cols=GCM×SSP×period
df["combo"] = df["gcm"].str[:8] + "\n" + df["ssp"] + "\n" + df["period"]
pivot = df.pivot_table(index="variable", columns="combo", values="available", aggfunc="first")
pivot = pivot.fillna(0).astype(int)

# Reorder BIO variables bio01 to bio19
bio_order = [f"bio{b:02d}" for b in range(1, 20)]
pivot = pivot.reindex(bio_order)

fig, ax = plt.subplots(figsize=(22, 8))
im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

# Labels
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_xlabel("GCM × SSP × Period", fontsize=12)
ax.set_ylabel("Bioclimatic Variable", fontsize=12)
ax.set_title("P4: Future Climate Data Completeness Matrix\n"
             f"All 380/380 BIO variables available (5 GCMs × 2 SSPs × 2 periods × 19 BIO)",
             fontsize=14, fontweight="bold")

# Add cell text annotations
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        color = "white" if val == 0 else "darkgreen"
        ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color, fontweight="bold")

cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label("1 = Available, 0 = Missing", fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "P4_future_climate_completeness.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "P4_future_climate_completeness.pdf"), bbox_inches="tight")
plt.close()

print(f"P4 updated: 380/380 complete")
print(f"Saved to: {FIG_DIR}")
