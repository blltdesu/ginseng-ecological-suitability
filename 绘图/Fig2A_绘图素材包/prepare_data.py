#!/usr/bin/env python3
"""Prepare Fig2A data package: consensus variable importance with group annotations."""
import os, pandas as pd

ROOT = r"E:\人参种在哪\实验2"
OUT_DIR = r"E:\人参种在哪\绘图\Fig2A_绘图素材包\data"
os.makedirs(OUT_DIR, exist_ok=True)

# Source data
df = pd.read_csv(os.path.join(ROOT, "03_global_importance", "ensemble_consensus_importance.csv"))

# Add group column
group_map = {
    "bio02": "Climate", "bio03": "Climate", "bio05": "Climate", "bio15": "Climate",
    "clay": "Soil", "sand": "Soil",
    "elevation": "Terrain", "northness": "Terrain",
}
df["group"] = df["variable"].map(group_map)

# Sort by importance descending
df = df.sort_values("importance_pct", ascending=False).reset_index(drop=True)

# Verify values match expected
print("Data verification:")
print(df[["variable", "importance_pct", "cumulative_pct", "group", "tier"]].to_string())

# Save clean plotting data
df.to_csv(os.path.join(OUT_DIR, "Fig2A_consensus_importance.csv"), index=False)

# Also save a simplified version with just plotting columns
plot_df = df[["variable", "group", "importance_pct", "cumulative_pct", "tier"]].copy()
plot_df.to_csv(os.path.join(OUT_DIR, "Fig2A_plot_data.csv"), index=False)

print(f"\nData saved to {OUT_DIR}")
print(f"Files:")
for f in os.listdir(OUT_DIR):
    print(f"  {f}")
