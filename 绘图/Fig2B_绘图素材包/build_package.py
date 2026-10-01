#!/usr/bin/env python3
"""
Build Fig2B plotting material package.
Extracts per-variable response curve data, thresholds, favorable ranges,
and rug/support data for the 5 Strong drivers: bio02, bio03, bio05, bio15, clay.
"""
import os
import numpy as np
import pandas as pd

EXP2 = r"E:\人参种在哪\实验2"
PKG = r"E:\人参种在哪\绘图\Fig2B_绘图素材包"
DATA_DIR = os.path.join(PKG, "data")
os.makedirs(DATA_DIR, exist_ok=True)

STRONG_DRIVERS = ["bio03", "bio15", "bio02", "bio05", "clay"]

UNITS = {
    "bio02": "°C",
    "bio03": "%",
    "bio05": "°C",
    "bio15": "CV",
    "clay": "%",
}

NAMES = {
    "bio02": "Mean diurnal range (BIO02)",
    "bio03": "Isothermality (BIO03)",
    "bio05": "Max temperature of warmest month (BIO05)",
    "bio15": "Precipitation seasonality (BIO15)",
    "clay": "Soil clay content (0–20 cm)",
}

# ---- 1. Response curve data (per variable) ----
ale = pd.read_csv(os.path.join(EXP2, "05_ale", "ensemble_consensus_ALE.csv"))
ale_full = pd.read_csv(os.path.join(EXP2, "05_ale", "ale_1d_all_models.csv"))

# P5/P95 per variable computed from raster sample (actual data support)
rs_path = os.path.join(EXP2, "02_analysis_dataset", "raster_explainability_sample.csv")
rs = pd.read_csv(rs_path)
pct = pd.DataFrame({
    "variable": STRONG_DRIVERS,
    "P5": [np.percentile(rs[v].dropna(), 5) for v in STRONG_DRIVERS],
    "P95": [np.percentile(rs[v].dropna(), 95) for v in STRONG_DRIVERS],
}).set_index("variable")

curve_frames = []
for var in STRONG_DRIVERS:
    v = ale[ale["variable"] == var].copy().sort_values("bin_center")
    v = v.rename(columns={"ensemble_ALE_z": "ensemble_response"})
    v["P5"] = pct.loc[var, "P5"]
    v["P95"] = pct.loc[var, "P95"]
    # Mark core vs tail region based on P5–P95
    v["core_region"] = np.where(
        (v["bin_center"] >= v["P5"]) & (v["bin_center"] <= v["P95"]),
        "core", "tail",
    )
    v = v[["variable", "bin_center", "ensemble_response", "ale_lower95",
           "ale_upper95", "n_support", "P5", "P95", "core_region"]]
    curve_frames.append(v)
    v.to_csv(os.path.join(DATA_DIR, f"response_curve_{var}.csv"), index=False)

all_curves = pd.concat(curve_frames, ignore_index=True)
all_curves.to_csv(os.path.join(DATA_DIR, "response_curves_all.csv"), index=False)

# ---- 2. Thresholds ----
th = pd.read_csv(os.path.join(EXP2, "06_thresholds", "threshold_consensus_summary.csv"))
th = th[th["variable"].isin(STRONG_DRIVERS)].copy()
th_out = th[["variable", "median_threshold", "transition_interval_low",
             "transition_interval_high", "lower95", "upper95",
             "model_consensus_n", "support_percent", "status"]].copy()
th_out["unit"] = th_out["variable"].map(UNITS)
th_out["variable_name"] = th_out["variable"].map(NAMES)
th_out.to_csv(os.path.join(DATA_DIR, "thresholds.csv"), index=False)

# ---- 3. Favorable ranges ----
fav = pd.read_csv(os.path.join(EXP2, "06_thresholds", "favorable_environment_ranges.csv"))
fav = fav[fav["variable"].isin(STRONG_DRIVERS)].copy()
fav["unit"] = fav["variable"].map(UNITS)
fav["variable_name"] = fav["variable"].map(NAMES)
fav.to_csv(os.path.join(DATA_DIR, "favorable_ranges.csv"), index=False)

# ---- 4. Rug / data-support values (sampled predictor values) ----
rng = np.random.default_rng(20260807)
for var in STRONG_DRIVERS:
    vals = rs[var].dropna().values
    n = min(1000, len(vals))
    sample = rng.choice(vals, size=n, replace=False)
    rug = pd.DataFrame({
        "variable": var,
        "value": np.sort(sample),
    })
    rug.to_csv(os.path.join(DATA_DIR, f"rug_values_{var}.csv"), index=False)

# ---- 5. Variable metadata ----
meta = pd.DataFrame({
    "variable": STRONG_DRIVERS,
    "variable_name": [NAMES[v] for v in STRONG_DRIVERS],
    "unit": [UNITS[v] for v in STRONG_DRIVERS],
    "panel_order": range(1, len(STRONG_DRIVERS) + 1),
})
meta.to_csv(os.path.join(DATA_DIR, "variable_metadata.csv"), index=False)

print("Package files written:")
for f in sorted(os.listdir(DATA_DIR)):
    print(" -", f)
print("\nThresholds check:")
print(th_out[["variable", "median_threshold", "status"]].to_string(index=False))
