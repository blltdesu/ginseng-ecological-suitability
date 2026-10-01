#!/usr/bin/env python3
"""
Step 2: Build predictor group mapping table.
Generates predictor_group_mapping.csv with variable, group, meaning, unit, source.
"""
import sys, os, logging
import pandas as pd

ROOT = r"E:\人参种在哪\实验2"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
OUT_DIR = os.path.join(ROOT, "02_analysis_dataset")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "experiment2_master.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Known variable metadata - sourced from Experiment 1 DATA_DICTIONARY
VARIABLE_META = {
    "bio02": {"group": "climate", "meaning": "Mean Diurnal Range (BIO02)", "unit": "°C", "source": "WorldClim 2.1"},
    "bio03": {"group": "climate", "meaning": "Isothermality (BIO03)", "unit": "%", "source": "WorldClim 2.1"},
    "bio05": {"group": "climate", "meaning": "Max Temperature of Warmest Month (BIO05)", "unit": "°C", "source": "WorldClim 2.1"},
    "bio15": {"group": "climate", "meaning": "Precipitation Seasonality (BIO15)", "unit": "CV", "source": "WorldClim 2.1"},
    "clay": {"group": "soil", "meaning": "Clay Content (0-20cm)", "unit": "%", "source": "SoilGrids 2.0"},
    "elevation": {"group": "terrain", "meaning": "Elevation", "unit": "m", "source": "SRTM DEM"},
    "northness": {"group": "terrain", "meaning": "Northness (cos(aspect))", "unit": "dimensionless", "source": "Derived from SRTM DEM"},
    "sand": {"group": "soil", "meaning": "Sand Content (0-20cm)", "unit": "%", "source": "SoilGrids 2.0"},
}

def main():
    log.info("Building predictor group mapping...")
    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))

    rows = []
    for _, row in pred_df.iterrows():
        var = row["variable"]
        meta = VARIABLE_META.get(var, {})
        if not meta:
            log.warning(f"⚠️ No metadata found for {var} - using defaults")
            meta = {"group": row.get("group", "unknown"), "meaning": var, "unit": "unknown", "source": "unknown"}

        rows.append({
            "variable": var,
            "group": meta["group"],
            "meaning": meta["meaning"],
            "unit": meta["unit"],
            "source": meta["source"],
            "selected_in_experiment1": True,
        })

    mapping = pd.DataFrame(rows)
    # Validate grouping
    valid_groups = {"climate", "soil", "terrain"}
    for _, r in mapping.iterrows():
        assert r["group"] in valid_groups, f"Invalid group for {r['variable']}: {r['group']}"

    out_path = os.path.join(OUT_DIR, "predictor_group_mapping.csv")
    mapping.to_csv(out_path, index=False)
    log.info(f"Predictor group mapping written to {out_path}")
    log.info(f"\nGroup counts:\n{mapping['group'].value_counts().to_string()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
