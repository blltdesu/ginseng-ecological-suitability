#!/usr/bin/env python3
"""
Step 3: Build explainability analysis datasets.
- explainability_dataset.csv (full training data with required columns)
- explainability_sample.csv (stratified sample for SHAP/ALE)
- raster_explainability_sample.csv (spatial sample from rasters)
"""
import sys, os, logging
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from scipy.spatial import KDTree

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

RANDOM_SEED = 20260807

def main():
    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    predictors = pred_df["variable"].tolist()
    log.info(f"Final predictors: {predictors}")

    # Load training matrix
    train = pd.read_csv(os.path.join(INPUT_DIR, "training_matrix_main.csv"))
    log.info(f"Training matrix: {train.shape}")

    # A. Full explainability dataset
    required_cols = ["sample_id", "label", "sample_type", "longitude", "latitude",
                     "outer_fold", "background_replicate"]
    # Add sample_weight if present
    if "sample_weight" in train.columns:
        required_cols.append("sample_weight")
    else:
        log.warning("sample_weight not in training matrix, using default 1.0")
        train["sample_weight"] = 1.0
        required_cols.append("sample_weight")

    cols_to_keep = required_cols + predictors
    missing = [c for c in cols_to_keep if c not in train.columns]
    if missing:
        log.error(f"Missing columns: {missing}")
        return 1

    explain_df = train[cols_to_keep].copy()
    exp_path = os.path.join(OUT_DIR, "explainability_dataset.csv")
    explain_df.to_csv(exp_path, index=False)
    log.info(f"Explainability dataset: {explain_df.shape} written to {exp_path}")

    # B. Explainability sample (stratified)
    np.random.seed(RANDOM_SEED)
    presence = explain_df[explain_df["label"] == 1].copy()
    background = explain_df[explain_df["label"] == 0].copy()

    n_bg_sample = min(5000, len(background))
    # Stratify by outer_fold × background_replicate
    bg_groups = background.groupby(["outer_fold", "background_replicate"])
    sampled_bg_parts = []
    # Proportional allocation
    total_bg = len(background)
    for name, group in bg_groups:
        n = max(1, int(n_bg_sample * len(group) / total_bg))
        sampled_bg_parts.append(group.sample(n=n, random_state=RANDOM_SEED))

    bg_sample = pd.concat(sampled_bg_parts, ignore_index=True)
    # Trim to exact count
    if len(bg_sample) > n_bg_sample:
        bg_sample = bg_sample.sample(n=n_bg_sample, random_state=RANDOM_SEED)

    explain_sample = pd.concat([presence, bg_sample], ignore_index=True)
    log.info(f"Explainability sample: {len(explain_sample)} total "
             f"(presence={len(presence)}, background={len(bg_sample)})")

    sample_path = os.path.join(OUT_DIR, "explainability_sample.csv")
    explain_sample.to_csv(sample_path, index=False)
    log.info(f"Explainability sample written to {sample_path}")

    # C. Raster explainability sample (spatial stratified from M area)
    log.info("Building raster explainability sample...")
    try:
        # Use the first predictor raster as reference
        ref_raster_path = os.path.join(INPUT_DIR, "predictors", f"{predictors[0]}.tif")
        with rasterio.open(ref_raster_path) as src:
            transform = src.transform
            width, height = src.width, src.height
            crs = src.crs

            # Read all valid pixels (non-nodata)
            # Use regular grid sampling
            n_target = min(50000, width * height)

            # Systematic grid sampling
            total_pixels = width * height
            step = max(1, int(np.sqrt(total_pixels / n_target)))

            # Read a block to find valid pixels
            ys = np.arange(0, height, step, dtype=int)
            xs = np.arange(0, width, step, dtype=int)

            # Read a sample of the raster
            raster_data = src.read(1)
            valid_mask = ~np.isnan(raster_data) & (raster_data != src.nodata) if src.nodata else ~np.isnan(raster_data)

            sampled_rows = []
            predictor_dir = os.path.join(INPUT_DIR, "predictors")
            # Pre-load required predictors
            pred_arrays = {}
            for p in predictors:
                with rasterio.open(os.path.join(predictor_dir, f"{p}.tif")) as ps:
                    pred_arrays[p] = ps.read(1)

            i = 0
            for y in ys:
                for x in xs:
                    if y < height and x < width and valid_mask[y, x]:
                        lon, lat = rasterio.transform.xy(transform, y, x)
                        row_data = {
                            "sample_id": f"raster_{i}",
                            "longitude": lon,
                            "latitude": lat,
                            "col": x,
                            "row": y,
                        }
                        valid = True
                        for p in predictors:
                            val = pred_arrays[p][y, x]
                            if np.isnan(val):
                                valid = False
                                break
                            row_data[p] = float(val)
                        if valid:
                            sampled_rows.append(row_data)
                            i += 1
                            if i >= n_target:
                                break
                if i >= n_target:
                    break

            raster_sample = pd.DataFrame(sampled_rows)
            # Ensure all predictor columns present
            for p in predictors:
                if p not in raster_sample.columns:
                    raster_sample[p] = np.nan

            raster_sample = raster_sample.dropna(subset=predictors)
            raster_path = os.path.join(OUT_DIR, "raster_explainability_sample.csv")
            raster_sample.to_csv(raster_path, index=False)
            log.info(f"Raster explainability sample: {len(raster_sample)} pixels written to {raster_path}")

    except Exception as e:
        log.error(f"Failed to build raster explainability sample: {e}")
        log.warning("Continuing without raster sample - spatial SHAP may be limited")
        import traceback
        traceback.print_exc()

    log.info("Step 2 (dataset building) complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
