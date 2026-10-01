#!/usr/bin/env python3
"""Fast spatial analysis using Random Forest only for key gain maps."""
import sys, os, logging, warnings, ast
import numpy as np
import pandas as pd
import rasterio
import joblib

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验3"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment2")
OUT_DIR = os.path.join(ROOT, "11_spatial_group_contribution")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(os.path.join(OUT_DIR, "subset_maps"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "spatial_contribution.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

RANDOM_SEED = 20260807


def _save_raster(data, path, transform, crs, width, height, dtype=np.float32):
    with rasterio.open(path, 'w', driver='GTiff', height=height, width=width,
                       count=1, dtype=dtype, crs=crs, transform=transform,
                       compress='lzw', nodata=np.nan) as dst:
        dst.write(data.astype(dtype), 1)


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV

    log.info("=" * 60)
    log.info("FAST SPATIAL ANALYSIS (RF only)")
    log.info("=" * 60)

    # Load data
    registry = pd.read_csv(os.path.join(ROOT, "03_subset_datasets", "subset_registry.csv"))
    train = pd.read_csv(os.path.join(INPUT_DIR, "training_matrix_main.csv"))

    # Load rasters
    pred_dir = os.path.join(INPUT_DIR, "predictors")
    pred_arrays = {}
    transform = None; crs = None; width = None; height = None

    for p in ["bio02","bio03","bio05","bio15","clay","elevation","northness","sand"]:
        path = os.path.join(pred_dir, f"{p}.tif")
        with rasterio.open(path) as src:
            if transform is None:
                transform = src.transform; crs = src.crs
                width = src.width; height = src.height
            pred_arrays[p] = src.read(1).astype(np.float64)
        log.info(f"  Loaded {p}")

    log.info(f"Raster: {height}x{width}, CRS: {crs}")

    # Build valid mask
    valid_mask = np.ones((height, width), dtype=bool)
    for p in pred_arrays:
        data = pred_arrays[p]
        valid_mask &= ~np.isnan(data) & (data > -1e38)

    n_valid = valid_mask.sum()
    log.info(f"Valid pixels: {n_valid} / {height*width}")

    # Train RF models for key subsets
    key_subsets = ["C", "S", "T", "CS", "CT", "ST", "CST"]
    subset_maps = {}

    for _, row in registry.iterrows():
        sid = row["subset_id"]
        if sid not in key_subsets:
            continue
        vl_str = row["variable_list"]
        if isinstance(vl_str, str) and vl_str.startswith("["):
            var_list = ast.literal_eval(vl_str)
        else:
            continue
        if not var_list:
            continue

        log.info(f"Training RF for {sid}: {var_list}")
        X_full = train[var_list].values.astype(np.float64)
        y_full = train["label"].values.astype(int)

        rf = RandomForestClassifier(n_estimators=500, max_depth=None, max_features="sqrt",
                                     min_samples_leaf=1, class_weight="balanced",
                                     random_state=RANDOM_SEED, n_jobs=-1)
        cal = CalibratedClassifierCV(estimator=rf, method='sigmoid', cv=3, n_jobs=-1)
        cal.fit(X_full, y_full)

        # Predict in batches
        pred_map = np.full((height, width), np.nan, dtype=np.float32)
        n_pixels = height * width
        batch_size = 200000
        valid_flat = valid_mask.flatten()

        for start in range(0, n_pixels, batch_size):
            end = min(start + batch_size, n_pixels)
            batch_valid = valid_flat[start:end]
            if batch_valid.sum() == 0:
                continue

            X_batch = np.zeros((batch_valid.sum(), len(var_list)), dtype=np.float64)
            for i, v in enumerate(var_list):
                X_batch[:, i] = pred_arrays[v].flatten()[start:end][batch_valid]

            pred_batch = cal.predict_proba(X_batch)[:, 1]
            pred_flat = pred_map.flatten()
            batch_indices = np.where(batch_valid)[0]
            pred_flat[start + batch_indices] = pred_batch
            pred_map = pred_flat.reshape(height, width)

        subset_maps[sid] = pred_map

        # Save subset map
        out_path = os.path.join(OUT_DIR, "subset_maps", f"{sid}_suitability.tif")
        _save_raster(pred_map, out_path, transform, crs, width, height)
        log.info(f"  Saved {sid}_suitability.tif")

    # Spatial gain maps
    log.info("\nComputing spatial gain maps...")
    gains = {}

    if "CST" in subset_maps and "ST" in subset_maps:
        c_gain = subset_maps["CST"] - subset_maps["ST"]
        _save_raster(c_gain, os.path.join(OUT_DIR, "climate_spatial_gain.tif"),
                    transform, crs, width, height)
        gains["Climate"] = c_gain
        log.info(f"  Climate gain: mean={np.nanmean(c_gain):.4f}")

    if "CST" in subset_maps and "CT" in subset_maps:
        s_gain = subset_maps["CST"] - subset_maps["CT"]
        _save_raster(s_gain, os.path.join(OUT_DIR, "soil_spatial_gain.tif"),
                    transform, crs, width, height)
        gains["Soil"] = s_gain
        log.info(f"  Soil gain: mean={np.nanmean(s_gain):.4f}")

    if "CST" in subset_maps and "CS" in subset_maps:
        t_gain = subset_maps["CST"] - subset_maps["CS"]
        _save_raster(t_gain, os.path.join(OUT_DIR, "terrain_spatial_gain.tif"),
                    transform, crs, width, height)
        gains["Terrain"] = t_gain
        log.info(f"  Terrain gain: mean={np.nanmean(t_gain):.4f}")

    # Dominant group
    if gains:
        groups = list(gains.keys())
        abs_stack = np.stack([np.abs(gains[g]) for g in groups], axis=0)
        dominant_idx = np.argmax(abs_stack, axis=0).astype(np.float32)
        all_nan = np.all(np.isnan(abs_stack), axis=0)
        dominant_idx[all_nan] = np.nan
        dominant_map = dominant_idx + 1
        dominant_strength = np.max(abs_stack, axis=0)

        _save_raster(dominant_map, os.path.join(OUT_DIR, "dominant_group_ablation.tif"),
                    transform, crs, width, height)
        _save_raster(dominant_strength, os.path.join(OUT_DIR, "dominant_group_strength.tif"),
                    transform, crs, width, height)

        for i, g in enumerate(groups):
            valid_px = (~all_nan).sum()
            pct = (dominant_idx == i).sum() / max(valid_px, 1) * 100
            log.info(f"  {g} dominant: {pct:.1f}% of valid pixels")

    log.info("\nFast spatial analysis complete!")
    return 0


if __name__ == "__main__":
    main()
