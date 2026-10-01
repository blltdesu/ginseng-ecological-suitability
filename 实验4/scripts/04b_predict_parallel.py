#!/usr/bin/env python3
"""
Experiment 4 — Parallel Future Prediction Runner
Spawns one process per GCM to parallelize predictions across GCMs.
"""
import os, sys, csv, json, logging, time, warnings
from pathlib import Path
import joblib
import rasterio
import numpy as np

warnings.filterwarnings("ignore")

EXP4_DIR = Path(r"E:\人参种在哪\实验4")
INPUT_DIR = EXP4_DIR / "00_input_from_experiment3"
PROCESSED_CLIMATE_DIR = EXP4_DIR / "04_future_climate_processed"
PRED_DIR = EXP4_DIR / "07_future_predictions_individual"
ENSEMBLE_DIR = EXP4_DIR / "08_future_predictions_ensemble"
CHANGE_DIR = EXP4_DIR / "09_suitability_change"
LOG_DIR = EXP4_DIR / "logs"

SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]
BIO_VARS = ["bio02", "bio03", "bio05", "bio15"]
STATIC_VARS = {"clay": "clay", "elevation": "elevation", "northness": "northness", "sand": "sand"}
FEATURE_ORDER = ["bio02", "bio03", "bio05", "bio15", "clay", "elevation", "northness", "sand"]
MODEL_NAMES = ["random_forest", "xgboost", "brt", "maxent"]

# Load shared state
MASK_PATH = INPUT_DIR / "common_valid_mask.tif"
CURRENT_ENSEMBLE = INPUT_DIR / "current_ensemble_suitability.tif"
CURRENT_BINARY = INPUT_DIR / "current_binary_suitability.tif"

with open(INPUT_DIR / "ensemble_threshold.json", "r") as f:
    THRESHOLD_DATA = json.load(f)
ENSEMBLE_THRESHOLD = THRESHOLD_DATA["ensemble_threshold"]

MODELS_MANIFEST = INPUT_DIR / "models_manifest.csv"
ensemble_weights = {}
with open(MODELS_MANIFEST, "r") as f:
    for row in csv.DictReader(f):
        ensemble_weights[row["model"]] = float(row["ensemble_weight"])

def predict_model_chunked(model, X, model_name, chunk_size=200000):
    if isinstance(model, dict):
        cal_model = model.get("calibrated_model", model)
    else:
        cal_model = model
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype=np.float32)
    try:
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            X_chunk = X[start:end]
            if hasattr(cal_model, "predict_proba"):
                proba = cal_model.predict_proba(X_chunk)
                if proba.shape[1] >= 2:
                    predictions[start:end] = proba[:, 1]
                else:
                    predictions[start:end] = proba[:, 0]
            elif hasattr(cal_model, "predict"):
                predictions[start:end] = cal_model.predict(X_chunk)
            else:
                return None
        out_of_range = ((predictions < 0) | (predictions > 1)).sum()
        if out_of_range > 0:
            predictions = np.clip(predictions, 0, 1)
        return predictions
    except Exception as e:
        print(f"Prediction error for {model_name}: {e}", flush=True)
        return None

def process_gcm(gcm_name):
    """Process all scenarios for a single GCM."""
    log_file = LOG_DIR / f"future_prediction_{gcm_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(f"predict_{gcm_name}")

    logger.info(f"=== Processing GCM: {gcm_name} ===")

    # Load static rasters
    with rasterio.open(str(MASK_PATH)) as msrc:
        valid_mask = msrc.read(1) > 0
        ref_profile = msrc.profile.copy()
        ref_shape = msrc.shape

    STATIC_RASTER_DIR = Path(r"E:\人参种在哪\实验3\00_input_from_experiment2\predictors")
    STATIC_DATA = {}
    for var_name in STATIC_VARS:
        for search_dir in [STATIC_RASTER_DIR, Path(r"E:\人参种在哪\实验1\03_predictor_screening\final_predictors"), INPUT_DIR]:
            if not search_dir.exists():
                continue
            for pattern in [f"{var_name}.tif", f"*{var_name}*.tif"]:
                matches = list(search_dir.glob(pattern))
                if matches:
                    try:
                        with rasterio.open(str(matches[0])) as s:
                            data = s.read(1)
                            if s.shape != ref_shape:
                                from rasterio.warp import reproject, Resampling
                                aligned = np.zeros(ref_shape, dtype=data.dtype)
                                reproject(source=data, destination=aligned,
                                         src_transform=s.transform, src_crs=s.crs,
                                         dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                                         resampling=Resampling.bilinear)
                                data = aligned
                            data[~valid_mask] = np.nan
                            STATIC_DATA[var_name] = data
                            logger.info(f"Loaded {var_name}: range=[{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
                    except Exception as e:
                        logger.warning(f"Failed to load {var_name}: {e}")
                    break
            if var_name in STATIC_DATA:
                break

    # Load models
    MODEL_DIR = INPUT_DIR / "models_for_future_projection"
    models = {}
    for mn in MODEL_NAMES:
        mf = MODEL_DIR / f"{mn}_cst_calibrated.joblib"
        if mf.exists():
            models[mn] = joblib.load(str(mf))
            logger.info(f"Loaded model: {mn}")

    # Load current binary
    with rasterio.open(str(CURRENT_BINARY)) as s:
        current_binary = s.read(1)
        current_binary = (current_binary > 0.5) & valid_mask

    total = len(SSPS) * len(PERIODS)
    completed = 0
    for ssp in SSPS:
        for period in PERIODS:
            completed += 1
            scenario_key = f"{gcm_name}|{ssp}|{period}"

            # Check if ensemble already exists
            ens_dir = ENSEMBLE_DIR / gcm_name / ssp / period
            if (ens_dir / "ensemble_suitability.tif").exists():
                logger.info(f"[{completed}/{total}] {scenario_key} — already done, skipping")
                continue

            # Check climate data
            climate_dir = PROCESSED_CLIMATE_DIR / gcm_name / ssp / period
            if not climate_dir.exists() or not all((climate_dir / f"{v}.tif").exists() for v in BIO_VARS):
                logger.warning(f"[{completed}/{total}] {scenario_key} — climate data missing, skipping")
                continue

            logger.info(f"[{completed}/{total}] {scenario_key}")

            # Build predictor stack
            stack = []
            for var in FEATURE_ORDER:
                if var in BIO_VARS:
                    tif_path = climate_dir / f"{var}.tif"
                    with rasterio.open(str(tif_path)) as s:
                        data = s.read(1)
                        if s.shape != ref_shape:
                            from rasterio.warp import reproject, Resampling
                            aligned = np.zeros(ref_shape, dtype=data.dtype)
                            reproject(source=data, destination=aligned,
                                     src_transform=s.transform, src_crs=s.crs,
                                     dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                                     resampling=Resampling.bilinear)
                            data = aligned
                    stack.append(data)
                elif var in STATIC_DATA:
                    stack.append(STATIC_DATA[var])
                else:
                    logger.error(f"Unknown variable: {var}")

            X = np.column_stack([s[valid_mask].ravel() for s in stack])
            logger.info(f"  Stack: {X.shape[0]} pixels x {X.shape[1]} features")

            valid_rows = ~np.isnan(X).any(axis=1)
            X_valid = X[valid_rows]
            logger.info(f"  Valid rows: {X_valid.shape[0]}")

            if X_valid.shape[0] == 0:
                logger.warning("  All NaN — skipping")
                continue

            # Predict with each model
            model_preds = {}
            for mn, model in models.items():
                t0 = time.time()
                pred = predict_model_chunked(model, X_valid, mn)
                dt = time.time() - t0

                if pred is None:
                    logger.error(f"  {mn}: FAILED")
                    continue

                # Expand to full grid
                full_pred = np.full(valid_mask.shape, np.nan, dtype=np.float32)
                full_vals = np.full(valid_mask.sum(), np.nan, dtype=np.float32)
                full_vals[valid_rows] = pred
                full_pred[valid_mask] = full_vals

                # Save individual
                out_dir = PRED_DIR / mn / gcm_name / ssp / period
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "suitability.tif"
                profile = ref_profile.copy()
                profile.pop('shape', None)
                profile.update(dtype=np.float32, compress='lzw', nodata=-3.4e38)
                with rasterio.open(str(out_path), 'w', **profile) as dst:
                    dst.write(full_pred, 1)

                model_preds[mn] = full_pred
                logger.info(f"  {mn}: range=[{np.nanmin(full_pred):.4f}, {np.nanmax(full_pred):.4f}], {dt:.1f}s")

            if len(model_preds) < 4:
                logger.warning(f"  Only {len(model_preds)}/4 models succeeded")
                if len(model_preds) == 0:
                    continue

            # Ensemble
            ensemble = np.zeros(valid_mask.shape, dtype=np.float32)
            for mn, pred in model_preds.items():
                w = ensemble_weights.get(mn, 0.25)
                ensemble = np.nansum([ensemble, w * pred], axis=0)

            ens_path = ens_dir / "ensemble_suitability.tif"
            ens_dir.mkdir(parents=True, exist_ok=True)
            profile = ref_profile.copy()
            profile.pop('shape', None)
            profile.update(dtype=np.float32, compress='lzw', nodata=-3.4e38)
            with rasterio.open(str(ens_path), 'w', **profile) as dst:
                dst.write(ensemble, 1)
            logger.info(f"  Ensemble: range=[{np.nanmin(ensemble):.4f}, {np.nanmax(ensemble):.4f}]")

            # Binary
            future_binary = ensemble >= ENSEMBLE_THRESHOLD
            binary_dir = ENSEMBLE_DIR / "binary" / gcm_name / ssp / period
            binary_dir.mkdir(parents=True, exist_ok=True)
            bin_path = binary_dir / "binary_suitability.tif"
            profile = ref_profile.copy()
            profile.pop('shape', None)
            profile.update(dtype=np.uint8, compress='lzw', nodata=255)
            bin_out = np.where(valid_mask, future_binary.astype(np.uint8), 255)
            with rasterio.open(str(bin_path), 'w', **profile) as dst:
                dst.write(bin_out, 1)

            # Change map
            change_map = np.full(valid_mask.shape, 255, dtype=np.uint8)
            change_map[current_binary & future_binary] = 1
            change_map[current_binary & ~future_binary] = 2
            change_map[~current_binary & future_binary] = 3
            change_map[~current_binary & ~future_binary] = 0

            change_dir = CHANGE_DIR / gcm_name / ssp / period
            change_dir.mkdir(parents=True, exist_ok=True)
            change_path = change_dir / "change_class.tif"
            profile = ref_profile.copy()
            profile.pop('shape', None)
            profile.update(dtype=np.uint8, compress='lzw', nodata=255)
            with rasterio.open(str(change_path), 'w', **profile) as dst:
                dst.write(change_map, 1)

            n_current = current_binary.sum()
            n_future = future_binary.sum()
            n_stable = (change_map == 1).sum()
            n_loss = (change_map == 2).sum()
            n_gain = (change_map == 3).sum()
            logger.info(f"  Change: cur={n_current}, fut={n_future}, stable={n_stable}, loss={n_loss}, gain={n_gain}")

    logger.info(f"=== GCM {gcm_name} complete ===")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("gcm", type=str, help="GCM name to process")
    args = parser.parse_args()
    process_gcm(args.gcm)
