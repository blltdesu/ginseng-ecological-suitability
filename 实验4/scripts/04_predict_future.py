#!/usr/bin/env python3
"""
Experiment 4 — Steps 10-16: Future Prediction Pipeline
Builds predictor stacks, runs 4 models, creates ensemble, binarizes,
and classifies stable/gain/loss for all scenarios.
"""
import os, sys, csv, json, logging, time, warnings
from pathlib import Path
import joblib
import rasterio
from rasterio.warp import Resampling
import numpy as np

warnings.filterwarnings("ignore")

# ——— Config ———
EXP4_DIR = Path(r"E:\人参种在哪\实验4")
INPUT_DIR = EXP4_DIR / "00_input_from_experiment3"
PROCESSED_CLIMATE_DIR = EXP4_DIR / "04_future_climate_processed"
STACK_DIR = EXP4_DIR / "05_future_predictor_stacks"
PRED_DIR = EXP4_DIR / "07_future_predictions_individual"
ENSEMBLE_DIR = EXP4_DIR / "08_future_predictions_ensemble"
CHANGE_DIR = EXP4_DIR / "09_suitability_change"
QC_DIR = EXP4_DIR / "06_projection_qc"
LOG_DIR = EXP4_DIR / "logs"

for d in [STACK_DIR, PRED_DIR, ENSEMBLE_DIR, CHANGE_DIR, QC_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "future_prediction.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("future_prediction")

# ——— Constants ———
GCMS = ["ACCESS-CM2", "BCC-CSM2-MR", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]
BIO_VARS = ["bio02", "bio03", "bio05", "bio15"]
STATIC_VARS = {"clay": "clay", "elevation": "elevation", "northness": "northness", "sand": "sand"}
FEATURE_ORDER = ["bio02", "bio03", "bio05", "bio15", "clay", "elevation", "northness", "sand"]
MODEL_NAMES = ["random_forest", "xgboost", "brt", "maxent"]

# ——— Load static data ———
MASK_PATH = INPUT_DIR / "common_valid_mask.tif"
CURRENT_ENSEMBLE = INPUT_DIR / "current_ensemble_suitability.tif"
CURRENT_BINARY = INPUT_DIR / "current_binary_suitability.tif"

with open(INPUT_DIR / "ensemble_threshold.json", "r") as f:
    THRESHOLD_DATA = json.load(f)
ENSEMBLE_THRESHOLD = THRESHOLD_DATA["ensemble_threshold"]

# Load ensemble weights from models_manifest
MODELS_MANIFEST = INPUT_DIR / "models_manifest.csv"
ensemble_weights = {}
with open(MODELS_MANIFEST, "r") as f:
    for row in csv.DictReader(f):
        ensemble_weights[row["model"]] = float(row["ensemble_weight"])

logger.info(f"Ensemble weights: {ensemble_weights}")
logger.info(f"Ensemble threshold: {ENSEMBLE_THRESHOLD}")

# ——— Load static rasters ———
logger.info("Loading static rasters...")
STATIC_DATA = {}
with rasterio.open(str(MASK_PATH)) as msrc:
    valid_mask = msrc.read(1) > 0
    ref_profile = msrc.profile.copy()
    ref_shape = msrc.shape
    ref_profile['shape'] = ref_shape

# Find static variable rasters from experiment 1 input or experiment 3
# These should be in the processed data from experiment 1
# For now, look in common locations
static_sources = {
    "elevation": INPUT_DIR.parent / "01_current_environment" if False else None,
}

# We need to find the static rasters. They might be in experiment 1's output.
# Let's look for them
STATIC_RASTER_DIR = Path(r"E:\人参种在哪\实验3\00_input_from_experiment2\predictors")
if not STATIC_RASTER_DIR.exists():
    STATIC_RASTER_DIR = Path(r"E:\人参种在哪\实验1\03_predictor_screening\final_predictors")
if not STATIC_RASTER_DIR.exists():
    STATIC_RASTER_DIR = INPUT_DIR

logger.info(f"Looking for static rasters in: {STATIC_RASTER_DIR}")

for var_name, var_key in STATIC_VARS.items():
    # Try various locations
    found = False
    for search_dir in [STATIC_RASTER_DIR,
                       Path(r"E:\人参种在哪\实验1\03_predictor_screening\final_predictors"),
                       Path(r"E:\人参种在哪\实验3\00_input_from_experiment2\predictors"),
                       INPUT_DIR]:
        if not search_dir.exists():
            continue
        for pattern in [f"{var_key}.tif", f"{var_name}.tif", f"*{var_key}*.tif"]:
            matches = list(search_dir.glob(pattern))
            if matches:
                try:
                    with rasterio.open(str(matches[0])) as s:
                        data = s.read(1)
                        src_shape = s.shape
                        # Align to reference if needed
                        if src_shape != ref_shape:
                            logger.warning(f"{var_name} shape mismatch: {s.shape} vs {ref_profile['shape']}")
                            # Resample
                            from rasterio.warp import reproject, Resampling
                            aligned = np.zeros(ref_profile["shape"], dtype=data.dtype)
                            reproject(source=data, destination=aligned,
                                     src_transform=s.transform, src_crs=s.crs,
                                     dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                                     resampling=Resampling.bilinear)
                            data = aligned
                        # Apply mask
                        data[~valid_mask] = np.nan
                        STATIC_DATA[var_name] = data
                        logger.info(f"  Loaded {var_name} from {matches[0]}, range=[{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
                        found = True
                except Exception as e:
                    logger.warning(f"  Failed to load {var_name} from {matches[0]}: {e}")
                break
        if found:
            break
    if not found:
        logger.warning(f"  Could not find {var_name} raster!")

def build_predictor_stack(gcm, ssp, period):
    """Build the predictor stack for a given scenario."""
    scenario_climate = PROCESSED_CLIMATE_DIR / gcm / ssp / period
    if not scenario_climate.exists():
        return None

    stack = []
    for var in FEATURE_ORDER:
        if var in BIO_VARS:
            # Future climate variable
            tif_path = scenario_climate / f"{var}.tif"
            if not tif_path.exists():
                logger.error(f"Missing climate raster: {tif_path}")
                return None
            with rasterio.open(str(tif_path)) as s:
                data = s.read(1)
                if s.shape != ref_shape:
                    from rasterio.warp import reproject
                    aligned = np.zeros(ref_profile["shape"], dtype=data.dtype)
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
            return None

    # Stack shape: (n_pixels, n_features)
    stack_2d = np.column_stack([s[valid_mask].ravel() for s in stack])
    return stack_2d

def predict_model_chunked(model, X, model_name, chunk_size=200000):
    """Run model prediction with Platt calibration in chunks to manage memory."""
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
                logger.error(f"No predict method for {model_name}")
                return None

        # Check range
        out_of_range = ((predictions < 0) | (predictions > 1)).sum()
        if out_of_range > 0:
            logger.warning(f"  {model_name}: {out_of_range} predictions outside [0,1] — clipping")
            predictions = np.clip(predictions, 0, 1)

        return predictions
    except Exception as e:
        logger.error(f"Prediction error for {model_name}: {e}")
        return None

def main():
    logger.info("=" * 60)
    logger.info("Future Prediction Pipeline — Experiment 4")
    logger.info(f"Design: {len(GCMS)} GCMs × {len(SSPS)} SSPs × {len(PERIODS)} periods = {len(GCMS)*len(SSPS)*len(PERIODS)} scenarios")
    logger.info("=" * 60)

    # ——— Load models ———
    MODEL_DIR = INPUT_DIR / "models_for_future_projection"
    models = {}
    for mn in MODEL_NAMES:
        mf = MODEL_DIR / f"{mn}_cst_calibrated.joblib"
        if mf.exists():
            models[mn] = joblib.load(str(mf))
            logger.info(f"Loaded model: {mn}")
        else:
            logger.error(f"Model not found: {mf}")

    if len(models) < 4:
        logger.critical("Not all models loaded — STOP_D")
        return

    # ——— Load current binary for change detection ———
    with rasterio.open(str(CURRENT_BINARY)) as s:
        current_binary = s.read(1)
        # Apply valid mask
        current_binary = (current_binary > 0.5) & valid_mask
    logger.info(f"Loaded current binary: {current_binary.sum()} suitable pixels")

    # ——— Process all scenarios ———
    prediction_qc_rows = []
    total = len(GCMS) * len(SSPS) * len(PERIODS)
    completed = 0

    for gcm in GCMS:
        for ssp in SSPS:
            for period in PERIODS:
                completed += 1
                logger.info(f"\n[{completed}/{total}] {gcm} | {ssp} | {period}")

                # Check climate data exists
                climate_dir = PROCESSED_CLIMATE_DIR / gcm / ssp / period
                if not climate_dir.exists() or not all((climate_dir / f"{v}.tif").exists() for v in BIO_VARS):
                    logger.warning(f"  Missing climate data — skipping")
                    continue

                # Build predictor stack
                X = build_predictor_stack(gcm, ssp, period)
                if X is None:
                    logger.error(f"  Failed to build predictor stack")
                    continue

                logger.info(f"  Predictor stack: {X.shape[0]} pixels × {X.shape[1]} features")

                # Remove NaN rows
                valid_rows = ~np.isnan(X).any(axis=1)
                X_valid = X[valid_rows]
                logger.info(f"  Valid (non-NaN) rows: {X_valid.shape[0]}")

                if X_valid.shape[0] == 0:
                    logger.warning(f"  All pixels NaN — skipping")
                    continue

                # ——— Model predictions ———
                model_preds = {}
                for mn, model in models.items():
                    t0 = time.time()
                    pred = predict_model_chunked(model, X_valid, mn)
                    dt = time.time() - t0

                    if pred is None:
                        logger.error(f"  {mn}: prediction FAILED")
                        continue

                    # Expand back to full grid
                    full_pred = np.full(valid_mask.shape, np.nan, dtype=np.float32)
                    full_pred[valid_mask] = np.nan
                    full_pred_vals = np.full(valid_mask.sum(), np.nan, dtype=np.float32)
                    full_pred_vals[valid_rows] = pred
                    full_pred[valid_mask] = full_pred_vals

                    # Save individual model prediction
                    out_dir = PRED_DIR / mn / gcm / ssp / period
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / "suitability.tif"
                    profile = ref_profile.copy()
                    profile.pop('shape', None)
                    profile.update(dtype=np.float32, compress='lzw', nodata=-3.4e38)
                    with rasterio.open(str(out_path), 'w', **profile) as dst:
                        dst.write(full_pred, 1)

                    model_preds[mn] = full_pred
                    logger.info(f"  {mn}: range=[{np.nanmin(full_pred):.4f}, {np.nanmax(full_pred):.4f}], {dt:.1f}s")

                    # QC
                    pred_valid = full_pred[valid_mask][valid_rows]
                    nan_pct = np.isnan(pred_valid).sum() / len(pred_valid) * 100 if len(pred_valid) > 0 else 100
                    out_of_range = ((pred_valid < 0) | (pred_valid > 1)).sum() / len(pred_valid) * 100 if len(pred_valid) > 0 else 0

                    prediction_qc_rows.append({
                        "gcm": gcm, "ssp": ssp, "period": period, "model": mn,
                        "nan_pct": f"{nan_pct:.4f}",
                        "out_of_range_pct": f"{out_of_range:.4f}",
                        "min": f"{np.nanmin(pred_valid):.6f}" if len(pred_valid) > 0 else "nan",
                        "max": f"{np.nanmax(pred_valid):.6f}" if len(pred_valid) > 0 else "nan",
                    })

                if len(model_preds) < 4:
                    logger.warning(f"  Only {len(model_preds)}/4 models succeeded")
                    if len(model_preds) == 0:
                        continue

                # ——— Ensemble prediction ———
                ensemble = np.zeros(valid_mask.shape, dtype=np.float32)
                weight_sum = 0
                for mn, pred in model_preds.items():
                    w = ensemble_weights.get(mn, 0.25)
                    ensemble = np.nansum([ensemble, w * pred], axis=0)
                    weight_sum += w

                if weight_sum > 0 and weight_sum != 1:
                    ensemble = ensemble / weight_sum

                # Save ensemble
                ens_dir = ENSEMBLE_DIR / gcm / ssp / period
                ens_dir.mkdir(parents=True, exist_ok=True)
                ens_path = ens_dir / "ensemble_suitability.tif"
                profile = ref_profile.copy()
                profile.pop('shape', None)
                profile.update(dtype=np.float32, compress='lzw', nodata=-3.4e38)
                with rasterio.open(str(ens_path), 'w', **profile) as dst:
                    dst.write(ensemble, 1)
                logger.info(f"  Ensemble: range=[{np.nanmin(ensemble):.4f}, {np.nanmax(ensemble):.4f}]")

                # ——— Binary ———
                future_binary = ensemble >= ENSEMBLE_THRESHOLD
                binary_dir = ENSEMBLE_DIR / "binary" / gcm / ssp / period
                binary_dir.mkdir(parents=True, exist_ok=True)
                bin_path = binary_dir / "binary_suitability.tif"
                profile = ref_profile.copy()
                profile.pop('shape', None)
                profile.update(dtype=np.uint8, compress='lzw', nodata=255)
                bin_out = np.where(valid_mask, future_binary.astype(np.uint8), 255)
                with rasterio.open(str(bin_path), 'w', **profile) as dst:
                    dst.write(bin_out, 1)

                # ——— Change classification ———
                change_map = np.full(valid_mask.shape, 255, dtype=np.uint8)
                # 0: persistent unsuitable, 1: stable suitable, 2: loss, 3: gain
                change_map[current_binary & future_binary] = 1   # stable
                change_map[current_binary & ~future_binary] = 2  # loss
                change_map[~current_binary & future_binary] = 3  # gain
                change_map[~current_binary & ~future_binary] = 0 # persistent unsuitable

                change_dir = CHANGE_DIR / gcm / ssp / period
                change_dir.mkdir(parents=True, exist_ok=True)
                change_path = change_dir / "change_class.tif"
                profile = ref_profile.copy()
                profile.pop('shape', None)
                profile.update(dtype=np.uint8, compress='lzw', nodata=255)
                with rasterio.open(str(change_path), 'w', **profile) as dst:
                    dst.write(change_map, 1)

                # Log change statistics
                n_current = current_binary.sum()
                n_future = future_binary.sum()
                n_stable = (change_map == 1).sum()
                n_loss = (change_map == 2).sum()
                n_gain = (change_map == 3).sum()
                logger.info(f"  Change: current={n_current}, future={n_future}, "
                          f"stable={n_stable}, loss={n_loss}, gain={n_gain}")

    # ——— Save QC ———
    if prediction_qc_rows:
        with open(QC_DIR / "future_prediction_qc.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=prediction_qc_rows[0].keys())
            w.writeheader()
            w.writerows(prediction_qc_rows)
        logger.info(f"Prediction QC saved: {len(prediction_qc_rows)} entries")

    logger.info("\n" + "=" * 60)
    logger.info("Prediction pipeline complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
