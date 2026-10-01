#!/usr/bin/env python3
"""
Spatial contribution analysis for Experiment 3.
Generate subset suitability maps, spatial gain maps, and dominant group maps.
"""
import sys, os, logging, warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
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


def get_model_constructor(model_name):
    """Get model constructor (same as core script)."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=1000, max_depth=None, max_features="sqrt",
            min_samples_leaf=1, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1
        )
    elif model_name == "xgboost":
        return XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0
        )
    elif model_name == "brt":
        return GradientBoostingClassifier(
            n_estimators=1000, max_depth=5, learning_rate=0.01,
            subsample=0.8, max_features="sqrt",
            random_state=RANDOM_SEED
        )
    elif model_name == "maxent":
        return LogisticRegression(
            C=1.0, penalty='l2', solver='lbfgs',
            max_iter=10000, random_state=RANDOM_SEED,
            class_weight='balanced'
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def calibrate_model(model, X_cal, y_cal):
    """Calibrate model."""
    from sklearn.calibration import CalibratedClassifierCV
    calibrated = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=3, n_jobs=-1)
    calibrated.fit(X_cal, y_cal)
    return calibrated


def predict_raster(model, predictor_arrays, var_list, transform, width, height, nodata=-9999):
    """Apply model prediction across a raster grid."""
    # Stack all predictors
    n_pixels = width * height
    X = np.zeros((n_pixels, len(var_list)), dtype=np.float64)
    valid = np.ones(n_pixels, dtype=bool)

    for i, var in enumerate(var_list):
        if var in predictor_arrays:
            data = predictor_arrays[var].flatten()
            X[:, i] = data
            # Invalidate nodata pixels
            valid = valid & (~np.isnan(data))
            if nodata is not None:
                valid = valid & (data != nodata)
        else:
            log.warning(f"Predictor {var} not found in raster arrays")
            valid[:] = False

    # Predict in batches
    pred = np.full(n_pixels, np.nan, dtype=np.float32)
    if valid.sum() > 0:
        batch_size = 200000
        for start in range(0, n_pixels, batch_size):
            end = min(start + batch_size, n_pixels)
            batch_valid = valid[start:end]
            if batch_valid.sum() > 0:
                X_batch = X[start:end][batch_valid]
                pred[start:end][batch_valid] = model.predict_proba(X_batch)[:, 1]

    return pred.reshape(height, width)


def main():
    log.info("=" * 60)
    log.info("SPATIAL GROUP CONTRIBUTION ANALYSIS")
    log.info("=" * 60)

    # Load data
    group_map = pd.read_csv(os.path.join(ROOT, "02_group_definition", "environment_group_definition.csv"))
    registry = pd.read_csv(os.path.join(ROOT, "03_subset_datasets", "subset_registry.csv"))
    weights = pd.read_csv(os.path.join(INPUT_DIR, "ensemble_weights.csv"))
    train = pd.read_csv(os.path.join(INPUT_DIR, "training_matrix_main.csv"))
    # Restore fold information
    fa = pd.read_csv(os.path.join(INPUT_DIR, "spatial_cv_fold_assignment.csv"))
    presence_idx = train[train["sample_type"] == "presence"].index
    train.loc[presence_idx, "outer_fold"] = fa["outer_fold"].values
    rng = np.random.RandomState(RANDOM_SEED)
    bg_idx = train[train["sample_type"] == "background"].index
    fold_props = fa["outer_fold"].value_counts(normalize=True).sort_index()
    for rep in sorted(train.loc[bg_idx, "background_replicate"].unique()):
        rep_idx = bg_idx[train.loc[bg_idx, "background_replicate"] == rep]
        folds = rng.choice(fold_props.index, size=len(rep_idx), p=fold_props.values)
        train.loc[rep_idx, "outer_fold"] = folds
    train["outer_fold"] = train["outer_fold"].astype(int)

    predictors = group_map["variable"].tolist()

    # Get model weights for ensemble
    w_dict = dict(zip(weights["model"], weights["weight"]))
    total_w = sum(w_dict.values())
    w_norm = {k: v / total_w for k, v in w_dict.items()}
    available_models = list(w_norm.keys())

    # Read all predictor rasters
    pred_dir = os.path.join(INPUT_DIR, "predictors")
    pred_arrays = {}
    ref_raster = None
    transform = None
    crs = None
    width, height = None, None

    log.info("Loading predictor rasters...")
    for p in predictors:
        path = os.path.join(pred_dir, f"{p}.tif")
        if os.path.exists(path):
            with rasterio.open(path) as src:
                if ref_raster is None:
                    transform = src.transform
                    crs = src.crs
                    width = src.width
                    height = src.height
                    ref_raster = p
                pred_arrays[p] = src.read(1).astype(np.float64)
            log.info(f"  Loaded {p}: {pred_arrays[p].shape}")
        else:
            log.warning(f"  Missing: {path}")

    log.info(f"Reference raster: {ref_raster}, shape: ({height}, {width})")
    log.info(f"Transform: {transform}")
    log.info(f"CRS: {crs}")

    # Train ensemble models for each subset on FULL data
    subset_ids = ["C", "S", "T", "CS", "CT", "ST", "CST"]
    subset_predictions = {}

    for _, row in registry.iterrows():
        subset_id = row["subset_id"]
        if subset_id not in subset_ids:
            continue
        # Parse variable list - may be stored as string representation of list
        var_list_str = row["variable_list"]
        if isinstance(var_list_str, str) and var_list_str.startswith("["):
            import ast
            var_list = ast.literal_eval(var_list_str)
        elif isinstance(var_list_str, str):
            var_list = [v.strip() for v in var_list_str.split("+") if v.strip() and v.strip() != "None"]
        else:
            var_list = var_list_str

        if not var_list:
            continue

        log.info(f"\nTraining ensemble for subset {subset_id} with {var_list}")

        # Train each model on full data (not fold-split)
        model_preds = []
        X_full = train[var_list].values.astype(np.float64)
        y_full = train["label"].values.astype(int)

        for model_name in available_models:
            try:
                base = get_model_constructor(model_name)
                cal = calibrate_model(base, X_full, y_full)

                # Predict on raster
                pred_map = predict_raster(cal, pred_arrays, var_list, transform, width, height)
                model_preds.append((pred_map, w_norm[model_name]))
                log.info(f"  {model_name}: raster predicted")
            except Exception as e:
                log.error(f"  {model_name}: FAILED - {e}")

        if model_preds:
            # Ensemble weighted average
            ensemble_map = np.zeros((height, width), dtype=np.float64)
            weight_sum = 0
            for pmap, w in model_preds:
                valid_mask = ~np.isnan(pmap)
                ensemble_map[valid_mask] += pmap[valid_mask] * w
                weight_sum += w
            if weight_sum > 0:
                ensemble_map = ensemble_map / weight_sum

            subset_predictions[subset_id] = ensemble_map

            # Save suitability map
            map_path = os.path.join(OUT_DIR, "subset_maps", f"{subset_id}_suitability.tif")
            with rasterio.open(
                map_path, 'w',
                driver='GTiff',
                height=height, width=width,
                count=1, dtype=np.float32,
                crs=crs, transform=transform,
                compress='lzw', nodata=np.nan
            ) as dst:
                dst.write(ensemble_map.astype(np.float32), 1)
            log.info(f"  Saved: {map_path}")

    # Compute spatial gain maps
    log.info("\n" + "=" * 60)
    log.info("COMPUTING SPATIAL GAIN MAPS")
    log.info("=" * 60)

    # Climate gain = CST - ST
    if "CST" in subset_predictions and "ST" in subset_predictions:
        climate_gain = subset_predictions["CST"] - subset_predictions["ST"]
        _save_raster(climate_gain, os.path.join(OUT_DIR, "climate_spatial_gain.tif"),
                    transform, crs, width, height)
        log.info("Climate spatial gain saved")

    # Soil gain = CST - CT
    if "CST" in subset_predictions and "CT" in subset_predictions:
        soil_gain = subset_predictions["CST"] - subset_predictions["CT"]
        _save_raster(soil_gain, os.path.join(OUT_DIR, "soil_spatial_gain.tif"),
                    transform, crs, width, height)
        log.info("Soil spatial gain saved")

    # Terrain gain = CST - CS
    if "CST" in subset_predictions and "CS" in subset_predictions:
        terrain_gain = subset_predictions["CST"] - subset_predictions["CS"]
        _save_raster(terrain_gain, os.path.join(OUT_DIR, "terrain_spatial_gain.tif"),
                    transform, crs, width, height)
        log.info("Terrain spatial gain saved")

    # Dominant group map
    log.info("\nComputing dominant group map...")
    gain_maps = {}
    if "CST" in subset_predictions and "ST" in subset_predictions:
        gain_maps["Climate"] = subset_predictions["CST"] - subset_predictions["ST"]
    if "CST" in subset_predictions and "CT" in subset_predictions:
        gain_maps["Soil"] = subset_predictions["CST"] - subset_predictions["CT"]
    if "CST" in subset_predictions and "CS" in subset_predictions:
        gain_maps["Terrain"] = subset_predictions["CST"] - subset_predictions["CS"]

    if gain_maps:
        # Stack absolute gains
        groups = list(gain_maps.keys())
        abs_stack = np.stack([np.abs(gain_maps[g]) for g in groups], axis=0)

        # Dominant group index
        dominant_idx = np.argmax(abs_stack, axis=0).astype(np.float32)
        dominant_idx[np.all(np.isnan(abs_stack), axis=0)] = np.nan

        # Dominant group category (1=Climate, 2=Soil, 3=Terrain)
        dominant_map = dominant_idx + 1

        # Dominant effect strength
        dominant_strength = np.max(abs_stack, axis=0)

        _save_raster(dominant_map, os.path.join(OUT_DIR, "dominant_group_ablation.tif"),
                    transform, crs, width, height, dtype=np.float32)
        _save_raster(dominant_strength, os.path.join(OUT_DIR, "dominant_group_strength.tif"),
                    transform, crs, width, height, dtype=np.float32)
        log.info("Dominant group maps saved")

        # Compute area percentages
        for i, g in enumerate(groups):
            area_pct = (dominant_idx == i).sum() / (~np.isnan(dominant_idx)).sum() * 100
            log.info(f"  {g} dominant: {area_pct:.1f}%")

    log.info("\nSpatial analysis complete!")
    return 0


def _save_raster(data, path, transform, crs, width, height, dtype=np.float32):
    """Helper to save a single-band raster."""
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=dtype,
        crs=crs, transform=transform,
        compress='lzw', nodata=np.nan if dtype == np.float32 else -9999
    ) as dst:
        dst.write(data.astype(dtype), 1)


if __name__ == "__main__":
    main()
