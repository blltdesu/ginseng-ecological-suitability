#!/usr/bin/env python3
"""
Save callable model objects for Experiment 4 future projections.

Per the operation manual (Section 64):
- Save Full CST models for each successfully reconstructed algorithm
- Include: base_model, Platt_calibrator, feature_order, hyperparameters, training_range
- Save to: 18_handoff\models_for_future_projection\
"""
import sys, os, logging, warnings, json, ast, hashlib
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验3"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment2")
OUT_DIR = os.path.join(ROOT, "18_handoff", "models_for_future_projection")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "save_models.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

RANDOM_SEED = 20260807


def get_model_constructor(model_name):
    """Get base model constructor matching experiment3_core.py."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression

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


def calibrate_model(model, X_train, y_train):
    """Calibrate model with Platt scaling (sklearn ≥1.6 compatible)."""
    from sklearn.calibration import CalibratedClassifierCV
    cal = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=3, n_jobs=-1)
    cal.fit(X_train, y_train)
    return cal


def compute_sha256(filepath):
    """Compute SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return "MISSING"
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    log.info("=" * 60)
    log.info("SAVING CALLABLE MODELS FOR EXPERIMENT 4")
    log.info("=" * 60)

    # Load training data
    train = pd.read_csv(os.path.join(INPUT_DIR, "training_matrix_main.csv"))
    group_map = pd.read_csv(os.path.join(ROOT, "02_group_definition", "environment_group_definition.csv"))
    predictors = group_map["variable"].tolist()
    log.info(f"Training data: {train.shape[0]} rows, {len(predictors)} predictors")
    log.info(f"Predictors: {predictors}")

    # Full CST model features
    cst_vars = predictors
    X_full = train[cst_vars].values.astype(np.float64)
    y_full = train["label"].values.astype(int)

    # Compute training ranges
    ranges = []
    for var in cst_vars:
        vals = train[var].values
        ranges.append({
            'variable': var,
            'min': float(np.nanmin(vals)),
            'p01': float(np.nanpercentile(vals, 1)),
            'p05': float(np.nanpercentile(vals, 5)),
            'median': float(np.nanmedian(vals)),
            'p95': float(np.nanpercentile(vals, 95)),
            'p99': float(np.nanpercentile(vals, 99)),
            'max': float(np.nanmax(vals)),
            'unit': '',
        })

    # Get weight info
    weights = pd.read_csv(os.path.join(INPUT_DIR, "ensemble_weights.csv"))
    w_dict = dict(zip(weights["model"], weights["weight"]))
    total_w = sum(w_dict.values())
    w_norm = {k: v / total_w for k, v in w_dict.items()}

    # Models to save
    models_to_save = ['random_forest', 'xgboost', 'brt', 'maxent']

    manifest = []
    saved_count = 0

    for model_name in models_to_save:
        weight = w_norm.get(model_name, 0)
        log.info(f"\n--- {model_name} (ensemble weight: {weight:.4f}) ---")

        try:
            # Get base model
            base = get_model_constructor(model_name)
            log.info(f"  Base model created")

            # Fit calibrated model on full training data
            cal = calibrate_model(base, X_full, y_full)
            log.info(f"  Calibrated model fitted")

            # Save model bundle
            model_bundle = {
                'model_type': model_name,
                'calibrated_model': cal,
                'feature_order': cst_vars,
                'hyperparameters': str(base.get_params()),
                'training_ranges': ranges,
                'ensemble_weight': weight,
                'random_seed': RANDOM_SEED,
                'creation_date': '2026-08-08',
            }

            out_path = os.path.join(OUT_DIR, f"{model_name}_cst_calibrated.joblib")
            joblib.dump(model_bundle, out_path, compress=3)
            sha = compute_sha256(out_path)
            file_size_mb = os.path.getsize(out_path) / (1024 * 1024)

            log.info(f"  Saved: {out_path}")
            log.info(f"  Size: {file_size_mb:.1f} MB")
            log.info(f"  SHA256: {sha[:16]}...")

            manifest.append({
                'model': model_name,
                'file': f"models_for_future_projection/{model_name}_cst_calibrated.joblib",
                'ensemble_weight': weight,
                'feature_count': len(cst_vars),
                'sha256': sha,
                'size_mb': round(file_size_mb, 2),
            })
            saved_count += 1

        except Exception as e:
            log.error(f"  FAILED: {e}", exc_info=True)
            manifest.append({
                'model': model_name,
                'file': 'FAILED',
                'ensemble_weight': weight,
                'sha256': f'ERROR: {str(e)[:100]}',
            })

    # Save training ranges
    ranges_df = pd.DataFrame(ranges)
    ranges_path = os.path.join(os.path.dirname(OUT_DIR), "current_environment_training_ranges.csv")
    ranges_df.to_csv(ranges_path, index=False)
    log.info(f"\nTraining ranges saved: {ranges_path}")

    # Save training sample for Experiment 4 novelty analysis
    sample_path = os.path.join(os.path.dirname(OUT_DIR), "training_environment_matrix_sample.csv")
    sample_n = min(5000, len(train))
    train[cst_vars + ['label', 'sample_type']].sample(sample_n, random_state=RANDOM_SEED).to_csv(
        sample_path, index=False)
    log.info(f"Training sample ({sample_n} rows) saved: {sample_path}")

    # Save feature order
    feature_order_path = os.path.join(os.path.dirname(OUT_DIR), "feature_order.json")
    with open(feature_order_path, 'w', encoding='utf-8') as f:
        json.dump({'feature_order': cst_vars, 'n_features': len(cst_vars)}, f, indent=2)
    log.info(f"Feature order saved: {feature_order_path}")

    # Save manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(os.path.dirname(OUT_DIR), "models_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    log.info(f"\nManifest saved: {manifest_path}")

    # Summary
    log.info(f"\n{'=' * 60}")
    log.info(f"MODEL SAVING COMPLETE: {saved_count}/{len(models_to_save)} models saved")
    log.info(f"Output directory: {OUT_DIR}")
    log.info(f"Files:")
    for f in os.listdir(OUT_DIR):
        fpath = os.path.join(OUT_DIR, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        log.info(f"  {f} ({size_mb:.1f} MB)")

    return 0 if saved_count >= 2 else 1


if __name__ == "__main__":
    main()
