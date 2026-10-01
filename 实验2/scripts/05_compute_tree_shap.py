#!/usr/bin/env python3
"""
Steps 6-8: TreeSHAP Analysis
Compute SHAP values for tree-based models (RF, XGBoost, BRT).
EXCLUDES MaxEnt (not tree-based).
Outputs SHAP values and global importance per model.
"""
import sys, os, logging, warnings
import numpy as np
import pandas as pd
import joblib
import shap

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验2"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
DATA_DIR = os.path.join(ROOT, "02_analysis_dataset")
OUT_DIR = os.path.join(ROOT, "04_shap")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "shap.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

RANDOM_SEED = 20260807
TREE_MODELS = ["random_forest", "xgboost", "brt"]

def main():
    log.info("=" * 60)
    log.info("TreeSHAP ANALYSIS")
    log.info("=" * 60)

    # Load predictors
    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    predictors = pred_df["variable"].tolist()
    log.info(f"Predictors: {predictors}")

    # Load explainability sample
    sample_path = os.path.join(DATA_DIR, "explainability_sample.csv")
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        log.info(f"Using explainability sample: {len(sample)} rows")
    else:
        sample = pd.read_csv(os.path.join(INPUT_DIR, "training_matrix_main.csv"))
        log.info(f"Using full training data: {len(sample)} rows")

    X_sample = sample[predictors].values.astype(np.float64)

    # Record SHAP scale definition
    shap_scale_md = []
    shap_scale_md.append("# SHAP Scale Definition\n")
    shap_scale_md.append("## SHAP Interpretation Scope\n\n")
    shap_scale_md.append("- **SHAP explains the base model (pre-calibration raw scores/margin)**, not the Platt-scaled calibrated predictions.\n")
    shap_scale_md.append("- **ALE and final prediction-based analyses use the calibrated prediction pipeline** (closest to ensemble suitability output).\n")
    shap_scale_md.append("- SHAP values represent log-odds or raw margin contributions (model-dependent).\n")
    shap_scale_md.append("- Do NOT directly compare SHAP magnitudes with ALE effect magnitudes.\n")
    shap_scale_md.append("- SHAP is used for: (1) variable importance ranking within tree models, (2) direction of effect, (3) interaction detection.\n")
    shap_scale_md.append("- Primary cross-model importance ranking remains **ensemble permutation importance**.\n")

    model_manifest = pd.read_csv(os.path.join(INPUT_DIR, "final_model_manifest.csv"))

    all_shap_importance = []
    all_shap_values = {}

    for _, mrow in model_manifest.iterrows():
        model_name = mrow["model"]
        if model_name == "maxent":
            log.info(f"Skipping {model_name} - not a tree model (TreeSHAP not applicable)")
            shap_scale_md.append(f"\n- **{model_name}**: TreeSHAP not computed (non-tree model).\n")
            continue

        model_path = os.path.join(INPUT_DIR, mrow["file"])
        log.info(f"\nProcessing {model_name}...")

        try:
            model = joblib.load(model_path)
        except Exception as e:
            log.error(f"  Cannot load {model_name}: {e}")
            shap_scale_md.append(f"\n- **{model_name}**: SHAP FAILED - model load error: {e}\n")
            continue

        # Attempt to detect base model vs calibrated wrapper
        # Platt scaling in sklearn often wraps as CalibratedClassifierCV
        base_model = model
        is_calibrated = False
        if hasattr(model, "calibrated_classifiers_"):
            is_calibrated = True
            log.info(f"  Detected CalibratedClassifierCV wrapper - extracting base estimator")
            base_model = model.calibrated_classifiers_[0].base_estimator
        elif hasattr(model, "base_estimator"):
            is_calibrated = True
            log.info(f"  Detected wrapper with base_estimator - extracting")
            base_model = model.base_estimator

        log.info(f"  Model type: {type(base_model).__name__}, calibrated_wrapper: {is_calibrated}")
        shap_scale_md.append(f"\n- **{model_name}**: calibrated_wrapper={is_calibrated}, "
                            f"base_model_type={type(base_model).__name__}, "
                            f"SHAP explains base model.\n")

        try:
            # Try TreeExplainer
            if hasattr(base_model, "get_booster") or "xgboost" in model_name:
                explainer = shap.TreeExplainer(base_model)
            elif hasattr(base_model, "estimators_"):
                # Random Forest
                explainer = shap.TreeExplainer(base_model)
            else:
                log.error(f"  {model_name}: Not a supported tree model for TreeSHAP")
                shap_scale_md.append(f"  TreeSHAP not supported for this model type.\n")
                continue

            # Compute SHAP - use a subset for performance
            n_shap = min(len(X_sample), 2000)
            X_shap = X_sample[:n_shap]

            log.info(f"  Computing SHAP for {n_shap} samples...")
            shap_values = explainer.shap_values(X_shap)

            # Handle multi-class / list output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (presence)

            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

            log.info(f"  SHAP values shape: {shap_values.shape}")

            # Store
            all_shap_values[model_name] = {
                "values": shap_values,
                "samples": X_shap[:n_shap],
                "ids": sample["sample_id"].values[:n_shap],
                "lat": sample["latitude"].values[:n_shap] if "latitude" in sample.columns else np.zeros(n_shap),
                "lon": sample["longitude"].values[:n_shap] if "longitude" in sample.columns else np.zeros(n_shap),
            }

            # Save SHAP values as CSV
            shap_df = pd.DataFrame()
            shap_df["sample_id"] = sample["sample_id"].values[:n_shap]
            if "label" in sample.columns:
                shap_df["label"] = sample["label"].values[:n_shap]
            if "longitude" in sample.columns:
                shap_df["longitude"] = sample["longitude"].values[:n_shap]
                shap_df["latitude"] = sample["latitude"].values[:n_shap]
            for i, p in enumerate(predictors):
                shap_df[f"predictor_value_{p}"] = X_shap[:, i]
                shap_df[f"shap_{p}"] = shap_values[:, i]
            shap_df["model"] = model_name

            csv_path = os.path.join(OUT_DIR, f"{model_name}_shap_values.csv")
            shap_df.to_csv(csv_path, index=False)
            log.info(f"  SHAP values saved to {csv_path}")

            # Also save as parquet for efficiency
            try:
                pq_path = os.path.join(OUT_DIR, f"{model_name}_shap_values.parquet")
                shap_df.to_parquet(pq_path, index=False)
                log.info(f"  SHAP values (parquet) saved to {pq_path}")
            except Exception:
                pass

            # Global SHAP importance
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            total_abs = mean_abs_shap.sum()
            for i, p in enumerate(predictors):
                all_shap_importance.append({
                    "model": model_name,
                    "variable": p,
                    "mean_abs_shap": mean_abs_shap[i],
                    "relative_shap_pct": (mean_abs_shap[i] / total_abs * 100) if total_abs > 0 else 0,
                })

            # Direction analysis
            direction_rows = []
            for i, p in enumerate(predictors):
                high_mask = X_shap[:, i] > np.median(X_shap[:, i])
                low_mask = ~high_mask
                shap_high = shap_values[high_mask, i].mean()
                shap_low = shap_values[low_mask, i].mean()
                # Direction: if high values -> positive SHAP, that's "positive" direction
                direction_rows.append({
                    "variable": p,
                    f"{model_name}_direction": "positive" if shap_high > 0 else "negative",
                    f"{model_name}_shap_high_mean": shap_high,
                    f"{model_name}_shap_low_mean": shap_low,
                })
            # Save direction data
            dir_df = pd.DataFrame(direction_rows)
            # Merge with existing if any
            dir_path = os.path.join(OUT_DIR, "shap_direction_summary.csv")
            if os.path.exists(dir_path):
                existing = pd.read_csv(dir_path)
                for col in dir_df.columns:
                    if col not in existing.columns:
                        existing[col] = dir_df[col]
                dir_df = existing
            dir_df.to_csv(dir_path, index=False)

        except Exception as e:
            log.error(f"  SHAP failed for {model_name}: {e}")
            shap_scale_md.append(f"  SHAP FAILED: {e}\n")
            import traceback
            traceback.print_exc()

    # Save global SHAP importance
    shap_imp_df = pd.DataFrame(all_shap_importance)
    if len(shap_imp_df) > 0:
        imp_path = os.path.join(OUT_DIR, "shap_global_importance_by_model.csv")
        shap_imp_df.to_csv(imp_path, index=False)
        log.info(f"\nSHAP global importance -> {imp_path}")

        # Tree SHAP consensus (average across tree models)
        consensus = shap_imp_df.groupby("variable").agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            mean_relative_pct=("relative_shap_pct", "mean"),
            n_models=("model", "nunique"),
        ).reset_index()
        consensus = consensus.sort_values("mean_relative_pct", ascending=False)
        consensus_path = os.path.join(OUT_DIR, "tree_shap_consensus_importance.csv")
        consensus.to_csv(consensus_path, index=False)
        log.info(f"Tree SHAP consensus -> {consensus_path}")
    else:
        log.warning("No SHAP importance results generated")

    # Write SHAP scale definition
    scale_path = os.path.join(OUT_DIR, "SHAP_SCALE_DEFINITION.md")
    with open(scale_path, "w", encoding="utf-8") as f:
        f.writelines(shap_scale_md)
    log.info(f"SHAP scale definition -> {scale_path}")

    log.info("\nTreeSHAP analysis complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
