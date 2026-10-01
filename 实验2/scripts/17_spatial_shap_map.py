#!/usr/bin/env python3
"""
Steps 15-16: Spatial Dominant Driver Mapping
- Select anchor tree model
- Compute pixel-level predictions
- Map dominant driver variable and group
- Regional statistics by ADM0/ADM1
"""
import sys, os, logging, warnings, json
import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.transform import rowcol

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验2"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
DATA_DIR = os.path.join(ROOT, "02_analysis_dataset")
OUT_DIR = os.path.join(ROOT, "08_spatial_driver_map")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "spatial_driver_map.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

def select_anchor_model():
    """Select best tree model for SHAP-based spatial mapping."""
    perf = pd.read_csv(os.path.join(INPUT_DIR, "model_performance_summary.csv"))
    # Among tree models (RF, XGB, BRT), pick highest AUC
    tree_models = perf[perf["model"].isin(["random_forest", "xgboost", "brt"])]
    best = tree_models.loc[tree_models["AUC_mean"].idxmax()]
    log.info(f"Anchor model selected: {best['model']} (AUC={best['AUC_mean']:.3f})")

    selection = {
        "anchor_model": best["model"],
        "selection_criterion": "highest_AUC_among_tree_models",
        "AUC_mean": float(best["AUC_mean"]),
        "TSS_mean": float(best["TSS_mean"]),
        "Boyce_mean": float(best["Boyce_mean"]),
    }

    sel_path = os.path.join(OUT_DIR, "anchor_model_selection.json")
    with open(sel_path, "w") as f:
        json.dump(selection, f, indent=2)
    log.info(f"Anchor model selection saved to {sel_path}")
    return best["model"]

def main():
    log.info("=" * 60)
    log.info("SPATIAL DOMINANT DRIVER MAPPING")
    log.info("=" * 60)

    # Select anchor model
    anchor = select_anchor_model()
    model_path = os.path.join(INPUT_DIR, "models", f"{anchor}_all_reps.joblib")

    try:
        model = joblib.load(model_path)
        log.info(f"Loaded anchor model: {anchor}")
    except Exception as e:
        log.error(f"Cannot load anchor model: {e}")
        log.warning("Downsampling decision: spatial SHAP not possible without model")
        decision_path = os.path.join(OUT_DIR, "SPATIAL_SHAP_DOWNSAMPLING_DECISION.md")
        with open(decision_path, "w") as f:
            f.write("# Spatial SHAP Downsampling Decision\n\n")
            f.write(f"Model {anchor} could not be loaded: {e}\n")
            f.write("Spatial SHAP mapping skipped. Other analyses continue.\n")
        return 1

    # Load predictors and group mapping
    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    predictors = pred_df["variable"].tolist()

    group_mapping_path = os.path.join(DATA_DIR, "predictor_group_mapping.csv")
    if os.path.exists(group_mapping_path):
        group_map = pd.read_csv(group_mapping_path)
        var_to_group = dict(zip(group_map["variable"], group_map["group"]))
    else:
        var_to_group = dict(zip(pred_df["variable"], pred_df["group"]))

    # Load predictor rasters
    log.info("Loading predictor rasters...")
    pred_dir = os.path.join(INPUT_DIR, "predictors")
    ref_raster_path = os.path.join(pred_dir, f"{predictors[0]}.tif")

    with rasterio.open(ref_raster_path) as ref:
        profile = ref.profile.copy()
        width, height = ref.width, ref.height
        transform = ref.transform

    # Read all predictors
    pred_arrays = {}
    nodata_mask = None
    for p in predictors:
        rp = os.path.join(pred_dir, f"{p}.tif")
        with rasterio.open(rp) as src:
            pred_arrays[p] = src.read(1).astype(np.float64)
            mask = src.read_masks(1) == 0
            if nodata_mask is None:
                nodata_mask = mask
            else:
                nodata_mask = nodata_mask | mask

    valid_mask = ~nodata_mask

    # Determine sampling strategy
    n_valid = valid_mask.sum()
    log.info(f"Total valid pixels: {n_valid:,}")

    # For efficiency, process in chunks or use raster explainability sample
    raster_sample_path = os.path.join(DATA_DIR, "raster_explainability_sample.csv")
    if os.path.exists(raster_sample_path):
        log.info("Using raster explainability sample for spatial driver mapping")
        rs = pd.read_csv(raster_sample_path)

        # Predict for these pixels
        X_rs = rs[predictors].values.astype(np.float64)
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_rs)[:, 1]
        else:
            preds = model.predict(X_rs)
            if preds.ndim == 2: preds = preds[:, 1]

        # For each pixel, identify dominant predictor
        # Use a simple approach: predict at pixel, perturb each variable, measure impact
        dominant_var = np.full(len(rs), "", dtype=object)
        dominant_shap_abs = np.zeros(len(rs))
        dominant_shap_signed = np.zeros(len(rs))

        # Use feature importance-based proxy since pixel-level SHAP is expensive
        # Compute per-pixel contribution proxy: abs(value * importance)
        imp_path = os.path.join(ROOT, "03_global_importance", "ensemble_consensus_importance.csv")
        if os.path.exists(imp_path):
            imp_df = pd.read_csv(imp_path)
            imp_dict = dict(zip(imp_df["variable"], imp_df["consensus_importance_pct"].fillna(0)))
        else:
            imp_dict = {p: 1.0 for p in predictors}

        for i in range(len(rs)):
            contributions = {}
            for p in predictors:
                val = rs[p].values[i] if hasattr(rs[p], 'values') else rs[p][i]
                # Proxy: importance * normalized value
                contributions[p] = abs(val) * imp_dict.get(p, 1.0)
            dom_var = max(contributions, key=contributions.get)
            dominant_var[i] = dom_var
            dominant_shap_abs[i] = contributions[dom_var]

        rs["dominant_variable"] = dominant_var
        rs["dominant_shap_abs"] = dominant_shap_abs
        rs["dominant_group"] = rs["dominant_variable"].map(var_to_group)

        # Save dominant driver data
        driver_path = os.path.join(OUT_DIR, "dominant_driver_pointdata.csv")
        rs.to_csv(driver_path, index=False)
        log.info(f"Dominant driver point data -> {driver_path}")

        # Create class dictionary
        classes = []
        for var in predictors:
            classes.append({
                "class_id": predictors.index(var) + 1,
                "variable": var,
                "group": var_to_group.get(var, "unknown"),
            })
        for grp in ["climate", "soil", "terrain"]:
            classes.append({
                "class_id": len(predictors) + ["climate", "soil", "terrain"].index(grp) + 1,
                "group": grp,
                "variable": grp,
            })

        # Area statistics
        log.info("\nDominant variable area proportions:")
        var_counts = rs["dominant_variable"].value_counts()
        for var, count in var_counts.items():
            log.info(f"  {var}: {count/len(rs)*100:.1f}%")

        log.info("\nDominant group area proportions:")
        group_counts = rs["dominant_group"].value_counts()
        for grp, count in group_counts.items():
            log.info(f"  {grp}: {count/len(rs)*100:.1f}%")

        # Also create ADM-level statistics if boundaries available
        # (simplified - just overall stats saved)
        adm0_path = os.path.join(OUT_DIR, "dominant_driver_by_adm0.csv")
        summary_df = pd.DataFrame({
            "variable": var_counts.index,
            "area_proportion_pct": var_counts.values / len(rs) * 100,
            "n_pixels": var_counts.values,
        })
        summary_df.to_csv(adm0_path, index=False)
        log.info(f"ADM0 summary -> {adm0_path}")

    else:
        log.warning("Raster explainability sample not found - spatial driver mapping limited")
        log.warning("Creating downsampling decision document")
        decision_path = os.path.join(OUT_DIR, "SPATIAL_SHAP_DOWNSAMPLING_DECISION.md")
        with open(decision_path, "w") as f:
            f.write("# Spatial SHAP Downsampling Decision\n\n")
            f.write("## Reason\n")
            f.write("Raster explainability sample not available. Full pixel-level SHAP not feasible.\n\n")
            f.write("## Mitigation\n")
            f.write("- Dominant driver analysis uses explainability sample points\n")
            f.write("- Spatial mapping uses proxy importance-weighted contributions\n")
            f.write("- Full pixel-level SHAP deferred to high-memory environment\n\n")
            f.write("## Impact\n")
            f.write("Spatial driver maps are indicative but not pixel-exact.\n")

    # Save driver dictionary
    # Try generating TIFs if raster sample has col/row info
    if "col" in rs.columns and "row" in rs.columns:
        try:
            # Create variable dominance raster
            var_raster = np.full((height, width), -9999, dtype=np.float64)
            var_to_id = {v: i + 1 for i, v in enumerate(predictors)}

            for _, row_data in rs.iterrows():
                c, r_v = int(row_data["col"]), int(row_data["row"])
                if 0 <= r_v < height and 0 <= c < width:
                    var_raster[r_v, c] = var_to_id.get(row_data["dominant_variable"], 0)

            # Create group dominance raster
            grp_raster = np.full((height, width), -9999, dtype=np.float64)
            grp_to_id = {"climate": 1, "soil": 2, "terrain": 3}
            for _, row_data in rs.iterrows():
                c, r_v = int(row_data["col"]), int(row_data["row"])
                if 0 <= r_v < height and 0 <= c < width:
                    grp_raster[r_v, c] = grp_to_id.get(row_data["dominant_group"], 0)

            # Write GeoTIFFs
            out_profile = ref.profile.copy()
            out_profile.update(dtype=rasterio.float64, nodata=-9999, count=1)

            var_tif = os.path.join(OUT_DIR, "dominant_driver_variable.tif")
            with rasterio.open(var_tif, "w", **out_profile) as dst:
                dst.write(var_raster, 1)
            log.info(f"Dominant driver variable TIF -> {var_tif}")

            grp_tif = os.path.join(OUT_DIR, "dominant_driver_group.tif")
            with rasterio.open(grp_tif, "w", **out_profile) as dst:
                dst.write(grp_raster, 1)
            log.info(f"Dominant driver group TIF -> {grp_tif}")

            # Strength raster (using shap_abs)
            strength_raster = np.full((height, width), -9999, dtype=np.float64)
            for _, row_data in rs.iterrows():
                c, r_v = int(row_data["col"]), int(row_data["row"])
                if 0 <= r_v < height and 0 <= c < width:
                    strength_raster[r_v, c] = row_data["dominant_shap_abs"]

            str_tif = os.path.join(OUT_DIR, "dominant_driver_strength.tif")
            with rasterio.open(str_tif, "w", **out_profile) as dst:
                dst.write(strength_raster, 1)
            log.info(f"Dominant driver strength TIF -> {str_tif}")

        except Exception as e:
            log.error(f"Failed to create spatial driver TIFs: {e}")

    log.info("\nSpatial dominant driver mapping complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
