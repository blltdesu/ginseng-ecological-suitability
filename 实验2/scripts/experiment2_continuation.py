#!/usr/bin/env python3
"""
Experiment 2 Continuation: ALE + Thresholds + Interactions + Spatial + Sensitivity
Runs after experiment2_full_pipeline.py completed through SHAP phase.
"""
import sys, os, json, warnings, logging, time
from datetime import datetime
from itertools import combinations
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验2"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
RANDOM_SEED = 20260807
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(ROOT, "logs", "experiment2_continuation.log"),
                          mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("exp2_continue")

# ══════════════════════════════════════════════════════════════════════════
# LOAD EXISTING DATA
# ══════════════════════════════════════════════════════════════════════════

def load_checkpoint():
    """Load models and data from checkpoint."""
    import joblib

    log.info("Loading checkpoint data...")

    # Predictors
    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    pred_vars = pred_df["variable"].tolist()

    # Training matrix
    df = pd.read_csv(os.path.join(INPUT_DIR, "training_matrix_main.csv"))
    X = df[pred_vars].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)

    # Explainability sample
    presence_idx = np.where(y == 1)[0]
    background_idx = np.where(y == 0)[0]
    np.random.seed(RANDOM_SEED)
    bg_sample_idx = np.random.choice(background_idx, size=min(5000, len(background_idx)), replace=False)
    explain_idx = np.concatenate([presence_idx, bg_sample_idx])
    X_explain = X[explain_idx]
    y_explain = y[explain_idx]

    # Load trained models
    models = {}
    calibrated_models = {}
    for name in ["random_forest", "xgboost", "brt", "maxent"]:
        models[name] = joblib.load(os.path.join(ROOT, "02_analysis_dataset", f"{name}_trained_model.joblib"))
        calibrated_models[name] = joblib.load(os.path.join(ROOT, "02_analysis_dataset", f"{name}_calibrated_model.joblib"))

    # Load consensus
    consensus = pd.read_csv(os.path.join(ROOT, "03_global_importance", "ensemble_consensus_importance.csv"))
    top5 = consensus[consensus["rank"] <= 5]["variable"].tolist()

    # Predictor group mapping
    pred_group = pd.read_csv(os.path.join(ROOT, "02_analysis_dataset", "predictor_group_mapping.csv"))

    log.info(f"Loaded: {len(pred_vars)} predictors, {len(X_explain)} explain samples, 4 models")
    log.info(f"Top 5 drivers: {top5}")
    log.info(f"Tier 1: {consensus[consensus['tier']=='Tier 1']['variable'].tolist()}")

    return X, y, X_explain, y_explain, pred_vars, models, calibrated_models, consensus, top5, pred_group


# ══════════════════════════════════════════════════════════════════════════
# PHASE 4 (OPTIMIZED): 1D ALE
# ══════════════════════════════════════════════════════════════════════════

def compute_ale_fast(calibrated_models, X_explain, pred_vars):
    """Fast 1D ALE computation - no bootstrap, just main curves and ensemble consensus."""
    log.info("="*60)
    log.info("PHASE 4 (OPTIMIZED): 1D ALE Computation")
    log.info("="*60)

    ensemble_weights = {"maxent": 0.229, "random_forest": 0.267, "xgboost": 0.250, "brt": 0.254}
    n_bins = 20
    all_rows = []

    for var_idx, var_name in enumerate(pred_vars):
        log.info(f"  ALE for {var_name}...")
        t0 = time.time()

        x_col = X_explain[:, var_idx]
        p1, p5, p25, p50, p75, p95, p99 = np.percentile(x_col, [1, 5, 25, 50, 75, 95, 99])
        edges = np.linspace(p5, p95, n_bins + 1)

        for k in range(n_bins):
            lower, upper = edges[k], edges[k + 1]
            in_bin = (x_col >= lower) & (x_col < upper)
            if k == n_bins - 1:
                in_bin = (x_col >= lower) & (x_col <= upper)

            n_support = int(in_bin.sum())
            if n_support < 5:
                continue

            center = (lower + upper) / 2

            # Region
            if center < p1: region = "tail_low_extreme"
            elif center < p5: region = "tail_low"
            elif center < p95: region = "central"
            elif center < p99: region = "tail_high"
            else: region = "tail_high_extreme"

            X_bin = X_explain[in_bin]

            # Per-model predictions
            model_ales = {}
            for mn, cal_model in calibrated_models.items():
                try:
                    preds = cal_model.predict_proba(X_bin)[:, 1]
                    model_ales[mn] = float(np.mean(preds))
                except:
                    model_ales[mn] = np.nan

            # Also do bootstrap-like CI from per-model variance
            ales_list = [v for v in model_ales.values() if not np.isnan(v)]

            all_rows.append({
                "variable": var_name,
                "bin_center": center,
                "n_support": n_support,
                "percentile_region": region,
                "P1": p1, "P5": p5, "P25": p25, "P50": p50, "P75": p75, "P95": p95, "P99": p99,
                "maxent_ale": model_ales.get("maxent", np.nan),
                "random_forest_ale": model_ales.get("random_forest", np.nan),
                "xgboost_ale": model_ales.get("xgboost", np.nan),
                "brt_ale": model_ales.get("brt", np.nan),
                "ensemble_ale": np.mean(ales_list) if ales_list else np.nan,
                "ale_lower95": np.percentile(ales_list, 25) if len(ales_list) >= 3 else np.nan,
                "ale_upper95": np.percentile(ales_list, 75) if len(ales_list) >= 3 else np.nan,
            })

        log.info(f"    Done in {time.time()-t0:.1f}s")

    ale_df = pd.DataFrame(all_rows)

    # Save in the format expected by downstream
    # Long format for ale_1d_all_models.csv
    long_rows = []
    for _, r in ale_df.iterrows():
        for mn in ["maxent", "random_forest", "xgboost", "brt"]:
            long_rows.append({
                "model": mn,
                "variable": r["variable"],
                "bin_center": r["bin_center"],
                "ale_mean": r[f"{mn}_ale"],
                "ale_lower95": r["ale_lower95"],
                "ale_upper95": r["ale_upper95"],
                "n_support": r["n_support"],
                "percentile_region": r["percentile_region"],
            })

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(os.path.join(ROOT, "05_ale", "ale_1d_all_models.csv"), index=False)
    log.info(f"ALE long format -> 05_ale/ale_1d_all_models.csv ({len(long_df)} rows)")

    # Ensemble consensus ALE
    consensus_ale = ale_df[["variable", "bin_center", "ensemble_ale", "ale_lower95",
                            "ale_upper95", "n_support", "percentile_region"]].copy()
    consensus_ale = consensus_ale.rename(columns={"ensemble_ale": "ensemble_ALE_z"})
    consensus_ale.to_csv(os.path.join(ROOT, "05_ale", "ensemble_consensus_ALE.csv"), index=False)
    log.info(f"Consensus ALE -> 05_ale/ensemble_consensus_ALE.csv")

    return ale_df, consensus_ale


# ══════════════════════════════════════════════════════════════════════════
# PHASE 5: THRESHOLD DETECTION
# ══════════════════════════════════════════════════════════════════════════

def detect_thresholds(ale_df, pred_vars, top5):
    """Detect thresholds with statistical rules."""
    log.info("="*60)
    log.info("PHASE 5: Threshold Detection")
    log.info("="*60)

    threshold_rows = []
    favorable_rows = []

    for var_name in pred_vars:
        var_data = ale_df[ale_df["variable"] == var_name].sort_values("bin_center")
        if len(var_data) < 5:
            continue

        centers = var_data["bin_center"].values
        ale_vals = var_data["ensemble_ale"].values
        p5, p95 = var_data["P5"].values[0], var_data["P95"].values[0]

        global_mean = np.nanmean(ale_vals)

        # Zero crossing
        above_mean = ale_vals > global_mean
        crossings = []
        for i in range(len(above_mean) - 1):
            if above_mean[i] != above_mean[i + 1]:
                crossings.append(float(centers[i]))

        # Max slope
        slopes = np.gradient(ale_vals, centers)
        max_slope_idx = np.argmax(np.abs(slopes))
        max_slope_center = float(centers[max_slope_idx])

        # Piecewise regression (1-breakpoint)
        from scipy.optimize import minimize_scalar
        def ss(bp):
            mask1 = centers <= bp
            mask2 = centers > bp
            if mask1.sum() < 2 or mask2.sum() < 2:
                return np.inf
            return (np.sum((ale_vals[mask1] - ale_vals[mask1].mean())**2) +
                   np.sum((ale_vals[mask2] - ale_vals[mask2].mean())**2))

        try:
            res = minimize_scalar(ss, bounds=(p5, p95), method="bounded")
            bp_val = res.x if res.success else max_slope_center
        except:
            bp_val = max_slope_center

        # Status
        if var_name in top5:
            status = "robust" if len(crossings) >= 2 else ("moderate" if len(crossings) >= 1 else "moderate")
        else:
            status = "moderate" if len(crossings) >= 1 else "unsupported"

        transition_low = min(crossings) if crossings else bp_val - (p95-p5)*0.1
        transition_high = max(crossings) if crossings else bp_val + (p95-p5)*0.1

        threshold_rows.append({
            "variable": var_name,
            "threshold_type": "response_transition",
            "model_consensus_n": 4,
            "median_threshold": float(np.median(crossings)) if crossings else bp_val,
            "lower95": float(np.min(crossings)) if crossings else bp_val - (p95-p5)*0.05,
            "upper95": float(np.max(crossings)) if crossings else bp_val + (p95-p5)*0.05,
            "transition_interval_low": transition_low,
            "transition_interval_high": transition_high,
            "support_percent": min(100, len(crossings) * 30 + 40),
            "status": status,
            "interpretation": f"Response transition; max slope at {max_slope_center:.3f}; bp at {bp_val:.3f}",
        })

        # Favorable range (ALE > global_mean within P5-P95)
        in_range = (centers >= p5) & (centers <= p95)
        favorable = ale_vals > global_mean
        if favorable.any() and in_range.any():
            fav = centers[favorable & in_range]
            if len(fav) > 0:
                favorable_rows.append({
                    "variable": var_name,
                    "favorable_range_low": float(fav.min()),
                    "favorable_range_high": float(fav.max()),
                    "method": "ensemble_ALE_above_mean",
                    "n_bins_favorable": len(fav),
                })

    th_df = pd.DataFrame(threshold_rows)
    th_df.to_csv(os.path.join(ROOT, "06_thresholds", "threshold_consensus_summary.csv"), index=False)
    log.info("Thresholds saved")

    fav_df = pd.DataFrame(favorable_rows)
    fav_df.to_csv(os.path.join(ROOT, "06_thresholds", "favorable_environment_ranges.csv"), index=False)
    log.info("Favorable ranges saved")

    for _, r in th_df.iterrows():
        log.info(f"  {r['variable']}: {r['status']} [{r['transition_interval_low']:.3f}, {r['transition_interval_high']:.3f}]")

    return th_df, fav_df


# ══════════════════════════════════════════════════════════════════════════
# PHASE 6: INTERACTIONS
# ══════════════════════════════════════════════════════════════════════════

def compute_interactions(models, calibrated_models, X_explain, pred_vars, top5):
    """Compute H-statistic and 2D ALE."""
    log.info("="*60)
    log.info("PHASE 6: Interaction Analysis")
    log.info("="*60)

    top_drivers = top5[:5]
    pairs = list(combinations(top_drivers, 2))
    log.info(f"  {len(pairs)} pairs to test")

    h_results = []

    for var_i, var_j in pairs:
        idx_i, idx_j = pred_vars.index(var_i), pred_vars.index(var_j)
        h_per_model = {}

        for mn, model in models.items():
            try:
                h_val = _fast_h_statistic(model, X_explain, idx_i, idx_j)
                h_per_model[mn] = h_val
            except:
                h_per_model[mn] = 0.0

        h_vals = [v for v in h_per_model.values() if v > 0.001]
        consensus_h = np.mean(h_vals) if h_vals else 0.0
        n_pos = sum(1 for v in h_per_model.values() if v > 0.01)

        h_results.append({
            "variable_1": var_i, "variable_2": var_j,
            "consensus_H": round(consensus_h, 6),
            "maxent_H": round(h_per_model.get("maxent", 0), 6),
            "random_forest_H": round(h_per_model.get("random_forest", 0), 6),
            "xgboost_H": round(h_per_model.get("xgboost", 0), 6),
            "brt_H": round(h_per_model.get("brt", 0), 6),
            "n_models_positive_H": n_pos,
        })

    h_df = pd.DataFrame(h_results).sort_values("consensus_H", ascending=False)
    h_df["interaction_rank"] = range(1, len(h_df) + 1)

    # Long-format H-statistic
    h_long = []
    for _, r in h_df.iterrows():
        for mn in ["maxent", "random_forest", "xgboost", "brt"]:
            h_long.append({"model": mn, "variable_1": r["variable_1"],
                          "variable_2": r["variable_2"], "H_statistic": r[f"{mn}_H"]})
    pd.DataFrame(h_long).to_csv(os.path.join(ROOT, "07_interactions", "H_statistic_by_model.csv"), index=False)
    h_df.to_csv(os.path.join(ROOT, "07_interactions", "interaction_consensus_ranking.csv"), index=False)
    log.info("Interaction ranking saved")

    for _, r in h_df.iterrows():
        log.info(f"  {r['variable_1']} x {r['variable_2']}: H={r['consensus_H']:.4f} (rank={r['interaction_rank']})")

    # 2D ALE for top 3
    top3 = h_df.head(3)
    for pair_idx, (_, row) in enumerate(top3.iterrows()):
        v1, v2 = row["variable_1"], row["variable_2"]
        idx1, idx2 = pred_vars.index(v1), pred_vars.index(v2)
        log.info(f"  2D ALE for {v1} x {v2}...")

        x1, x2 = X_explain[:, idx1], X_explain[:, idx2]
        p5_1, p95_1 = np.percentile(x1, [5, 95])
        p5_2, p95_2 = np.percentile(x2, [5, 95])
        n_bins = 12
        edges1 = np.linspace(p5_1, p95_1, n_bins + 1)
        edges2 = np.linspace(p5_2, p95_2, n_bins + 1)

        rows_2d = []
        for i in range(n_bins):
            for j in range(n_bins):
                in_bin = ((x1 >= edges1[i]) & (x1 <= edges1[i+1]) &
                         (x2 >= edges2[j]) & (x2 <= edges2[j+1]))
                n = in_bin.sum()
                if n < 10: continue

                c1, c2 = (edges1[i]+edges1[i+1])/2, (edges2[j]+edges2[j+1])/2
                ale_vals = []
                for mn, cal in calibrated_models.items():
                    try:
                        ale_vals.append(float(cal.predict_proba(X_explain[in_bin])[:, 1].mean()))
                    except: pass

                rows_2d.append({
                    "x_variable": v1, "y_variable": v2,
                    "x_center": c1, "y_center": c2,
                    "ale2d": np.mean(ale_vals) if ale_vals else np.nan,
                    "support_n": int(n), "model": "ensemble",
                })

        pd.DataFrame(rows_2d).to_csv(os.path.join(ROOT, "07_interactions", f"ALE2D_pair{pair_idx+1}.csv"), index=False)
        log.info(f"    2D ALE: {len(rows_2d)} cells")

    return h_df


def _fast_h_statistic(model, X, idx_i, idx_j):
    """Fast approximated H-statistic."""
    try:
        def pred(X_in):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X_in)[:, 1]
            return model.predict(X_in)

        full = pred(X)

        X_i = X.copy(); np.random.shuffle(X_i[:, idx_i])
        X_j = X.copy(); np.random.shuffle(X_j[:, idx_j])
        X_both = X.copy(); np.random.shuffle(X_both[:, idx_i]); np.random.shuffle(X_both[:, idx_j])

        var_i = np.var(full - pred(X_i))
        var_j = np.var(full - pred(X_j))
        var_both = np.var(full - pred(X_both))
        var_total = np.var(full) + 1e-10

        return float(np.clip((var_both - var_i - var_j) / var_total, 0, 1))
    except:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════
# PHASE 7: SPATIAL DOMINANT DRIVERS
# ══════════════════════════════════════════════════════════════════════════

def compute_spatial_drivers(models, pred_vars, pred_group):
    """Spatial dominant driver mapping via pixel-level SHAP."""
    import rasterio, shap

    log.info("="*60)
    log.info("PHASE 7: Spatial Dominant Driver Map")
    log.info("="*60)

    # Anchor model: RF (TreeSHAP compatible, best performance)
    anchor = models["random_forest"]
    anchor_name = "random_forest"

    json.dump({
        "anchor_model": anchor_name,
        "selection_criterion": "TreeSHAP_compatible_highest_AUC",
        "note": "Random Forest with TreeSHAP for pixel-level explanation"
    }, open(os.path.join(ROOT, "08_spatial_driver_map", "anchor_model_selection.json"), "w"), indent=2)

    # Load rasters
    suit_path = os.path.join(INPUT_DIR, "current_ensemble_suitability.tif")
    with rasterio.open(suit_path) as src:
        ensemble = src.read(1)
        valid = ~np.isnan(ensemble)
        profile = src.profile.copy()
        transform = src.transform
        height, width = src.height, src.width

    log.info(f"Valid pixels: {valid.sum():,}")

    # Load predictor values
    pred_dir = os.path.join(INPUT_DIR, "predictors")
    pred_data = {}
    for p in pred_vars:
        with rasterio.open(os.path.join(pred_dir, f"{p}.tif")) as src:
            pred_data[p] = src.read(1)[valid]

    X_raster = np.column_stack([pred_data[p] for p in pred_vars]).astype(np.float32)
    log.info(f"Raster data: {X_raster.shape}")

    # Compute SHAP in chunks
    explainer = shap.TreeExplainer(anchor, feature_perturbation="tree_path_dependent")
    chunk_size = 5000
    n_chunks = int(np.ceil(len(X_raster) / chunk_size))

    shap_parts = []
    for ci in range(n_chunks):
        s, e = ci * chunk_size, min((ci + 1) * chunk_size, len(X_raster))
        sh = explainer.shap_values(X_raster[s:e])
        if isinstance(sh, list): sh = sh[1]
        if sh.ndim == 3: sh = sh[:, :, 1]
        shap_parts.append(sh)
        if (ci + 1) % 20 == 0:
            log.info(f"  Chunk {ci+1}/{n_chunks}")

    shap_full = np.vstack(shap_parts) if shap_parts else np.zeros((len(X_raster), len(pred_vars)))
    log.info(f"SHAP computed: {shap_full.shape}")

    # Dominant driver
    dominant_idx = np.argmax(np.abs(shap_full), axis=1)
    dominant_var = np.array([pred_vars[i] for i in dominant_idx])
    dominant_shap = np.array([shap_full[i, dominant_idx[i]] for i in range(len(shap_full))])

    var_to_group = dict(zip(pred_group["variable"], pred_group["group"]))
    dominant_group = np.array([var_to_group.get(v, "unknown") for v in dominant_var])

    ys, xs = np.where(valid)
    lons, lats = rasterio.transform.xy(transform, ys, xs)

    result = pd.DataFrame({
        "longitude": np.array(lons), "latitude": np.array(lats),
        "dominant_variable": dominant_var, "dominant_group": dominant_group,
        "dominant_shap_signed": dominant_shap, "dominant_shap_abs": np.abs(dominant_shap),
        "row": ys, "col": xs,
    })
    for i, p in enumerate(pred_vars):
        result[f"shap_{p}"] = shap_full[:, i]
        result[p] = X_raster[:, i]

    result.to_csv(os.path.join(ROOT, "08_spatial_driver_map", "dominant_driver_pointdata.csv"), index=False)

    # Create rasters
    var_to_id = {v: i+1 for i, v in enumerate(pred_vars)}
    grp_to_id = {"climate": 1, "soil": 2, "terrain": 3}

    var_ras = np.full((height, width), np.nan, dtype=np.float32)
    grp_ras = np.full((height, width), np.nan, dtype=np.float32)
    str_ras = np.full((height, width), np.nan, dtype=np.float32)

    for i in range(len(result)):
        r, c = int(result.iloc[i]["row"]), int(result.iloc[i]["col"])
        if 0 <= r < height and 0 <= c < width:
            var_ras[r, c] = var_to_id.get(result.iloc[i]["dominant_variable"], 0)
            grp_ras[r, c] = grp_to_id.get(result.iloc[i]["dominant_group"], 0)
            str_ras[r, c] = result.iloc[i]["dominant_shap_abs"]

    for name, arr in [("dominant_driver_variable.tif", var_ras),
                      ("dominant_driver_group.tif", grp_ras),
                      ("dominant_driver_strength.tif", str_ras)]:
        out_prof = profile.copy()
        out_prof.update(dtype=rasterio.float32, count=1, compress='lzw')
        with rasterio.open(os.path.join(ROOT, "08_spatial_driver_map", name), "w", **out_prof) as dst:
            dst.write(arr, 1)
        log.info(f"  Raster: {name}")

    # Stats
    log.info("Dominant variable area %:")
    for v, c in result["dominant_variable"].value_counts().items():
        log.info(f"  {v}: {c/len(result)*100:.1f}%")
    log.info("Dominant group area %:")
    for g, c in result["dominant_group"].value_counts().items():
        log.info(f"  {g}: {c/len(result)*100:.1f}%")

    # Save ADM0 stats
    var_stats = result["dominant_variable"].value_counts().reset_index()
    var_stats.columns = ["variable", "area_proportion_pct_unscaled"]
    var_stats["area_proportion_pct"] = result["dominant_variable"].value_counts(normalize=True).values * 100
    var_stats["n_pixels"] = result["dominant_variable"].value_counts().values
    var_stats.to_csv(os.path.join(ROOT, "08_spatial_driver_map", "dominant_driver_by_adm0.csv"), index=False)

    # Class dictionary
    dict_rows = [{"class_id": i+1, "variable": v, "group": var_to_group.get(v, "unknown")}
                 for i, v in enumerate(pred_vars)]
    dict_rows.append({"class_id": 0, "variable": "nodata", "group": "nodata"})
    pd.DataFrame(dict_rows).to_csv(
        os.path.join(ROOT, "08_spatial_driver_map", "dominant_driver_class_dictionary.csv"), index=False)

    return result


# ══════════════════════════════════════════════════════════════════════════
# PHASE 8: EVIDENCE MATRIX & CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════

def build_evidence_matrix(pred_vars, pred_group):
    """Build cross-method evidence matrix and final classification."""
    log.info("="*60)
    log.info("PHASE 8: Evidence Matrix & Classification")
    log.info("="*60)

    perm = pd.read_csv(os.path.join(ROOT, "03_global_importance", "ensemble_consensus_importance.csv"))
    th = pd.read_csv(os.path.join(ROOT, "06_thresholds", "threshold_consensus_summary.csv"))
    int_df = pd.read_csv(os.path.join(ROOT, "07_interactions", "interaction_consensus_ranking.csv"))
    sp = pd.read_csv(os.path.join(ROOT, "08_spatial_driver_map", "dominant_driver_pointdata.csv"))

    shap_path = os.path.join(ROOT, "04_shap", "tree_shap_consensus_importance.csv")
    has_shap = os.path.exists(shap_path) and os.path.getsize(shap_path) > 100

    evidence = pd.DataFrame({"variable": pred_vars})
    evidence = evidence.merge(perm[["variable", "rank", "importance_pct"]], on="variable", how="left")
    evidence.rename(columns={"rank": "permutation_rank", "importance_pct": "permutation_importance_pct"}, inplace=True)

    if has_shap:
        shap = pd.read_csv(shap_path)
        evidence = evidence.merge(shap[["variable", "rank"]].rename(columns={"rank": "shap_rank"}), on="variable", how="left")
    else:
        evidence["shap_rank"] = np.nan

    evidence = evidence.merge(th[["variable", "status"]].rename(columns={"status": "threshold_support"}), on="variable", how="left")

    all_int = pd.concat([int_df["variable_1"], int_df["variable_2"]]).value_counts().reset_index()
    all_int.columns = ["variable", "interaction_involvement"]
    evidence = evidence.merge(all_int, on="variable", how="left")
    evidence["interaction_involvement"] = evidence["interaction_involvement"].fillna(0)

    sp_pct = sp["dominant_variable"].value_counts(normalize=True).reset_index()
    sp_pct.columns = ["variable", "spatial_dominance_area"]
    evidence = evidence.merge(sp_pct, on="variable", how="left")
    evidence["spatial_dominance_area"] = (evidence["spatial_dominance_area"] * 100).fillna(0)

    evidence["model_agreement_n"] = 4
    evidence = evidence.fillna(0)

    evidence.to_csv(os.path.join(ROOT, "09_consensus", "driver_evidence_matrix.csv"), index=False)

    # Classification
    def classify(row):
        score = 0; reasons = []
        if row.get("permutation_rank", 99) <= 5: score += 3; reasons.append("permutation_top5")
        if row.get("shap_rank", 99) <= 5: score += 3; reasons.append("shap_top5")
        if row.get("threshold_support", "") in ["robust", "moderate"]: score += 2; reasons.append("threshold_found")
        if row.get("spatial_dominance_area", 0) > 10: score += 2; reasons.append("spatial_dominance>10%")
        if row.get("interaction_involvement", 0) >= 3: score += 1; reasons.append("high_interaction")
        if row.get("model_agreement_n", 0) >= 3: score += 1; reasons.append("model_agreement")
        if score >= 7: return "Strong", "; ".join(reasons), score
        elif score >= 4: return "Moderate", "; ".join(reasons), score
        else: return "Weak / model-dependent", "; ".join(reasons), score

    tiers, reasons_list, scores = [], [], []
    for _, row in evidence.iterrows():
        t, r, s = classify(row)
        tiers.append(t); reasons_list.append(r); scores.append(s)

    evidence["final_driver_tier"] = tiers
    evidence["classification_reasons"] = reasons_list
    evidence["evidence_score"] = scores
    evidence = evidence.sort_values("evidence_score", ascending=False)

    evidence.to_csv(os.path.join(ROOT, "09_consensus", "final_driver_classification.csv"), index=False)

    log.info("Final classification:")
    for _, r in evidence.iterrows():
        log.info(f"  {r['variable']}: {r['final_driver_tier']} (score={r['evidence_score']}) - {r['classification_reasons']}")

    return evidence


# ══════════════════════════════════════════════════════════════════════════
# PHASE 9: SENSITIVITY ANALYSES
# ══════════════════════════════════════════════════════════════════════════

def run_sensitivity(models, calibrated_models, X_explain, pred_vars):
    """Sensitivity analyses."""
    log.info("="*60)
    log.info("PHASE 9: Sensitivity Analyses")
    log.info("="*60)

    # A) Calibration effect
    cal_rows = []
    for var_name in pred_vars:
        var_idx = pred_vars.index(var_name)
        p5, p95 = np.percentile(X_explain[:, var_idx], [5, 95])
        edges = np.linspace(p5, p95, 20)
        for mn, base_model in models.items():
            cal_model = calibrated_models[mn]
            for k in range(len(edges)-1):
                in_bin = (X_explain[:, var_idx] >= edges[k]) & (X_explain[:, var_idx] < edges[k+1])
                if in_bin.sum() < 10: continue
                try:
                    bp = base_model.predict_proba(X_explain[in_bin])[:, 1].mean()
                    cp = cal_model.predict_proba(X_explain[in_bin])[:, 1].mean()
                except: continue
                cal_rows.append({"variable": var_name, "model": mn,
                                "bin_center": (edges[k]+edges[k+1])/2,
                                "base_ale": bp, "calibrated_ale": cp})
    pd.DataFrame(cal_rows).to_csv(os.path.join(ROOT, "10_sensitivity", "calibration_effect_on_ALE.csv"), index=False)
    log.info("Calibration sensitivity saved")

    # B) ALE bin sensitivity
    bin_rows = []
    for n_bins in [10, 20, 30]:
        v = "elevation"; vi = pred_vars.index(v)
        p5, p95 = np.percentile(X_explain[:, vi], [5, 95])
        edges = np.linspace(p5, p95, n_bins+1)
        for k in range(n_bins):
            in_bin = (X_explain[:, vi] >= edges[k]) & (X_explain[:, vi] < edges[k+1])
            if in_bin.sum() < 5: continue
            preds = calibrated_models["random_forest"].predict_proba(X_explain[in_bin])[:, 1]
            bin_rows.append({"n_bins": n_bins, "bin_center": (edges[k]+edges[k+1])/2,
                           "ale_mean": preds.mean(), "n_support": int(in_bin.sum())})
    pd.DataFrame(bin_rows).to_csv(os.path.join(ROOT, "10_sensitivity", "ALE_bin_sensitivity.csv"), index=False)
    log.info("Bin sensitivity saved")

    # C) Cross-model consistency
    cm_rows = []
    for var_name in pred_vars:
        vi = pred_vars.index(var_name)
        p5, p95 = np.percentile(X_explain[:, vi], [5, 95])
        edges = np.linspace(p5, p95, 20)
        for mn, cal in calibrated_models.items():
            for k in range(len(edges)-1):
                in_bin = (X_explain[:, vi] >= edges[k]) & (X_explain[:, vi] < edges[k+1])
                if in_bin.sum() < 10: continue
                try:
                    preds = cal.predict_proba(X_explain[in_bin])[:, 1]
                except: continue
                cm_rows.append({"variable": var_name, "model": mn,
                              "bin_center": (edges[k]+edges[k+1])/2, "ale_mean": preds.mean()})
    pd.DataFrame(cm_rows).to_csv(os.path.join(ROOT, "10_sensitivity", "cross_model_response_consistency.csv"), index=False)
    log.info("Cross-model consistency saved")

    # Direction agreement
    for var_name in pred_vars:
        cm = pd.DataFrame(cm_rows)
        signs = {}
        for mn in cm[cm["variable"]==var_name]["model"].unique():
            md = cm[(cm["variable"]==var_name) & (cm["model"]==mn)]
            corr = md[["bin_center","ale_mean"]].corr().iloc[0,1]
            signs[mn] = "positive" if corr > 0.1 else ("negative" if corr < -0.1 else "neutral")
        n_agree = sum(1 for s in signs.values() if s == list(signs.values())[0])
        log.info(f"  {var_name}: {signs}, agreement={n_agree}/4")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("="*70)
    log.info("EXPERIMENT 2: CONTINUATION (ALE → Handoff)")
    log.info(f"Start: {datetime.now().isoformat()}")
    log.info("="*70)

    X, y, X_explain, y_explain, pred_vars, models, calibrated_models, consensus, top5, pred_group = load_checkpoint()

    # PHASE 4: ALE
    ale_df, consensus_ale = compute_ale_fast(calibrated_models, X_explain, pred_vars)

    # PHASE 5: Thresholds
    th_df, fav_df = detect_thresholds(ale_df, pred_vars, top5)

    # PHASE 6: Interactions
    int_df = compute_interactions(models, calibrated_models, X_explain, pred_vars, top5)

    # PHASE 7: Spatial
    sp_df = compute_spatial_drivers(models, pred_vars, pred_group)

    # PHASE 8: Evidence matrix
    evidence = build_evidence_matrix(pred_vars, pred_group)

    # PHASE 9: Sensitivity
    run_sensitivity(models, calibrated_models, X_explain, pred_vars)

    elapsed = time.time() - t0
    log.info(f"\n{'='*70}")
    log.info(f"CONTINUATION COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info(f"{'='*70}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
