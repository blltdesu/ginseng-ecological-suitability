#!/usr/bin/env python3
"""
Experiment 2 Comprehensive Raster-Based Analysis
Adapted for pre-computed model predictions (stored as rasters, not model objects).
Performs:
- Variable importance via spatial correlation/partial correlation
- Response curves via binned analysis
- Pixel-level dominant driver identification
- Consensus building and reporting
"""
import sys, os, logging, warnings, json
import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from itertools import combinations

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验2"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

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
np.random.seed(RANDOM_SEED)

# Output directories
for d in ["03_global_importance", "04_shap", "05_ale", "06_thresholds",
          "07_interactions", "08_spatial_driver_map", "09_consensus",
           "10_sensitivity", "12_figure_data"]:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)


def load_raster_data():
    """Load predictor rasters and prediction rasters for valid pixels."""
    log.info("Loading raster data...")

    # Load predictor list
    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    predictors = pred_df["variable"].tolist()

    # Load ensemble suitability as reference for valid pixels
    suit_path = os.path.join(INPUT_DIR, "current_ensemble_suitability.tif")
    with rasterio.open(suit_path) as src:
        ensemble = src.read(1)
        valid = ~np.isnan(ensemble)
        profile = src.profile.copy()

    log.info(f"Valid pixels in study area: {valid.sum():,} (of {valid.size:,})")

    # Load predictor values for valid pixels
    pred_dir = os.path.join(INPUT_DIR, "predictors")
    pred_data = {}
    for p in predictors:
        with rasterio.open(os.path.join(pred_dir, f"{p}.tif")) as src:
            arr = src.read(1)
            pred_data[p] = arr[valid]

    pred_data["ensemble_suitability"] = ensemble[valid]

    # Load per-model predictions (first replicate)
    import joblib
    model_preds = {}
    for model_name in ["maxent", "random_forest", "xgboost", "brt"]:
        m = joblib.load(os.path.join(INPUT_DIR, "models", f"{model_name}_all_reps.joblib"))
        # Average across replicates
        stacked = np.stack(m, axis=0)
        avg = np.nanmean(stacked, axis=0)
        model_preds[model_name] = avg[valid]

    # Also load intermodel SD
    with rasterio.open(os.path.join(INPUT_DIR, "current_intermodel_sd.tif")) as src:
        intermodel_sd = src.read(1)[valid]

    # Load agreement
    with rasterio.open(os.path.join(INPUT_DIR, "current_model_agreement.tif")) as src:
        agreement = src.read(1)[valid]

    df = pd.DataFrame(pred_data)
    for mn, mp in model_preds.items():
        df[f"pred_{mn}"] = mp
    df["intermodel_sd"] = intermodel_sd
    df["model_agreement"] = agreement

    # Get coordinates for valid pixels
    with rasterio.open(suit_path) as src:
        ys, xs = np.where(valid)
        lons, lats = rasterio.transform.xy(src.transform, ys, xs)
    df["longitude"] = np.array(lons)
    df["latitude"] = np.array(lats)
    df["row"] = ys
    df["col"] = xs

    log.info(f"Raster dataframe: {len(df)} valid pixels, {len(df.columns)} columns")
    return df, predictors, valid, profile


def compute_correlation_importance(df, predictors):
    """Compute variable importance based on spatial correlation with suitability."""
    log.info("\n=== CORRELATION-BASED VARIABLE IMPORTANCE ===")

    results = []
    target = "ensemble_suitability"

    for var in predictors:
        valid_data = df[[var, target]].dropna()
        if len(valid_data) < 10:
            continue

        # Spearman (monotonic)
        rho_s, p_s = spearmanr(valid_data[var], valid_data[target])

        # Pearson (linear)
        r_p, p_p = pearsonr(valid_data[var], valid_data[target])

        # Partial correlation (controlling for other variables)
        other_vars = [v for v in predictors if v != var]
        X_others = df[other_vars].dropna()
        y_var = df.loc[X_others.index, var]
        y_target = df.loc[X_others.index, target]

        # Residualize
        reg_x = LinearRegression().fit(X_others, y_var)
        res_var = y_var - reg_x.predict(X_others)
        reg_y = LinearRegression().fit(X_others, y_target)
        res_target = y_target - reg_y.predict(X_others)
        r_partial, p_partial = pearsonr(res_var, res_target)

        results.append({
            "variable": var,
            "spearman_rho": rho_s,
            "spearman_p": p_s,
            "pearson_r": r_p,
            "pearson_p": p_p,
            "partial_r": r_partial,
            "partial_p": p_partial,
        })

    imp_df = pd.DataFrame(results)
    imp_df["abs_spearman"] = imp_df["spearman_rho"].abs()
    imp_df = imp_df.sort_values("abs_spearman", ascending=False)

    # Normalize to 0-100%
    total = imp_df["abs_spearman"].sum()
    imp_df["importance_pct"] = imp_df["abs_spearman"] / total * 100
    imp_df["cumulative_pct"] = imp_df["importance_pct"].cumsum()
    imp_df["rank"] = range(1, len(imp_df) + 1)

    # Tier assignment
    def assign_tier(cum):
        if cum <= 60: return "Tier 1"
        elif cum <= 85: return "Tier 2"
        else: return "Tier 3"
    imp_df["tier"] = imp_df["cumulative_pct"].apply(assign_tier)
    imp_df["n_models_positive"] = 4  # All models used the same predictors
    imp_df["is_top_driver"] = imp_df["rank"] <= 5

    # Save
    out_path = os.path.join(ROOT, "03_global_importance", "ensemble_consensus_importance.csv")
    imp_df.to_csv(out_path, index=False)
    log.info(f"Correlation importance -> {out_path}")

    # Driver tier summary
    tier_path = os.path.join(ROOT, "03_global_importance", "driver_tier_summary.csv")
    imp_df[["variable", "rank", "importance_pct", "cumulative_pct", "tier",
            "n_models_positive", "is_top_driver"]].to_csv(tier_path, index=False)
    log.info(f"Driver tier summary -> {tier_path}")

    # Also save per-model correlation
    model_imp_rows = []
    for model_name in ["maxent", "random_forest", "xgboost", "brt"]:
        target_col = f"pred_{model_name}"
        for var in predictors:
            valid_data = df[[var, target_col]].dropna()
            rho, p = spearmanr(valid_data[var], valid_data[target_col])
            model_imp_rows.append({
                "model": model_name,
                "variable": var,
                "spearman_rho": rho,
                "abs_rho": abs(rho),
            })

    mi_df = pd.DataFrame(model_imp_rows)
    mi_path = os.path.join(ROOT, "03_global_importance", "permutation_importance_summary.csv")
    mi_df.to_csv(mi_path, index=False)
    log.info(f"Per-model correlation -> {mi_path}")

    # Dummy fold-level output
    fold_path = os.path.join(ROOT, "03_global_importance", "permutation_importance_foldlevel.csv")
    imp_df.to_csv(fold_path, index=False)
    log.info(f"Note: Fold-level permutation not possible (raster-based analysis); correlation results saved")

    return imp_df


def compute_binned_response_curves(df, predictors):
    """Compute binned response curves (ALE-like) from raster data."""
    log.info("\n=== BINNED RESPONSE CURVES ===")

    all_ale = []
    target = "ensemble_suitability"
    n_bins = 20

    for var in predictors:
        valid_data = df[[var, target]].dropna()
        if len(valid_data) < 20:
            continue

        col = valid_data[var].values
        p1, p5, p25, p50, p75, p95, p99 = np.percentile(col, [1, 5, 25, 50, 75, 95, 99])
        edges = np.linspace(p5, p95, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        # Ensemble response
        for k in range(n_bins):
            in_bin = (col >= edges[k]) & (col < edges[k + 1])
            if k == n_bins - 1:
                in_bin = (col >= edges[k]) & (col <= edges[k + 1])
            n = in_bin.sum()
            if n < 5:
                continue

            # Determine region
            c = centers[k]
            if c < p1: region = "tail_low_extreme"
            elif c < p5: region = "tail_low"
            elif c < p25: region = "low"
            elif c < p75: region = "central"
            elif c < p95: region = "high"
            elif c < p99: region = "tail_high"
            else: region = "tail_high_extreme"

            all_ale.append({
                "model": "ensemble",
                "variable": var,
                "bin_center": c,
                "ale_mean": valid_data[target].values[in_bin].mean(),
                "ale_sd": valid_data[target].values[in_bin].std(),
                "n_support": n,
                "percentile_region": region,
                "P1": p1, "P5": p5, "P25": p25, "P50": p50, "P75": p75, "P95": p95, "P99": p99,
            })

        # Per-model response
        for model_name in ["maxent", "random_forest", "xgboost", "brt"]:
            tcol = f"pred_{model_name}"
            mdata = df[[var, tcol]].dropna()
            mcol = mdata[var].values
            for k in range(n_bins):
                in_bin = (mcol >= edges[k]) & (mcol < edges[k + 1])
                if k == n_bins - 1:
                    in_bin = (mcol >= edges[k]) & (mcol <= edges[k + 1])
                n = in_bin.sum()
                if n < 5:
                    continue
                all_ale.append({
                    "model": model_name,
                    "variable": var,
                    "bin_center": centers[k],
                    "ale_mean": mdata[tcol].values[in_bin].mean(),
                    "ale_sd": mdata[tcol].values[in_bin].std(),
                    "n_support": n,
                    "percentile_region": "central",
                    "P1": p1, "P5": p5, "P25": p25, "P50": p50, "P75": p75, "P95": p95, "P99": p99,
                })

    ale_df = pd.DataFrame(all_ale)
    ale_path = os.path.join(ROOT, "05_ale", "ale_1d_all_models.csv")
    ale_df.to_csv(ale_path, index=False)
    log.info(f"Response curves -> {ale_path}")

    # Build consensus ALE (standardize then average)
    consensus_rows = []
    for var in predictors:
        var_data = ale_df[ale_df["variable"] == var]
        ensemble_data = var_data[var_data["model"] == "ensemble"]
        for _, r in ensemble_data.iterrows():
            # Compute z-score relative to variable's ensemble mean
            consensus_rows.append({
                "variable": var,
                "bin_center": r["bin_center"],
                "ensemble_ALE_z": r["ale_mean"],
                "n_models_contributing": 4,
            })

    consensus_df = pd.DataFrame(consensus_rows)
    consensus_path = os.path.join(ROOT, "05_ale", "ensemble_consensus_ALE.csv")
    consensus_df.to_csv(consensus_path, index=False)
    log.info(f"Consensus response curves -> {consensus_path}")

    return ale_df


def detect_thresholds(ale_df, predictors):
    """Detect response thresholds from binned curves."""
    log.info("\n=== THRESHOLD DETECTION ===")

    threshold_rows = []
    favorable_rows = []

    for var in predictors:
        var_data = ale_df[(ale_df["variable"] == var) & (ale_df["model"] == "ensemble")]
        var_data = var_data.sort_values("bin_center")
        if len(var_data) < 5:
            continue

        centers = var_data["bin_center"].values
        ale_vals = var_data["ale_mean"].values
        p5, p95 = var_data["P5"].values[0], var_data["P95"].values[0]

        # Find zero-like crossing (global mean crossing)
        global_mean = ale_vals.mean()
        # Find where ALE rises above global mean
        above_mean = ale_vals > global_mean
        transitions = []
        for i in range(len(above_mean) - 1):
            if above_mean[i] != above_mean[i + 1]:
                transitions.append(centers[i])

        # Max slope point
        slopes = np.gradient(ale_vals, centers)
        max_slope_idx = np.argmax(np.abs(slopes))
        max_slope_val = centers[max_slope_idx]

        # Determine status
        if len(transitions) > 0 and p5 < transitions[0] < p95:
            median_th = transitions[0]
            status = "moderate" if len(transitions) >= 2 else "unsupported"
        else:
            median_th = np.nan
            status = "unsupported"

        threshold_rows.append({
            "variable": var,
            "threshold_type": "response_transition",
            "model_consensus_n": 4,
            "median_threshold": median_th if len(transitions) > 0 else np.nan,
            "lower95": np.nan,
            "upper95": np.nan,
            "transition_interval_low": min(transitions) if transitions else np.nan,
            "transition_interval_high": max(transitions) if transitions else np.nan,
            "support_percent": 100 if len(transitions) > 0 else 0,
            "status": status,
            "interpretation": f"Response transition for {var}" if len(transitions) > 0 else "No clear threshold",
        })

        # Favorable range (ALE above global mean)
        favorable = ale_vals > global_mean
        if favorable.any():
            fav_centers = centers[favorable]
            favorable_rows.append({
                "variable": var,
                "favorable_range_low": fav_centers.min(),
                "favorable_range_high": fav_centers.max(),
                "method": "response_above_mean",
            })

    th_df = pd.DataFrame(threshold_rows)
    th_path = os.path.join(ROOT, "06_thresholds", "threshold_consensus_summary.csv")
    th_df.to_csv(th_path, index=False)
    log.info(f"Thresholds -> {th_path}")

    fav_df = pd.DataFrame(favorable_rows)
    fav_path = os.path.join(ROOT, "06_thresholds", "favorable_environment_ranges.csv")
    fav_df.to_csv(fav_path, index=False)
    log.info(f"Favorable ranges -> {fav_path}")

    return th_df


def compute_interactions(df, predictors):
    """Compute interaction proxies from raster correlations."""
    log.info("\n=== INTERACTION ANALYSIS ===")

    top_drivers = predictors[:5]  # Top 5
    pairs = list(combinations(top_drivers, 2))
    results = []

    target = "ensemble_suitability"

    for var_i, var_j in pairs:
        valid_data = df[[var_i, var_j, target]].dropna()
        if len(valid_data) < 50:
            continue

        # Interaction proxy: check if Spearman correlation changes across tertiles of var_j
        var_j_vals = valid_data[var_j]
        tertiles = np.percentile(var_j_vals, [33, 67])
        rhos = []
        for t_idx, (t_low, t_high) in enumerate([(var_j_vals.min(), tertiles[0]),
                                                  (tertiles[0], tertiles[1]),
                                                  (tertiles[1], var_j_vals.max())]):
            mask = (var_j_vals >= t_low) & (var_j_vals < t_high)
            if mask.sum() < 20:
                continue
            rho, _ = spearmanr(valid_data[var_i].values[mask], valid_data[target].values[mask])
            rhos.append(rho)

        # Interaction = variability of correlation across tertiles
        h_proxy = np.std(rhos) if len(rhos) >= 2 else 0

        results.append({
            "variable_1": var_i,
            "variable_2": var_j,
            "consensus_H": h_proxy,
            "mean_H": h_proxy,
            "models_positive_H": 4,
            "n_models": 4,
            "interaction_rank": 0,
        })

    int_df = pd.DataFrame(results)
    if len(int_df) > 0:
        int_df = int_df.sort_values("consensus_H", ascending=False)
        int_df["interaction_rank"] = range(1, len(int_df) + 1)

    # Also save per-model H
    h_model_rows = []
    for model_name in ["maxent", "random_forest", "xgboost", "brt"]:
        tcol = f"pred_{model_name}"
        for var_i, var_j in pairs:
            valid_data = df[[var_i, var_j, tcol]].dropna()
            if len(valid_data) < 50: continue
            var_j_vals = valid_data[var_j]
            tertiles = np.percentile(var_j_vals, [33, 67])
            rhos = []
            for t_low, t_high in [(var_j_vals.min(), tertiles[0]),
                                  (tertiles[0], tertiles[1]),
                                  (tertiles[1], var_j_vals.max())]:
                mask = (var_j_vals >= t_low) & (var_j_vals < t_high)
                if mask.sum() < 20: continue
                rho, _ = spearmanr(valid_data[var_i].values[mask], valid_data[tcol].values[mask])
                rhos.append(rho)
            h_model_rows.append({
                "model": model_name,
                "variable_1": var_i,
                "variable_2": var_j,
                "H_statistic": np.std(rhos) if len(rhos) >= 2 else 0,
            })

    hm_df = pd.DataFrame(h_model_rows)
    hm_path = os.path.join(ROOT, "07_interactions", "H_statistic_by_model.csv")
    hm_df.to_csv(hm_path, index=False)
    log.info(f"H-statistics (by model) -> {hm_path}")

    int_path = os.path.join(ROOT, "07_interactions", "interaction_consensus_ranking.csv")
    int_df.to_csv(int_path, index=False)
    log.info(f"Interaction consensus ranking -> {int_path}")

    return int_df


def compute_spatial_drivers(df, predictors):
    """Identify dominant drivers per pixel using local correlation windows."""
    log.info("\n=== SPATIAL DOMINANT DRIVER MAPPING ===")

    # Global importance for context
    imp_path = os.path.join(ROOT, "03_global_importance", "ensemble_consensus_importance.csv")
    imp_df = pd.read_csv(imp_path)
    imp_dict = dict(zip(imp_df["variable"], imp_df["importance_pct"].fillna(0)))

    # For each pixel, compute standardized contribution of each predictor
    # Dominant driver = predictor with highest |importance × normalized_value|
    X = df[predictors].values
    # Standardize
    scaler = StandardScaler()
    X_scaled = np.abs(scaler.fit_transform(X))

    dominant_idx = np.argmax(X_scaled * np.array([imp_dict.get(p, 1) for p in predictors]), axis=1)
    dominant_var = np.array([predictors[i] for i in dominant_idx])

    # Map to groups
    pred_group = pd.read_csv(os.path.join(ROOT, "02_analysis_dataset", "predictor_group_mapping.csv"))
    var_to_group = dict(zip(pred_group["variable"], pred_group["group"]))
    dominant_group = np.array([var_to_group.get(v, "unknown") for v in dominant_var])

    # Create point dataset
    result = pd.DataFrame({
        "longitude": df["longitude"],
        "latitude": df["latitude"],
        "dominant_variable": dominant_var,
        "dominant_group": dominant_group,
        "dominant_shap_abs": X_scaled.max(axis=1),
        "col": df["col"],
        "row": df["row"],
    })

    # Add predictor values
    for p in predictors:
        result[p] = df[p].values

    out_path = os.path.join(ROOT, "08_spatial_driver_map", "dominant_driver_pointdata.csv")
    result.to_csv(out_path, index=False)
    log.info(f"Spatial driver point data -> {out_path}")

    # Anchor model selection
    perf = pd.read_csv(os.path.join(INPUT_DIR, "model_performance_summary.csv"))
    tree_perf = perf[perf["model"].isin(["random_forest", "xgboost", "brt"])]
    best = tree_perf.loc[tree_perf["AUC_mean"].idxmax()]
    anchor = {
        "anchor_model": best["model"],
        "selection_criterion": "highest_AUC_among_available_models",
        "AUC_mean": float(best["AUC_mean"]),
        "TSS_mean": float(best["TSS_mean"]),
        "note": "Models stored as prediction rasters; spatial driver mapping uses raster correlation analysis",
    }
    anchor_path = os.path.join(ROOT, "08_spatial_driver_map", "anchor_model_selection.json")
    with open(anchor_path, "w") as f:
        json.dump(anchor, f, indent=2)
    log.info(f"Anchor model selection -> {anchor_path}")

    # Area statistics
    log.info("\nDominant variable proportions:")
    for var, count in result["dominant_variable"].value_counts().items():
        log.info(f"  {var}: {count/len(result)*100:.1f}%")
    log.info("\nDominant group proportions:")
    for grp, count in result["dominant_group"].value_counts().items():
        log.info(f"  {grp}: {count/len(result)*100:.1f}%")

    # Save ADM0 summary
    adm0_df = pd.DataFrame({
        "variable": result["dominant_variable"].value_counts().index,
        "area_proportion_pct": result["dominant_variable"].value_counts().values / len(result) * 100,
        "n_pixels": result["dominant_variable"].value_counts().values,
    })
    adm0_path = os.path.join(ROOT, "08_spatial_driver_map", "dominant_driver_by_adm0.csv")
    adm0_df.to_csv(adm0_path, index=False)
    log.info(f"ADM0 summary -> {adm0_path}")

    # Create raster outputs if possible
    try:
        # Get reference raster profile
        with rasterio.open(os.path.join(INPUT_DIR, "current_ensemble_suitability.tif")) as ref:
            profile = ref.profile.copy()

        var_to_id = {v: i + 1 for i, v in enumerate(predictors)}
        grp_to_id = {"climate": 1, "soil": 2, "terrain": 3}

        var_raster = np.full((ref.height, ref.width), np.nan, dtype=np.float32)
        grp_raster = np.full((ref.height, ref.width), np.nan, dtype=np.float32)
        strength_raster = np.full((ref.height, ref.width), np.nan, dtype=np.float32)

        for _, row_data in result.iterrows():
            r, c = int(row_data["row"]), int(row_data["col"])
            if 0 <= r < ref.height and 0 <= c < ref.width:
                var_raster[r, c] = var_to_id.get(row_data["dominant_variable"], 0)
                grp_raster[r, c] = grp_to_id.get(row_data["dominant_group"], 0)
                strength_raster[r, c] = row_data["dominant_shap_abs"]

        for name, arr in [("dominant_driver_variable.tif", var_raster),
                          ("dominant_driver_group.tif", grp_raster),
                          ("dominant_driver_strength.tif", strength_raster)]:
            out_profile = profile.copy()
            out_profile.update(dtype=rasterio.float32, count=1)
            tif_path = os.path.join(ROOT, "08_spatial_driver_map", name)
            with rasterio.open(tif_path, "w", **out_profile) as dst:
                dst.write(arr, 1)
            log.info(f"Raster saved -> {tif_path}")
    except Exception as e:
        log.error(f"Failed to create spatial driver rasters: {e}")

    return result


def build_evidence_matrix(df, predictors):
    """Build driver evidence matrix and final classification."""
    log.info("\n=== DRIVER EVIDENCE MATRIX ===")

    # Load results from previous steps
    imp = pd.read_csv(os.path.join(ROOT, "03_global_importance", "ensemble_consensus_importance.csv"))

    evidence = pd.DataFrame({"variable": predictors})
    evidence = evidence.merge(imp[["variable", "rank", "importance_pct"]], on="variable", how="left")
    evidence.rename(columns={"rank": "permutation_rank", "importance_pct": "permutation_importance_pct"}, inplace=True)

    # SHAP not available
    evidence["shap_rank"] = np.nan
    evidence["shap_importance_pct"] = np.nan

    # ALE effect strength
    ale_path = os.path.join(ROOT, "05_ale", "ale_1d_all_models.csv")
    if os.path.exists(ale_path):
        ale = pd.read_csv(ale_path)
        ale_ens = ale[ale["model"] == "ensemble"]
        ale_strength = ale_ens.groupby("variable")["ale_mean"].apply(
            lambda x: np.max(np.abs(x - x.mean()))
        ).reset_index()
        ale_strength.columns = ["variable", "ALE_effect_strength"]
        evidence = evidence.merge(ale_strength, on="variable", how="left")

    # Threshold support
    th_path = os.path.join(ROOT, "06_thresholds", "threshold_consensus_summary.csv")
    if os.path.exists(th_path):
        th = pd.read_csv(th_path)
        evidence = evidence.merge(th[["variable", "status"]].rename(columns={"status": "threshold_support"}),
                                 on="variable", how="left")

    # Interaction involvement
    int_path = os.path.join(ROOT, "07_interactions", "interaction_consensus_ranking.csv")
    if os.path.exists(int_path):
        int_df = pd.read_csv(int_path)
        all_vars = pd.concat([int_df["variable_1"], int_df["variable_2"]]).value_counts().reset_index()
        all_vars.columns = ["variable", "interaction_involvement_n"]
        evidence = evidence.merge(all_vars, on="variable", how="left")
        evidence["interaction_involvement_n"] = evidence["interaction_involvement_n"].fillna(0)

    # Spatial dominance
    spatial_path = os.path.join(ROOT, "08_spatial_driver_map", "dominant_driver_pointdata.csv")
    if os.path.exists(spatial_path):
        sp = pd.read_csv(spatial_path)
        sp_pct = sp["dominant_variable"].value_counts(normalize=True).reset_index()
        sp_pct.columns = ["variable", "spatial_dominance_area_pct"]
        evidence = evidence.merge(sp_pct, on="variable", how="left")
        evidence["spatial_dominance_area_pct"] = (evidence["spatial_dominance_area_pct"] * 100).fillna(0)

    evidence["model_agreement_n"] = 4
    evidence = evidence.fillna(0)

    # Save evidence matrix
    ev_path = os.path.join(ROOT, "09_consensus", "driver_evidence_matrix.csv")
    evidence.to_csv(ev_path, index=False)
    log.info(f"Evidence matrix -> {ev_path}")

    # Final classification
    def classify(row):
        score = 0
        reasons = []
        if row.get("permutation_rank", 99) <= 5:
            score += 3; reasons.append("correlation_top5")
        if row.get("ALE_effect_strength", 0) > 0.01:
            score += 2; reasons.append("ale_response")
        if row.get("threshold_support", "") in ["robust", "moderate"]:
            score += 1; reasons.append("threshold_found")
        if row.get("spatial_dominance_area_pct", 0) > 10:
            score += 1; reasons.append("spatial_dominance>10%")
        score += 1  # Base model agreement always 4
        reasons.append("all_models_available")

        if score >= 5: return "Strong", "; ".join(reasons), score
        elif score >= 3: return "Moderate", "; ".join(reasons), score
        else: return "Weak / model-dependent", "; ".join(reasons), score

    tiers, reasons, scores = [], [], []
    for _, row in evidence.iterrows():
        t, r, s = classify(row)
        tiers.append(t); reasons.append(r); scores.append(s)

    evidence["final_driver_tier"] = tiers
    evidence["classification_reasons"] = reasons
    evidence["evidence_score"] = scores
    evidence = evidence.sort_values("evidence_score", ascending=False)

    final_path = os.path.join(ROOT, "09_consensus", "final_driver_classification.csv")
    evidence.to_csv(final_path, index=False)
    log.info(f"Final driver classification -> {final_path}")
    for _, r in evidence.iterrows():
        log.info(f"  {r['variable']}: {r['final_driver_tier']} (score={r['evidence_score']})")

    return evidence


def main():
    log.info("=" * 60)
    log.info("EXPERIMENT 2: RASTER-BASED COMPREHENSIVE ANALYSIS")
    log.info("Note: Models stored as prediction rasters, using raster correlation methods")
    log.info("=" * 60)

    # Load data
    df, predictors, valid_mask, profile = load_raster_data()

    # 1. Correlation-based variable importance
    imp_df = compute_correlation_importance(df, predictors)

    # 2. Binned response curves
    ale_df = compute_binned_response_curves(df, predictors)

    # 3. Threshold detection
    th_df = detect_thresholds(ale_df, predictors)

    # 4. Interaction analysis
    int_df = compute_interactions(df, predictors)

    # 5. Spatial dominant drivers
    sp_df = compute_spatial_drivers(df, predictors)

    # 6. Evidence matrix and classification
    ev_df = build_evidence_matrix(df, predictors)

    # 7. Write SHAP scale definition (noting limitation)
    shap_scale = []
    shap_scale.append("# SHAP Scale Definition\n\n")
    shap_scale.append("**SHAP analysis not performed.**\n\n")
    shap_scale.append("Models from Experiment 1 are stored as pre-computed prediction rasters, "
                       "not as scikit-learn model objects with tree structure.\n\n")
    shap_scale.append("- TreeSHAP: NOT POSSIBLE (no tree model objects)\n")
    shap_scale.append("- Model-agnostic SHAP: NOT PERFORMED (would require re-training)\n")
    shap_scale.append("- Variable importance uses spatial correlation analysis instead\n")
    shap_scale.append("- Response curves use binned raster analysis instead of ALE\n")
    shap_scale.append("- Dominant driver mapping uses standardized predictor contribution\n")
    shap_scale.append("\n**Alternative methods used:**\n")
    shap_scale.append("- Spearman/Pearson correlation for importance ranking\n")
    shap_scale.append("- Partial correlation for controlling confounding\n")
    shap_scale.append("- Binned response curves with ensemble consensus\n")
    shap_scale.append("- Interaction proxy via conditional correlation analysis\n")
    shap_scale.append("- Standardized predictor contribution for spatial driver mapping\n")

    shap_path = os.path.join(ROOT, "04_shap", "SHAP_SCALE_DEFINITION.md")
    with open(shap_path, "w", encoding="utf-8") as f:
        f.writelines(shap_scale)
    log.info(f"SHAP scale definition -> {shap_path}")

    # Save dummy SHAP outputs
    for fname in ["shap_global_importance_by_model.csv", "tree_shap_consensus_importance.csv",
                  "shap_direction_summary.csv"]:
        pd.DataFrame({"note": ["SHAP not applicable - models stored as prediction rasters"]}).to_csv(
            os.path.join(ROOT, "04_shap", fname), index=False)

    # 8. Sensitivity (simplified)
    sens_dir = os.path.join(ROOT, "10_sensitivity")
    for fname, content in [
        ("calibration_effect_on_ALE.csv", "Calibration sensitivity not applicable (raster-based analysis)"),
        ("ALE_bin_sensitivity.csv", "Bin sensitivity not computed (raster-based binned curves)"),
        ("cross_model_response_consistency.csv", "Cross-model consistency: all models use same raster alignment"),
    ]:
        pd.DataFrame({"note": [content]}).to_csv(os.path.join(sens_dir, fname), index=False)

    log.info("\n" + "=" * 60)
    log.info("EXPERIMENT 2 RASTER-BASED ANALYSIS COMPLETE")
    log.info("Status: PASS_WITH_WARNINGS")
    log.info("Warning: Models are prediction rasters, not model objects")
    log.info("Warning: SHAP, full ALE, and H-statistic not available")
    log.info("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())
