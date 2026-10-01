#!/usr/bin/env python3
"""
Steps 9-10: 1D ALE (Accumulated Local Effects)
Computes ALE for all final predictors across all 4 models.
Includes ensemble-weighted consensus ALE.
Uses calibrated prediction pipeline where possible.
"""
import sys, os, logging, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验2"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
DATA_DIR = os.path.join(ROOT, "02_analysis_dataset")
OUT_DIR = os.path.join(ROOT, "05_ale")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "ale.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

RANDOM_SEED = 20260807
N_BINS = 20

def predict_calibrated(model, X):
    """Get calibrated prediction (probability scale)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:
        raw = model.predict(X)
        if raw.ndim == 2 and raw.shape[1] == 2:
            return raw[:, 1]
        return raw

def compute_1d_ale(model, X, variable_idx, variable_name, bins=N_BINS):
    """
    Compute 1D ALE for a single variable.
    Returns arrays: bin_centers, ale_values, n_support
    """
    n = X.shape[0]
    # Create bins based on percentiles
    col = X[:, variable_idx]
    # Use P1-P99 for binning to avoid extremes
    p1, p99 = np.percentile(col, [1, 99])
    bin_edges = np.linspace(p1, p99, bins + 1)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ale = np.zeros(bins)
    n_support = np.zeros(bins, dtype=int)

    for k in range(bins):
        # Indices where x falls in this bin
        in_bin = (col >= bin_edges[k]) & (col < bin_edges[k + 1])
        if k == bins - 1:
            in_bin = (col >= bin_edges[k]) & (col <= bin_edges[k + 1])

        n_support[k] = in_bin.sum()
        if n_support[k] < 2:
            continue

        # Replace values with bin edges
        X_low = X.copy()
        X_high = X.copy()
        X_low[:, variable_idx] = bin_edges[k]
        X_high[:, variable_idx] = bin_edges[k + 1]

        pred_low = predict_calibrated(model, X_low)
        pred_high = predict_calibrated(model, X_high)

        # Local effect = difference in prediction
        local_effects = pred_high - pred_low
        ale[k] = local_effects.mean()

    # Accumulate
    ale_cumsum = np.cumsum(ale)
    # Center around mean
    ale_centered = ale_cumsum - np.mean(ale_cumsum)

    return bin_centers, ale_centered, n_support, col

def compute_percentile_regions(col):
    """Compute percentile boundaries for data support annotation."""
    return {
        "P1": np.percentile(col, 1),
        "P5": np.percentile(col, 5),
        "P25": np.percentile(col, 25),
        "P50": np.percentile(col, 50),
        "P75": np.percentile(col, 75),
        "P95": np.percentile(col, 95),
        "P99": np.percentile(col, 99),
    }

def main():
    np.random.seed(RANDOM_SEED)
    log.info("=" * 60)
    log.info("1D ALE ANALYSIS")
    log.info("=" * 60)

    # Load data
    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    predictors = pred_df["variable"].tolist()
    log.info(f"Predictors: {predictors}")

    # Use explainability sample for ALE
    sample_path = os.path.join(DATA_DIR, "explainability_sample.csv")
    if os.path.exists(sample_path):
        data = pd.read_csv(sample_path)
        log.info(f"Using explainability sample: {len(data)} rows")
    else:
        data = pd.read_csv(os.path.join(INPUT_DIR, "training_matrix_main.csv"))
        log.info(f"Using full training data: {len(data)} rows")

    X = data[predictors].values.astype(np.float64)

    # Load ensemble weights
    ew = pd.read_csv(os.path.join(INPUT_DIR, "ensemble_weights.csv"))
    ew_dict = ew.set_index("model")["weight"].to_dict()

    model_manifest = pd.read_csv(os.path.join(INPUT_DIR, "final_model_manifest.csv"))

    all_ale = []
    model_ale_z = {}  # Per-model standardized ALE

    for _, mrow in model_manifest.iterrows():
        model_name = mrow["model"]
        model_path = os.path.join(INPUT_DIR, mrow["file"])
        log.info(f"\nProcessing {model_name}...")

        try:
            model = joblib.load(model_path)
        except Exception as e:
            log.error(f"  Cannot load {model_name}: {e}")
            continue

        model_ale_z[model_name] = {}

        for var_idx, var in enumerate(predictors):
            log.info(f"  Variable: {var}")
            try:
                centers, ale_vals, support, col = compute_1d_ale(model, X, var_idx, var)
                pct = compute_percentile_regions(col)

                for k in range(len(centers)):
                    # Determine percentile region
                    if centers[k] < pct["P1"]:
                        region = "tail_low_extreme"
                    elif centers[k] < pct["P5"]:
                        region = "tail_low"
                    elif centers[k] < pct["P25"]:
                        region = "low"
                    elif centers[k] < pct["P75"]:
                        region = "central"
                    elif centers[k] < pct["P95"]:
                        region = "high"
                    elif centers[k] < pct["P99"]:
                        region = "tail_high"
                    else:
                        region = "tail_high_extreme"

                    all_ale.append({
                        "model": model_name,
                        "variable": var,
                        "bin_center": centers[k],
                        "ale_mean": ale_vals[k],
                        "n_support": support[k],
                        "percentile_region": region,
                        "P1": pct["P1"], "P5": pct["P5"],
                        "P25": pct["P25"], "P50": pct["P50"],
                        "P75": pct["P75"], "P95": pct["P95"], "P99": pct["P99"],
                    })

                # Store for standardization
                # Only use central region (P5-P95) for z-scoring
                central_mask = np.array([
                    pct["P5"] <= c <= pct["P95"] for c in centers
                ])
                if central_mask.sum() > 2:
                    ale_central = ale_vals[central_mask]
                    mu, sigma = ale_central.mean(), ale_central.std()
                    if sigma > 0:
                        model_ale_z[model_name][var] = {
                            "centers": centers,
                            "ale_z": (ale_vals - mu) / sigma,
                            "ale_raw": ale_vals,
                        }
                    else:
                        model_ale_z[model_name][var] = {
                            "centers": centers,
                            "ale_z": np.zeros_like(ale_vals),
                            "ale_raw": ale_vals,
                        }

            except Exception as e:
                log.error(f"  ALE failed for {var} in {model_name}: {e}")

    # Save all ALE results
    ale_df = pd.DataFrame(all_ale)
    ale_path = os.path.join(OUT_DIR, "ale_1d_all_models.csv")
    ale_df.to_csv(ale_path, index=False)
    log.info(f"\nAll ALE results -> {ale_path}")

    # Build ensemble consensus ALE
    log.info("\nBuilding ensemble consensus ALE...")
    consensus_rows = []
    for var in predictors:
        # Collect all models' ALE_z for this variable
        models_contributing = []

        # Define common bin grid (20 bins across P5-P95)
        var_data = data[var].values
        p5, p95 = np.percentile(var_data, [5, 95])
        common_centers = np.linspace(p5, p95, N_BINS)

        weighted_ale = np.zeros(N_BINS)
        total_weight = 0

        for model_name in model_ale_z:
            if var in model_ale_z[model_name]:
                w = ew_dict.get(model_name, 0.25)
                # Interpolate ALE_z to common bins
                mc = model_ale_z[model_name][var]["centers"]
                az = model_ale_z[model_name][var]["ale_z"]
                ar = model_ale_z[model_name][var]["ale_raw"]

                # Linear interpolation
                interp_z = np.interp(common_centers, mc, az)
                weighted_ale += w * interp_z
                total_weight += w
                models_contributing.append(model_name)

        if total_weight > 0:
            weighted_ale /= total_weight

        for k in range(N_BINS):
            consensus_rows.append({
                "variable": var,
                "bin_center": common_centers[k],
                "ensemble_ALE_z": weighted_ale[k],
                "n_models_contributing": len(models_contributing),
            })

    consensus_df = pd.DataFrame(consensus_rows)
    consensus_path = os.path.join(OUT_DIR, "ensemble_consensus_ALE.csv")
    consensus_df.to_csv(consensus_path, index=False)
    log.info(f"Ensemble consensus ALE -> {consensus_path}")

    log.info("\n1D ALE analysis complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
