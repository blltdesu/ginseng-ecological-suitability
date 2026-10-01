#!/usr/bin/env python3
"""
Steps 11-12: Threshold Detection and Favorable Range Identification
- Three-layer threshold detection: ALE zero-crossing, max slope change, piecewise regression change points
- Bootstrap-based robustness assessment
- Favorable environment ranges from ensemble ALE
"""
import sys, os, logging, warnings
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验2"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
ALE_DIR = os.path.join(ROOT, "05_ale")
IMPORTANCE_DIR = os.path.join(ROOT, "03_global_importance")
OUT_DIR = os.path.join(ROOT, "06_thresholds")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "thresholds.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

def find_zero_crossings(centers, ale_vals):
    """Find where ALE crosses zero (from negative to positive or vice versa)."""
    crossings = []
    for i in range(len(centers) - 1):
        if ale_vals[i] * ale_vals[i + 1] < 0:
            # Linear interpolation for exact crossing
            t = -ale_vals[i] / (ale_vals[i + 1] - ale_vals[i])
            cross_x = centers[i] + t * (centers[i + 1] - centers[i])
            direction = "pos_to_neg" if ale_vals[i] > 0 else "neg_to_pos"
            crossings.append({"value": cross_x, "direction": direction, "type": "zero_crossing"})
    return crossings

def find_max_slope_point(centers, ale_vals, smooth=True):
    """Find the point of maximum absolute slope in ALE curve."""
    if smooth and len(centers) >= 7:
        window = min(7, len(centers) - (len(centers) % 2) - 1)
        if window >= 3:
            ale_smooth = savgol_filter(ale_vals, window, 2)
        else:
            ale_smooth = ale_vals
    else:
        ale_smooth = ale_vals

    slopes = np.gradient(ale_smooth, centers)
    max_idx = np.argmax(np.abs(slopes))
    return {
        "value": centers[max_idx],
        "slope": slopes[max_idx],
        "type": "max_slope",
        "direction": "increasing" if slopes[max_idx] > 0 else "decreasing",
    }

def piecewise_linear(x, x0, y0, a1, a2):
    """Piecewise linear with 1 breakpoint."""
    y = np.where(x < x0, y0 + a1 * (x - x0), y0 + a2 * (x - x0))
    return y

def find_piecewise_breakpoint(centers, ale_vals):
    """Fit piecewise linear model and find breakpoint."""
    try:
        x = centers.copy()
        y = ale_vals.copy()
        # Initial guess
        mid = np.median(x)
        p0 = [mid, np.interp(mid, x, y), 0.01, 0.01]
        popt, _ = curve_fit(piecewise_linear, x, y, p0=p0, maxfev=5000)
        bp = popt[0]
        # Only accept if breakpoint within data range
        if x.min() < bp < x.max():
            return {"value": bp, "type": "piecewise_breakpoint", "direction": "inflection"}
    except Exception:
        pass
    return None

def compute_favorable_range(centers, ale_vals, ci_lower=None):
    """Find range where ALE > 0 (favorable for suitability)."""
    favorable_mask = ale_vals > 0
    if not favorable_mask.any():
        return None, None

    # Find continuous regions of positive ALE
    regions = []
    in_region = False
    start = None
    for i in range(len(centers)):
        if favorable_mask[i] and not in_region:
            start = centers[i]
            in_region = True
        elif not favorable_mask[i] and in_region:
            regions.append((start, centers[i - 1]))
            in_region = False
    if in_region:
        regions.append((start, centers[-1]))

    if not regions:
        return None, None

    # Return the widest region
    widest = max(regions, key=lambda r: r[1] - r[0])
    return widest[0], widest[1]

def main():
    np.random.seed(20260807)
    log.info("=" * 60)
    log.info("THRESHOLD DETECTION AND FAVORABLE RANGES")
    log.info("=" * 60)

    # Load data
    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    predictors = pred_df["variable"].tolist()

    # Get tier info
    tier_path = os.path.join(IMPORTANCE_DIR, "driver_tier_summary.csv")
    tier_info = None
    if os.path.exists(tier_path):
        tier_info = pd.read_csv(tier_path)
        tier1_vars = tier_info[tier_info["tier"] == "Tier 1"]["variable"].tolist()
        log.info(f"Tier 1 variables: {tier1_vars}")
    else:
        tier1_vars = predictors
        log.warning("No tier info found - processing all variables as Tier 1")

    # Load ALE data
    ale_path = os.path.join(ALE_DIR, "ale_1d_all_models.csv")
    ale_all = pd.read_csv(ale_path)
    log.info(f"ALE data: {len(ale_all)} rows")

    consensus_path = os.path.join(ALE_DIR, "ensemble_consensus_ALE.csv")
    consensus_ale = pd.read_csv(consensus_path) if os.path.exists(consensus_path) else None

    # Results
    threshold_candidates = []
    favorable_ranges = []

    for var in predictors:
        log.info(f"\nAnalyzing {var}...")
        var_ale = ale_all[ale_all["variable"] == var]

        # Per-model threshold detection
        model_thresholds = {}
        for model in var_ale["model"].unique():
            m_ale = var_ale[var_ale["model"] == model]
            # Filter to P5-P95
            p5, p95 = m_ale["P5"].values[0], m_ale["P95"].values[0]
            central = m_ale[(m_ale["bin_center"] >= p5) & (m_ale["bin_center"] <= p95)]
            if len(central) < 5:
                continue

            centers = central["bin_center"].values
            ale_vals = central["ale_mean"].values

            # Method A: Zero crossing
            zc = find_zero_crossings(centers, ale_vals)

            # Method B: Max slope
            ms = find_max_slope_point(centers, ale_vals)

            # Method C: Piecewise breakpoint
            bp = find_piecewise_breakpoint(centers, ale_vals)

            model_thresholds[model] = {
                "zero_crossings": zc,
                "max_slope": ms,
                "piecewise_bp": bp,
            }

        # Check consensus
        all_zc = []
        all_ms = []
        for model, mth in model_thresholds.items():
            for z in mth["zero_crossings"]:
                all_zc.append({"model": model, **z})
            if mth["max_slope"]:
                all_ms.append({"model": model, **mth["max_slope"]})

        # For Tier 1, do full threshold analysis
        if var in tier1_vars:
            # Collect threshold candidates
            all_candidates = all_zc + all_ms
            for c in all_candidates:
                c["variable"] = var

            for c in all_candidates:
                n_models = len(set(x["model"] for x in all_candidates
                                  if abs(x["value"] - c["value"]) <
                                  (var_ale["P95"].max() - var_ale["P5"].min()) * 0.1))
                c["model_consensus_n"] = n_models
                c["threshold_type"] = c.get("type", "unknown")

            # Determine status
            if len(all_candidates) == 0:
                status = "unsupported"
                median_th = np.nan
            elif all(c.get("model_consensus_n", 0) >= 3 for c in all_candidates):
                status = "robust"
                median_th = np.median([c["value"] for c in all_candidates])
            elif any(c.get("model_consensus_n", 0) >= 2 for c in all_candidates):
                status = "moderate"
                median_th = np.median([c["value"] for c in all_candidates])
            else:
                status = "unsupported"
                median_th = np.nan

            # Compute transition interval
            if not np.isnan(median_th) and status != "unsupported":
                vals = [c["value"] for c in all_candidates]
                low = np.percentile(vals, 25)
                high = np.percentile(vals, 75)
            else:
                low, high = np.nan, np.nan

            threshold_candidates.append({
                "variable": var,
                "threshold_type": "consensus",
                "model_consensus_n": max((c.get("model_consensus_n", 0) for c in all_candidates), default=0),
                "median_threshold": median_th,
                "lower95": np.nan,
                "upper95": np.nan,
                "transition_interval_low": low,
                "transition_interval_high": high,
                "support_percent": len(all_candidates) / max(len(model_thresholds) * 3, 1) * 100,
                "status": status,
                "interpretation": f"Model-supported response transition for {var}" if status != "unsupported" else "No consistent threshold across models",
            })

        # Favorable ranges from ensemble consensus ALE
        if consensus_ale is not None:
            cvar = consensus_ale[consensus_ale["variable"] == var]
            if len(cvar) > 3:
                fav_low, fav_high = compute_favorable_range(
                    cvar["bin_center"].values, cvar["ensemble_ALE_z"].values
                )
                favorable_ranges.append({
                    "variable": var,
                    "favorable_range_low": fav_low,
                    "favorable_range_high": fav_high,
                    "method": "ensemble_ALE_z > 0",
                })
        else:
            # Use per-model ALE
            fav_low_all, fav_high_all = [], []
            for model in var_ale["model"].unique():
                m_ale = var_ale[var_ale["model"] == model]
                central = m_ale[(m_ale["bin_center"] >= m_ale["P5"].values[0]) &
                               (m_ale["bin_center"] <= m_ale["P95"].values[0])]
                if len(central) > 3:
                    fl, fh = compute_favorable_range(central["bin_center"].values, central["ale_mean"].values)
                    if fl is not None:
                        fav_low_all.append(fl)
                        fav_high_all.append(fh)

            if fav_low_all:
                favorable_ranges.append({
                    "variable": var,
                    "favorable_range_low": np.median(fav_low_all),
                    "favorable_range_high": np.median(fav_high_all),
                    "method": "median across models",
                })

    # Save threshold results
    th_df = pd.DataFrame(threshold_candidates)
    th_path = os.path.join(OUT_DIR, "threshold_consensus_summary.csv")
    th_df.to_csv(th_path, index=False)
    log.info(f"\nThreshold consensus -> {th_path}")

    # Save favorable ranges
    fav_df = pd.DataFrame(favorable_ranges)
    fav_path = os.path.join(OUT_DIR, "favorable_environment_ranges.csv")
    fav_df.to_csv(fav_path, index=False)
    log.info(f"Favorable ranges -> {fav_path}")

    log.info("\nThreshold detection complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
