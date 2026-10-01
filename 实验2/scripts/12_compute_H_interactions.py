#!/usr/bin/env python3
"""
Steps 13-14: Interaction Analysis
- Friedman's H-statistic for pairwise interactions
- SHAP interaction values
- 2D ALE for top interaction pairs
"""
import sys, os, logging, warnings
import numpy as np
import pandas as pd
import joblib
from itertools import combinations

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验2"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
DATA_DIR = os.path.join(ROOT, "02_analysis_dataset")
IMPORTANCE_DIR = os.path.join(ROOT, "03_global_importance")
OUT_DIR = os.path.join(ROOT, "07_interactions")
ALE_DIR = os.path.join(ROOT, "05_ale")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "interactions.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

RANDOM_SEED = 20260807

def partial_dependence(model, X, features):
    """Compute partial dependence for feature set."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].mean()
    else:
        p = model.predict(X)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1].mean()
        return p.mean()

def compute_H_statistic(model, X, var_i, var_j, n_samples=500):
    """Friedman's H-statistic for pairwise interaction."""
    np.random.seed(RANDOM_SEED)
    n = min(len(X), n_samples)
    idx = np.random.choice(len(X), n, replace=False)
    X_sub = X[idx]

    # Total variance of partial dependence
    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(X_sub)[:, 1]
    else:
        preds = model.predict(X_sub)
        if preds.ndim == 2:
            preds = preds[:, 1]

    total_var = np.var(preds)

    if total_var == 0:
        return 0.0

    # PD for var_i
    col_i = X_sub[:, var_i]
    uniq_i = np.unique(col_i)
    pd_i = np.zeros(n)
    for val in uniq_i:
        mask = col_i == val
        if mask.sum() > 0:
            X_temp = X_sub.copy()
            X_temp[:, var_i] = val
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_temp)[:, 1]
            else:
                p = model.predict(X_temp)
                if p.ndim == 2: p = p[:, 1]
            pd_i[mask] = p[mask].mean() if len(p[mask]) > 0 else 0

    # PD for var_j
    col_j = X_sub[:, var_j]
    uniq_j = np.unique(col_j)
    pd_j = np.zeros(n)
    for val in uniq_j:
        mask = col_j == val
        if mask.sum() > 0:
            X_temp = X_sub.copy()
            X_temp[:, var_j] = val
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_temp)[:, 1]
            else:
                p = model.predict(X_temp)
                if p.ndim == 2: p = p[:, 1]
            pd_j[mask] = p[mask].mean() if len(p[mask]) > 0 else 0

    # PD for (i,j)
    pd_ij = np.zeros(n)
    for val_i in uniq_i:
        for val_j in uniq_j:
            mask = (col_i == val_i) & (col_j == val_j)
            if mask.sum() > 0:
                X_temp = X_sub.copy()
                X_temp[:, var_i] = val_i
                X_temp[:, var_j] = val_j
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(X_temp)[:, 1]
                else:
                    p = model.predict(X_temp)
                    if p.ndim == 2: p = p[:, 1]
                pd_ij[mask] = p[mask].mean() if len(p[mask]) > 0 else 0

    # H = sqrt( sum( (pd_ij - pd_i - pd_j)^2 ) / sum(pd_ij^2) )
    numerator = np.sum((pd_ij - pd_i - pd_j) ** 2)
    denominator = np.sum(pd_ij ** 2)
    if denominator == 0:
        return 0.0
    return np.sqrt(numerator / denominator)

def compute_2d_ale(model, X, var_i, var_j, name_i, name_j, bins=15):
    """Compute 2D ALE for a pair of variables."""
    n = X.shape[0]
    col_i, col_j = X[:, var_i], X[:, var_j]

    pi1, pi99 = np.percentile(col_i, [1, 99])
    pj1, pj99 = np.percentile(col_j, [1, 99])
    edges_i = np.linspace(pi1, pi99, bins + 1)
    edges_j = np.linspace(pj1, pj99, bins + 1)
    centers_i = (edges_i[:-1] + edges_i[1:]) / 2
    centers_j = (edges_j[:-1] + edges_j[1:]) / 2

    ale2d = np.zeros((bins, bins))
    support = np.zeros((bins, bins), dtype=int)

    for ki in range(bins):
        for kj in range(bins):
            in_bin = ((col_i >= edges_i[ki]) & (col_i < edges_i[ki + 1]) &
                      (col_j >= edges_j[kj]) & (col_j < edges_j[kj + 1]))
            if ki == bins - 1:
                in_bin = in_bin | ((col_i >= edges_i[ki]) & (col_i <= edges_i[ki + 1]) &
                                   (col_j >= edges_j[kj]) & (col_j < edges_j[kj + 1]))
            if kj == bins - 1:
                in_bin = in_bin | ((col_i >= edges_i[ki]) & (col_i < edges_i[ki + 1]) &
                                   (col_j >= edges_j[kj]) & (col_j <= edges_j[kj + 1]))

            support[ki, kj] = in_bin.sum()
            if support[ki, kj] < 2:
                continue

            # Compute local effect
            X_ll = X.copy()
            X_lh = X.copy()
            X_hl = X.copy()
            X_hh = X.copy()
            X_ll[:, var_i] = edges_i[ki]; X_ll[:, var_j] = edges_j[kj]
            X_lh[:, var_i] = edges_i[ki]; X_lh[:, var_j] = edges_j[kj + 1]
            X_hl[:, var_i] = edges_i[ki + 1]; X_hl[:, var_j] = edges_j[kj]
            X_hh[:, var_i] = edges_i[ki + 1]; X_hh[:, var_j] = edges_j[kj + 1]

            if hasattr(model, "predict_proba"):
                p_ll = model.predict_proba(X_ll)[:, 1]
                p_lh = model.predict_proba(X_lh)[:, 1]
                p_hl = model.predict_proba(X_hl)[:, 1]
                p_hh = model.predict_proba(X_hh)[:, 1]
            else:
                p_ll = model.predict(X_ll); p_lh = model.predict(X_lh)
                p_hl = model.predict(X_hl); p_hh = model.predict(X_hh)

            # 2D local effect = second-order difference
            local_effect = (p_hh - p_hl - p_lh + p_ll) / (
                (edges_i[ki + 1] - edges_i[ki]) * (edges_j[kj + 1] - edges_j[kj])
            )
            ale2d[ki, kj] = local_effect.mean()

    return centers_i, centers_j, ale2d, support

def main():
    np.random.seed(RANDOM_SEED)
    log.info("=" * 60)
    log.info("INTERACTION ANALYSIS")
    log.info("=" * 60)

    # Load data
    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    predictors = pred_df["variable"].tolist()

    sample_path = os.path.join(DATA_DIR, "explainability_sample.csv")
    if os.path.exists(sample_path):
        data = pd.read_csv(sample_path)
    else:
        data = pd.read_csv(os.path.join(INPUT_DIR, "training_matrix_main.csv"))
    X = data[predictors].values.astype(np.float64)
    log.info(f"Data: {X.shape}")

    # Get top 5 drivers for interaction analysis
    tier_path = os.path.join(IMPORTANCE_DIR, "driver_tier_summary.csv")
    if os.path.exists(tier_path):
        tier_df = pd.read_csv(tier_path)
        top5 = tier_df[tier_df["rank"] <= 5]["variable"].tolist()
    else:
        top5 = predictors[:5]
    log.info(f"Top 5 drivers for interactions: {top5}")

    pairs = list(combinations(top5, 2))
    log.info(f"Interaction pairs to evaluate: {len(pairs)}")

    # Load ensemble weights
    ew = pd.read_csv(os.path.join(INPUT_DIR, "ensemble_weights.csv"))
    ew_dict = ew.set_index("model")["weight"].to_dict()

    model_manifest = pd.read_csv(os.path.join(INPUT_DIR, "final_model_manifest.csv"))

    all_h_stats = []

    for _, mrow in model_manifest.iterrows():
        model_name = mrow["model"]
        model_path = os.path.join(INPUT_DIR, mrow["file"])
        log.info(f"\nComputing H-statistic for {model_name}...")

        try:
            model = joblib.load(model_path)
        except Exception as e:
            log.error(f"  Cannot load {model_name}: {e}")
            continue

        for var_i, var_j in pairs:
            idx_i = predictors.index(var_i)
            idx_j = predictors.index(var_j)
            try:
                h = compute_H_statistic(model, X, idx_i, idx_j, n_samples=1000)
                all_h_stats.append({
                    "model": model_name,
                    "variable_1": var_i,
                    "variable_2": var_j,
                    "H_statistic": h,
                })
                log.info(f"  {var_i} × {var_j}: H = {h:.4f}")
            except Exception as e:
                log.error(f"  H failed for {var_i}×{var_j} in {model_name}: {e}")

    # Save H-statistic results
    h_df = pd.DataFrame(all_h_stats)
    h_path = os.path.join(OUT_DIR, "H_statistic_by_model.csv")
    h_df.to_csv(h_path, index=False)
    log.info(f"\nH-statistics -> {h_path}")

    # Build interaction consensus ranking
    if len(h_df) > 0:
        # Normalize per model
        h_df["H_normalized"] = h_df.groupby("model")["H_statistic"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )

        # Weighted consensus
        h_df["ensemble_weight"] = h_df["model"].map(ew_dict).fillna(0.25)
        h_df["weighted_H"] = h_df["H_normalized"] * h_df["ensemble_weight"]

        consensus_h = h_df.groupby(["variable_1", "variable_2"]).agg(
            consensus_H=("weighted_H", "sum"),
            mean_H=("H_statistic", "mean"),
            models_positive_H=("H_statistic", lambda x: (x > 0).sum()),
            n_models=("model", "nunique"),
        ).reset_index()
        consensus_h = consensus_h.sort_values("consensus_H", ascending=False)
        consensus_h["interaction_rank"] = range(1, len(consensus_h) + 1)

        consensus_path = os.path.join(OUT_DIR, "interaction_consensus_ranking.csv")
        consensus_h.to_csv(consensus_path, index=False)
        log.info(f"Interaction consensus ranking -> {consensus_path}")

        # Top 3 pairs for 2D ALE
        top3 = consensus_h.head(3)
        log.info(f"\nTop 3 interactions:")
        for _, r in top3.iterrows():
            log.info(f"  {r['interaction_rank']}: {r['variable_1']} × {r['variable_2']} "
                     f"(H={r['consensus_H']:.4f}, models_positive={r['models_positive_H']})")

        # 2D ALE for Top 3 using anchor model (RF)
        try:
            rf_model = joblib.load(os.path.join(INPUT_DIR, "models", "random_forest_all_reps.joblib"))
            log.info("\nComputing 2D ALE for Top 3 pairs...")

            for pair_idx, (_, r) in enumerate(top3.iterrows()):
                var_i, var_j = r["variable_1"], r["variable_2"]
                idx_i = predictors.index(var_i)
                idx_j = predictors.index(var_j)
                log.info(f"  2D ALE: {var_i} × {var_j}")

                try:
                    ci, cj, ale2d, support = compute_2d_ale(rf_model, X, idx_i, idx_j, var_i, var_j)

                    # Save as long-format CSV
                    rows = []
                    for ki in range(len(ci)):
                        for kj in range(len(cj)):
                            rows.append({
                                "x_variable": var_i,
                                "y_variable": var_j,
                                "x_center": ci[ki],
                                "y_center": cj[kj],
                                "ale2d": ale2d[ki, kj],
                                "support_n": support[ki, kj],
                                "model": "random_forest",
                            })
                    pair_df = pd.DataFrame(rows)
                    pair_path = os.path.join(OUT_DIR, f"ALE2D_pair{pair_idx + 1}.csv")
                    pair_df.to_csv(pair_path, index=False)
                    log.info(f"  2D ALE saved to {pair_path}")
                except Exception as e:
                    log.error(f"  2D ALE failed for {var_i}×{var_j}: {e}")

        except Exception as e:
            log.error(f"Cannot compute 2D ALE: {e}")

    log.info("\nInteraction analysis complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
