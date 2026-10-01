#!/usr/bin/env python3
"""
Steps 4-5: Permutation Importance
- Per-model, per-fold, per-variable permutation with 20 repeats
- Ensemble consensus importance with weighting
- Driver tier classification
"""
import sys, os, logging, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验2"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
OUT_DIR = os.path.join(ROOT, "03_global_importance")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "permutation_importance.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

N_PERMUTATIONS = 20
RANDOM_SEED = 20260807

def boyce_index(y_true, y_pred):
    """Continuous Boyce index."""
    try:
        from sklearn.isotonic import IsotonicRegression
        n_bins = 10
        # Focus on presence predictions
        pres_pred = y_pred[y_true == 1]
        if len(pres_pred) < 10:
            return np.nan
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        # Predicted-to-expected ratio
        total = len(y_pred)
        ratios = []
        for j in range(n_bins):
            in_bin = (y_pred >= bins[j]) & (y_pred < bins[j + 1])
            if j == n_bins - 1:
                in_bin = (y_pred >= bins[j]) & (y_pred <= bins[j + 1])
            n_pred = in_bin.sum()
            expected = n_pred / total if total > 0 else 0
            n_pres = (y_pred[y_true == 1] >= bins[j]).sum() - (y_pred[y_true == 1] >= bins[j + 1]).sum() if j < n_bins - 1 else 0
            if expected > 0 and n_pred > 0:
                ratios.append(n_pres / n_pred / expected if expected > 0 else 0)
        if len(ratios) < 3:
            return np.nan
        rho, _ = spearmanr(range(len(ratios)), ratios)
        return rho
    except Exception:
        return np.nan

def tss_score(y_true, y_pred):
    """True Skill Statistic."""
    from sklearn.metrics import confusion_matrix
    thresh = np.median(y_pred)
    y_hat = (y_pred >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens + spec - 1

def compute_metrics(y_true, y_pred):
    return {
        "auc": roc_auc_score(y_true, y_pred),
        "tss": tss_score(y_true, y_pred),
        "boyce": boyce_index(y_true, y_pred),
    }

def load_model_and_get_predict_fn(model_path, model_name):
    """Load a model and return a prediction function (predict_proba for [:,1])."""
    m = joblib.load(model_path)
    if hasattr(m, "predict_proba"):
        return lambda X: m.predict_proba(X)[:, 1]
    elif hasattr(m, "predict"):
        raw = m.predict(X)
        if raw.ndim == 2 and raw.shape[1] == 2:
            return lambda X: m.predict(X)[:, 1]
        return lambda X: m.predict(X)
    else:
        raise ValueError(f"Cannot get prediction function from {model_name}")

def main():
    np.random.seed(RANDOM_SEED)
    log.info("=" * 60)
    log.info("PERMUTATION IMPORTANCE ANALYSIS")
    log.info("=" * 60)

    # Load data
    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    predictors = pred_df["variable"].tolist()
    train = pd.read_csv(os.path.join(INPUT_DIR, "training_matrix_main.csv"))
    log.info(f"Training data: {train.shape}, predictors: {predictors}")

    ensemble_weights = pd.read_csv(os.path.join(INPUT_DIR, "ensemble_weights.csv"))
    model_manifest = pd.read_csv(os.path.join(INPUT_DIR, "final_model_manifest.csv"))

    # Prepare data
    X = train[predictors].values.astype(np.float64)
    y = train["label"].values.astype(int)

    # Create 5-fold stratified spatial holdout folds
    # Since training matrix has outer_fold=-1 (final merged data),
    # we create ad-hoc spatial folds based on latitude bands for held-out evaluation
    from sklearn.model_selection import StratifiedKFold
    n_folds = 5
    lats = train["latitude"].values
    lons = train["longitude"].values
    # Create spatial stratum from rounded lat/lon grid
    spatial_stratum = np.round(lats, 1).astype(str) + "_" + np.round(lons, 1).astype(str)
    stratum_ids = pd.factorize(spatial_stratum)[0]

    # Use StratifiedKFold on spatial strata to approximate spatial blocking
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_assignments = np.full(len(y), -1)
    for fold_idx, (_, test_idx) in enumerate(skf.split(X, y)):
        fold_assignments[test_idx] = fold_idx

    # Verify folds have both classes
    for f in range(n_folds):
        f_mask = fold_assignments == f
        log.info(f"  Created fold {f}: {f_mask.sum()} samples, "
                 f"presence={(y[f_mask]==1).sum()}, background={(y[f_mask]==0).sum()}")

    folds = fold_assignments
    unique_folds = list(range(n_folds))

    # Results collector
    fold_results = []

    for _, mrow in model_manifest.iterrows():
        model_name = mrow["model"]
        model_path = os.path.join(INPUT_DIR, mrow["file"])
        log.info(f"\nProcessing model: {model_name}")

        try:
            predict_fn = load_model_and_get_predict_fn(model_path, model_name)
        except Exception as e:
            log.error(f"  Cannot load {model_name}: {e}")
            continue

        log.info(f"  Using {n_folds}-fold holdout (ad-hoc spatial stratified)")

        for fold in unique_folds:
            test_mask = folds == fold
            X_test = X[test_mask]
            y_test = y[test_mask]

            if len(y_test) < 10 or len(np.unique(y_test)) < 2:
                log.warning(f"  Fold {fold}: insufficient test data, skipping")
                continue

            # Baseline prediction and metrics
            base_pred = predict_fn(X_test)
            base_metrics = compute_metrics(y_test, base_pred)
            log.info(f"  Fold {fold}: baseline AUC={base_metrics['auc']:.4f}, "
                     f"TSS={base_metrics['tss']:.4f}, Boyce={base_metrics['boyce']:.4f}")

            for var in predictors:
                var_idx = predictors.index(var)
                for rep in range(N_PERMUTATIONS):
                    X_perm = X_test.copy()
                    np.random.shuffle(X_perm[:, var_idx])
                    perm_pred = predict_fn(X_perm)
                    perm_metrics = compute_metrics(y_test, perm_pred)

                    fold_results.append({
                        "model": model_name,
                        "outer_fold": fold,
                        "variable": var,
                        "repeat": rep,
                        "baseline_auc": base_metrics["auc"],
                        "permuted_auc": perm_metrics["auc"],
                        "delta_auc": base_metrics["auc"] - perm_metrics["auc"],
                        "baseline_tss": base_metrics["tss"],
                        "permuted_tss": perm_metrics["tss"],
                        "delta_tss": base_metrics["tss"] - perm_metrics["tss"],
                        "baseline_boyce": base_metrics["boyce"],
                        "permuted_boyce": perm_metrics["boyce"],
                        "delta_boyce": base_metrics["boyce"] - perm_metrics["boyce"],
                    })

    # Save fold-level results
    fold_df = pd.DataFrame(fold_results)
    fold_path = os.path.join(OUT_DIR, "permutation_importance_foldlevel.csv")
    fold_df.to_csv(fold_path, index=False)
    log.info(f"\nFold-level results: {len(fold_df)} rows -> {fold_path}")

    # Summary: mean per model per variable
    summary = fold_df.groupby(["model", "variable"]).agg(
        mean_delta_auc=("delta_auc", "mean"),
        sd_delta_auc=("delta_auc", "std"),
        mean_delta_tss=("delta_tss", "mean"),
        sd_delta_tss=("delta_tss", "std"),
        mean_delta_boyce=("delta_boyce", "mean"),
        sd_delta_boyce=("delta_boyce", "std"),
        n_repeats=("delta_boyce", "count"),
    ).reset_index()
    summary_path = os.path.join(OUT_DIR, "permutation_importance_summary.csv")
    summary.to_csv(summary_path, index=False)
    log.info(f"Summary: {len(summary)} rows -> {summary_path}")

    # Build ensemble consensus importance
    log.info("\nBuilding ensemble consensus importance...")
    ew = ensemble_weights.set_index("model")["weight"].to_dict()

    consensus_rows = []
    for var in predictors:
        var_summary = summary[summary["variable"] == var]
        total_importance = 0
        weighted_importances = {}
        for _, vs in var_summary.iterrows():
            model = vs["model"]
            delta = max(0, vs["mean_delta_boyce"])  # Clip negative
            w = ew.get(model, 0.25)
            weighted_importances[model] = delta
            total_importance += delta

        # Per-model relative importance
        for model, delta in weighted_importances.items():
            model_total = summary[(summary["model"] == model)]["mean_delta_boyce"].clip(lower=0).sum()
            rel_imp = delta / model_total if model_total > 0 else 0
            consensus_rows.append({
                "variable": var,
                "model": model,
                "mean_delta_boyce": delta,
                "relative_importance_model": rel_imp,
                "ensemble_weight": ew.get(model, 0.25),
                "weighted_importance": rel_imp * ew.get(model, 0.25),
            })

    consensus_detail = pd.DataFrame(consensus_rows)

    # Cross-model consensus
    final_consensus = consensus_detail.groupby("variable").agg(
        consensus_importance_raw=("weighted_importance", "sum"),
        n_models_positive=("mean_delta_boyce", lambda x: (x > 0).sum()),
        mean_delta_boyce=("mean_delta_boyce", "mean"),
    ).reset_index()

    # Normalize to 0-100%
    total = final_consensus["consensus_importance_raw"].sum()
    final_consensus["consensus_importance_pct"] = (
        final_consensus["consensus_importance_raw"] / total * 100 if total > 0 else 0
    )
    final_consensus = final_consensus.sort_values("consensus_importance_pct", ascending=False).reset_index(drop=True)
    final_consensus["rank"] = range(1, len(final_consensus) + 1)

    # Cumulative
    final_consensus["cumulative_pct"] = final_consensus["consensus_importance_pct"].cumsum()

    # Tier assignment
    def assign_tier(cum_pct):
        if cum_pct <= 60:
            return "Tier 1"
        elif cum_pct <= 85:
            return "Tier 2"
        else:
            return "Tier 3"

    final_consensus["tier"] = final_consensus["cumulative_pct"].apply(assign_tier)

    # Determine top drivers
    top5 = final_consensus.head(5)["variable"].tolist()
    final_consensus["is_top_driver"] = final_consensus["variable"].apply(
        lambda v: v in top5 and
        final_consensus[final_consensus["variable"] == v]["n_models_positive"].values[0] >= 3
    )

    consensus_path = os.path.join(OUT_DIR, "ensemble_consensus_importance.csv")
    final_consensus.to_csv(consensus_path, index=False)
    log.info(f"Ensemble consensus importance -> {consensus_path}")

    # Driver tier summary
    tier_summary = final_consensus[["variable", "rank", "consensus_importance_pct",
                                     "cumulative_pct", "tier", "n_models_positive",
                                     "is_top_driver"]].copy()
    tier_path = os.path.join(OUT_DIR, "driver_tier_summary.csv")
    tier_summary.to_csv(tier_path, index=False)
    log.info(f"Driver tier summary -> {tier_path}")

    log.info("\n" + "=" * 60)
    log.info("Tier 1 Drivers (Top 60% cumulative):")
    for _, r in final_consensus[final_consensus["tier"] == "Tier 1"].iterrows():
        log.info(f"  {r['variable']}: {r['consensus_importance_pct']:.1f}% "
                 f"(rank={r['rank']}, models_positive={r['n_models_positive']})")
    log.info(f"Top drivers: {top5}")

    log.info("\nPermutation importance analysis complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
