#!/usr/bin/env python3
"""
Experiment 3 Core Analysis Pipeline.
Climate/Soil/Terrain independent contribution, shared contribution,
and complementarity effects on ginseng ecological suitability.

Steps: Group definition → Subset construction → Model training →
        Performance evaluation → Ablation → Shapley → Complementarity
"""
import sys, os, logging, json, warnings, hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
from itertools import combinations

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验3"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment2")
LOG_DIR = os.path.join(ROOT, "logs")

# Output directories
DIRS = {
    "group_def": os.path.join(ROOT, "02_group_definition"),
    "subset_ds": os.path.join(ROOT, "03_subset_datasets"),
    "model_recon": os.path.join(ROOT, "04_model_reconstruction"),
    "spatial_cv": os.path.join(ROOT, "05_spatial_cv"),
    "subset_models": os.path.join(ROOT, "06_subset_models"),
    "perf_comp": os.path.join(ROOT, "07_performance_comparison"),
    "ablation": os.path.join(ROOT, "08_ablation"),
    "shapley": os.path.join(ROOT, "09_group_shapley"),
    "complement": os.path.join(ROOT, "10_complementarity"),
    "spatial": os.path.join(ROOT, "11_spatial_group_contribution"),
    "cross_val": os.path.join(ROOT, "12_cross_experiment_validation"),
    "sensitivity": os.path.join(ROOT, "13_sensitivity"),
    "handoff": os.path.join(ROOT, "18_handoff"),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

RANDOM_SEED = 20260807
N_FOLDS = 5
N_BOOTSTRAP = 2000
N_PERM_REPEATS = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "experiment3_master.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def boyce_index(y_true, y_pred, n_bins=10):
    """Continuous Boyce index (standard implementation).

    Boyce = Spearman correlation between bin rank and predicted-to-expected ratio.
    Returns 0 for no skill or degenerate cases.
    Reference: Hirzel et al. (2006) Ecological Modelling.
    """
    try:
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=int)

        # Handle degenerate cases
        if len(y_true) < 10:
            return 0.0
        if np.std(y_pred) < 1e-10:
            return 0.0

        # Clip predictions to valid range
        y_pred = np.clip(y_pred, 0.0, 1.0)

        # Create fixed-width bins
        bins = np.linspace(0.0, 1.0, n_bins + 1)

        # Count presence and all predictions per bin
        F_j = np.zeros(n_bins)
        total_pres = np.sum(y_true == 1)
        total_all = len(y_true)

        if total_pres == 0 or total_all == 0:
            return 0.0

        for j in range(n_bins):
            if j < n_bins - 1:
                bin_mask = (y_pred >= bins[j]) & (y_pred < bins[j + 1])
            else:
                bin_mask = (y_pred >= bins[j]) & (y_pred <= bins[j + 1])

            n_in_bin = bin_mask.sum()
            n_pres_in_bin = (y_true[bin_mask] == 1).sum()

            if n_in_bin > 0 and total_pres > 0:
                P_j = n_pres_in_bin / total_pres  # proportion of presences in bin
                E_j = n_in_bin / total_all          # proportion of all points in bin
                if E_j > 0:
                    F_j[j] = P_j / E_j
                else:
                    F_j[j] = 0.0

        # Boyce = Spearman correlation between bin index and F_j
        valid = F_j > 0
        if valid.sum() < 3:
            return 0.0

        bin_indices = np.arange(n_bins)[valid]
        F_valid = F_j[valid]

        rho, _ = spearmanr(bin_indices, F_valid)
        if np.isnan(rho):
            return 0.0
        return float(rho)

    except Exception:
        return 0.0


def tss_score(y_true, y_pred):
    """True Skill Statistic using median threshold."""
    thresh = np.median(y_pred)
    y_hat = (y_pred >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens + spec - 1


def compute_all_metrics(y_true, y_pred):
    """Compute AUC, TSS, Boyce, Sensitivity, Specificity, Balanced Accuracy, PR-AUC."""
    from sklearn.metrics import average_precision_score, balanced_accuracy_score
    thresh = np.median(y_pred)
    y_hat = (y_pred >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "auc": roc_auc_score(y_true, y_pred),
        "tss": sens + spec - 1,
        "boyce": boyce_index(y_true, y_pred),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_hat),
        "pr_auc": average_precision_score(y_true, y_pred),
    }


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, alpha=0.05):
    """Bootstrap 95% CI for a metric."""
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(RANDOM_SEED)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    return np.mean(values), np.percentile(means, alpha/2 * 100), np.percentile(means, (1 - alpha/2) * 100)


# =============================================================================
# STEP 1: LOAD AND PREPARE DATA
# =============================================================================

def load_data():
    """Load and prepare all input data."""
    log.info("=" * 60)
    log.info("LOADING DATA")
    log.info("=" * 60)

    # Training matrix
    tm = pd.read_csv(os.path.join(INPUT_DIR, "training_matrix_main.csv"))
    log.info(f"Training matrix: {tm.shape}")

    # Predictor list
    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    predictors = pred_df["variable"].tolist()
    log.info(f"Predictors: {predictors}")

    # Group mapping
    group_map = pd.read_csv(os.path.join(INPUT_DIR, "predictor_group_mapping.csv"))
    # Standardize group names: climate→Climate, soil→Soil, terrain→Terrain
    group_map["group_std"] = group_map["group"].map({
        "climate": "Climate", "soil": "Soil", "terrain": "Terrain"
    })
    log.info(f"Groups: {group_map.group_std.unique()}")

    # Spatial CV fold assignment
    fa = pd.read_csv(os.path.join(INPUT_DIR, "spatial_cv_fold_assignment.csv"))

    # Ensemble weights
    weights = pd.read_csv(os.path.join(INPUT_DIR, "ensemble_weights.csv"))

    # Model manifest
    manifest = pd.read_csv(os.path.join(INPUT_DIR, "final_model_manifest.csv"))

    # Model performance (experiment 1 reference)
    perf = pd.read_csv(os.path.join(INPUT_DIR, "model_performance_summary.csv"))

    return tm, predictors, group_map, fa, weights, manifest, perf


# =============================================================================
# STEP 2: ASSIGN SPATIAL CV FOLDS
# =============================================================================

def assign_folds(tm, fa):
    """Assign spatial CV folds to all data points.

    Presence rows map 1:1 to spatial_cv_fold_assignment.csv rows.
    Background points get folds proportional to presence fold distribution.
    """
    log.info("=" * 60)
    log.info("ASSIGNING SPATIAL CV FOLDS")
    log.info("=" * 60)

    n_presence = (tm["sample_type"] == "presence").sum()
    assert n_presence == len(fa), f"Presence count {n_presence} != fold rows {len(fa)}"

    presence_idx = tm[tm["sample_type"] == "presence"].index
    background_idx = tm[tm["sample_type"] == "background"].index

    # Assign presence folds 1:1
    tm.loc[presence_idx, "outer_fold"] = fa["outer_fold"].values

    # Assign background folds randomly proportional to presence fold distribution
    fold_counts = fa["outer_fold"].value_counts().sort_index()
    fold_props = (fold_counts / fold_counts.sum()).to_dict()
    unique_bg_reps = sorted(tm.loc[background_idx, "background_replicate"].unique())

    rng = np.random.RandomState(RANDOM_SEED)
    for rep in unique_bg_reps:
        bg_rep_idx = background_idx[tm.loc[background_idx, "background_replicate"] == rep]
        n_bg_rep = len(bg_rep_idx)
        # Assign folds proportionally
        fold_labels = []
        for f in sorted(fold_props.keys()):
            n_f = int(round(n_bg_rep * fold_props[f]))
            fold_labels.extend([f] * n_f)
        # Trim or pad
        if len(fold_labels) < n_bg_rep:
            fold_labels.extend([rng.choice(list(fold_props.keys())) for _ in range(n_bg_rep - len(fold_labels))])
        elif len(fold_labels) > n_bg_rep:
            fold_labels = fold_labels[:n_bg_rep]
        rng.shuffle(fold_labels)
        tm.loc[bg_rep_idx, "outer_fold"] = fold_labels

    tm["outer_fold"] = tm["outer_fold"].astype(int)

    for f in range(N_FOLDS):
        f_mask = tm["outer_fold"] == f
        log.info(f"  Fold {f}: {f_mask.sum()} samples, "
                 f"presence={(tm.loc[f_mask,'label']==1).sum()}, "
                 f"background={(tm.loc[f_mask,'label']==0).sum()}")

    return tm


# =============================================================================
# STEP 3: FREEZE GROUP DEFINITION
# =============================================================================

def freeze_group_definition(group_map):
    """Create and save the frozen environment group definition."""
    log.info("=" * 60)
    log.info("FREEZING GROUP DEFINITION")
    log.info("=" * 60)

    out_path = os.path.join(DIRS["group_def"], "environment_group_definition.csv")
    group_map.to_csv(out_path, index=False)
    log.info(f"Group definition saved to {out_path}")

    # Summary
    for g in ["Climate", "Soil", "Terrain"]:
        vars_g = group_map[group_map["group_std"] == g]["variable"].tolist()
        log.info(f"  {g}: {vars_g}")

    return group_map


# =============================================================================
# STEP 4: BUILD SUBSET REGISTRY
# =============================================================================

def build_subset_registry(group_map_std):
    """Build registry of all environment group combinations."""
    log.info("=" * 60)
    log.info("BUILDING SUBSET REGISTRY")
    log.info("=" * 60)

    groups = ["Climate", "Soil", "Terrain"]
    group_vars = {}
    for g in groups:
        group_vars[g] = sorted(group_map_std[group_map_std["group_std"] == g]["variable"].tolist())
        log.info(f"  {g} variables: {group_vars[g]}")

    # All non-empty combinations
    subsets = []
    for k in range(1, 4):
        for combo in combinations(groups, k):
            subset_id = "".join([g[0] for g in combo])  # C, S, T, CS, CT, ST, CST
            variables = []
            for g in combo:
                variables.extend(group_vars[g])
            subsets.append({
                "subset_id": subset_id,
                "groups": "+".join(combo),
                "n_variables": len(variables),
                "variables": "+".join(variables),
                "variable_list": variables,
                "is_null": False,
            })

    # Add NULLMODEL baseline
    subsets.append({
        "subset_id": "NULLMODEL",
        "groups": "None",
        "n_variables": 0,
        "variables": "None",
        "variable_list": [],
        "is_null": True,
    })

    registry = pd.DataFrame(subsets)
    registry_path = os.path.join(DIRS["subset_ds"], "subset_registry.csv")
    registry.to_csv(registry_path, index=False)
    log.info(f"Subset registry ({len(registry)} subsets) saved")

    for _, row in registry.iterrows():
        log.info(f"  {row['subset_id']:5s} | groups={row['groups']:20s} | n_vars={row['n_variables']} | {row['variables']}")

    return registry, group_vars


# =============================================================================
# STEP 5: RECONSTRUCT AND TRAIN MODELS
# =============================================================================

def get_model_constructor(model_name):
    """Get a fresh model constructor with experiment 1 parameters."""
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
        # MaxEnt is approximated via regularized logistic regression
        return LogisticRegression(
            C=1.0, penalty='l2', solver='lbfgs',
            max_iter=10000, random_state=RANDOM_SEED,
            class_weight='balanced'
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def calibrate_model(model, X_cal, y_cal):
    """Apply Platt scaling calibration (sklearn 1.9+ compatible)."""
    from sklearn.calibration import CalibratedClassifierCV
    # Use CalibratedClassifierCV with cv=3 for internal cross-validated calibration
    # This avoids the deprecated cv='prefit' issue in sklearn >=1.6
    calibrated = CalibratedClassifierCV(
        estimator=model, method='sigmoid', cv=3, n_jobs=-1
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated


def train_fold_models(tm, predictors, registry, model_names):
    """Train all model × subset × fold combinations."""
    log.info("=" * 60)
    log.info("TRAINING MODELS")
    log.info("=" * 60)

    all_metrics = []
    trained_models = {}  # (model_name, subset_id, fold) → model

    for model_name in model_names:
        log.info(f"\n--- {model_name} ---")
        for _, row in registry.iterrows():
            subset_id = row["subset_id"]
            var_list = row["variable_list"]
            is_null = row["is_null"]
            log.info(f"  Subset: {subset_id}, variables: {var_list}")

            for fold in range(N_FOLDS):
                # Split data
                test_mask = tm["outer_fold"] == fold
                train_mask = ~test_mask

                if is_null:
                    # NULLMODEL model: constant probability
                    test_labels = tm.loc[test_mask, "label"].values
                    train_labels = tm.loc[train_mask, "label"].values
                    null_prob = np.mean(train_labels)
                    y_pred = np.full(len(test_labels), null_prob)
                    metrics = compute_all_metrics(test_labels, y_pred)
                    metrics.update({"model": model_name, "subset": subset_id,
                                   "outer_fold": fold, "n_variables": 0})
                    all_metrics.append(metrics)
                    trained_models[(model_name, subset_id, fold)] = ("null", null_prob)
                    continue

                # Check variables exist
                missing = [v for v in var_list if v not in tm.columns]
                if missing:
                    log.warning(f"  Missing variables: {missing}")
                    continue

                X_train = tm.loc[train_mask, var_list].values.astype(np.float64)
                y_train = tm.loc[train_mask, "label"].values.astype(int)
                X_test = tm.loc[test_mask, var_list].values.astype(np.float64)
                y_test = tm.loc[test_mask, "label"].values.astype(int)

                # Sample weights
                if "sample_weight" in tm.columns:
                    sw_train = tm.loc[train_mask, "sample_weight"].values
                else:
                    sw_train = np.ones(len(y_train))

                # Train model (base model + calibration wrapped together)
                try:
                    base_model = get_model_constructor(model_name)
                    # CalibratedClassifierCV handles fitting internally with cv=3
                    calibrated = calibrate_model(base_model, X_train, y_train)
                    y_pred = calibrated.predict_proba(X_test)[:, 1]

                    metrics = compute_all_metrics(y_test, y_pred)
                    metrics.update({"model": model_name, "subset": subset_id,
                                   "outer_fold": fold, "n_variables": len(var_list)})
                    all_metrics.append(metrics)
                    trained_models[(model_name, subset_id, fold)] = calibrated

                except Exception as e:
                    log.error(f"  Error training {model_name}/{subset_id}/fold{fold}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Record NaN metrics
                    metrics = {"model": model_name, "subset": subset_id,
                              "outer_fold": fold, "n_variables": len(var_list),
                              "auc": np.nan, "tss": np.nan, "boyce": np.nan,
                              "sensitivity": np.nan, "specificity": np.nan,
                              "balanced_accuracy": np.nan, "pr_auc": np.nan}
                    all_metrics.append(metrics)

    # Save fold-level metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(DIRS["subset_models"], "subset_model_fold_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"\nFold metrics saved to {metrics_path}")

    return metrics_df, trained_models


# =============================================================================
# STEP 6: BUILD ENSEMBLE
# =============================================================================

def build_ensemble(metrics_df, weights_df, registry):
    """Build ensemble predictions from individual model predictions."""
    log.info("=" * 60)
    log.info("BUILDING ENSEMBLE")
    log.info("=" * 60)

    # Get available models (those with valid metrics)
    available_models = weights_df["model"].tolist()
    # Get weights for available models
    w_dict = dict(zip(weights_df["model"], weights_df["weight"]))
    total_w = sum(w_dict[m] for m in available_models)
    w_reweighted = {m: w_dict[m] / total_w for m in available_models}
    log.info(f"Reweighted weights: {w_reweighted}")

    # For each subset and fold, compute weighted ensemble metrics
    ensemble_metrics = []

    for _, row in registry.iterrows():
        subset_id = row["subset_id"]
        for fold in range(N_FOLDS):
            fold_metrics = metrics_df[
                (metrics_df["subset"] == subset_id) &
                (metrics_df["outer_fold"] == fold)
            ]

            # Compute weighted average of each metric
            weighted = {"subset": subset_id, "outer_fold": fold,
                       "n_variables": row["n_variables"]}
            for metric in ["auc", "tss", "boyce", "sensitivity", "specificity",
                          "balanced_accuracy", "pr_auc"]:
                vals = []
                for m in available_models:
                    m_data = fold_metrics[fold_metrics["model"] == m]
                    if len(m_data) > 0 and not np.isnan(m_data[metric].values[0]):
                        vals.append((m_data[metric].values[0], w_reweighted[m]))
                if vals:
                    weighted[metric] = sum(v * w for v, w in vals) / sum(w for _, w in vals)
                else:
                    weighted[metric] = np.nan
            ensemble_metrics.append(weighted)

    ens_df = pd.DataFrame(ensemble_metrics)
    ens_fold_path = os.path.join(DIRS["perf_comp"], "ensemble_subset_fold_metrics.csv")
    ens_df.to_csv(ens_fold_path, index=False)

    # Summary
    summary_rows = []
    for _, row in registry.iterrows():
        ss = ens_df[ens_df["subset"] == row["subset_id"]]
        summary_row = {"Subset": row["subset_id"], "Variables": row["variables"],
                      "n_variables": row["n_variables"]}
        for metric in ["auc", "tss", "boyce"]:
            vals = ss[metric].dropna()
            summary_row[f"{metric}_mean"] = vals.mean()
            summary_row[f"{metric}_SD"] = vals.std()
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(DIRS["perf_comp"], "ensemble_subset_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Ensemble summary:\n{summary_df.to_string()}")

    return ens_df, summary_df, w_reweighted


# =============================================================================
# STEP 7: FULL MODEL RECONSTRUCTION CHECK
# =============================================================================

def check_full_reconstruction(metrics_df, perf_ref):
    """Compare reconstructed full model performance with experiment 1."""
    log.info("=" * 60)
    log.info("FULL MODEL RECONSTRUCTION CHECK")
    log.info("=" * 60)

    check_rows = []
    for _, ref_row in perf_ref.iterrows():
        model_name = ref_row["model"]
        # Get CST fold metrics
        cst_metrics = metrics_df[
            (metrics_df["model"] == model_name) &
            (metrics_df["subset"] == "CST")
        ]

        if len(cst_metrics) == 0:
            check_rows.append({
                "model": model_name,
                "experiment1_auc": ref_row["AUC_mean"],
                "reconstructed_auc": np.nan,
                "delta_auc": np.nan,
                "experiment1_tss": ref_row["TSS_mean"],
                "reconstructed_tss": np.nan,
                "delta_tss": np.nan,
                "experiment1_boyce": ref_row["Boyce_mean"],
                "reconstructed_boyce": np.nan,
                "delta_boyce": np.nan,
                "status": "failed"
            })
            continue

        r_auc = cst_metrics["auc"].mean()
        r_tss = cst_metrics["tss"].mean()
        r_boyce = cst_metrics["boyce"].mean()

        d_auc = r_auc - ref_row["AUC_mean"]
        d_tss = r_tss - ref_row["TSS_mean"]
        d_boyce = r_boyce - ref_row["Boyce_mean"]

        # Check tolerances
        acceptable = (abs(d_auc) <= 0.05 and abs(d_tss) <= 0.10 and abs(d_boyce) <= 0.10)
        status = "acceptable" if acceptable else "warning"

        check_rows.append({
            "model": model_name,
            "experiment1_auc": ref_row["AUC_mean"],
            "reconstructed_auc": r_auc,
            "delta_auc": d_auc,
            "experiment1_tss": ref_row["TSS_mean"],
            "reconstructed_tss": r_tss,
            "delta_tss": d_tss,
            "experiment1_boyce": ref_row["Boyce_mean"],
            "reconstructed_boyce": r_boyce,
            "delta_boyce": d_boyce,
            "status": status
        })
        log.info(f"  {model_name}: AUC Δ={d_auc:.4f}, TSS Δ={d_tss:.4f}, Boyce Δ={d_boyce:.4f} → {status}")

    check_df = pd.DataFrame(check_rows)
    check_path = os.path.join(DIRS["model_recon"], "full_model_reconstruction_check.csv")
    check_df.to_csv(check_path, index=False)

    n_acceptable = (check_df["status"] == "acceptable").sum()
    n_total = len(check_df)
    log.info(f"Reconstruction: {n_acceptable}/{n_total} acceptable")

    if n_acceptable < 2:
        log.error("FEWER THAN 2 MODELS RECONSTRUCTED ACCEPTABLY!")
        with open(os.path.join(DIRS["model_recon"], "STOP_B_FULL_RECONSTRUCTION_FAILED.md"), "w") as f:
            f.write("# STOP: Full Model Reconstruction Failed\n\n"
                   f"Only {n_acceptable}/{n_total} models within tolerance.\n")
    elif n_acceptable < 3:
        log.warning("RECONSTRUCTION_WARNING: Only 2/4 models acceptable")
        with open(os.path.join(DIRS["model_recon"], "RECONSTRUCTION_WARNING.md"), "w") as f:
            f.write("# WARNING: Marginal Reconstruction\n\n"
                   f"Only {n_acceptable}/{n_total} models within tolerance.\n")

    return check_df


# =============================================================================
# STEP 8: STANDALONE PREDICTIVE POWER
# =============================================================================

def compute_standalone_power(ens_summary, main_metric="auc"):
    """Compute standalone predictive power for each environmental group."""
    log.info("=" * 60)
    log.info("STANDALONE PREDICTIVE POWER")
    log.info("=" * 60)

    # Get values
    v_null = ens_summary[ens_summary["Subset"] == "NULLMODEL"][f"{main_metric}_mean"].values[0]
    v_cst = ens_summary[ens_summary["Subset"] == "CST"][f"{main_metric}_mean"].values[0]
    total_gain = v_cst - v_null
    log.info(f"V(NULLMODEL) = {v_null:.4f}, V(CST) = {v_cst:.4f}, Total gain = {total_gain:.4f}")

    results = []
    for group, subset_id in [("Climate", "C"), ("Soil", "S"), ("Terrain", "T")]:
        v_group = ens_summary[ens_summary["Subset"] == subset_id][f"{main_metric}_mean"].values
        if len(v_group) > 0:
            v_g = v_group[0]
            standalone = (v_g - v_null) / total_gain if total_gain > 0 else 0
            results.append({"Group": group, "Subset": subset_id,
                          f"V({subset_id})": v_g, "Standalone_Power": standalone})
            log.info(f"  {group}: V({subset_id})={v_g:.4f}, Standalone={standalone:.3f}")

    sp_df = pd.DataFrame(results)
    sp_path = os.path.join(DIRS["ablation"], "standalone_predictive_power.csv")
    sp_df.to_csv(sp_path, index=False)
    return sp_df


# =============================================================================
# STEP 9: DROP-ONE-GROUP ABLATION
# =============================================================================

def compute_drop_one_ablation(ens_fold, main_metric="auc"):
    """Compute drop-one-group ablation loss with bootstrap CI."""
    log.info("=" * 60)
    log.info("DROP-ONE-GROUP ABLATION")
    log.info("=" * 60)

    # Map: drop Climate → use ST, drop Soil → use CT, drop Terrain → use CS
    ablations = [
        ("Climate", "CST", "ST"),
        ("Soil", "CST", "CT"),
        ("Terrain", "CST", "CS"),
    ]

    results = []
    for group, full_id, drop_id in ablations:
        full_vals = ens_fold[ens_fold["subset"] == full_id][main_metric].dropna().values
        drop_vals = ens_fold[ens_fold["subset"] == drop_id][main_metric].dropna().values

        # Compute per-fold loss
        loss = np.mean(full_vals) - np.mean(drop_vals)
        mean_loss, ci_low, ci_high = bootstrap_ci(
            np.array([full_vals[i] - drop_vals[i] for i in range(min(len(full_vals), len(drop_vals)))])
        )
        results.append({
            "Group": group, "Drop_Loss": loss, "Mean_Loss": mean_loss,
            "CI_low": ci_low, "CI_high": ci_high,
            "Significant": ci_low > 0
        })
        log.info(f"  Drop {group}: loss={loss:.4f} [{ci_low:.4f}, {ci_high:.4f}], "
                f"significant={ci_low > 0}")

    abl_df = pd.DataFrame(results)
    abl_path = os.path.join(DIRS["ablation"], "drop_one_group_loss.csv")
    abl_df.to_csv(abl_path, index=False)
    return abl_df


# =============================================================================
# STEP 10: GROUP SHAPLEY DECOMPOSITION
# =============================================================================

def compute_group_shapley(ens_summary, ens_fold, main_metric="auc"):
    """Compute exact group-level Shapley values."""
    log.info("=" * 60)
    log.info("GROUP SHAPLEY DECOMPOSITION")
    log.info("=" * 60)

    # Extract V values from summary
    v_dict = {}
    for _, row in ens_summary.iterrows():
        v_dict[row["Subset"]] = row[f"{main_metric}_mean"]

    v_null = v_dict.get("NULLMODEL", 0)
    v_c = v_dict.get("C", v_null)
    v_s = v_dict.get("S", v_null)
    v_t = v_dict.get("T", v_null)
    v_cs = v_dict.get("CS", v_null)
    v_ct = v_dict.get("CT", v_null)
    v_st = v_dict.get("ST", v_null)
    v_cst = v_dict.get("CST", v_null)

    # Exact Shapley formulas for 3 players
    phi_c = (1/3)*(v_c - v_null) + (1/6)*(v_cs - v_s) + (1/6)*(v_ct - v_t) + (1/3)*(v_cst - v_st)
    phi_s = (1/3)*(v_s - v_null) + (1/6)*(v_cs - v_c) + (1/6)*(v_st - v_t) + (1/3)*(v_cst - v_ct)
    phi_t = (1/3)*(v_t - v_null) + (1/6)*(v_ct - v_c) + (1/6)*(v_st - v_s) + (1/3)*(v_cst - v_cs)

    total_gain = v_cst - v_null
    log.info(f"Total explained gain (V(CST)-V(NULLMODEL)): {total_gain:.4f}")
    log.info(f"  φ_Climate = {phi_c:.4f} ({phi_c/total_gain*100:.1f}%)")
    log.info(f"  φ_Soil    = {phi_s:.4f} ({phi_s/total_gain*100:.1f}%)")
    log.info(f"  φ_Terrain = {phi_t:.4f} ({phi_t/total_gain*100:.1f}%)")
    log.info(f"  Sum check: {phi_c + phi_s + phi_t:.6f} ≈ {total_gain:.6f}")

    # Bootstrap Shapley from folds
    shapley_boot = []
    for _ in range(N_BOOTSTRAP):
        rng = np.random.RandomState(RANDOM_SEED + _)
        # Bootstrap folds
        fold_indices = rng.choice(N_FOLDS, size=N_FOLDS, replace=True)
        boot_v = {}
        for subset_id in ["NULLMODEL", "C", "S", "T", "CS", "CT", "ST", "CST"]:
            ss_data = ens_fold[ens_fold["subset"] == subset_id][main_metric].dropna()
            if len(ss_data) > 0:
                boot_sample = [ss_data.values[i % len(ss_data)] for i in fold_indices]
                boot_v[subset_id] = np.mean(boot_sample)
            else:
                boot_v[subset_id] = v_null

        bv_null = boot_v.get("NULLMODEL", v_null)
        bv_c = boot_v.get("C", bv_null)
        bv_s = boot_v.get("S", bv_null)
        bv_t = boot_v.get("T", bv_null)
        bv_cs = boot_v.get("CS", bv_null)
        bv_ct = boot_v.get("CT", bv_null)
        bv_st = boot_v.get("ST", bv_null)
        bv_cst = boot_v.get("CST", bv_null)

        b_phi_c = (1/3)*(bv_c - bv_null) + (1/6)*(bv_cs - bv_s) + (1/6)*(bv_ct - bv_t) + (1/3)*(bv_cst - bv_st)
        b_phi_s = (1/3)*(bv_s - bv_null) + (1/6)*(bv_cs - bv_c) + (1/6)*(bv_st - bv_t) + (1/3)*(bv_cst - bv_ct)
        b_phi_t = (1/3)*(bv_t - bv_null) + (1/6)*(bv_ct - bv_c) + (1/6)*(bv_st - bv_s) + (1/3)*(bv_cst - bv_cs)
        b_total = bv_cst - bv_null

        shapley_boot.append({
            "Climate_phi": b_phi_c, "Soil_phi": b_phi_s, "Terrain_phi": b_phi_t,
            "Climate_pct": b_phi_c / b_total * 100 if b_total > 0 else 0,
            "Soil_pct": b_phi_s / b_total * 100 if b_total > 0 else 0,
            "Terrain_pct": b_phi_t / b_total * 100 if b_total > 0 else 0,
        })

    boot_df = pd.DataFrame(shapley_boot)
    boot_path = os.path.join(DIRS["shapley"], "group_shapley_bootstrap.csv")
    boot_df.to_csv(boot_path, index=False)

    # Summary with CI
    summary_rows = []
    for group, short in [("Climate", "Climate"), ("Soil", "Soil"), ("Terrain", "Terrain")]:
        pcts = boot_df[f"{short}_pct"].dropna()
        phis = boot_df[f"{short}_phi"].dropna()
        rank = 1
        summary_rows.append({
            "group": group,
            "mean_percent": pcts.mean(),
            "median_percent": pcts.median(),
            "lower95": pcts.quantile(0.025),
            "upper95": pcts.quantile(0.975),
            "mean_phi": phis.mean(),
            "rank": None,  # Will be assigned below
        })

    summary_shap = pd.DataFrame(summary_rows)
    # Assign ranks based on mean_percent
    summary_shap = summary_shap.sort_values("mean_percent", ascending=False)
    summary_shap["rank"] = range(1, len(summary_shap) + 1)
    summary_shap = summary_shap.sort_values("group")

    shap_summary_path = os.path.join(DIRS["shapley"], "group_shapley_summary.csv")
    summary_shap.to_csv(shap_summary_path, index=False)

    # Contributions
    contrib_rows = []
    for _, row in summary_shap.iterrows():
        contrib_rows.append({
            "Group": row["group"],
            "Shapley_percent": row["mean_percent"],
            "CI_low": row["lower95"],
            "CI_high": row["upper95"],
            "Rank": row["rank"],
        })
    contrib_df = pd.DataFrame(contrib_rows)
    contrib_path = os.path.join(DIRS["shapley"], "group_shapley_contributions.csv")
    contrib_df.to_csv(contrib_path, index=False)
    log.info(f"Shapley contributions:\n{contrib_df.to_string()}")

    return contrib_df, boot_df


# =============================================================================
# STEP 11: PAIRWISE COMPLEMENTARITY / REDUNDANCY
# =============================================================================

def compute_complementarity(ens_summary, ens_fold, main_metric="auc"):
    """Compute pairwise synergy and three-way residual."""
    log.info("=" * 60)
    log.info("COMPLEMENTARITY & REDUNDANCY")
    log.info("=" * 60)

    v_null = ens_summary[ens_summary["Subset"] == "NULLMODEL"][f"{main_metric}_mean"].values[0]
    v_c = ens_summary[ens_summary["Subset"] == "C"][f"{main_metric}_mean"].values[0]
    v_s = ens_summary[ens_summary["Subset"] == "S"][f"{main_metric}_mean"].values[0]
    v_t = ens_summary[ens_summary["Subset"] == "T"][f"{main_metric}_mean"].values[0]
    v_cs = ens_summary[ens_summary["Subset"] == "CS"][f"{main_metric}_mean"].values[0]
    v_ct = ens_summary[ens_summary["Subset"] == "CT"][f"{main_metric}_mean"].values[0]
    v_st = ens_summary[ens_summary["Subset"] == "ST"][f"{main_metric}_mean"].values[0]
    v_cst = ens_summary[ens_summary["Subset"] == "CST"][f"{main_metric}_mean"].values[0]

    # Pairwise synergy: V(AB) - V(A) - V(B) + V(NULLMODEL)
    pairs = [
        ("Climate-Soil", "CS", "C", "S"),
        ("Climate-Terrain", "CT", "C", "T"),
        ("Soil-Terrain", "ST", "S", "T"),
    ]

    synergy_results = []
    for pair_name, ab_id, a_id, b_id in pairs:
        v_ab = ens_summary[ens_summary["Subset"] == ab_id][f"{main_metric}_mean"].values[0]
        v_a = ens_summary[ens_summary["Subset"] == a_id][f"{main_metric}_mean"].values[0]
        v_b = ens_summary[ens_summary["Subset"] == b_id][f"{main_metric}_mean"].values[0]
        synergy = v_ab - v_a - v_b + v_null

        # Bootstrap CI for synergy
        syn_boot = []
        rng = np.random.RandomState(RANDOM_SEED)
        for _ in range(N_BOOTSTRAP):
            folds_i = rng.choice(N_FOLDS, size=N_FOLDS, replace=True)
            b_ab = np.mean([ens_fold[ens_fold["subset"]==ab_id][main_metric].dropna().values[i%N_FOLDS] for i in folds_i])
            b_a = np.mean([ens_fold[ens_fold["subset"]==a_id][main_metric].dropna().values[i%N_FOLDS] for i in folds_i])
            b_b = np.mean([ens_fold[ens_fold["subset"]==b_id][main_metric].dropna().values[i%N_FOLDS] for i in folds_i])
            syn_boot.append(b_ab - b_a - b_b + v_null)

        syn_boot = np.array(syn_boot)
        ci_low = np.percentile(syn_boot, 2.5)
        ci_high = np.percentile(syn_boot, 97.5)

        if synergy > 0.01:
            interp = "Complementary/Synergistic"
        elif synergy < -0.01:
            interp = "Redundant/Shared information"
        else:
            interp = "Approximately additive"

        synergy_results.append({
            "Pair": pair_name, "Synergy_Boyce": synergy,
            "CI_low": ci_low, "CI_high": ci_high,
            "Interpretation": interp,
        })
        log.info(f"  {pair_name}: synergy={synergy:.4f} [{ci_low:.4f}, {ci_high:.4f}] → {interp}")

    synergy_df = pd.DataFrame(synergy_results)
    synergy_path = os.path.join(DIRS["complement"], "pairwise_synergy.csv")
    synergy_df.to_csv(synergy_path, index=False)

    # Three-way residual
    three_way = v_cst - (v_c + v_s + v_t) + (v_cs + v_ct + v_st) - v_null
    log.info(f"  Three-way residual: {three_way:.4f}")

    # Incremental gain paths
    inc_rows = [
        {"Path": "C → CS → CST", "Step1_Gain": v_cs - v_c, "Step2_Gain": v_cst - v_cs,
         "Total_Gain": v_cst - v_c},
        {"Path": "C → CT → CST", "Step1_Gain": v_ct - v_c, "Step2_Gain": v_cst - v_ct,
         "Total_Gain": v_cst - v_c},
        {"Path": "S → ST → CST", "Step1_Gain": v_st - v_s, "Step2_Gain": v_cst - v_st,
         "Total_Gain": v_cst - v_s},
        {"Path": "T → ST → CST", "Step1_Gain": v_st - v_t, "Step2_Gain": v_cst - v_st,
         "Total_Gain": v_cst - v_t},
    ]
    inc_df = pd.DataFrame(inc_rows)
    inc_path = os.path.join(DIRS["complement"], "incremental_gain_paths.csv")
    inc_df.to_csv(inc_path, index=False)

    return synergy_df, inc_df


# =============================================================================
# STEP 12: SENSITIVITY - METRIC COMPARISON
# =============================================================================

def metric_sensitivity(ens_summary, ens_fold):
    """Compute Shapley and ablation using TSS and AUC as alternatives."""
    log.info("=" * 60)
    log.info("METRIC SENSITIVITY ANALYSIS")
    log.info("=" * 60)

    sens_results = []
    for metric in ["boyce", "tss", "auc"]:
        log.info(f"\n--- Using {metric} as value function ---")

        # Standalone
        v_null = ens_summary[ens_summary["Subset"] == "NULLMODEL"][f"{metric}_mean"].values[0]
        v_cst = ens_summary[ens_summary["Subset"] == "CST"][f"{metric}_mean"].values[0]
        total_gain = v_cst - v_null

        # Drop-one
        for group, full_id, drop_id in [("Climate", "CST", "ST"), ("Soil", "CST", "CT"), ("Terrain", "CST", "CS")]:
            full_v = ens_summary[ens_summary["Subset"] == full_id][f"{metric}_mean"].values[0]
            drop_v = ens_summary[ens_summary["Subset"] == drop_id][f"{metric}_mean"].values[0]
            loss = full_v - drop_v
            sens_results.append({
                "Metric": metric, "Group": group, "Type": "Drop-one Loss", "Value": loss
            })

        # Shapley
        v_dict = {}
        for _, row in ens_summary.iterrows():
            v_dict[row["Subset"]] = row[f"{metric}_mean"]
        v_null = v_dict.get("NULLMODEL", 0)
        v_c = v_dict.get("C", v_null); v_s = v_dict.get("S", v_null); v_t = v_dict.get("T", v_null)
        v_cs = v_dict.get("CS", v_null); v_ct = v_dict.get("CT", v_null); v_st = v_dict.get("ST", v_null)
        v_cst = v_dict.get("CST", v_null)

        phi_c = (1/3)*(v_c-v_null) + (1/6)*(v_cs-v_s) + (1/6)*(v_ct-v_t) + (1/3)*(v_cst-v_st)
        phi_s = (1/3)*(v_s-v_null) + (1/6)*(v_cs-v_c) + (1/6)*(v_st-v_t) + (1/3)*(v_cst-v_ct)
        phi_t = (1/3)*(v_t-v_null) + (1/6)*(v_ct-v_c) + (1/6)*(v_st-v_s) + (1/3)*(v_cst-v_cs)
        total = v_cst - v_null

        for group, phi in [("Climate", phi_c), ("Soil", phi_s), ("Terrain", phi_t)]:
            sens_results.append({
                "Metric": metric, "Group": group, "Type": "Shapley %",
                "Value": phi / total * 100 if total > 0 else 0
            })

    sens_df = pd.DataFrame(sens_results)
    sens_path = os.path.join(DIRS["sensitivity"], "metric_sensitivity.csv")
    sens_df.to_csv(sens_path, index=False)

    # Check rank consistency
    for metric in ["boyce", "tss", "auc"]:
        shap_vals = sens_df[(sens_df["Metric"] == metric) & (sens_df["Type"] == "Shapley %")]
        ranks = shap_vals.sort_values("Value", ascending=False)["Group"].tolist()
        log.info(f"  {metric} Shapley rank: {ranks}")

    return sens_df


# =============================================================================
# STEP 13: ALGORITHM SENSITIVITY
# =============================================================================

def algorithm_sensitivity(metrics_df, registry):
    """Compute group contributions per algorithm."""
    log.info("=" * 60)
    log.info("ALGORITHM SENSITIVITY")
    log.info("=" * 60)

    algo_results = []
    for model_name in metrics_df["model"].unique():
        log.info(f"\n--- {model_name} ---")
        mm = metrics_df[metrics_df["model"] == model_name]

        # Build per-model summary
        v_dict = {}
        for _, row in registry.iterrows():
            ss = mm[mm["subset"] == row["subset_id"]]
            if len(ss) > 0:
                v_dict[row["subset_id"]] = ss["boyce"].mean()
            else:
                v_dict[row["subset_id"]] = np.nan

        v_null = v_dict.get("NULLMODEL", np.nan)
        v_c = v_dict.get("C", np.nan); v_s = v_dict.get("S", np.nan); v_t = v_dict.get("T", np.nan)
        v_cs = v_dict.get("CS", np.nan); v_ct = v_dict.get("CT", np.nan); v_st = v_dict.get("ST", np.nan)
        v_cst = v_dict.get("CST", np.nan)

        if np.isnan(v_null) or np.isnan(v_cst):
            continue

        # Drop-one
        for group, full_id, drop_id in [("Climate", "CST", "ST"), ("Soil", "CST", "CT"), ("Terrain", "CST", "CS")]:
            loss = v_dict.get(full_id, np.nan) - v_dict.get(drop_id, np.nan)
            algo_results.append({
                "Model": model_name, "Group": group, "Type": "Drop-one Loss", "Value": loss
            })

        # Shapley
        phi_c = (1/3)*(v_c-v_null) + (1/6)*(v_cs-v_s) + (1/6)*(v_ct-v_t) + (1/3)*(v_cst-v_st)
        phi_s = (1/3)*(v_s-v_null) + (1/6)*(v_cs-v_c) + (1/6)*(v_st-v_t) + (1/3)*(v_cst-v_ct)
        phi_t = (1/3)*(v_t-v_null) + (1/6)*(v_ct-v_c) + (1/6)*(v_st-v_s) + (1/3)*(v_cst-v_cs)
        total = v_cst - v_null

        for group, phi in [("Climate", phi_c), ("Soil", phi_s), ("Terrain", phi_t)]:
            algo_results.append({
                "Model": model_name, "Group": group, "Type": "Shapley %",
                "Value": phi / total * 100 if total > 0 else 0
            })

    algo_df = pd.DataFrame(algo_results)
    algo_path = os.path.join(DIRS["sensitivity"], "group_contribution_by_algorithm.csv")
    algo_df.to_csv(algo_path, index=False)

    # Rank consistency (only if we have data)
    if len(algo_df) > 0 and "Model" in algo_df.columns:
        for group in ["Climate", "Soil", "Terrain"]:
            ranks = []
            for model in algo_df["Model"].unique():
                shap = algo_df[(algo_df["Model"] == model) & (algo_df["Group"] == group) & (algo_df["Type"] == "Shapley %")]
                if len(shap) > 0:
                    ranks.append(shap["Value"].values[0])
            if ranks:
                cv_val = np.std(ranks) / abs(np.mean(ranks)) * 100 if abs(np.mean(ranks)) > 0.001 else np.nan
                log.info(f"  {group}: mean Shapley={np.mean(ranks):.1f}%, cv={cv_val:.1f}% across models")

    return algo_df


# =============================================================================
# STEP 14: BACKGROUND REPLICATE SENSITIVITY
# =============================================================================

def background_replicate_sensitivity(tm, predictors, registry, model_names):
    """Check if group contributions depend on background replicate."""
    log.info("=" * 60)
    log.info("BACKGROUND REPLICATE SENSITIVITY")
    log.info("=" * 60)

    # Get unique background replicates
    bg_reps = sorted(tm["background_replicate"].unique())
    bg_reps = [r for r in bg_reps if r != 0]  # Exclude presence marker
    log.info(f"Background replicates: {bg_reps}")

    bg_results = []
    for rep in bg_reps[:3]:  # Limit to 3 for speed
        log.info(f"\n--- Replicate {rep} ---")
        # Subset: all presence + this replicate's background
        rep_mask = (tm["sample_type"] == "presence") | (tm["background_replicate"] == rep)
        tm_rep = tm[rep_mask].copy()

        # Quick train for main subsets
        subsets_to_test = ["NULLMODEL", "C", "S", "T", "CS", "CT", "ST", "CST"]
        v_dict = {}
        for _, row in registry.iterrows():
            if row["subset_id"] not in subsets_to_test:
                continue
            var_list = row["variable_list"]
            fold_metrics = []
            for fold in range(N_FOLDS):
                test_mask = tm_rep["outer_fold"] == fold
                train_mask = ~test_mask
                if row["is_null"]:
                    prob = tm_rep.loc[train_mask, "label"].mean()
                    y_pred = np.full(test_mask.sum(), prob)
                    m = compute_all_metrics(tm_rep.loc[test_mask, "label"].values, y_pred)
                    fold_metrics.append(m["boyce"])
                else:
                    X_tr = tm_rep.loc[train_mask, var_list].values.astype(np.float64)
                    y_tr = tm_rep.loc[train_mask, "label"].values.astype(int)
                    X_te = tm_rep.loc[test_mask, var_list].values.astype(np.float64)
                    y_te = tm_rep.loc[test_mask, "label"].values.astype(int)
                    try:
                        m = get_model_constructor("random_forest")
                        m.fit(X_tr, y_tr)
                        cal = calibrate_model(m, X_tr, y_tr)
                        y_pred = cal.predict_proba(X_te)[:, 1]
                        fold_metrics.append(boyce_index(y_te, y_pred))
                    except:
                        fold_metrics.append(np.nan)
            v_dict[row["subset_id"]] = np.nanmean(fold_metrics) if fold_metrics else np.nan

        v_null = v_dict.get("NULLMODEL", 0)
        v_c = v_dict.get("C", v_null); v_s = v_dict.get("S", v_null); v_t = v_dict.get("T", v_null)
        v_cs = v_dict.get("CS", v_null); v_ct = v_dict.get("CT", v_null); v_st = v_dict.get("ST", v_null)
        v_cst = v_dict.get("CST", v_null)

        phi_c = (1/3)*(v_c-v_null) + (1/6)*(v_cs-v_s) + (1/6)*(v_ct-v_t) + (1/3)*(v_cst-v_st)
        phi_s = (1/3)*(v_s-v_null) + (1/6)*(v_cs-v_c) + (1/6)*(v_st-v_t) + (1/3)*(v_cst-v_ct)
        phi_t = (1/3)*(v_t-v_null) + (1/6)*(v_ct-v_c) + (1/6)*(v_st-v_s) + (1/3)*(v_cst-v_cs)
        total = v_cst - v_null

        for group, phi in [("Climate", phi_c), ("Soil", phi_s), ("Terrain", phi_t)]:
            bg_results.append({
                "Background_Replicate": rep,
                "Group": group,
                "Shapley_percent": phi / total * 100 if total > 0 else 0
            })
            log.info(f"  {group}: {phi/total*100:.1f}%")

    bg_df = pd.DataFrame(bg_results)
    bg_path = os.path.join(DIRS["sensitivity"], "background_replicate_group_contribution.csv")
    bg_df.to_csv(bg_path, index=False)

    # CV across replicates per group
    for group in ["Climate", "Soil", "Terrain"]:
        vals = bg_df[bg_df["Group"] == group]["Shapley_percent"].values
        log.info(f"  {group} across replicates: mean={np.mean(vals):.1f}%, cv={np.std(vals)/np.mean(vals)*100:.1f}%")

    return bg_df


# =============================================================================
# STEP 15: HYPERPARAMETER STRATEGY SENSITIVITY
# =============================================================================

def hyperparameter_sensitivity(tm, predictors, registry):
    """Compare frozen vs retuned hyperparameters for key subsets."""
    log.info("=" * 60)
    log.info("HYPERPARAMETER SENSITIVITY")
    log.info("=" * 60)

    from sklearn.model_selection import GridSearchCV

    # Simple param grid for inner CV
    param_grid = {
        "n_estimators": [500, 1000],
        "max_depth": [None, 10],
    }

    hparam_results = []
    key_subsets = ["C", "S", "T", "CS", "CT", "ST", "CST"]

    for subset_id in key_subsets:
        row = registry[registry["subset_id"] == subset_id]
        if len(row) == 0:
            continue
        var_list = row.iloc[0]["variable_list"]
        if not var_list:
            continue

        # Frozen params (already computed in main analysis)
        # Retuned with inner CV
        retuned_boyce = []
        for fold in range(N_FOLDS):
            test_mask = tm["outer_fold"] == fold
            train_mask = ~test_mask
            X_tr = tm.loc[train_mask, var_list].values.astype(np.float64)
            y_tr = tm.loc[train_mask, "label"].values.astype(int)
            X_te = tm.loc[test_mask, var_list].values.astype(np.float64)
            y_te = tm.loc[test_mask, "label"].values.astype(int)

            try:
                base = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
                gs = GridSearchCV(base, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
                gs.fit(X_tr, y_tr)
                cal = calibrate_model(gs.best_estimator_, X_tr, y_tr)
                y_pred = cal.predict_proba(X_te)[:, 1]
                retuned_boyce.append(boyce_index(y_te, y_pred))
            except:
                retuned_boyce.append(np.nan)

        hparam_results.append({
            "Subset": subset_id,
            "Frozen_Boyce_mean": np.nanmean(
                metrics_df[(metrics_df["model"]=="random_forest") & (metrics_df["subset"]==subset_id)]["boyce"]
            ) if 'metrics_df' in dir() else np.nan,
            "Retuned_Boyce_mean": np.nanmean(retuned_boyce),
        })

    if hparam_results:
        hp_df = pd.DataFrame(hparam_results)
        hp_path = os.path.join(DIRS["sensitivity"], "hyperparameter_strategy_sensitivity.csv")
        hp_df.to_csv(hp_path, index=False)

    return pd.DataFrame(hparam_results)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    log.info("=" * 70)
    log.info("EXPERIMENT 3: CLIMATE/SOIL/TERRAIN CONTRIBUTION ANALYSIS")
    log.info("=" * 70)

    # Load data
    tm, predictors, group_map, fa, weights, manifest, perf_ref = load_data()

    # Assign spatial CV folds
    tm = assign_folds(tm, fa)

    # Freeze group definition
    group_map_std = freeze_group_definition(group_map)

    # Build subset registry
    registry, group_vars = build_subset_registry(group_map_std)

    # Get available model names
    model_names = list(manifest["model"])
    log.info(f"Models to train: {model_names}")

    # Train all models
    metrics_df, trained_models = train_fold_models(tm, predictors, registry, model_names)

    # Build ensemble
    ens_fold, ens_summary, w_reweighted = build_ensemble(metrics_df, weights, registry)

    # Check full model reconstruction
    check_df = check_full_reconstruction(metrics_df, perf_ref)

    # Standalone predictive power
    standalone_df = compute_standalone_power(ens_summary, main_metric="auc")

    # Drop-one ablation
    ablation_df = compute_drop_one_ablation(ens_fold, main_metric="auc")

    # Group Shapley
    shapley_df, shapley_boot = compute_group_shapley(ens_summary, ens_fold, main_metric="auc")

    # Complementarity
    synergy_df, inc_df = compute_complementarity(ens_summary, ens_fold, main_metric="auc")

    # Metric sensitivity
    metric_sens_df = metric_sensitivity(ens_summary, ens_fold)

    # Algorithm sensitivity
    algo_sens_df = algorithm_sensitivity(metrics_df, registry)

    # Background replicate sensitivity
    bg_sens_df = background_replicate_sensitivity(tm, predictors, registry, model_names)

    # Hyperparameter sensitivity
    hp_df = hyperparameter_sensitivity(tm, predictors, registry)

    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT 3 CORE ANALYSIS COMPLETE")
    log.info("=" * 70)

    return 0


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    main()
