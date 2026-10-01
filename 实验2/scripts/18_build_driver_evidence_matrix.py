#!/usr/bin/env python3
"""
Step 16: Build Driver Evidence Matrix and Final Classification
Aggregates results from permutation, SHAP, ALE, thresholds, interactions, and spatial mapping.
Produces final driver classification (Strong / Moderate / Weak).
"""
import sys, os, logging, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验2"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
DATA_DIR = os.path.join(ROOT, "02_analysis_dataset")
IMP_DIR = os.path.join(ROOT, "03_global_importance")
SHAP_DIR = os.path.join(ROOT, "04_shap")
ALE_DIR = os.path.join(ROOT, "05_ale")
TH_DIR = os.path.join(ROOT, "06_thresholds")
INT_DIR = os.path.join(ROOT, "07_interactions")
SPATIAL_DIR = os.path.join(ROOT, "08_spatial_driver_map")
OUT_DIR = os.path.join(ROOT, "09_consensus")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "experiment2_master.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

def main():
    log.info("=" * 60)
    log.info("BUILDING DRIVER EVIDENCE MATRIX AND FINAL CLASSIFICATION")
    log.info("=" * 60)

    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    predictors = pred_df["variable"].tolist()

    evidence = pd.DataFrame({"variable": predictors})

    # 1. Ensemble permutation rank
    imp_path = os.path.join(IMP_DIR, "ensemble_consensus_importance.csv")
    if os.path.exists(imp_path):
        imp = pd.read_csv(imp_path)
        evidence = evidence.merge(
            imp[["variable", "rank", "consensus_importance_pct", "n_models_positive"]],
            on="variable", how="left"
        )
        evidence.rename(columns={
            "rank": "permutation_rank",
            "consensus_importance_pct": "permutation_importance_pct",
            "n_models_positive": "permutation_models_positive",
        }, inplace=True)

    # 2. Tree SHAP rank
    shap_path = os.path.join(SHAP_DIR, "tree_shap_consensus_importance.csv")
    if os.path.exists(shap_path):
        shap = pd.read_csv(shap_path)
        shap["shap_rank"] = range(1, len(shap) + 1)
        evidence = evidence.merge(
            shap[["variable", "shap_rank", "mean_relative_pct"]],
            on="variable", how="left"
        )
        evidence.rename(columns={"mean_relative_pct": "shap_importance_pct"}, inplace=True)

    # 3. ALE effect strength
    ale_path = os.path.join(ALE_DIR, "ensemble_consensus_ALE.csv")
    if os.path.exists(ale_path):
        ale = pd.read_csv(ale_path)
        ale_strength = ale.groupby("variable")["ensemble_ALE_z"].apply(
            lambda x: np.max(np.abs(x))
        ).reset_index()
        ale_strength.columns = ["variable", "ALE_effect_strength"]
        evidence = evidence.merge(ale_strength, on="variable", how="left")

    # 4. Threshold support
    th_path = os.path.join(TH_DIR, "threshold_consensus_summary.csv")
    if os.path.exists(th_path):
        th = pd.read_csv(th_path)
        th_info = th[["variable", "status"]].drop_duplicates()
        evidence = evidence.merge(th_info.rename(columns={"status": "threshold_support"}), on="variable", how="left")

    # 5. Interaction involvement
    int_path = os.path.join(INT_DIR, "interaction_consensus_ranking.csv")
    if os.path.exists(int_path):
        int_df = pd.read_csv(int_path)
        # Count how many interactions each variable participates in
        all_vars = pd.concat([
            int_df["variable_1"], int_df["variable_2"]
        ]).value_counts().reset_index()
        all_vars.columns = ["variable", "interaction_involvement_n"]
        evidence = evidence.merge(all_vars, on="variable", how="left")
        evidence["interaction_involvement_n"] = evidence["interaction_involvement_n"].fillna(0)

    # 6. Spatial dominance area
    spatial_path = os.path.join(SPATIAL_DIR, "dominant_driver_pointdata.csv")
    if os.path.exists(spatial_path):
        sp = pd.read_csv(spatial_path)
        sp_counts = sp["dominant_variable"].value_counts(normalize=True).reset_index()
        sp_counts.columns = ["variable", "spatial_dominance_area_pct"]
        evidence = evidence.merge(sp_counts, on="variable", how="left")
        evidence["spatial_dominance_area_pct"] = (evidence["spatial_dominance_area_pct"] * 100).fillna(0)

    # 7. Model agreement (from permutation)
    evidence["model_agreement_n"] = evidence.get("permutation_models_positive", 0).fillna(0)

    # Fill missing
    for col in evidence.columns:
        if col != "variable":
            evidence[col] = evidence[col].fillna(0)

    # Save evidence matrix
    ev_path = os.path.join(OUT_DIR, "driver_evidence_matrix.csv")
    evidence.to_csv(ev_path, index=False)
    log.info(f"Driver evidence matrix -> {ev_path}")

    # ===== Final Driver Classification =====
    classification = evidence.copy()

    # Criteria for Strong Driver:
    # - Permutation rank Top 5
    # - SHAP rank Top 5 (if available)
    # - Threshold support is 'robust' or 'moderate'
    # - At least 3/4 models direction consistent
    def classify(row):
        score = 0
        reasons = []

        # Permutation check
        if row.get("permutation_rank", 99) <= 5:
            score += 3
            reasons.append("permutation_top5")

        # SHAP check
        if row.get("shap_rank", 99) <= 5:
            score += 2
            reasons.append("shap_top5")

        # Threshold check
        if row.get("threshold_support", "") in ["robust", "moderate"]:
            score += 2
            reasons.append("threshold_supported")

        # ALE strength check
        if row.get("ALE_effect_strength", 0) > 0.5:
            score += 1
            reasons.append("ale_clear_response")

        # Model agreement
        if row.get("model_agreement_n", 0) >= 3:
            score += 1
            reasons.append("model_agreement>=3")

        # Spatial dominance
        if row.get("spatial_dominance_area_pct", 0) > 10:
            score += 1
            reasons.append("spatial_dominance>10%")

        if score >= 7:
            tier = "Strong"
        elif score >= 3:
            tier = "Moderate"
        else:
            tier = "Weak / model-dependent"

        return tier, "; ".join(reasons), score

    tiers = []
    reason_list = []
    scores = []
    for _, row in classification.iterrows():
        t, r, s = classify(row)
        tiers.append(t)
        reason_list.append(r)
        scores.append(s)

    classification["final_driver_tier"] = tiers
    classification["classification_reasons"] = reason_list
    classification["evidence_score"] = scores

    # Sort by score descending
    classification = classification.sort_values("evidence_score", ascending=False).reset_index(drop=True)

    final_path = os.path.join(OUT_DIR, "final_driver_classification.csv")
    classification.to_csv(final_path, index=False)
    log.info(f"Final driver classification -> {final_path}")

    log.info("\nFinal Driver Classification:")
    for _, r in classification.iterrows():
        log.info(f"  {r['variable']}: {r['final_driver_tier']} "
                 f"(score={r['evidence_score']}, reasons: {r['classification_reasons']})")

    log.info("\nEvidence matrix complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
