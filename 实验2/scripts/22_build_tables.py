#!/usr/bin/env python3
"""
Generate E2-Table1, E2-Table2, E2-Table3
"""
import sys, os
import numpy as np
import pandas as pd

ROOT = r"E:\人参种在哪\实验2"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
IMP_DIR = os.path.join(ROOT, "03_global_importance")
SHAP_DIR = os.path.join(ROOT, "04_shap")
ALE_DIR = os.path.join(ROOT, "05_ale")
TH_DIR = os.path.join(ROOT, "06_thresholds")
INT_DIR = os.path.join(ROOT, "07_interactions")
CONS_DIR = os.path.join(ROOT, "09_consensus")
OUT_DIR = os.path.join(ROOT, "13_tables")
os.makedirs(OUT_DIR, exist_ok=True)

def ensure_path(p):
    return p if os.path.exists(p) else None

def main():
    print("Building tables...")

    # Load key data
    consensus_imp = pd.read_csv(os.path.join(IMP_DIR, "ensemble_consensus_importance.csv"))
    tier_info = pd.read_csv(ensure_path(os.path.join(IMP_DIR, "driver_tier_summary.csv")) or
                           os.path.join(IMP_DIR, "ensemble_consensus_importance.csv"))
    final_class = pd.read_csv(ensure_path(os.path.join(CONS_DIR, "final_driver_classification.csv")) or
                             os.path.join(IMP_DIR, "ensemble_consensus_importance.csv"))

    thresholds = pd.read_csv(ensure_path(os.path.join(TH_DIR, "threshold_consensus_summary.csv")))
    interactions = pd.read_csv(ensure_path(os.path.join(INT_DIR, "interaction_consensus_ranking.csv")))

    pred_df = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))

    group_mapping_path = os.path.join(ROOT, "02_analysis_dataset", "predictor_group_mapping.csv")
    if os.path.exists(group_mapping_path):
        group_map = pd.read_csv(group_mapping_path)
        var_to_group = dict(zip(group_map["variable"], group_map["group"]))
        var_to_unit = dict(zip(group_map["variable"], group_map["unit"]))
    else:
        var_to_group = dict(zip(pred_df["variable"], pred_df["group"]))
        var_to_unit = {v: "unknown" for v in pred_df["variable"]}

    # SHAP consensus
    shap_consensus = None
    shap_path = os.path.join(SHAP_DIR, "tree_shap_consensus_importance.csv")
    if os.path.exists(shap_path):
        shap_consensus = pd.read_csv(shap_path)

    # ===== Table E2-1: Driver Ranking =====
    t1_rows = []
    for _, r in consensus_imp.iterrows():
        var = r["variable"]
        row_data = {
            "Variable": var,
            "Group": var_to_group.get(var, "unknown"),
            "Permutation_rank": r.get("rank", ""),
            "Permutation_percent": f"{r.get('consensus_importance_pct', 0):.1f}",
            "SHAP_rank": "",
            "Models_supporting": int(r.get("n_models_positive", 0)),
            "ALE_pattern": "",
            "Threshold_status": "",
            "Final_driver_tier": r.get("tier", ""),
        }

        if shap_consensus is not None and "variable" in shap_consensus.columns:
            sc = shap_consensus[shap_consensus["variable"] == var]
            if len(sc) > 0:
                row_data["SHAP_rank"] = sc.index[0] + 1  # Position in sorted df

        if thresholds is not None:
            vt = thresholds[thresholds["variable"] == var]
            if len(vt) > 0:
                row_data["Threshold_status"] = vt["status"].values[0]

        # ALE pattern
        ale_path = os.path.join(ALE_DIR, "ensemble_consensus_ALE.csv")
        if os.path.exists(ale_path):
            ale_data = pd.read_csv(ale_path)
            av = ale_data[ale_data["variable"] == var]
            if len(av) > 3:
                ale_vals = av["ensemble_ALE_z"].values
                if np.all(ale_vals > 0):
                    row_data["ALE_pattern"] = "monotonic_positive"
                elif np.all(ale_vals < 0):
                    row_data["ALE_pattern"] = "monotonic_negative"
                elif ale_vals[0] < 0 and ale_vals[-1] > 0:
                    row_data["ALE_pattern"] = "increasing_crossing"
                elif ale_vals[0] > 0 and ale_vals[-1] < 0:
                    row_data["ALE_pattern"] = "decreasing_crossing"
                elif np.max(ale_vals) > 0 and np.min(ale_vals) < 0:
                    row_data["ALE_pattern"] = "nonlinear_crossing"
                else:
                    row_data["ALE_pattern"] = "flat"

        t1_rows.append(row_data)

    t1_df = pd.DataFrame(t1_rows)
    t1_path = os.path.join(OUT_DIR, "E2_Table1_driver_ranking.csv")
    t1_df.to_csv(t1_path, index=False)
    print(f"Table E2-1 -> {t1_path}")

    # ===== Table E2-2: Thresholds and Ranges =====
    t2_rows = []
    if thresholds is not None:
        fav_path = os.path.join(TH_DIR, "favorable_environment_ranges.csv")
        fav_ranges = pd.read_csv(fav_path) if os.path.exists(fav_path) else None

        for _, r in consensus_imp.iterrows():
            var = r["variable"]
            vt = thresholds[thresholds["variable"] == var]
            row_data = {
                "Variable": var,
                "Unit": var_to_unit.get(var, "unknown"),
                "Transition_low": "",
                "Transition_high": "",
                "Median_threshold": "",
                "CI_low": "",
                "CI_high": "",
                "Favorable_range_low": "",
                "Favorable_range_high": "",
                "Model_support": "",
                "Bootstrap_support": "",
                "Status": "",
            }
            if len(vt) > 0:
                row_data["Transition_low"] = f"{vt['transition_interval_low'].values[0]:.3f}" if not np.isnan(vt["transition_interval_low"].values[0]) else ""
                row_data["Transition_high"] = f"{vt['transition_interval_high'].values[0]:.3f}" if not np.isnan(vt["transition_interval_high"].values[0]) else ""
                row_data["Median_threshold"] = f"{vt['median_threshold'].values[0]:.3f}" if not np.isnan(vt["median_threshold"].values[0]) else ""
                row_data["Model_support"] = str(int(vt["model_consensus_n"].values[0]))
                row_data["Status"] = vt["status"].values[0]

            if fav_ranges is not None:
                fvr = fav_ranges[fav_ranges["variable"] == var]
                if len(fvr) > 0:
                    row_data["Favorable_range_low"] = f"{fvr['favorable_range_low'].values[0]:.3f}" if not (fvr['favorable_range_low'].isna().values[0] if hasattr(fvr['favorable_range_low'], 'isna') else pd.isna(fvr['favorable_range_low'].values[0])) else ""
                    row_data["Favorable_range_high"] = f"{fvr['favorable_range_high'].values[0]:.3f}" if not (fvr['favorable_range_high'].isna().values[0] if hasattr(fvr['favorable_range_high'], 'isna') else pd.isna(fvr['favorable_range_high'].values[0])) else ""

            t2_rows.append(row_data)

    t2_df = pd.DataFrame(t2_rows)
    t2_path = os.path.join(OUT_DIR, "E2_Table2_thresholds_and_ranges.csv")
    t2_df.to_csv(t2_path, index=False)
    print(f"Table E2-2 -> {t2_path}")

    # ===== Table E2-3: Interactions =====
    t3_rows = []
    if interactions is not None and len(interactions) > 0:
        for _, r in interactions.iterrows():
            t3_rows.append({
                "Variable_1": r["variable_1"],
                "Variable_2": r["variable_2"],
                "Consensus_H": f"{r['consensus_H']:.4f}",
                "SHAP_interaction": "",
                "Models_supporting": int(r.get("models_positive_H", r.get("n_models", 0))),
                "Interaction_rank": int(r.get("interaction_rank", 0)),
                "Interpretation": f"Interaction between {r['variable_1']} and {r['variable_2']}",
            })

    t3_df = pd.DataFrame(t3_rows)
    t3_path = os.path.join(OUT_DIR, "E2_Table3_interactions.csv")
    t3_df.to_csv(t3_path, index=False)
    print(f"Table E2-3 -> {t3_path}")

    print("Table generation complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
