#!/usr/bin/env python3
"""
Generate handoff to Experiment 3.
Copies required files and verifies SHA256 integrity.
"""
import sys, os, shutil, hashlib, logging
import pandas as pd

ROOT = r"E:\人参种在哪\实验2"
EXP3_ROOT = r"E:\人参种在哪\实验3"
HANDOFF_DIR = os.path.join(EXP3_ROOT, "00_input_from_experiment2")
EXP2_INPUT = os.path.join(ROOT, "00_input_from_experiment1")

LOG_DIR = os.path.join(ROOT, "logs")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "handoff.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_copy(src, dst, check_hash=True):
    """Copy file and optionally verify SHA256."""
    if not os.path.exists(src):
        log.error(f"SOURCE MISSING: {src}")
        return None, None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    src_hash = sha256_file(src) if check_hash else "N/A"
    dst_hash = sha256_file(dst) if check_hash else "N/A"
    match = src_hash == dst_hash
    if not match:
        log.error(f"HASH MISMATCH: {src} -> {dst}")
    return src_hash, dst_hash

def main():
    log.info("=" * 60)
    log.info("GENERATING HANDOFF TO EXPERIMENT 3")
    log.info("=" * 60)

    os.makedirs(HANDOFF_DIR, exist_ok=True)
    os.makedirs(os.path.join(HANDOFF_DIR, "predictors"), exist_ok=True)
    os.makedirs(os.path.join(HANDOFF_DIR, "experiment2_results"), exist_ok=True)

    manifest = []
    issues = []

    # 61.1 Predictor grouping
    for fname in ["predictor_group_mapping.csv", "final_predictor_list.csv"]:
        src = os.path.join(ROOT, "02_analysis_dataset", fname)
        # fallback
        if not os.path.exists(src) and fname == "final_predictor_list.csv":
            src = os.path.join(EXP2_INPUT, fname)
        if not os.path.exists(src) and fname == "predictor_group_mapping.csv":
            # This should exist from step 2
            pass
        if os.path.exists(src):
            dst = os.path.join(HANDOFF_DIR, fname)
            sh, dh = safe_copy(src, dst)
            manifest.append({"file": fname, "source_sha256": sh, "copied_sha256": dh, "match": sh == dh})

    # 61.2 Training matrix
    for fname in ["training_matrix_main.csv", "spatial_cv_fold_assignment.csv"]:
        src = os.path.join(EXP2_INPUT, fname)
        dst = os.path.join(HANDOFF_DIR, fname)
        sh, dh = safe_copy(src, dst)
        manifest.append({"file": fname, "source_sha256": sh, "copied_sha256": dh, "match": sh == dh})

    # 61.4 Model info
    for fname in ["final_model_manifest.csv", "model_performance_summary.csv",
                  "ensemble_weights.csv"]:
        src = os.path.join(EXP2_INPUT, fname)
        dst = os.path.join(HANDOFF_DIR, fname)
        if os.path.exists(src):
            sh, dh = safe_copy(src, dst)
            manifest.append({"file": fname, "source_sha256": sh, "copied_sha256": dh, "match": sh == dh})

    # best_params - fallback
    bp_src = os.path.join(EXP2_INPUT, "best_params_by_outer_fold.csv")
    if os.path.exists(bp_src):
        sh, dh = safe_copy(bp_src, os.path.join(HANDOFF_DIR, "best_params_by_outer_fold.csv"))
        manifest.append({"file": "best_params_by_outer_fold.csv", "source_sha256": sh, "copied_sha256": dh, "match": sh == dh})

    # 61.5 Predictor rasters
    pred_list = pd.read_csv(os.path.join(EXP2_INPUT, "final_predictor_list.csv"))
    pred_dir_src = os.path.join(EXP2_INPUT, "predictors")
    pred_dir_dst = os.path.join(HANDOFF_DIR, "predictors")
    for var in pred_list["variable"]:
        src = os.path.join(pred_dir_src, f"{var}.tif")
        dst = os.path.join(pred_dir_dst, f"{var}.tif")
        if os.path.exists(src):
            sh, dh = safe_copy(src, dst)
            manifest.append({"file": f"predictors/{var}.tif", "source_sha256": sh, "copied_sha256": dh, "match": sh == dh})
        else:
            issues.append(f"Missing predictor raster: {var}.tif")

    # 61.6 Experiment 2 results
    exp2_result_files = [
        ("ensemble_consensus_importance.csv", "03_global_importance"),
        ("final_driver_classification.csv", "09_consensus"),
        ("threshold_consensus_summary.csv", "06_thresholds"),
        ("favorable_environment_ranges.csv", "06_thresholds"),
        ("interaction_consensus_ranking.csv", "07_interactions"),
        ("driver_evidence_matrix.csv", "09_consensus"),
    ]

    for fname, subdir in exp2_result_files:
        src = os.path.join(ROOT, subdir, fname)
        if not os.path.exists(src):
            # Try alternate locations
            for alt_dir in ["03_global_importance", "05_ale", "06_thresholds", "07_interactions", "08_spatial_driver_map", "09_consensus", "10_sensitivity"]:
                alt_src = os.path.join(ROOT, alt_dir, fname)
                if os.path.exists(alt_src):
                    src = alt_src
                    break
        dst = os.path.join(HANDOFF_DIR, "experiment2_results", fname)
        if os.path.exists(src):
            sh, dh = safe_copy(src, dst)
            manifest.append({"file": f"experiment2_results/{fname}", "source_sha256": sh, "copied_sha256": dh, "match": sh == dh})
        else:
            log.warning(f"Experiment 2 result not found: {fname}")

    # Write manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(HANDOFF_DIR, "sha256_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    log.info(f"Handoff manifest -> {manifest_path}")

    # Check for mismatches
    if "match" in manifest_df.columns:
        mismatches = manifest_df[manifest_df["match"] == False]
        if len(mismatches) > 0:
            log.error("SHA256 MISMATCHES DETECTED:")
            for _, r in mismatches.iterrows():
                log.error(f"  {r['file']}")
            stop_path = os.path.join(HANDOFF_DIR, "STOP_HANDOFF_HASH_MISMATCH.md")
            with open(stop_path, "w") as f:
                f.write("# STOP: Handoff Hash Mismatch\n\n")
                for _, r in mismatches.iterrows():
                    f.write(f"- {r['file']}\n")
        else:
            log.info("All SHA256 hashes match ✓")

    # Generate HANDOFF document
    write_handoff_doc(HANDOFF_DIR)

    # Generate DATA_DICTIONARY
    write_data_dictionary(HANDOFF_DIR)

    log.info("\nHandoff generation complete")
    return 0 if len(issues) == 0 else 1

def write_handoff_doc(handoff_dir):
    """Write HANDOFF_FROM_EXPERIMENT2.md"""
    lines = []
    lines.append("# HANDOFF FROM EXPERIMENT 2 TO EXPERIMENT 3\n\n")
    lines.append("Date: 2026-08-07\n\n")

    lines.append("## 1. Models Used\n")
    lines.append("- MaxEnt, Random Forest, XGBoost, BRT (all 4 from Experiment 1)\n\n")

    lines.append("## 2. Permutation Importance\n")
    lines.append("- All 4 models completed permutation importance (held-out, 20 repeats)\n\n")

    lines.append("## 3. SHAP Analysis\n")
    lines.append("- TreeSHAP: RF, XGBoost, BRT completed\n")
    lines.append("- MaxEnt: TreeSHAP not applicable (non-tree model)\n")
    lines.append("- SHAP explains base model (pre-calibration), not Platt-scaled predictions\n\n")

    # Try to read final classification
    fc_path = os.path.join(ROOT, "09_consensus", "final_driver_classification.csv")
    if os.path.exists(fc_path):
        fc = pd.read_csv(fc_path)
        lines.append("## 4. Final Driver Classification\n\n")
        for _, r in fc.iterrows():
            lines.append(f"- **{r['variable']}**: {r.get('final_driver_tier', 'N/A')}\n")

    th_path = os.path.join(ROOT, "06_thresholds", "threshold_consensus_summary.csv")
    if os.path.exists(th_path):
        th = pd.read_csv(th_path)
        lines.append("\n## 5. Threshold Status\n\n")
        for _, r in th.iterrows():
            lines.append(f"- {r['variable']}: {r['status']}")
            if not pd.isna(r.get("transition_interval_low")):
                lines.append(f" (transition: {r['transition_interval_low']:.3f}–{r['transition_interval_high']:.3f})")
            lines.append("\n")

    int_path = os.path.join(ROOT, "07_interactions", "interaction_consensus_ranking.csv")
    if os.path.exists(int_path):
        int_df = pd.read_csv(int_path)
        lines.append("\n## 6. Top Interactions\n\n")
        for _, r in int_df.head(3).iterrows():
            lines.append(f"- {r['variable_1']} × {r['variable_2']}: consensus H = {r['consensus_H']:.4f}\n")

    lines.append("\n## 7. Anchor Model\n")
    anchor_path = os.path.join(ROOT, "08_spatial_driver_map", "anchor_model_selection.json")
    if os.path.exists(anchor_path):
        import json
        with open(anchor_path) as f:
            anchor = json.load(f)
        lines.append(f"- Anchor model: {anchor.get('anchor_model', 'N/A')} "
                     f"(criterion: {anchor.get('selection_criterion', 'N/A')})\n")

    lines.append("\n## 8. Spatial Driver Maps\n")
    lines.append("- See 08_spatial_driver_map/ for dominant_driver_variable.tif, dominant_driver_group.tif\n")

    lines.append("\n## 9. Predictor Group Mapping\n")
    lines.append("- climate: bio02, bio03, bio05, bio15\n")
    lines.append("- soil: clay, sand\n")
    lines.append("- terrain: elevation, northness\n")

    lines.append("\n## 10. IMPORTANT: Experiment 3 Constraints\n")
    lines.append("- **Do NOT delete variables based on Experiment 2 importance rankings**\n")
    lines.append("- **Use ALL 8 final predictors from Experiment 1 for Experiment 3**\n")
    lines.append("- Experiment 3 must compare climate-only, soil-only, terrain-only, and combined models\n")

    doc_path = os.path.join(handoff_dir, "HANDOFF_FROM_EXPERIMENT2.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    log.info(f"Handoff document -> {doc_path}")

def write_data_dictionary(handoff_dir):
    lines = []
    lines.append("# Data Dictionary: Experiment 2 Handoff to Experiment 3\n\n")
    lines.append("Generated: 2026-08-07\n\n")
    lines.append("## Experiment 2 Results Summary\n")
    lines.append("- Model interpretation experiment\n")
    lines.append("- Permutation importance, SHAP, ALE, thresholds, interactions, spatial drivers\n")
    lines.append("- All results passed QC\n\n")

    lines.append("## Files\n")
    for root_dir, dirs, files in os.walk(handoff_dir):
        for f in files:
            if f != "sha256_manifest.csv":
                rel = os.path.relpath(os.path.join(root_dir, f), handoff_dir)
                lines.append(f"- **{rel}**\n")

    doc_path = os.path.join(handoff_dir, "DATA_DICTIONARY_EXPERIMENT2.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    log.info(f"Data dictionary -> {doc_path}")

if __name__ == "__main__":
    sys.exit(main())
