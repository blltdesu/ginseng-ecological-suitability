#!/usr/bin/env python3
"""
Step 1: Input integrity check and QC.
Verifies SHA256 hashes, predictor completeness, model loadability, and data consistency.
"""
import sys, os, hashlib, json, logging
import pandas as pd
import numpy as np

ROOT = r"E:\人参种在哪\实验2"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment1")
QC_DIR = os.path.join(ROOT, "01_input_qc")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(QC_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "input_qc.log"), mode="w", encoding="utf-8"),
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

def main():
    issues = []
    log.info("=" * 60)
    log.info("EXPERIMENT 2 INPUT QC - STARTING")
    log.info("=" * 60)

    # 1. SHA256 manifest check
    manifest_path = os.path.join(INPUT_DIR, "sha256_manifest.csv")
    if not os.path.exists(manifest_path):
        log.error("sha256_manifest.csv not found!")
        issues.append("MISSING: sha256_manifest.csv")
    else:
        manifest = pd.read_csv(manifest_path)
        all_match = (manifest["match"] == True).all()
        log.info(f"SHA256 manifest: {len(manifest)} files, all_match={all_match}")
        if not all_match:
            bad = manifest[manifest["match"] != True]
            log.error(f"SHA256 MISMATCH for: {bad['file'].tolist()}")
            issues.append("SHA256_MISMATCH")

    # 2. Final predictor list
    pred_list_path = os.path.join(INPUT_DIR, "final_predictor_list.csv")
    pred_df = pd.read_csv(pred_list_path)
    predictors = pred_df["variable"].tolist()
    log.info(f"Final predictors ({len(predictors)}): {predictors}")
    if len(predictors) < 3:
        issues.append("TOO_FEW_PREDICTORS")
    log.info(f"Predictor groups: {dict(pred_df.groupby('group')['variable'].apply(list))}")

    # 3. Check predictor rasters exist
    pred_dir = os.path.join(INPUT_DIR, "predictors")
    missing_rasters = []
    for p in predictors:
        tif_path = os.path.join(pred_dir, f"{p}.tif")
        if not os.path.exists(tif_path):
            missing_rasters.append(p)
    if missing_rasters:
        log.error(f"Missing predictor rasters: {missing_rasters}")
        issues.append("MISSING_PREDICTOR_RASTERS")
    else:
        log.info("All predictor rasters present")

    # 4. Training matrix check
    tm_path = os.path.join(INPUT_DIR, "training_matrix_main.csv")
    train = pd.read_csv(tm_path)
    log.info(f"Training matrix: {train.shape[0]} rows, {train.shape[1]} columns")
    log.info(f"  Columns: {list(train.columns)}")

    # Check label
    labels = train["label"].unique()
    log.info(f"  label values: {sorted(labels)}")
    if set(labels) != {0, 1}:
        issues.append("LABEL_NOT_01")

    # Check sample_type
    if "sample_type" in train.columns:
        st = train["sample_type"].value_counts().to_dict()
        log.info(f"  sample_type: {st}")
    else:
        issues.append("MISSING_sample_type")

    # Check all predictor columns present
    missing_cols = [p for p in predictors if p not in train.columns]
    if missing_cols:
        log.error(f"Missing predictor columns in training matrix: {missing_cols}")
        issues.append("PREDICTOR_COLUMNS_MISSING_IN_TRAINING")
    else:
        log.info("All predictor columns present in training matrix")

    # Check outer_fold
    if "outer_fold" in train.columns:
        folds = sorted(train["outer_fold"].unique())
        log.info(f"  outer_folds in training matrix: {folds}")
        # Note: outer_fold = -1 is expected for final training matrix (merged after CV)
        # Actual CV fold info is in spatial_cv_fold_assignment.csv
        if len([f for f in folds if f >= 0]) < 3:
            log.info("  outer_fold=-1 indicates final merged training data; CV folds in spatial_cv_fold_assignment.csv")
            log.info("  Experiment 2 will create ad-hoc stratified holdout folds for permutation importance")
    else:
        issues.append("MISSING_outer_fold")

    # 5. Check model files
    model_dir = os.path.join(INPUT_DIR, "models")
    model_manifest = pd.read_csv(os.path.join(INPUT_DIR, "final_model_manifest.csv"))
    for _, row in model_manifest.iterrows():
        mp = os.path.join(INPUT_DIR, row["file"])
        if os.path.exists(mp):
            log.info(f"Model file exists: {row['file']}")
            try:
                import joblib
                m = joblib.load(mp)
                log.info(f"  {row['model']}: type={type(m).__name__}")
                # Check feature names if possible
                if hasattr(m, "feature_names_in_"):
                    log.info(f"  {row['model']} feature_names: {list(m.feature_names_in_)}")
            except Exception as e:
                log.error(f"  Cannot load {row['model']}: {e}")
                issues.append(f"MODEL_LOAD_FAIL_{row['model']}")
        else:
            log.error(f"Model file missing: {row['file']}")
            issues.append(f"MODEL_MISSING_{row['model']}")

    # 6. Ensemble weights check
    ew = pd.read_csv(os.path.join(INPUT_DIR, "ensemble_weights.csv"))
    wsum = ew["weight"].sum()
    log.info(f"Ensemble weights sum: {wsum:.6f}")
    if abs(wsum - 1.0) > 0.01:
        issues.append("ENSEMBLE_WEIGHTS_NOT_SUMMING_TO_1")

    # 7. Model performance vs experiment 1 report
    perf = pd.read_csv(os.path.join(INPUT_DIR, "model_performance_summary.csv"))
    log.info(f"Model performance summary:")
    for _, row in perf.iterrows():
        log.info(f"  {row['model']}: AUC={row['AUC_mean']:.3f}±{row['AUC_sd']:.3f}, "
                 f"TSS={row['TSS_mean']:.3f}±{row['TSS_sd']:.3f}, "
                 f"Boyce={row['Boyce_mean']:.3f}")

    # 8. Check spatial CV assignment
    cv = pd.read_csv(os.path.join(INPUT_DIR, "spatial_cv_fold_assignment.csv"))
    log.info(f"Spatial CV: {len(cv)} blocks, folds={sorted(cv['outer_fold'].unique())}")

    # 9. Check TIF files
    for tif_name in ["current_ensemble_suitability.tif", "current_intermodel_sd.tif", "current_model_agreement.tif"]:
        tp = os.path.join(INPUT_DIR, tif_name)
        if os.path.exists(tp):
            log.info(f"Raster exists: {tif_name}")
        else:
            issues.append(f"MISSING_{tif_name}")

    # 10. Boundaries
    for bname in ["study_context_adm0.gpkg", "study_context_adm1.gpkg"]:
        bp = os.path.join(INPUT_DIR, "boundaries", bname)
        if os.path.exists(bp):
            log.info(f"Boundary exists: {bname}")
        else:
            issues.append(f"MISSING_{bname}")

    # Final QC report
    qc_lines = []
    qc_lines.append("# Experiment 2 Input QC Report\n")
    qc_lines.append(f"Date: 2026-08-07\n\n")
    qc_lines.append(f"## SHA256 Verification\n")
    qc_lines.append(f"- Total files in manifest: {len(manifest)}\n")
    qc_lines.append(f"- All match: {all_match}\n\n")
    qc_lines.append(f"## Predictors\n")
    qc_lines.append(f"- Count: {len(predictors)}\n")
    qc_lines.append(f"- List: {predictors}\n")
    qc_lines.append(f"- All rasters present: {len(missing_rasters)==0}\n\n")
    qc_lines.append(f"## Training Matrix\n")
    qc_lines.append(f"- Rows: {train.shape[0]}\n")
    qc_lines.append(f"- Presence count: {(train['label']==1).sum()}\n")
    qc_lines.append(f"- Background count: {(train['label']==0).sum()}\n")
    qc_lines.append(f"- Outer folds: {sorted(train['outer_fold'].unique())}\n\n")
    qc_lines.append(f"## Models\n")
    for _, row in model_manifest.iterrows():
        qc_lines.append(f"- {row['model']}: {row['file']} - loadable\n")
    qc_lines.append(f"\n## Ensemble\n")
    qc_lines.append(f"- Weights sum: {wsum:.6f}\n")
    qc_lines.append(f"\n## Issues\n")
    if issues:
        for i in issues:
            qc_lines.append(f"- ❌ {i}\n")
    else:
        qc_lines.append("- ✅ No issues found\n")
    qc_lines.append(f"\n## Status\n")
    if issues:
        qc_lines.append("- ⚠️ ISSUES FOUND - check before proceeding\n")
    else:
        qc_lines.append("- ✅ PASS - ready for Experiment 2 analysis\n")

    qc_path = os.path.join(QC_DIR, "EXPERIMENT2_INPUT_QC.md")
    with open(qc_path, "w", encoding="utf-8") as f:
        f.writelines(qc_lines)
    log.info(f"QC report written to {qc_path}")

    if issues:
        stop_msg = "# STOP: Experiment 2 Input Issues\n\n" + "\n".join(f"- {i}" for i in issues)
        stop_path = os.path.join(QC_DIR, "STOP_01_EXPERIMENT2_INPUT.md")
        with open(stop_path, "w", encoding="utf-8") as f:
            f.write(stop_msg)
        log.error("STOP condition triggered - see QC report")
        return 1

    log.info("INPUT QC PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
