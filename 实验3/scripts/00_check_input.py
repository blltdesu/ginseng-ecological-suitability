#!/usr/bin/env python3
"""
Step 1: Input QC and hash verification for Experiment 3.
"""
import sys, os, logging, hashlib
import pandas as pd
import numpy as np

ROOT = r"E:\人参种在哪\实验3"
INPUT_DIR = os.path.join(ROOT, "00_input_from_experiment2")
OUT_DIR = os.path.join(ROOT, "01_input_qc")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(OUT_DIR, exist_ok=True)
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
    log.info("=" * 60)
    log.info("EXPERIMENT 3: INPUT QC")
    log.info("=" * 60)

    qc_results = []

    # 1. Verify SHA256 manifest
    manifest_path = os.path.join(INPUT_DIR, "sha256_manifest.csv")
    if not os.path.exists(manifest_path):
        log.error("sha256_manifest.csv NOT FOUND!")
        with open(os.path.join(OUT_DIR, "STOP_01_EXPERIMENT3_INPUT.md"), "w") as f:
            f.write("# STOP: Experiment 3 Input Missing\n\nsha256_manifest.csv not found.\n")
        return 1

    manifest = pd.read_csv(manifest_path)
    all_match = True
    for _, row in manifest.iterrows():
        fname = row["file"]
        expected = row["copied_sha256"]
        fpath = os.path.join(INPUT_DIR, fname)
        if os.path.exists(fpath):
            actual = sha256_file(fpath)
            match = actual == expected
            if not match:
                log.error(f"HASH MISMATCH: {fname}")
                all_match = False
            qc_results.append({"file": fname, "expected_sha256": expected, "actual_sha256": actual, "match": match})
        else:
            log.warning(f"File not found for hash check: {fname}")

    log.info(f"Hash verification: {'ALL MATCH' if all_match else 'MISMATCHES FOUND'}")

    # 2. Verify training matrix
    tm_path = os.path.join(INPUT_DIR, "training_matrix_main.csv")
    tm = pd.read_csv(tm_path)
    log.info(f"Training matrix: {tm.shape[0]} rows, {tm.shape[1]} cols")

    # Check required columns
    required = ["sample_id", "label", "sample_type", "longitude", "latitude",
                "background_replicate", "outer_fold"]
    for col in required:
        if col not in tm.columns:
            log.error(f"Missing column: {col}")

    # Check label values
    labels = tm["label"].unique()
    log.info(f"Label values: {labels}")
    if set(labels) != {0, 1}:
        log.warning("Labels not 0/1 only")

    # Check presence count
    n_presence = (tm["sample_type"] == "presence").sum()
    n_background = (tm["sample_type"] == "background").sum()
    log.info(f"Presence: {n_presence}, Background: {n_background}")

    # 3. Verify 8 predictors
    pred_list = pd.read_csv(os.path.join(INPUT_DIR, "final_predictor_list.csv"))
    predictors = pred_list["variable"].tolist()
    log.info(f"Predictors ({len(predictors)}): {predictors}")

    missing_preds = [p for p in predictors if p not in tm.columns]
    if missing_preds:
        log.error(f"Missing predictors in training matrix: {missing_preds}")
    else:
        log.info("All 8 predictors present in training matrix")

    # 4. Verify predictor group mapping
    group_map = pd.read_csv(os.path.join(INPUT_DIR, "predictor_group_mapping.csv"))
    groups = group_map["group"].unique()
    log.info(f"Environment groups: {list(groups)}")

    for g in ["climate", "soil", "terrain"]:
        n_vars = (group_map["group"] == g).sum()
        log.info(f"  {g}: {n_vars} variables")
        if n_vars == 0:
            log.error(f"EMPTY GROUP: {g}")
            with open(os.path.join(OUT_DIR, "STOP_A_EMPTY_GROUP.md"), "w") as f:
                f.write(f"# STOP: Empty Group\n\nGroup '{g}' has 0 variables.\n")

    # 5. Verify spatial CV fold assignment
    fa_path = os.path.join(INPUT_DIR, "spatial_cv_fold_assignment.csv")
    fa = pd.read_csv(fa_path)
    n_folds = fa["outer_fold"].nunique()
    log.info(f"Spatial CV folds: {n_folds}, rows: {fa.shape[0]}")

    # 6. Verify all rasters exist
    pred_dir = os.path.join(INPUT_DIR, "predictors")
    missing_rasters = []
    for p in predictors:
        rast_path = os.path.join(pred_dir, f"{p}.tif")
        if not os.path.exists(rast_path):
            missing_rasters.append(p)
    if missing_rasters:
        log.warning(f"Missing rasters: {missing_rasters}")
    else:
        log.info("All predictor rasters present")

    # Write QC report
    report_lines = [
        "# Experiment 3 Input QC Report",
        "",
        f"Date: 2026-08-07",
        "",
        "## Hash Verification",
        f"- All files match: {all_match}",
        "",
        "## Training Matrix",
        f"- Rows: {tm.shape[0]}",
        f"- Presence: {n_presence}",
        f"- Background: {n_background}",
        f"- Labels: {list(labels)}",
        "",
        "## Predictors",
        f"- Count: {len(predictors)}",
        f"- Variables: {predictors}",
        "",
        "## Environment Groups",
    ]
    for g in ["climate", "soil", "terrain"]:
        vars_g = group_map[group_map["group"] == g]["variable"].tolist()
        report_lines.append(f"- {g}: {vars_g}")

    report_lines += [
        "",
        "## Spatial CV",
        f"- Folds: {n_folds}",
        f"- Assignment rows: {fa.shape[0]}",
        "",
        "## Status",
        "PASS" if all_match else "FAIL - hash mismatch",
    ]

    report_path = os.path.join(OUT_DIR, "EXPERIMENT3_INPUT_QC.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log.info(f"QC report written to {report_path}")

    return 0 if all_match else 1

if __name__ == "__main__":
    sys.exit(main())
