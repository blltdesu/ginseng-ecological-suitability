#!/usr/bin/env python3
"""
Experiment 4 — Step 1: Upstream Input QC
Validates all files transferred from Experiment 3.
"""
import json
import hashlib
import csv
import sys
import os
import warnings
import logging
from pathlib import Path

import numpy as np
import rasterio
import joblib

# ——— Config ———
EXP4_DIR = Path(r"E:\人参种在哪\实验4")
INPUT_DIR = EXP4_DIR / "00_input_from_experiment3"
QC_DIR = EXP4_DIR / "01_input_qc"
LOG_DIR = EXP4_DIR / "logs"
REPORT_PATH = QC_DIR / "EXPERIMENT4_INPUT_QC.md"
STOP_PATH = QC_DIR / "STOP_A_MODEL_OBJECT_INVALID.md"

LOG_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "input_qc.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("input_qc")

# ——— Helpers ———
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

class QCResults:
    def __init__(self):
        self.checks = []
    def record(self, name, passed, detail=""):
        symbol = "✅" if passed else "❌"
        self.checks.append((name, passed, symbol, detail))
        logger.info(f"{symbol} {name}: {detail}")
    def any_failed(self):
        return any(not p for _, p, _, _ in self.checks)
    def markdown(self):
        lines = [
            "# Experiment 4 — Input QC Report\n",
            f"Date: 2026-08-08\n",
            f"Input directory: `{INPUT_DIR}`\n\n",
            "## Check Results\n",
            "| # | Check | Status | Detail |",
            "|---|-------|--------|--------|",
        ]
        for i, (name, passed, symbol, detail) in enumerate(self.checks, 1):
            lines.append(f"| {i} | {name} | {symbol} {['FAIL','PASS'][passed]} | {detail} |")
        lines.append("")
        lines.append(f"**Overall**: {'PASS' if not self.any_failed() else 'HAS FAILURES — review before continuing'}")
        lines.append("")
        return "\n".join(lines)

qc = QCResults()

# ——— 1. SHA256 Verification ———
logger.info("=== SHA256 Verification ===")
manifest = load_csv(INPUT_DIR / "sha256_manifest.csv")
all_sha_pass = True
for row in manifest:
    fpath = INPUT_DIR / row["relative_path"]
    if not fpath.exists():
        qc.record(f"SHA256: {row['relative_path']}", False, "FILE MISSING")
        all_sha_pass = False
        continue
    actual = sha256_file(fpath)
    expected = row["sha256"]
    ok = actual == expected
    if not ok:
        qc.record(f"SHA256: {row['relative_path']}", False,
                  f"MISMATCH — expected {expected[:16]}…, got {actual[:16]}…")
        all_sha_pass = False
if all_sha_pass:
    qc.record("SHA256 all files", True, f"All {len(manifest)} files verified")

# ——— 2. Model Objects Loadable ———
logger.info("=== Model Loadability ===")
MODEL_DIR = INPUT_DIR / "models_for_future_projection"
model_files = sorted(MODEL_DIR.glob("*.joblib"))
model_names = [p.stem.replace("_cst_calibrated", "") for p in model_files]
models = {}
model_load_ok = True
for mf in model_files:
    mn = mf.stem.replace("_cst_calibrated", "")
    try:
        obj = joblib.load(str(mf))
        models[mn] = obj
        qc.record(f"Load model: {mn}", True, f"type={type(obj).__name__}")
    except Exception as e:
        qc.record(f"Load model: {mn}", False, str(e))
        model_load_ok = False

if not model_load_ok:
    logger.error("At least one model failed to load — writing STOP file")
    with open(STOP_PATH, "w", encoding="utf-8") as f:
        f.write("# STOP_A_MODEL_OBJECT_INVALID\n\nOne or more models cannot be loaded.\n")
    sys.exit(1)

# ——— 3. Feature Order in Each Model ———
logger.info("=== Feature Order ===")
expected_order = ["bio02", "bio03", "bio05", "bio15", "clay", "elevation", "northness", "sand"]
feature_ok = True
for mn, obj in models.items():
    # Models saved as dict or as CalibratedClassifierCV
    fo = None
    if isinstance(obj, dict):
        fo = obj.get("feature_order")
    elif hasattr(obj, "feature_order"):
        fo = obj.feature_order
    elif hasattr(obj, "estimator") and hasattr(obj.estimator, "feature_order"):
        fo = obj.estimator.feature_order

    if fo is None:
        qc.record(f"Feature order: {mn}", False, "feature_order NOT FOUND in model object")
        feature_ok = False
    else:
        fo_list = list(fo)
        match = fo_list == expected_order
        qc.record(f"Feature order: {mn}", match,
                  f"order={fo_list} | {'MATCH' if match else 'MISMATCH'}")
        if not match:
            feature_ok = False

if not feature_ok:
    logger.error("Feature order mismatch — would write STOP_C, continuing QC for full report")

# ——— 4. Platt Calibrator Presence ———
logger.info("=== Platt Calibrator ===")
for mn, obj in models.items():
    has_platt = False
    detail = ""
    if isinstance(obj, dict):
        if "calibrated_model" in obj:
            cm = obj["calibrated_model"]
            has_platt = hasattr(cm, "calibrated_classifiers_") or "CalibratedClassifierCV" in str(type(cm))
            detail = f"type={type(cm).__name__}"
    elif "CalibratedClassifierCV" in str(type(obj)):
        has_platt = True
        detail = f"type={type(obj).__name__}, base_estimator={type(obj.estimator).__name__}"
    elif hasattr(obj, "calibrated_classifiers_"):
        has_platt = True
        detail = "has calibrated_classifiers_"
    qc.record(f"Platt calibrator: {mn}", has_platt, detail)

# ——— 5. Final Predictor List ———
logger.info("=== Final Predictor List ===")
fp = load_csv(INPUT_DIR / "final_predictor_list.csv")
fp_vars = [r["variable"] for r in fp]
fp_ok = set(fp_vars) == set(expected_order) and len(fp_vars) == 8
qc.record("Final predictor list", fp_ok,
          f"variables={fp_vars} | {'MATCH' if fp_ok else 'MISMATCH'}")

# ——— 6. Current Suitability Rasters ———
logger.info("=== Current Suitability Rasters ===")
raster_checks = {
    "current_ensemble_suitability.tif": "current_ensemble_suitability.tif",
    "current_binary_suitability.tif": "current_binary_suitability.tif",
}
for label, fname in raster_checks.items():
    path = INPUT_DIR / fname
    try:
        with rasterio.open(str(path)) as src:
            arr = src.read(1, masked=True)
            valid_pct = arr.count() / arr.size * 100
            dmin, dmax = arr.min(), arr.max()
        qc.record(f"Raster readable: {label}", True,
                  f"shape={arr.shape}, valid={valid_pct:.1f}%, range=[{dmin:.4f},{dmax:.4f}]")
    except Exception as e:
        qc.record(f"Raster readable: {label}", False, str(e))

# ——— 7. Ensemble Threshold ———
logger.info("=== Ensemble Threshold ===")
th_path = INPUT_DIR / "ensemble_threshold.json"
try:
    with open(th_path, "r") as f:
        th = json.load(f)
    et = th["ensemble_threshold"]
    ok = 0 < et < 1
    qc.record("Ensemble threshold", ok, f"threshold={et} | {'valid' if ok else 'out of range'}")
    for mk, mv in th.get("model_thresholds", {}).items():
        qc.record(f"Model threshold: {mk}", 0 < float(mv) < 1, f"threshold={mv}")
except Exception as e:
    qc.record("Ensemble threshold", False, str(e))

# ——— 8. Reference Grid ———
logger.info("=== Reference Grid ===")
ref_path = INPUT_DIR / "reference_grid_template.tif"
try:
    with rasterio.open(str(ref_path)) as src:
        qc.record("Reference grid", True,
                  f"CRS={src.crs}, shape={src.shape}, res={src.res}, "
                  f"transform={src.transform}, bounds={src.bounds}")
except Exception as e:
    qc.record("Reference grid", False, str(e))

# ——— 9. Common Valid Mask ———
logger.info("=== Common Valid Mask ===")
mask_path = INPUT_DIR / "common_valid_mask.tif"
try:
    with rasterio.open(str(mask_path)) as src:
        arr = src.read(1)
        valid_count = int(arr.sum()) if arr.dtype in (np.uint8, np.int32, np.int64) else (arr > 0).sum()
        qc.record("Common valid mask", True,
                  f"valid_pixels={valid_count}, shape={arr.shape}")
except Exception as e:
    qc.record("Common valid mask", False, str(e))

# ——— 10. Training Ranges ———
logger.info("=== Training Ranges ===")
tr_path = INPUT_DIR / "current_environment_training_ranges.csv"
try:
    tr = load_csv(tr_path)
    tr_vars = [r["variable"] for r in tr]
    tr_ok = len(tr_vars) == 8 and all(v in tr_vars for v in expected_order)
    qc.record("Training ranges", tr_ok,
              f"variables={len(tr_vars)} | {'complete' if tr_ok else 'incomplete'}")
except Exception as e:
    qc.record("Training ranges", False, str(e))

# ——— 11. Training Sample Matrix ———
ts_path = INPUT_DIR / "training_environment_matrix_sample.csv"
try:
    ts = np.loadtxt(str(ts_path), delimiter=",", skiprows=1)
    qc.record("Training sample matrix", ts.shape[0] > 0,
              f"shape={ts.shape}")
except Exception as e:
    qc.record("Training sample matrix", False, str(e))

# ——— 12. Experiment 3 Results ———
logger.info("=== Experiment 3 Results ===")
e3_dir = INPUT_DIR / "experiment3_results"
for f in ["group_shapley_summary.csv", "drop_one_group_loss.csv", "pairwise_synergy.csv", "dominant_group_ablation.tif"]:
    p = e3_dir / f
    exists = p.exists()
    detail = f"{p.stat().st_size} bytes" if exists else "MISSING"
    qc.record(f"E3 result: {f}", exists, detail)

# ——— 13. Metadata files ———
for f in ["DATA_DICTIONARY_EXPERIMENT3.md", "HANDOFF_FROM_EXPERIMENT3.md",
           "REQUIRED_EXTERNAL_INPUTS_EXPERIMENT4.md", "future_projection_variable_policy.csv"]:
    p = INPUT_DIR / f
    exists = p.exists()
    qc.record(f"Metadata: {f}", exists, f"{p.stat().st_size} bytes" if exists else "MISSING")

# ——— Final Report ———
report = qc.markdown()
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

logger.info(f"QC report written to {REPORT_PATH}")

if qc.any_failed():
    logger.warning("Some QC checks FAILED — review before continuing.")
    # But don't stop unless it's a model-load failure (already handled above)
    # Write STOP_C if feature order mismatch
    if feature_ok is False:
        stop_c = QC_DIR / "STOP_C_FEATURE_ORDER_MISMATCH.md"
        with open(stop_c, "w", encoding="utf-8") as f:
            f.write("# STOP_C_FEATURE_ORDER_MISMATCH\n\nFeature order in model does not match expected order.\n")
        logger.error(f"Written {stop_c}")
else:
    logger.info("All QC checks PASSED.")

print("\n=== QC Summary ===")
for _, (name, passed, symbol, detail) in enumerate(qc.checks):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
