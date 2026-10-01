#!/usr/bin/env python3
"""
实验6 - Step 0: 输入QC
检查SHA256、栅格对齐、值域范围
"""
import os, sys, json, hashlib, csv
import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path

ROOT = Path(r"E:\人参种在哪\实验6")
INPUT_DIR = ROOT / "00_input_from_experiment5"
QC_DIR = ROOT / "01_input_qc"
LOG_DIR = ROOT / "logs"

os.makedirs(QC_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- 1. SHA256 verification ---
manifest_path = INPUT_DIR / "sha256_manifest.csv"
sha_results = []
with open(manifest_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        file_rel = row["file"].replace("\\", "/")
        file_path = INPUT_DIR / file_rel
        expected = row.get("copied_sha256", row.get("source_sha256", ""))
        status = row.get("status", "UNKNOWN")

        if status == "MISSING_SOURCE":
            sha_results.append({"file": file_rel, "status": "MISSING_SOURCE", "match": "N/A"})
            continue

        if not file_path.exists():
            sha_results.append({"file": file_rel, "status": "FILE_NOT_FOUND", "match": False})
            continue

        try:
            import hashlib
            with open(file_path, "rb") as bf:
                actual = hashlib.sha256(bf.read()).hexdigest()
            match = (actual == expected)
            sha_results.append({"file": file_rel, "status": "OK" if match else "MISMATCH", "match": match})
        except Exception as e:
            sha_results.append({"file": file_rel, "status": f"ERROR: {e}", "match": False})

sha_all_ok = all(r["match"] in [True, "N/A"] for r in sha_results)

# --- 2. Raster alignment check ---
# Need to also copy reference_grid and common_valid_mask from experiment 5 input
REF_GRID_SRC = Path(r"E:\人参种在哪\实验5\00_input_from_experiment4\current_baseline\reference_grid_template.tif")
COMMON_MASK_SRC = Path(r"E:\人参种在哪\实验5\00_input_from_experiment4\current_baseline\common_valid_mask.tif")

# List all GeoTIFFs to check
tif_files = list(INPUT_DIR.glob("**/*.tif"))

alignment_results = []
ref_meta = None
ref_path = None

for tif_path in sorted(tif_files):
    try:
        with rasterio.open(tif_path) as src:
            meta = {
                "file": str(tif_path.relative_to(INPUT_DIR)).replace("\\", "/"),
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "transform": list(src.transform)[:6],
                "resolution": (abs(src.transform[0]), abs(src.transform[4])),
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
            }
            if ref_meta is None:
                ref_meta = meta
                ref_path = meta["file"]
            # Check alignment
            aligned = (
                meta["crs"] == ref_meta["crs"]
                and meta["width"] == ref_meta["width"]
                and meta["height"] == ref_meta["height"]
                and np.allclose(meta["transform"], ref_meta["transform"], atol=1e-9)
            )
            meta["aligned"] = aligned
            alignment_results.append(meta)
    except Exception as e:
        alignment_results.append({"file": str(tif_path.relative_to(INPUT_DIR)).replace("\\", "/"), "error": str(e)})

all_aligned = all(r.get("aligned", False) for r in alignment_results if "error" not in r)

# --- 3. Value range checks ---
range_checks = []
for tif_path in sorted(tif_files):
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            valid = data[data != src.nodata] if src.nodata is not None else data[~np.isnan(data)]
            if len(valid) == 0:
                range_checks.append({"file": str(tif_path.relative_to(INPUT_DIR)).replace("\\", "/"), "valid_pixels": 0, "range": "EMPTY"})
                continue
            range_checks.append({
                "file": str(tif_path.relative_to(INPUT_DIR)).replace("\\", "/"),
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "valid_pixels": int(len(valid)),
                "nodata": src.nodata,
            })
    except Exception as e:
        range_checks.append({"file": str(tif_path.relative_to(INPUT_DIR)).replace("\\", "/"), "error": str(e)})

# --- Generate QC report ---
qc_report_path = QC_DIR / "EXPERIMENT6_INPUT_QC.md"
align_csv_path = QC_DIR / "raster_alignment_check.csv"

# Write alignment CSV
with open(align_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "crs", "width", "height", "resolution", "dtype", "nodata", "aligned"])
    writer.writeheader()
    for r in alignment_results:
        writer.writerow({
            "file": r.get("file", ""),
            "crs": r.get("crs", ""),
            "width": r.get("width", ""),
            "height": r.get("height", ""),
            "resolution": r.get("resolution", ""),
            "dtype": r.get("dtype", ""),
            "nodata": r.get("nodata", ""),
            "aligned": r.get("aligned", ""),
        })

# Write QC markdown report
with open(qc_report_path, "w", encoding="utf-8") as f:
    f.write("# Experiment 6 Input QC Report\n\n")
    f.write(f"**Date**: 2026-08-08\n")
    f.write(f"**Reference file**: {ref_path}\n")
    f.write(f"**Reference CRS**: {ref_meta['crs']}\n")
    f.write(f"**Reference shape**: {ref_meta['width']} x {ref_meta['height']}\n")
    f.write(f"**Reference resolution**: {ref_meta['resolution']}\n\n")

    f.write("## SHA256 Verification\n\n")
    f.write("| File | Status | Match |\n|------|--------|-------|\n")
    for r in sha_results:
        f.write(f"| {r['file']} | {r['status']} | {r['match']} |\n")
    f.write(f"\n**All SHA256 OK**: {sha_all_ok}\n\n")

    f.write("## Raster Alignment\n\n")
    f.write(f"| File | CRS | Width | Height | Resolution | Aligned |\n")
    f.write(f"|------|-----|-------|--------|------------|--------|\n")
    for r in alignment_results:
        if "error" in r:
            f.write(f"| {r['file']} | ERROR: {r['error']} | - | - | - | - |\n")
        else:
            f.write(f"| {r['file']} | {r['crs']} | {r['width']} | {r['height']} | {r['resolution']} | {r['aligned']} |\n")
    f.write(f"\n**All aligned**: {all_aligned}\n\n")

    f.write("## Value Range Checks\n\n")
    f.write("| File | Min | Max | Mean | Valid Pixels |\n")
    f.write("|------|-----|-----|------|-------------|\n")
    for r in range_checks:
        if "error" in r:
            f.write(f"| {r['file']} | ERROR: {r['error']} | - | - | - |\n")
        else:
            f.write(f"| {r['file']} | {r.get('min', 'N/A')} | {r.get('max', 'N/A')} | {r.get('mean', 'N/A')} | {r.get('valid_pixels', 'N/A')} |\n")

    # Final verdict
    f.write("\n## Final QC Verdict\n\n")
    if sha_all_ok and all_aligned:
        f.write("**PASS** - All checks passed.\n")
    elif not all_aligned:
        f.write("**STOP_A_INPUT_ALIGNMENT_FAILED** - Raster alignment mismatch detected.\n")
        # Write stop file
        stop_path = ROOT / "STOP_A_INPUT_ALIGNMENT_FAILED.md"
        with open(stop_path, "w") as sf:
            sf.write("# STOP_A_INPUT_ALIGNMENT_FAILED\n\nRaster alignment check failed in Experiment 6 input QC.\n")
    else:
        f.write("**WARNING** - Some SHA256 checks failed but alignment is OK.\n")

print("=== QC Summary ===")
print(f"SHA256 all OK: {sha_all_ok}")
print(f"Alignment all OK: {all_aligned}")
print(f"Total rasters checked: {len(tif_files)}")

# Check for missing critical files
critical_files = [
    "current/current_ensemble_suitability.tif",
    "current/current_binary_suitability.tif",
    "future_stability/future_stability_index.tif",
    "future_stability/loss_frequency_current_suitable.tif",
    "uncertainty/prediction_confidence.tif",
    "uncertainty/high_uncertainty_zone.tif",
    "novelty/novelty_frequency.tif",
    "vulnerability/robust_climatic_core.tif",
    "vulnerability/high_confidence_loss_zone.tif",
    "landcover/landcover_aligned.tif",
    "landcover/landcover_class_dictionary.csv",
]
for cf in critical_files:
    p = INPUT_DIR / cf
    if not p.exists():
        print(f"CRITICAL MISSING: {cf}")

print("\nDone. Reports written to 01_input_qc/")
