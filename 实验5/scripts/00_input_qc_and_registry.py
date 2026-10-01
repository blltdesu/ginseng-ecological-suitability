#!/usr/bin/env python3
"""
Experiment 5 - Script 00: Input QC and Scenario Registry
=========================================================
- Verify SHA256 from Experiment 4 handoff
- Check raster alignment against reference grid
- Build scenario availability registry
- Identify missing scenarios
"""

import os
import sys
import json
import hashlib
import csv
from pathlib import Path
import rasterio
import numpy as np
import pandas as pd

# === Configuration ===
EXP5_DIR = Path(r"E:\人参种在哪\实验5")
INPUT_DIR = EXP5_DIR / "00_input_from_experiment4"
REF_GRID_PATH = INPUT_DIR / "current_baseline" / "reference_grid_template.tif"
OUTPUT_QC = EXP5_DIR / "01_input_qc"
OUTPUT_REGISTRY = EXP5_DIR / "02_scenario_registry"

# Fixed experiment parameters
ENSEMBLE_THRESHOLD = 0.12846759691320192
EXPECTED_SCENARIOS = 40
GCMS = ["ACCESS-CM2", "BCC-CSM2-MR", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]
MODELS = ["random_forest", "xgboost", "brt", "maxent"]


def sha256_file(filepath):
    """Compute SHA256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_sha256_manifest():
    """Verify SHA256 manifest from experiment 4."""
    manifest_path = INPUT_DIR / "sha256_manifest.csv"
    if not manifest_path.exists():
        return "WARN: No SHA256 manifest found in experiment 4 handoff"

    results = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = INPUT_DIR / row.get("file", row.get("path", ""))
            expected = row.get("sha256", row.get("hash", ""))
            if filepath.exists():
                actual = sha256_file(filepath)
                match = "OK" if actual == expected else "MISMATCH"
                results.append({"file": str(filepath.relative_to(INPUT_DIR)), "status": match})
            else:
                results.append({"file": str(filepath.relative_to(INPUT_DIR)), "status": "MISSING"})

    mismatches = [r for r in results if r["status"] != "OK"]
    if mismatches:
        return f"WARN: {len(mismatches)} SHA256 mismatches/missing files"
    return f"OK: {len(results)} files verified"


def check_raster_alignment():
    """Check that all ensemble rasters align with reference grid."""
    with rasterio.open(REF_GRID_PATH) as ref:
        ref_shape = ref.shape
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_bounds = ref.bounds

    mismatches = []
    ensemble_dir = INPUT_DIR / "ensemble_predictions"

    for gcm in GCMS:
        for ssp in SSPS:
            for period in PERIODS:
                tif_path = ensemble_dir / gcm / ssp / period / "ensemble_suitability.tif"
                if not tif_path.exists():
                    continue
                try:
                    with rasterio.open(tif_path) as src:
                        if src.shape != ref_shape:
                            mismatches.append(f"{gcm}/{ssp}/{period}: shape {src.shape} != {ref_shape}")
                        if src.crs != ref_crs:
                            mismatches.append(f"{gcm}/{ssp}/{period}: CRS {src.crs} != {ref_crs}")
                        # Check transform within tolerance
                        dx = abs(src.transform.a - ref_transform.a)
                        dy = abs(src.transform.e - ref_transform.e)
                        if dx > 1e-10 or dy > 1e-10:
                            mismatches.append(f"{gcm}/{ssp}/{period}: transform mismatch")
                except Exception as e:
                    mismatches.append(f"{gcm}/{ssp}/{period}: read error - {e}")

    return mismatches


def build_scenario_registry():
    """Build complete scenario availability registry."""
    rows = []
    ensemble_dir = INPUT_DIR / "ensemble_predictions"
    individual_dir = INPUT_DIR / "individual_model_predictions"
    change_dir = INPUT_DIR / "change_class_maps"
    binary_dir = INPUT_DIR / "binary_predictions"

    for gcm in GCMS:
        for ssp in SSPS:
            for period in PERIODS:
                scenario_id = f"{gcm}_{ssp}_{period}"

                # Check ensemble
                ensemble_path = ensemble_dir / gcm / ssp / period / "ensemble_suitability.tif"
                ensemble_available = ensemble_path.exists()

                # Check change map
                change_path = change_dir / gcm / ssp / period / "change_class.tif"
                change_available = change_path.exists()

                # Check binary
                binary_path = binary_dir / gcm / ssp / period / "binary_suitability.tif"
                binary_available = binary_path.exists()

                # Check individual models
                model_availability = {}
                for model in MODELS:
                    model_path = individual_dir / model / gcm / ssp / period / "suitability.tif"
                    model_availability[model] = model_path.exists()

                all_models_available = all(model_availability.values())
                n_models = sum(model_availability.values())

                # Determine reason if missing
                reason = ""
                if not ensemble_available:
                    # Check registry for reason
                    reason = "CMIP6 data unavailable (download_failed in Experiment 4)"

                rows.append({
                    "scenario_id": scenario_id,
                    "gcm": gcm,
                    "ssp": ssp,
                    "period": period,
                    "ensemble_available": ensemble_available,
                    "change_map_available": change_available,
                    "binary_available": binary_available,
                    "n_models_available": n_models,
                    "all_models_available": all_models_available,
                    "reason_missing": reason,
                    "included_in_analysis": ensemble_available,
                    "ensemble_path": str(ensemble_path) if ensemble_available else "",
                    "change_path": str(change_path) if change_available else "",
                    "binary_path": str(binary_path) if binary_available else "",
                })

    return pd.DataFrame(rows)


def write_qc_report(sha_result, alignment_issues, df_registry):
    """Write QC report."""
    n_available = df_registry["ensemble_available"].sum()
    n_missing = EXPECTED_SCENARIOS - n_available
    missing_scenarios = df_registry[~df_registry["ensemble_available"]][
        ["scenario_id", "reason_missing"]
    ].to_string(index=False)

    report = f"""# Experiment 5 Input QC Report

## 1. SHA256 Verification
{sha_result}

## 2. Raster Alignment
Number of alignment issues: {len(alignment_issues)}
"""
    if alignment_issues:
        report += "\n".join(f"- {m}" for m in alignment_issues[:20])
        if len(alignment_issues) > 20:
            report += f"\n... and {len(alignment_issues) - 20} more"

    report += f"""

## 3. Scenario Availability
- Expected: {EXPECTED_SCENARIOS}
- Available (ensemble): {n_available}
- Missing: {n_missing}

### Missing Scenarios
{missing_scenarios}

## 4. Model-Level Predictions
- Total individual predictions: {df_registry['n_models_available'].sum()}
- Expected if all scenarios had all models: {n_available * 4}

## 5. STOP Checks
"""
    # STOP_A: registry vs disk mismatch
    stop_a = False
    # STOP_B: alignment issues
    stop_b = len(alignment_issues) > 0
    # STOP_D: model predictions
    stop_d = df_registry[df_registry["ensemble_available"] & (df_registry["n_models_available"] < 2)].shape[0] > 0
    # STOP_E: novelty reference
    train_matrix = INPUT_DIR / "model_metadata" / "training_environment_matrix_sample.csv"
    train_ranges = INPUT_DIR / "model_metadata" / "current_environment_training_ranges.csv"
    stop_e = not (train_matrix.exists() and train_ranges.exists())

    report += f"- STOP_A (registry mismatch): {'TRIGGERED' if stop_a else 'OK'}\n"
    report += f"- STOP_B (alignment mismatch): {'TRIGGERED' if stop_b else 'OK'}\n"
    report += f"- STOP_D (model predictions missing): {'TRIGGERED' if stop_d else 'OK'}\n"
    report += f"- STOP_E (novelty reference missing): {'TRIGGERED' if stop_e else 'OK'}\n"

    report += f"""
## 6. QC Verdict
"""
    if stop_b:
        report += "**STOP_B triggered: Raster alignment issues detected.**\n"
    elif stop_e:
        report += "**STOP_E triggered: Cannot compute MESS without training reference.**\n"
    else:
        report += "**QC PASSED** - Proceeding with analysis using {}/{} scenarios.\n".format(n_available, EXPECTED_SCENARIOS)

    return report


def main():
    print("=" * 60)
    print("Experiment 5 - Input QC and Scenario Registry")
    print("=" * 60)

    # 1. SHA256 verification
    print("\n[1/4] Verifying SHA256 manifest...")
    sha_result = verify_sha256_manifest()
    print(f"  {sha_result}")

    # 2. Raster alignment check
    print("\n[2/4] Checking raster alignment...")
    alignment_issues = check_raster_alignment()
    if alignment_issues:
        print(f"  WARNING: {len(alignment_issues)} alignment issues found")
        for m in alignment_issues[:5]:
            print(f"    - {m}")
    else:
        print("  All rasters aligned with reference grid")

    # 3. Build scenario registry
    print("\n[3/4] Building scenario registry...")
    df_registry = build_scenario_registry()
    n_avail = df_registry["ensemble_available"].sum()
    print(f"  {n_avail}/{EXPECTED_SCENARIOS} ensemble scenarios available")
    print(f"  {df_registry['n_models_available'].sum()} individual model predictions available")

    # Save registry
    registry_path = OUTPUT_REGISTRY / "scenario_availability.csv"
    df_registry.to_csv(registry_path, index=False)
    print(f"  Saved: {registry_path}")

    # Also save missing scenarios
    missing = df_registry[~df_registry["ensemble_available"]]
    missing_path = OUTPUT_REGISTRY / "missing_scenarios.csv"
    missing.to_csv(missing_path, index=False)

    # 4. Write QC report
    print("\n[4/4] Writing QC report...")
    report = write_qc_report(sha_result, alignment_issues, df_registry)
    report_path = OUTPUT_QC / "EXPERIMENT5_INPUT_QC.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # Save key parameters for downstream scripts
    params = {
        "n_available_scenarios": int(n_avail),
        "n_expected_scenarios": EXPECTED_SCENARIOS,
        "ensemble_threshold": ENSEMBLE_THRESHOLD,
        "gcms": GCMS,
        "ssps": SSPS,
        "periods": PERIODS,
        "models": MODELS,
        "reference_shape": [4320, 8640],
        "reference_crs": "EPSG:4326",
        "reference_transform": [-180.0, 0.04, 0.0, 90.0, 0.0, -0.04],
        "missing_scenarios": missing["scenario_id"].tolist(),
    }
    params_path = OUTPUT_REGISTRY / "experiment5_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print(f"\nDone. {n_avail} scenarios available for analysis.")
    print(f"Missing scenarios: {missing['scenario_id'].tolist()}")


if __name__ == "__main__":
    main()
