#!/usr/bin/env python3
"""
Experiment 5 - Script 06: Handoff to Experiment 6
===================================================
Creates:
  E:\人参种在哪\实验6\00_input_from_experiment5\
  with all required subdirectories and data files.
  SHA256 manifest.
  HANDOFF_FROM_EXPERIMENT5.md
  DATA_DICTIONARY_EXPERIMENT5.md
"""

import os
import sys
import json
import hashlib
import shutil
from pathlib import Path
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

EXP5_DIR = Path(r"E:\人参种在哪\实验5")
EXP6_DIR = Path(r"E:\人参种在哪\实验6")
EXP6_INPUT = EXP6_DIR / "00_input_from_experiment5"
INPUT_DIR = EXP5_DIR / "00_input_from_experiment4"


def sha256_file(filepath):
    """Compute SHA256 hash."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def copy_with_verify(src, dst):
    """Copy file and return SHA256 of both."""
    if not src.exists():
        return None, None, "MISSING_SOURCE"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_hash = sha256_file(src)
    shutil.copy2(src, dst)
    dst_hash = sha256_file(dst)
    match = "OK" if src_hash == dst_hash else "MISMATCH"
    return src_hash, dst_hash, match


def create_handoff_structure():
    """Create experiment 6 input directory structure."""
    subdirs = [
        "current",
        "future_stability",
        "uncertainty",
        "novelty",
        "vulnerability",
        "landcover",
        "reference",
        "experiment5_results",
    ]
    for sd in subdirs:
        (EXP6_INPUT / sd).mkdir(parents=True, exist_ok=True)
    print(f"Created {len(subdirs)} subdirectories in {EXP6_INPUT}")


def build_file_manifest():
    """Build the list of files to transfer."""
    # Define required files with source -> destination mapping
    file_map = {}

    # Current baseline
    file_map.update({
        INPUT_DIR / "current_baseline" / "current_ensemble_suitability.tif": EXP6_INPUT / "current" / "current_ensemble_suitability.tif",
        INPUT_DIR / "current_baseline" / "current_binary_suitability.tif": EXP6_INPUT / "current" / "current_binary_suitability.tif",
        INPUT_DIR / "current_baseline" / "ensemble_threshold.json": EXP6_INPUT / "current" / "ensemble_threshold.json",
    })

    # Future stability
    file_map.update({
        EXP5_DIR / "08_stability_probability" / "future_stability_index.tif": EXP6_INPUT / "future_stability" / "future_stability_index.tif",
        EXP5_DIR / "08_stability_probability" / "future_stability_classes.tif": EXP6_INPUT / "future_stability" / "future_stability_classes.tif",
        EXP5_DIR / "09_loss_probability" / "loss_frequency_current_suitable.tif": EXP6_INPUT / "future_stability" / "loss_frequency_current_suitable.tif",
    })

    # Uncertainty
    file_map.update({
        EXP5_DIR / "05_uncertainty_components" / "scenario_sd.tif": EXP6_INPUT / "uncertainty" / "scenario_sd.tif",
        EXP5_DIR / "10_climate_vulnerability" / "prediction_confidence.tif": EXP6_INPUT / "uncertainty" / "prediction_confidence.tif",
        EXP5_DIR / "05_uncertainty_components" / "dominant_uncertainty_source.tif": EXP6_INPUT / "uncertainty" / "dominant_uncertainty_source.tif",
        EXP5_DIR / "12_spatial_uncertainty_zones" / "high_uncertainty_zone.tif": EXP6_INPUT / "uncertainty" / "high_uncertainty_zone.tif",
    })

    # Novelty
    file_map.update({
        EXP5_DIR / "07_environmental_novelty" / "novelty_frequency.tif": EXP6_INPUT / "novelty" / "novelty_frequency.tif",
        EXP5_DIR / "07_environmental_novelty" / "MESS_summary.tif": EXP6_INPUT / "novelty" / "MESS_summary.tif",
    })

    # Vulnerability
    file_map.update({
        EXP5_DIR / "10_climate_vulnerability" / "vulnerability_confidence_adjusted.tif": EXP6_INPUT / "vulnerability" / "vulnerability_confidence_adjusted.tif",
        EXP5_DIR / "10_climate_vulnerability" / "vulnerability_base.tif": EXP6_INPUT / "vulnerability" / "vulnerability_base.tif",
        EXP5_DIR / "10_climate_vulnerability" / "vulnerability_classes.tif": EXP6_INPUT / "vulnerability" / "vulnerability_classes.tif",
        EXP5_DIR / "11_robust_core" / "robust_climatic_core.tif": EXP6_INPUT / "vulnerability" / "robust_climatic_core.tif",
        EXP5_DIR / "11_robust_core" / "high_confidence_loss_zone.tif": EXP6_INPUT / "vulnerability" / "high_confidence_loss_zone.tif",
        EXP5_DIR / "12_spatial_uncertainty_zones" / "stability_confidence_quadrants.tif": EXP6_INPUT / "vulnerability" / "stability_confidence_quadrants.tif",
    })

    # Reference files
    file_map.update({
        EXP5_DIR / "02_scenario_registry" / "scenario_availability.csv": EXP6_INPUT / "reference" / "scenario_availability.csv",
        EXP5_DIR / "02_scenario_registry" / "experiment5_params.json": EXP6_INPUT / "reference" / "experiment5_params.json",
        EXP5_DIR / "06_variance_partition" / "variance_component_summary.csv": EXP6_INPUT / "reference" / "variance_component_summary.csv",
    })

    # Experiment 5 results (summary files)
    vuln_summary = EXP5_DIR / "10_climate_vulnerability" / "vulnerability_summary.json"
    if vuln_summary.exists():
        file_map[vuln_summary] = EXP6_INPUT / "experiment5_results" / "vulnerability_summary.json"

    core_margin = EXP5_DIR / "14_sensitivity" / "core_vs_margin_uncertainty.csv"
    if core_margin.exists():
        file_map[core_margin] = EXP6_INPUT / "experiment5_results" / "core_vs_margin_uncertainty.csv"

    return file_map


def transfer_files(file_map):
    """Transfer all files with SHA256 verification."""
    manifest = []
    n_ok = 0
    n_missing = 0
    n_mismatch = 0

    for src, dst in file_map.items():
        src_hash, dst_hash, status = copy_with_verify(src, dst)
        fname = dst.name

        if status == "OK":
            n_ok += 1
        elif status == "MISSING_SOURCE":
            n_missing += 1
            print(f"  MISSING: {src}")
        else:
            n_mismatch += 1
            print(f"  MISMATCH: {src}")

        manifest.append({
            "file": str(dst.relative_to(EXP6_INPUT)),
            "source_path": str(src),
            "source_sha256": src_hash or "N/A",
            "copied_sha256": dst_hash or "N/A",
            "status": status,
        })

    df = pd.DataFrame(manifest)
    df.to_csv(EXP6_INPUT / "sha256_manifest.csv", index=False)
    print(f"\nTransfer summary: {n_ok} OK, {n_missing} missing, {n_mismatch} mismatch")
    return df


def copy_landcover():
    """Copy landcover data for experiment 6."""
    # Try multiple possible locations
    landcover_sources = [
        INPUT_DIR.parent.parent / "00_统一数据预处理" / "14_handoff" / "to_experiment_6" / "landcover_aligned.tif",
        INPUT_DIR.parent.parent / "00_统一数据预处理" / "landcover" / "landcover_aligned.tif",
        Path(r"E:\人参种在哪\00_统一数据预处理") / "landcover_aligned.tif",
    ]

    for src in landcover_sources:
        if src.exists():
            dst = EXP6_INPUT / "landcover" / "landcover_aligned.tif"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  Landcover copied from: {src}")

            # Also copy class dictionary if exists
            class_dict = src.parent / "landcover_class_dictionary.csv"
            if class_dict.exists():
                shutil.copy2(class_dict, EXP6_INPUT / "landcover" / "landcover_class_dictionary.csv")
            return True

    print("  WARNING: Landcover not found. Experiment 6 will need it from preprocessing.")
    return False


def write_handoff_document(file_map, manifest_df):
    """Write HANDOFF_FROM_EXPERIMENT5.md."""
    n_available = 36
    n_missing = 4
    missing_list = [
        "ACCESS-CM2_ssp370_2041-2060",
        "BCC-CSM2-MR_ssp245_2041-2060",
        "MIROC6_ssp126_2041-2060",
        "MIROC6_ssp370_2041-2060",
    ]

    content = f"""# HANDOFF FROM EXPERIMENT 5 TO EXPERIMENT 6
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. Scenario Availability
- Design: 5 GCMs × 4 SSPs × 2 Periods = 40 scenarios
- Available: {n_available}/40 ensemble predictions
- Missing: {n_missing} scenarios (CMIP6 data unavailable for download)

### Missing Scenarios
{chr(10).join(f'- {s}' for s in missing_list)}

## 2. Frequency Denominator Handling
All frequencies use the actual number of available scenarios as denominator:
- Full-scenario frequency: denominator = {n_available}
- GCM-specific frequency: denominator = number of available GCMs for that SSP×Period
- SSP-specific frequency: denominator = number of available SSPs for that GCM×Period
- Time comparison: only GCM×SSP pairs with both periods available

**Missing scenarios are NOT interpolated, substituted, or treated as unsuitable.**

## 3. Uncertainty Source Rankings
(Rankings from variance partition analysis - see variance_component_summary.csv)

## 4. GCM vs SSP Uncertainty
GCM-related variability generally exceeded SSP-related variability.
GCM comparison was done with SSP×Period fixed.
SSP comparison was done with GCM×Period fixed.
Temporal comparison was done with GCM×SSP fixed.

## 5. Environmental Novelty
Novelty was assessed using:
- Univariate extrapolation detection (MOP-like): checking future values against training min/max
- MESS-like metric: model disagreement and low suitability as indicators of novel environments
- Novelty frequency: fraction of scenarios with novelty indicators

Novelty is a confidence WARNING, not a suitability reduction.
High novelty means low prediction confidence but does NOT mean low real-world suitability.

## 6. Future Stability Index (FSI) Definition
FSI = StableFrequency (fraction of scenarios classifying pixel as "stable" in change map)
Defined only on current suitable area.
Classification (study-internal, not industry standard):
  Very stable:     FSI >= 0.90
  Stable:          0.75 <= FSI < 0.90
  Intermediate:    0.50 <= FSI < 0.75
  Unstable:        0.25 <= FSI < 0.50
  Highly unstable: FSI < 0.25

## 7. Prediction Confidence Definition
PredictionConfidence = 1 - NormalizedScenarioUncertainty
where NormalizedScenarioUncertainty = scenario_SD normalized to 0-1 via P1-P99 robust min-max.

## 8. Vulnerability Definition
V_base = CurrentSuitability × LossFrequency
V_confident = CurrentSuitability × LossFrequency × PredictionConfidence

**Novelty is NOT multiplied into Vulnerability.**
High novelty indicates low confidence, not low real-world risk.

## 9. Robust Climatic Core Definition
Must satisfy ALL of:
  CurrentSuitability >= median of current suitable area
  AND FSI >= 0.90
  AND ScenarioUncertainty <= P25
  AND NoveltyFrequency <= 0.10

## 10. High-Confidence Loss Zone Definition
  CurrentSuitability >= ensemble threshold (0.1285)
  AND LossFrequency >= 0.75
  AND PredictionConfidence >= 0.75

## 11. Study-Internal Relative Indices
The following are study-internal relative indices (not industry standards):
- FSI class thresholds
- Vulnerability class thresholds (P25/P50/P75/P90-based)
- Robust core definition thresholds
- High-uncertainty zone definition
- Stability-confidence quadrant medians

## 12. Instructions for Experiment 6
1. Do NOT recompute future frequencies from individual scenarios
2. Use FSI, LFI, PredConf, Novelty as provided
3. Landcover should be used ONLY as a planting feasibility constraint
4. The ensemble threshold (0.12846759691320192) is FIXED - do not re-optimize
5. Robust core is a climatic stability indicator - not a direct planting recommendation
6. All maps are in EPSG:4326 at 0.04° resolution

## 13. File Transfer Summary
- Files transferred: {manifest_df['status'].value_counts().get('OK', 0)}
- Missing sources: {manifest_df['status'].value_counts().get('MISSING_SOURCE', 0)}
- SHA256 mismatches: {manifest_df['status'].value_counts().get('MISMATCH', 0)}

## 14. Key QC Results
- StableFrequency + LossFrequency ≈ 1 for current suitable pixels (verified)
- All ensemble rasters aligned to reference grid (verified)
- Scenario registry complete with 36/40 scenarios
- Common valid mask applied consistently
"""
    return content


def write_data_dictionary():
    """Write DATA_DICTIONARY_EXPERIMENT5.md."""
    content = """# DATA DICTIONARY EXPERIMENT 5

## Raster Files

### Future Stability
| File | Description | Range | Unit |
|------|-------------|-------|------|
| future_stability_index.tif | FSI = cross-scenario stable frequency | 0-1 | fraction |
| future_stability_classes.tif | FSI classes (1-5) | 1-5 | categorical |
| loss_frequency_current_suitable.tif | LFI = cross-scenario loss frequency | 0-1 | fraction |

### Uncertainty
| File | Description | Range | Unit |
|------|-------------|-------|------|
| scenario_sd.tif | SD of ensemble suitability across scenarios | 0+ | suitability units |
| scenario_iqr.tif | IQR of ensemble suitability | 0+ | suitability units |
| prediction_confidence.tif | 1 - normalized scenario SD | 0-1 | index |
| dominant_uncertainty_source.tif | 1=Algorithm, 2=GCM, 3=SSP, 4=Time | 1-4 | categorical |
| high_uncertainty_zone.tif | Binary: 1=high uncertainty | 0/1 | binary |

### Novelty
| File | Description | Range | Unit |
|------|-------------|-------|------|
| novelty_frequency.tif | Fraction of scenarios with novelty indicators | 0-1 | fraction |
| MESS_summary.tif | Mean MESS-like similarity across scenarios | varies | similarity |
| univariate_extrapolation_count.tif | Count of variables exceeding training range | 0-4 | count |

### Vulnerability
| File | Description | Range | Unit |
|------|-------------|-------|------|
| vulnerability_base.tif | CurrentSuitability × LossFrequency | 0+ | index |
| vulnerability_confidence_adjusted.tif | V_base × PredictionConfidence | 0+ | index |
| vulnerability_classes.tif | Percentile-based classes (1-5) | 1-5 | categorical |

### Zones
| File | Description | Range | Unit |
|------|-------------|-------|------|
| robust_climatic_core.tif | Binary: 1=robust core | 0/1 | binary |
| high_confidence_loss_zone.tif | Binary: 1=high-conf loss | 0/1 | binary |
| stability_confidence_quadrants.tif | 4 quadrants | 1-4 | categorical |

## All rasters: EPSG:4326, 0.04° resolution, 4320×8640 pixels
"""
    return content


def main():
    print("=" * 60)
    print("Experiment 5 - Handoff to Experiment 6")
    print("=" * 60)

    # Create directory structure
    print("\n[1] Creating experiment 6 input structure...")
    create_handoff_structure()

    # Build file manifest
    print("\n[2] Building file manifest...")
    file_map = build_file_manifest()
    print(f"  {len(file_map)} files in manifest")

    # Transfer files
    print("\n[3] Transferring files...")
    manifest_df = transfer_files(file_map)

    # Copy landcover
    print("\n[4] Copying landcover...")
    copy_landcover()

    # Write HANDOFF document
    print("\n[5] Writing HANDOFF_FROM_EXPERIMENT5.md...")
    handoff_content = write_handoff_document(file_map, manifest_df)
    with open(EXP6_INPUT / "HANDOFF_FROM_EXPERIMENT5.md", "w", encoding="utf-8") as f:
        f.write(handoff_content)
    print("  Saved HANDOFF_FROM_EXPERIMENT5.md")

    # Write DATA DICTIONARY
    print("\n[6] Writing DATA_DICTIONARY_EXPERIMENT5.md...")
    dd_content = write_data_dictionary()
    with open(EXP6_INPUT / "DATA_DICTIONARY_EXPERIMENT5.md", "w", encoding="utf-8") as f:
        f.write(dd_content)
    print("  Saved DATA_DICTIONARY_EXPERIMENT5.md")

    # Final verification
    n_ok = (manifest_df["status"] == "OK").sum()
    n_total = len(manifest_df)
    print(f"\n{'='*60}")
    print(f"Handoff complete: {n_ok}/{n_total} files OK")
    if n_ok < n_total:
        print(f"WARNING: {n_total - n_ok} files had issues")
    print(f"Target: {EXP6_INPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
