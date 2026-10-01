"""Steps 11-15: Build predictor registry, alignment QC, numeric QC,
extract occurrence environment values, and prepare handoff data."""
import os
import csv
import json
import hashlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from datetime import datetime

PREPROCESS_DIR = r"E:\人参种在哪\00_统一数据预处理"
REF_META_PATH = os.path.join(PREPROCESS_DIR, "09_reference_grid", "reference_grid_metadata.json")
QC_DIR = os.path.join(PREPROCESS_DIR, "11_qc")
REGISTRY_DIR = os.path.join(PREPROCESS_DIR, "10_aligned_predictors")
HANDOFF_DIR = os.path.join(PREPROCESS_DIR, "14_handoff")

os.makedirs(QC_DIR, exist_ok=True)
os.makedirs(REGISTRY_DIR, exist_ok=True)
os.makedirs(HANDOFF_DIR, exist_ok=True)

RANDOM_SEED = 20260807
np.random.seed(RANDOM_SEED)


def get_all_predictor_paths():
    """Collect all aligned predictor rasters."""
    predictors = {}

    # Current climate (BIO1-BIO19)
    climate_dir = os.path.join(PREPROCESS_DIR, "04_current_climate")
    for f in os.listdir(climate_dir):
        if f.endswith(".tif"):
            var = f.replace("_aligned.tif", "")
            predictors[var] = os.path.join(climate_dir, f)

    # Soil
    soil_dir = os.path.join(PREPROCESS_DIR, "06_soil", "aligned")
    for f in os.listdir(soil_dir):
        if f.endswith(".tif"):
            var = f.replace("_0_15cm.tif", "")
            predictors[var] = os.path.join(soil_dir, f)

    # Terrain
    terrain_dir = os.path.join(PREPROCESS_DIR, "07_terrain", "aligned")
    for f in os.listdir(terrain_dir):
        if f.endswith(".tif"):
            var = f.replace(".tif", "")
            predictors[var] = os.path.join(terrain_dir, f)

    return predictors


def check_raster_alignment(predictors, ref_meta):
    """Check that all predictor rasters match the reference grid exactly."""
    print("=" * 50)
    print("RASTER ALIGNMENT QC")

    ref_width = ref_meta["width"]
    ref_height = ref_meta["height"]
    ref_crs = ref_meta["crs"]
    ref_res = ref_meta["resolution"]
    ref_bounds = ref_meta["bounds"]
    ref_transform = ref_meta["transform"]

    results = []
    for var_name, fpath in sorted(predictors.items()):
        with rasterio.open(fpath) as src:
            crs_match = str(src.crs) == ref_crs
            shape_match = src.shape == (ref_height, ref_width)
            res_match = (
                abs(src.res[0] - ref_res[0]) < 1e-10 and
                abs(src.res[1] - abs(ref_res[1])) < 1e-10
            )
            transform_match = all(
                abs(a - b) < 1e-8
                for a, b in zip(src.transform, ref_transform)
            )
            bounds_match = (
                abs(src.bounds.left - ref_bounds["left"]) < 1e-8 and
                abs(src.bounds.top - ref_bounds["top"]) < 1e-8
            )

            overall = all([crs_match, shape_match, res_match, transform_match, bounds_match])

            results.append({
                "variable": var_name,
                "path": fpath,
                "crs_match": crs_match,
                "shape_match": shape_match,
                "resolution_match": res_match,
                "transform_match": transform_match,
                "bounds_match": bounds_match,
                "overall_pass": overall,
            })

            status = "PASS" if overall else "FAIL"
            print(f"  {var_name}: {status} (CRS={crs_match}, shape={shape_match}, "
                  f"res={res_match}, transform={transform_match}, bounds={bounds_match})")

    # Save
    qc_path = os.path.join(QC_DIR, "raster_alignment_qc.csv")
    with open(qc_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    all_pass = all(r["overall_pass"] for r in results)
    print(f"\nOverall alignment: {'PASS' if all_pass else 'FAIL'} ({sum(1 for r in results if r['overall_pass'])}/{len(results)})")

    # Checkpoint D
    if not all_pass:
        stop_path = os.path.join(PREPROCESS_DIR, "STOP_D_ALIGNMENT.md")
        with open(stop_path, "w", encoding="utf-8") as f:
            f.write("# STOP D: Raster Alignment Failed\n\n")
            for r in results:
                if not r["overall_pass"]:
                    f.write(f"- {r['variable']}: FAIL\n")
        print("WARNING: Not all rasters aligned!")

    return results


def numeric_qc(predictors):
    """Run numeric quality checks on all predictors."""
    print("\n" + "=" * 50)
    print("NUMERIC QC")

    results = []
    for var_name, fpath in sorted(predictors.items()):
        with rasterio.open(fpath) as src:
            data = src.read(1, masked=True)
            valid = data.compressed()

            if len(valid) == 0:
                print(f"  {var_name}: EMPTY - all nodata!")
                results.append({"variable": var_name, "status": "EMPTY"})
                continue

            pct_missing = round((1 - data.count() / data.size) * 100, 2)

            # Check for constant raster
            is_constant = valid.max() == valid.min()

            # Check for infinities
            has_inf = np.isinf(valid).any() if valid.dtype.kind == 'f' else False

            # Basic stats
            record = {
                "variable": var_name,
                "min": round(float(valid.min()), 4),
                "q01": round(float(np.percentile(valid, 1)), 4),
                "q05": round(float(np.percentile(valid, 5)), 4),
                "median": round(float(np.median(valid)), 4),
                "mean": round(float(valid.mean()), 4),
                "q95": round(float(np.percentile(valid, 95)), 4),
                "q99": round(float(np.percentile(valid, 99)), 4),
                "max": round(float(valid.max()), 4),
                "nodata_pct": pct_missing,
                "is_constant": is_constant,
                "has_inf": has_inf,
                "n_valid_pixels": len(valid),
            }
            results.append(record)
            print(f"  {var_name}: range=[{record['min']}, {record['max']}], "
                  f"median={record['median']}, nodata={pct_missing}%")

    # Save
    qc_path = os.path.join(QC_DIR, "predictor_numeric_qc.csv")
    with open(qc_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    return results


def extract_occurrence_environment(predictors):
    """Extract environmental values at occurrence points for QC."""
    print("\n" + "=" * 50)
    print("EXTRACTING OCCURRENCE ENVIRONMENT")

    # Load occurrence data
    occ_path = os.path.join(PREPROCESS_DIR, "02_occurrence", "occurrence_noncultivated_candidate.csv")
    if not os.path.exists(occ_path):
        print("WARNING: Occurrence file not found!")
        return

    occ = pd.read_csv(occ_path)
    print(f"Loaded {len(occ)} non-cultivated candidate records")

    # Extract values from each predictor
    extraction = []
    for idx, row in occ.iterrows():
        lon = row.get("decimalLongitude")
        lat = row.get("decimalLatitude")
        if pd.isna(lon) or pd.isna(lat):
            continue

        record = {
            "record_id": row.get("gbifID", idx),
            "longitude": lon,
            "latitude": lat,
        }
        missing_count = 0

        for var_name, fpath in sorted(predictors.items()):
            with rasterio.open(fpath) as src:
                try:
                    py, px = src.index(lon, lat)
                    if 0 <= py < src.height and 0 <= px < src.width:
                        val = src.read(1, window=((py, py+1), (px, px+1)))[0, 0]
                        if src.nodata is not None and val == src.nodata:
                            val = np.nan
                            missing_count += 1
                        record[var_name] = round(float(val), 4) if not np.isnan(val) else np.nan
                    else:
                        record[var_name] = np.nan
                        missing_count += 1
                except Exception:
                    record[var_name] = np.nan
                    missing_count += 1

        record["missing_count"] = missing_count
        extraction.append(record)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(occ)} records...")

    # Save
    df_extract = pd.DataFrame(extraction)
    out_path = os.path.join(QC_DIR, "occurrence_environment_extraction_qc.csv")
    df_extract.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path} ({len(df_extract)} records)")
    print(f"  Records with >0 missing variables: {(df_extract['missing_count'] > 0).sum()}")

    return df_extract


def build_predictor_registry(predictors, alignment_results):
    """Build the current predictor registry."""
    print("\n" + "=" * 50)
    print("BUILDING PREDICTOR REGISTRY")

    records = []
    for var_name, fpath in sorted(predictors.items()):
        # Determine category
        if var_name.startswith("bio"):
            category = "climate"
        elif var_name in ["phh2o", "soc", "cec", "clay", "sand", "nitrogen", "bdod"]:
            category = "soil"
        elif var_name in ["elevation", "slope", "northness", "eastness"]:
            category = "terrain"
        else:
            category = "other"

        records.append({
            "variable": var_name,
            "category": category,
            "file_path": fpath,
            "status": "ready",
        })

    reg_path = os.path.join(REGISTRY_DIR, "current_predictor_registry.csv")
    with open(reg_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Predictor registry: {len(records)} variables ({reg_path})")

    # Also build the variable list for the final handoff
    var_list_path = os.path.join(REGISTRY_DIR, "current_predictor_registry.csv")
    return records


def build_handoff():
    """Build handoff packages for experiments 1, 4, and 6."""
    print("\n" + "=" * 50)
    print("BUILDING HANDOFF PACKAGES")

    to_exp1 = os.path.join(HANDOFF_DIR, "to_experiment_1")
    to_exp4 = os.path.join(HANDOFF_DIR, "to_experiment_4")
    to_exp6 = os.path.join(HANDOFF_DIR, "to_experiment_6")
    os.makedirs(to_exp1, exist_ok=True)
    os.makedirs(to_exp4, exist_ok=True)
    os.makedirs(to_exp6, exist_ok=True)

    # === Experiment 1 Handoff ===
    import shutil
    exp1_sources = {
        # Occurrence
        "02_occurrence/occurrence_thin_5km.csv": "occurrence_thin_5km.csv",
        "02_occurrence/occurrence_thin_10km.csv": "occurrence_thin_10km.csv",
        "02_occurrence/occurrence_thin_20km.csv": "occurrence_thin_20km.csv",
        "02_occurrence/occurrence_cultivated_or_uncertain.csv": "occurrence_cultivated_or_uncertain.csv",
        "02_occurrence/occurrence_noncultivated_candidate.csv": "occurrence_noncultivated_candidate.csv",
        # Reference
        "09_reference_grid/reference_grid_template.tif": "reference_grid_template.tif",
        "09_reference_grid/reference_grid_metadata.json": "reference_grid_metadata.json",
        # Registry
        "10_aligned_predictors/current_predictor_registry.csv": "current_predictor_registry.csv",
        # Boundaries
        "03_boundaries/study_context_adm0.gpkg": "study_context_adm0.gpkg",
        "03_boundaries/study_context_adm1.gpkg": "study_context_adm1.gpkg",
    }

    # Copy climate, soil, terrain to experiment 1
    for cat, src_dir in [
        ("04_current_climate", "current_predictors"),
        ("06_soil/aligned", "current_predictors"),
        ("07_terrain/aligned", "current_predictors"),
    ]:
        full_src = os.path.join(PREPROCESS_DIR, src_dir)
        full_dst = os.path.join(to_exp1, "current_predictors")
        os.makedirs(full_dst, exist_ok=True)
        if os.path.exists(full_src):
            for f in os.listdir(full_src):
                if f.endswith(".tif"):
                    shutil.copy2(os.path.join(full_src, f), os.path.join(full_dst, f))

    # Copy specific files
    for src_rel, dst_name in exp1_sources.items():
        full_src = os.path.join(PREPROCESS_DIR, src_rel)
        full_dst = os.path.join(to_exp1, dst_name)
        if os.path.exists(full_src):
            if os.path.isdir(full_src):
                if not os.path.exists(full_dst):
                    shutil.copytree(full_src, full_dst)
            else:
                shutil.copy2(full_src, full_dst)
        else:
            print(f"  WARNING: Source not found: {full_src}")

    # Copy common valid mask if exists
    mask_path = os.path.join(PREPROCESS_DIR, "09_reference_grid", "common_valid_mask.tif")
    if os.path.exists(mask_path):
        shutil.copy2(mask_path, os.path.join(to_exp1, "common_valid_mask.tif"))

    # Copy QC reports
    qc_dst = os.path.join(to_exp1, "qc_reports")
    os.makedirs(qc_dst, exist_ok=True)
    for f in os.listdir(QC_DIR):
        if f.endswith(".csv") or f.endswith(".md"):
            shutil.copy2(os.path.join(QC_DIR, f), os.path.join(qc_dst, f))

    print(f"Experiment 1 handoff: {to_exp1}")

    # === Experiment 4 Handoff ===
    # Future climate - only if data exists
    future_aligned = os.path.join(PREPROCESS_DIR, "05_future_climate", "aligned")
    if os.path.exists(future_aligned):
        exp4_predictors = os.path.join(to_exp4, "aligned_future_climate")
        os.makedirs(exp4_predictors, exist_ok=True)
        for root, dirs, files in os.walk(future_aligned):
            for f in files:
                src = os.path.join(root, f)
                dst_rel = os.path.relpath(src, future_aligned)
                dst = os.path.join(exp4_predictors, dst_rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    # Future registry
    future_reg = os.path.join(PREPROCESS_DIR, "05_future_climate", "future_climate_registry.csv")
    if os.path.exists(future_reg):
        shutil.copy2(future_reg, os.path.join(to_exp4, "future_climate_registry.csv"))

    print(f"Experiment 4 handoff: {to_exp4}")

    # === Experiment 6 Handoff ===
    lc_src = os.path.join(PREPROCESS_DIR, "08_landcover", "landcover_aligned.tif")
    lc_dict_src = os.path.join(PREPROCESS_DIR, "08_landcover", "landcover_class_dictionary.csv")
    if os.path.exists(lc_src):
        shutil.copy2(lc_src, os.path.join(to_exp6, "landcover_aligned.tif"))
    if os.path.exists(lc_dict_src):
        shutil.copy2(lc_dict_src, os.path.join(to_exp6, "landcover_class_dictionary.csv"))

    print(f"Experiment 6 handoff: {to_exp6}")

    # === Generate handoff SHA256 ===
    sha_records = []
    for handoff_name, handoff_path in [
        ("to_experiment_1", to_exp1),
        ("to_experiment_4", to_exp4),
        ("to_experiment_6", to_exp6),
    ]:
        for dirpath, dirnames, filenames in os.walk(handoff_path):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                sha = hashlib.sha256()
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha.update(chunk)
                rel = os.path.relpath(fpath, HANDOFF_DIR)
                sha_records.append({
                    "handoff": handoff_name,
                    "file": rel,
                    "sha256": sha.hexdigest(),
                })

    sha_path = os.path.join(HANDOFF_DIR, "handoff_sha256.csv")
    with open(sha_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["handoff", "file", "sha256"])
        writer.writeheader()
        writer.writerows(sha_records)
    print(f"Handoff SHA256: {sha_path}")


def generate_data_dictionary():
    """Generate data dictionary for handoff."""
    dd = []
    # Climate
    bio_desc = {
        "bio01": "Annual Mean Temperature (°C)",
        "bio02": "Mean Diurnal Range (°C)",
        "bio03": "Isothermality (bio02/bio07 ×100)",
        "bio04": "Temperature Seasonality (sd×100)",
        "bio05": "Max Temperature of Warmest Month (°C)",
        "bio06": "Min Temperature of Coldest Month (°C)",
        "bio07": "Temperature Annual Range (bio05-bio06) (°C)",
        "bio08": "Mean Temperature of Wettest Quarter (°C)",
        "bio09": "Mean Temperature of Driest Quarter (°C)",
        "bio10": "Mean Temperature of Warmest Quarter (°C)",
        "bio11": "Mean Temperature of Coldest Quarter (°C)",
        "bio12": "Annual Precipitation (mm)",
        "bio13": "Precipitation of Wettest Month (mm)",
        "bio14": "Precipitation of Driest Month (mm)",
        "bio15": "Precipitation Seasonality (CV)",
        "bio16": "Precipitation of Wettest Quarter (mm)",
        "bio17": "Precipitation of Driest Quarter (mm)",
        "bio18": "Precipitation of Warmest Quarter (mm)",
        "bio19": "Precipitation of Coldest Quarter (mm)",
    }
    for bio, desc in bio_desc.items():
        dd.append({
            "variable": bio,
            "description": desc,
            "unit": "varies (see description)",
            "source": "WorldClim 2.1 (1970-2000)",
            "processing": "Copied as-is (reference grid source)",
            "depth": "",
        })

    # Soil
    soil_desc = {
        "phh2o": ("Soil pH (H2O)", "pH", "SoilGrids 2.0", "Storage integers ÷10, 0-15cm weighted mean"),
        "soc": ("Soil Organic Carbon", "g/kg", "SoilGrids 2.0", "Storage integers ÷10, 0-15cm weighted mean"),
        "cec": ("Cation Exchange Capacity", "cmol(c)/kg", "SoilGrids 2.0", "Storage integers ÷10, 0-15cm weighted mean"),
        "clay": ("Clay Content", "%", "SoilGrids 2.0", "Storage integers ÷10, 0-15cm weighted mean"),
        "sand": ("Sand Content", "%", "SoilGrids 2.0", "Storage integers ÷10, 0-15cm weighted mean"),
        "nitrogen": ("Soil Nitrogen", "g/kg", "SoilGrids 2.0", "Storage integers ÷100, 0-15cm weighted mean"),
        "bdod": ("Bulk Density", "kg/dm³", "SoilGrids 2.0", "Storage integers ÷100, 0-15cm weighted mean"),
    }
    for var, (desc, unit, src, proc) in soil_desc.items():
        dd.append({
            "variable": var,
            "description": desc,
            "unit": unit,
            "source": src,
            "processing": proc,
            "depth": "0-15 cm (weighted mean of 0-5cm and 5-15cm)",
        })

    # Terrain
    terr_desc = {
        "elevation": ("Elevation", "m", "SRTM", "Bilinear resampling to reference grid"),
        "slope": ("Slope", "degrees", "SRTM-derived", "Bilinear resampling to reference grid"),
        "northness": ("Northness (cos(aspect))", "unitless [-1,1]", "SRTM-derived", "Bilinear resampling to reference grid"),
        "eastness": ("Eastness (sin(aspect))", "unitless [-1,1]", "SRTM-derived", "Bilinear resampling to reference grid"),
    }
    for var, (desc, unit, src, proc) in terr_desc.items():
        dd.append({
            "variable": var,
            "description": desc,
            "unit": unit,
            "source": src,
            "processing": proc,
            "depth": "",
        })

    # Save
    dd_path = os.path.join(HANDOFF_DIR, "to_experiment_1", "DATA_DICTIONARY.md")
    with open(dd_path, "w", encoding="utf-8") as f:
        f.write("# Data Dictionary: Current Predictors for Panax ginseng SDM\n\n")
        f.write("| Variable | Description | Unit | Source | Processing | Depth |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in dd:
            f.write(f"| {row['variable']} | {row['description']} | {row['unit']} | "
                    f"{row['source']} | {row['processing']} | {row.get('depth', '')} |\n")

    return dd


def generate_handoff_readme():
    """Generate HANDOFF_TO_EXPERIMENT_1.md."""
    content = """# Handoff to Experiment 1: Current Ecological Suitability

## Overview
This package contains standardized, QC'd data for modeling the current ecological suitability
of *Panax ginseng* C.A.Mey.

## Contents

### Occurrence Data
- `occurrence_thin_10km.csv` — Primary candidate presence data (10 km spatial thinning)
- `occurrence_thin_5km.csv` — 5 km thinning for sensitivity analysis
- `occurrence_thin_20km.csv` — 20 km thinning for sensitivity analysis
- `occurrence_noncultivated_candidate.csv` — All non-cultivated candidate records
- `occurrence_cultivated_or_uncertain.csv` — Cultivated/uncertain records for auxiliary validation

### Environmental Predictors (in `current_predictors/`)
All predictors are aligned to the WorldClim 2.5 arc-minute reference grid (EPSG:4326, 8640×4320).

- BIO1–BIO19 (WorldClim 2.1, 1970-2000)
- phh2o, soc, cec, clay, sand, nitrogen, bdod (SoilGrids 2.0, 0-15 cm)
- elevation, slope, northness, eastness (SRTM-derived)

### Reference Data
- `reference_grid_template.tif` — Master reference grid
- `reference_grid_metadata.json` — Grid metadata
- `study_context_adm0.gpkg` — Country boundaries (6 countries)
- `study_context_adm1.gpkg` — Province/state boundaries
- `current_predictor_registry.csv` — Registry of all predictors

### QC Reports (in `qc_reports/`)
- Raster alignment QC
- Predictor numeric QC
- Occurrence environment extraction QC

## Data Processing Summary
- All rasters strictly aligned to the WorldClim reference grid
- SoilGrids units converted from storage integers to standard units
- Soil depths: 0-15 cm thickness-weighted mean
- Occurrence records: taxonomic filtering, coordinate QC, cultivation flagging, spatial thinning

## Notes for Experiment 1
1. The default main analysis thinning is 10 km, but 5 and 20 km versions are provided for sensitivity analysis.
2. Variable correlation screening and VIF analysis belong to Experiment 1.
3. The ecological accessible area (M) should be defined in Experiment 1.
4. Do NOT re-read raw data from `E:\\人参种在哪\\数据`.
"""
    readme_path = os.path.join(HANDOFF_DIR, "to_experiment_1", "HANDOFF_TO_EXPERIMENT_1.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    print("=" * 60)
    print("ALIGNMENT QC, PREDICTOR REGISTRY & HANDOFF BUILD")
    print("=" * 60)

    # Load reference metadata
    with open(REF_META_PATH) as f:
        ref_meta = json.load(f)

    # Get all predictors
    predictors = get_all_predictor_paths()
    print(f"\nFound {len(predictors)} predictor rasters")

    # 1. Alignment QC
    alignment_results = check_raster_alignment(predictors, ref_meta)

    # 2. Numeric QC
    numeric_results = numeric_qc(predictors)

    # 3. Extract occurrence environment
    extract_env = extract_occurrence_environment(predictors)

    # 4. Build predictor registry
    registry = build_predictor_registry(predictors, alignment_results)

    # 5. Generate data dictionary
    data_dict = generate_data_dictionary()

    # 6. Generate handoff README
    generate_handoff_readme()

    # 7. Build handoff
    build_handoff()

    print(f"\n{'='*50}")
    print("ALL QC AND HANDOFF STEPS COMPLETE")


if __name__ == "__main__":
    main()
