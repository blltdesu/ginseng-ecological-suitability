#!/usr/bin/env python3
"""
Experiment 4 — Steps 5-9: Align future climate to reference grid,
check units, run numeric QC, and check range exceedance.
"""
import os, sys, csv, json, logging
from pathlib import Path
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
import numpy as np

# ——— Config ———
EXP4_DIR = Path(r"E:\人参种在哪\实验4")
INPUT_DIR = EXP4_DIR / "00_input_from_experiment3"
PROCESSED_DIR = EXP4_DIR / "04_future_climate_processed"
ALIGNED_DIR = EXP4_DIR / "04_future_climate_processed"  # Align in-place
QC_DIR = EXP4_DIR / "06_projection_qc"
LOG_DIR = EXP4_DIR / "logs"
REF_GRID = INPUT_DIR / "reference_grid_template.tif"
TRAINING_RANGES = INPUT_DIR / "current_environment_training_ranges.csv"
MASK = INPUT_DIR / "common_valid_mask.tif"

for d in [QC_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "climate_alignment.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("climate_alignment")

GCMS = ["ACCESS-CM2", "BCC-CSM2-MR", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]
BIO_VARS = ["bio02", "bio03", "bio05", "bio15"]

def align_raster(src_path, ref_path, dst_path):
    """Align a raster to the reference grid."""
    with rasterio.open(str(ref_path)) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(str(src_path)) as src:
        if (src.crs == ref_crs and src.transform == ref_transform
            and src.width == ref_width and src.height == ref_height):
            # Already aligned — just copy if path differs
            return "already_aligned"

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, ref_crs, src.width, src.height,
            left=ref_transform[2], bottom=ref_transform[5] + ref_height * ref_transform[4],
            right=ref_transform[2] + ref_width * ref_transform[0], top=ref_transform[5],
            dst_width=ref_width, dst_height=ref_height
        )

        profile = src.profile.copy()
        profile.update(crs=ref_crs, transform=dst_transform,
                       width=dst_width, height=dst_height,
                       compress='lzw', tiled=True,
                       blockxsize=256, blockysize=256)

        with rasterio.open(str(dst_path), 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                data = src.read(i)
                reproject(
                    source=data, destination=rasterio.band(dst, i),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )
    return "aligned"

def main():
    logger.info("=" * 60)
    logger.info("Future Climate Alignment & QC — Experiment 4")
    logger.info("=" * 60)

    # Verify reference grid
    if not REF_GRID.exists():
        logger.error(f"Reference grid not found: {REF_GRID}")
        return

    with rasterio.open(str(REF_GRID)) as ref:
        ref_profile = {
            "crs": str(ref.crs), "width": ref.width, "height": ref.height,
            "transform": list(ref.transform), "bounds": list(ref.bounds),
            "resolution": ref.res
        }
    logger.info(f"Reference grid: {ref_profile}")

    # Load training ranges
    training_ranges = {}
    with open(TRAINING_RANGES, "r") as f:
        for row in csv.DictReader(f):
            training_ranges[row["variable"]] = {k: float(v) if v.replace('.','').replace('-','').replace('e','').replace('+','').isdigit() else v
                                                  for k, v in row.items() if k != "variable"}

    # Load valid mask
    with rasterio.open(str(MASK)) as msrc:
        valid_mask = msrc.read(1) > 0
    logger.info(f"Valid mask: {valid_mask.sum()} pixels")

    # ——— Process each scenario ———
    unit_check_rows = []
    numeric_qc_rows = []
    exceedance_rows = []

    for gcm in GCMS:
        for ssp in SSPS:
            for period in PERIODS:
                scenario_dir = PROCESSED_DIR / gcm / ssp / period
                if not scenario_dir.exists():
                    logger.warning(f"Missing: {gcm}/{ssp}/{period}")
                    continue

                logger.info(f"\nProcessing: {gcm}/{ssp}/{period}")

                for bio_var in BIO_VARS:
                    src_path = scenario_dir / f"{bio_var}.tif"
                    if not src_path.exists():
                        logger.warning(f"  Missing file: {bio_var}.tif")
                        continue

                    # ——— Alignment ———
                    dst_path = scenario_dir / f"{bio_var}_aligned.tif"
                    # Actually, let's use the source directly if already aligned
                    with rasterio.open(str(src_path)) as s:
                        needs_align = (str(s.crs) != str(ref_profile["crs"]) or
                                       s.width != ref_profile["width"] or
                                       s.height != ref_profile["height"])
                    if needs_align:
                        align_result = align_raster(src_path, REF_GRID, dst_path)
                        logger.info(f"  {bio_var}: {align_result}")
                        data_path = dst_path
                    else:
                        align_result = "already_aligned"
                        data_path = src_path

                    if align_result == "already_aligned":
                        logger.info(f"  {bio_var}: already aligned to reference grid")

                    # ——— Numeric QC ———
                    with rasterio.open(str(data_path)) as s:
                        data = s.read(1)
                        # Apply valid mask
                        data_valid = data[valid_mask]
                        # Filter out NaN
                        data_valid = data_valid[~np.isnan(data_valid)]
                        # Filter out extreme NoData
                        data_valid = data_valid[data_valid > -3e38]

                        nodata_pct = (data[valid_mask].size - data_valid.size) / data[valid_mask].size * 100 if data[valid_mask].size > 0 else 100

                        stats = {
                            "gcm": gcm, "ssp": ssp, "period": period,
                            "variable": bio_var,
                            "min": float(np.min(data_valid)) if data_valid.size > 0 else np.nan,
                            "p01": float(np.percentile(data_valid, 1)) if data_valid.size > 0 else np.nan,
                            "p05": float(np.percentile(data_valid, 5)) if data_valid.size > 0 else np.nan,
                            "median": float(np.median(data_valid)) if data_valid.size > 0 else np.nan,
                            "p95": float(np.percentile(data_valid, 95)) if data_valid.size > 0 else np.nan,
                            "p99": float(np.percentile(data_valid, 99)) if data_valid.size > 0 else np.nan,
                            "max": float(np.max(data_valid)) if data_valid.size > 0 else np.nan,
                            "nodata_pct": float(nodata_pct),
                            "alignment": align_result,
                        }
                        numeric_qc_rows.append(stats)
                        logger.info(f"  {bio_var}: min={stats['min']:.2f}, median={stats['median']:.2f}, max={stats['max']:.2f}")

                    # ——— Unit Check (compare with training ranges) ———
                    tr = training_ranges.get(bio_var, {})
                    unit_note = "OK"
                    # WorldClim BIO variables use standard units:
                    # bio02: °C (Mean Diurnal Range)
                    # bio03: % (Isothermality) — sometimes stored as fraction
                    # bio05: °C (Max Temp of Warmest Month)
                    # bio15: CV (Precipitation Seasonality)
                    # Check if values are in expected range
                    if data_valid.size > 0:
                        med = np.median(data_valid)
                        tr_med = float(tr.get("median", 0))
                        if abs(med - tr_med) > 100 and tr_med != 0:
                            # Possible unit mismatch
                            unit_note = f"POTENTIAL_MISMATCH: future_median={med:.2f}, training_median={tr_med:.2f}"

                    unit_check_rows.append({
                        "gcm": gcm, "ssp": ssp, "period": period,
                        "variable": bio_var,
                        "training_min": tr.get("min", ""),
                        "training_max": tr.get("max", ""),
                        "future_median": stats["median"],
                        "unit_check": unit_note,
                    })

                    # ——— Range Exceedance ———
                    if data_valid.size > 0:
                        tr_min = float(tr.get("min", -np.inf))
                        tr_max = float(tr.get("max", np.inf))
                        tr_p01 = float(tr.get("p01", np.inf))
                        tr_p99 = float(tr.get("p99", -np.inf))

                        below_min_pct = (data_valid < tr_min).sum() / data_valid.size * 100
                        above_max_pct = (data_valid > tr_max).sum() / data_valid.size * 100
                        outside_p01p99_pct = ((data_valid < tr_p01) | (data_valid > tr_p99)).sum() / data_valid.size * 100

                        exceedance_rows.append({
                            "gcm": gcm, "ssp": ssp, "period": period,
                            "variable": bio_var,
                            "below_training_min_pct": f"{below_min_pct:.2f}",
                            "above_training_max_pct": f"{above_max_pct:.2f}",
                            "outside_p01_p99_pct": f"{outside_p01p99_pct:.2f}",
                        })
                        warning_level = "Low" if below_min_pct < 5 and above_max_pct < 5 else (
                            "Moderate" if below_min_pct < 20 and above_max_pct < 20 else "High")
                        logger.info(f"  {bio_var} exceedance: below_min={below_min_pct:.1f}%, above_max={above_max_pct:.1f}% ({warning_level})")

    # ——— Save outputs ———
    # Unit check
    with open(QC_DIR / "future_climate_unit_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=unit_check_rows[0].keys())
        w.writeheader()
        w.writerows(unit_check_rows)
    logger.info(f"Unit check saved to {QC_DIR / 'future_climate_unit_check.csv'}")

    # Numeric QC
    with open(QC_DIR / "future_climate_numeric_qc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=numeric_qc_rows[0].keys())
        w.writeheader()
        w.writerows(numeric_qc_rows)
    logger.info(f"Numeric QC saved to {QC_DIR / 'future_climate_numeric_qc.csv'}")

    # Range exceedance
    with open(QC_DIR / "future_range_exceedance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=exceedance_rows[0].keys())
        w.writeheader()
        w.writerows(exceedance_rows)
    logger.info(f"Range exceedance saved to {QC_DIR / 'future_range_exceedance.csv'}")

    # ——— Summary ———
    # Count high-exceedance scenarios
    high_warnings = [r for r in exceedance_rows if
                     float(r["below_training_min_pct"]) > 20 or float(r["above_training_max_pct"]) > 20]
    logger.info(f"\nScenarios with >20% exceedance: {len(high_warnings)}")
    if high_warnings:
        for r in high_warnings:
            logger.warning(f"  {r['gcm']}/{r['ssp']}/{r['period']} {r['variable']}: "
                          f"below_min={r['below_training_min_pct']}%, above_max={r['above_training_max_pct']}%")

    logger.info("\nAlignment & QC complete!")

if __name__ == "__main__":
    main()
