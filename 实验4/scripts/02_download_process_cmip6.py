#!/usr/bin/env python3
"""
Experiment 4 — Steps 3-7: Download & Process CMIP6 Future Climate Data
Downloads bioclim GeoTIFFs from geodata.ucdavis.edu, extracts needed BIO bands,
and saves them to the processed directory.

Strategy:
  - Download full bioc GeoTIFF (19 bands, ~438 MB each) via Python urllib
  - Extract only the needed bands (bio02=band2, bio03=band3, bio05=band5, bio15=band15)
  - Delete the large source file to save disk space
  - Each GCM/SSP/period = 4 output files totaling ~90 MB (vs 438 MB source)
"""
import os, sys, csv, json, time, hashlib, logging, subprocess, shutil
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import HTTPError, URLError
import rasterio
import numpy as np

# ——— Config ———
EXP4_DIR = Path(r"E:\人参种在哪\实验4")
RAW_DIR = EXP4_DIR / "02_future_climate_raw"
PROCESSED_DIR = EXP4_DIR / "04_future_climate_processed"
REGISTRY_DIR = EXP4_DIR / "03_future_climate_registry"
QC_DIR = EXP4_DIR / "06_projection_qc"
LOG_DIR = EXP4_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, REGISTRY_DIR, QC_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "future_climate_processing.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("cmip6_processing")

# ——— Design ———
GCMS = ["ACCESS-CM2", "BCC-CSM2-MR", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]
BASE_URL = "https://geodata.ucdavis.edu/cmip6/2.5m"

# BIO band mapping: variable -> band number (1-indexed) in WorldClim
BIO_BANDS = {"bio02": 2, "bio03": 3, "bio05": 5, "bio15": 15}

# Reference grid from experiment 3
REF_GRID = EXP4_DIR / "00_input_from_experiment3" / "reference_grid_template.tif"

# ——— Helpers ———
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def check_url_exists(url, timeout=10):
    try:
        urlopen(url, timeout=timeout)
        return True
    except:
        return False

def build_url(gcm, ssp, period):
    fname = f"wc2.1_2.5m_bioc_{gcm}_{ssp}_{period}.tif"
    return f"{BASE_URL}/{gcm}/{ssp}/{fname}", fname

def download_with_progress(url, dest_path):
    """Download a file with progress logging."""
    tmp_path = str(dest_path) + ".tmp"
    try:
        logger.info(f"  Downloading: {url}")
        t0 = time.time()
        urlretrieve(url, tmp_path)
        elapsed = time.time() - t0
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        speed_mb = size_mb / elapsed if elapsed > 0 else 0
        logger.info(f"  Downloaded {size_mb:.1f} MB in {elapsed:.0f}s ({speed_mb:.1f} MB/s)")
        os.replace(tmp_path, dest_path)
        return True, size_mb
    except HTTPError as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        logger.error(f"  HTTP {e.code}: {url}")
        return False, 0
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        logger.error(f"  Error: {e}")
        return False, 0

# ——— Main Pipeline ———
def process_scenario(gcm, ssp, period):
    """Download and extract needed BIO bands for one scenario."""
    scenario_dir = PROCESSED_DIR / gcm / ssp / period
    raw_dir = RAW_DIR / gcm / ssp / period
    raw_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    url, fname = build_url(gcm, ssp, period)
    raw_path = raw_dir / fname

    # — Check if already processed ———
    all_bands_exist = all((scenario_dir / f"{bio_var}.tif").exists() for bio_var in BIO_BANDS)
    if all_bands_exist:
        logger.info(f"  [SKIP] All bands already extracted for {gcm}/{ssp}/{period}")
        return "already_processed"

    # — Download if needed ———
    if not raw_path.exists():
        # Check if URL is available
        if not check_url_exists(url):
            logger.warning(f"  [SKIP] URL not available: {url}")
            return "url_unavailable"

        success, size_mb = download_with_progress(url, raw_path)
        if not success:
            return "download_failed"
    else:
        size_mb = os.path.getsize(raw_path) / (1024 * 1024)
        logger.info(f"  Using existing file: {size_mb:.1f} MB")

    # — Extract bands ———
    try:
        with rasterio.open(str(raw_path)) as src:
            profile = src.profile.copy()
            profile.update(count=1, compress='lzw', tiled=True, blockxsize=256, blockysize=256)

            for bio_var, band_idx in BIO_BANDS.items():
                out_path = scenario_dir / f"{bio_var}.tif"
                if out_path.exists():
                    continue

                data = src.read(band_idx)
                # Replace NoData with a standard value
                if src.nodata is not None:
                    data = np.where(data == src.nodata, np.nan, data)

                with rasterio.open(str(out_path), 'w', **profile) as dst:
                    dst.write(data, 1)
                    dst.set_band_description(1, f"{bio_var} — {gcm} {ssp} {period}")
                    dst.update_tags(
                        source=url,
                        gcm=gcm, ssp=ssp, period=period,
                        variable=bio_var, band=str(band_idx)
                    )

                file_mb = os.path.getsize(out_path) / (1024 * 1024)
                logger.info(f"    Extracted {bio_var} ({file_mb:.1f} MB)")

    except Exception as e:
        logger.error(f"  Extraction error: {e}")
        return "extraction_failed"

    # — Optionally delete raw file to save space ———
    # raw_path.unlink()  # Uncomment if disk space is tight
    # logger.info(f"  Deleted raw file to save disk space")

    return "complete"

def main():
    logger.info("=" * 60)
    logger.info("CMIP6 Future Climate Processing — Experiment 4")
    logger.info(f"Design: {len(GCMS)} GCMs × {len(SSPS)} SSPs × {len(PERIODS)} periods = {len(GCMS)*len(SSPS)*len(PERIODS)} scenarios")
    logger.info(f"Needed bands: {list(BIO_BANDS.keys())}")
    logger.info("=" * 60)

    # — Check if reference grid exists ———
    if not REF_GRID.exists():
        logger.error("Reference grid not found!")
        return

    # — Check connectivity ———
    test_url = f"{BASE_URL}/ACCESS-CM2/ssp126/"
    if not check_url_exists(test_url):
        logger.critical("Cannot reach geodata.ucdavis.edu server!")
        logger.info("Attempting download anyway...")

    # — Process all scenarios ———
    registry_rows = []
    total = len(GCMS) * len(SSPS) * len(PERIODS)
    stats = {"complete": 0, "already_processed": 0, "download_failed": 0,
             "url_unavailable": 0, "extraction_failed": 0}

    for idx, gcm in enumerate(GCMS):
        for ssp in SSPS:
            for period in PERIODS:
                logger.info(f"\n[{sum(stats.values())+1}/{total}] {gcm} | {ssp} | {period}")
                result = process_scenario(gcm, ssp, period)
                stats[result] = stats.get(result, 0) + 1

                # Record in registry
                scenario_dir = PROCESSED_DIR / gcm / ssp / period
                bio_status = {}
                for bio_var in BIO_BANDS:
                    p = scenario_dir / f"{bio_var}.tif"
                    bio_status[f"{bio_var}_available"] = 1 if p.exists() else 0

                registry_rows.append({
                    "gcm": gcm,
                    "ssp": ssp,
                    "period": period,
                    "source_url": f"{BASE_URL}/{gcm}/{ssp}/wc2.1_2.5m_bioc_{gcm}_{ssp}_{period}.tif",
                    **bio_status,
                    "status": result,
                })

    # — Save registry ———
    registry_path = REGISTRY_DIR / "future_climate_registry.csv"
    with open(registry_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=registry_rows[0].keys())
        writer.writeheader()
        writer.writerows(registry_rows)
    logger.info(f"\nRegistry saved to {registry_path}")

    # — Save completeness matrix ———
    matrix_path = REGISTRY_DIR / "future_climate_completeness_matrix.csv"
    with open(matrix_path, "w", newline="", encoding="utf-8") as f:
        fields = ["gcm", "ssp", "period", "expected_bios", "available_bios",
                  "bio02", "bio03", "bio05", "bio15", "scenario_status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in registry_rows:
            available = sum(row[f"{b}_available"] for b in BIO_BANDS)
            row_out = {
                "gcm": row["gcm"], "ssp": row["ssp"], "period": row["period"],
                "expected_bios": 4, "available_bios": available,
                "bio02": row["bio02_available"],
                "bio03": row["bio03_available"],
                "bio05": row["bio05_available"],
                "bio15": row["bio15_available"],
                "scenario_status": "complete" if available == 4 else "incomplete",
            }
            writer.writerow(row_out)
    logger.info(f"Completeness matrix saved to {matrix_path}")

    # — Summary ———
    logger.info("\n" + "=" * 60)
    logger.info("Processing Summary:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  Total: {sum(stats.values())}/{total}")

    incomplete = stats.get("download_failed", 0) + stats.get("extraction_failed", 0) + stats.get("url_unavailable", 0)
    if incomplete > (total * 0.5):
        logger.critical("More than 50% of scenarios failed — STOP_A_CMIP6_NOT_READY")
    elif incomplete > 0:
        logger.warning(f"{incomplete} scenarios incomplete — continuing with warnings")
    else:
        logger.info("All scenarios complete!")

    return stats, registry_rows

if __name__ == "__main__":
    main()
