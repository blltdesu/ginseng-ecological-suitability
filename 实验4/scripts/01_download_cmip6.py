#!/usr/bin/env python3
"""
Experiment 4 — Step 3: Download CMIP6 future climate data from WorldClim v2.1
Downloads 2.5m resolution future bioclimatic variables.
"""
import os
import sys
import hashlib
import zipfile
import logging
import time
import csv
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import HTTPError, URLError

# ——— Config ———
EXP4_DIR = Path(r"E:\人参种在哪\实验4")
RAW_DIR = EXP4_DIR / "02_future_climate_raw"
LOG_DIR = EXP4_DIR / "logs"
REGISTRY_DIR = EXP4_DIR / "03_future_climate_registry"

for d in [RAW_DIR, LOG_DIR, REGISTRY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "future_climate_download.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("cmip6_download")

# ——— WorldClim CMIP6 v2.1 2.5m base URL ———
BASE_URL = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/fut/2.5m"

# ——— Scenario design ———
GCMS = [
    "ACCESS-CM2",
    "BCC-CSM2-MR",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
]

SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]

PERIODS = ["2041-2060", "2061-2080"]

# Also try 2081-2100 as optional supplement
PERIODS_EXTRA = ["2021-2040", "2081-2100"]

# BIO variables we need (from the 19 WorldClim bioclimatic variables)
NEEDED_BIOS = {"bio02", "bio03", "bio05", "bio15"}

# ——— Download with retry ———
def download_file(url, dest_path, max_retries=3):
    """Download a file with retry logic."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading: {url}")
            tmp_path = str(dest_path) + ".tmp"
            urlretrieve(url, tmp_path)
            os.replace(tmp_path, dest_path)
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            logger.info(f"  -> Downloaded {size_mb:.1f} MB")
            return True
        except HTTPError as e:
            if e.code == 404:
                logger.warning(f"  -> 404 Not Found: {url}")
                return False
            logger.warning(f"  -> HTTP {e.code}, attempt {attempt+1}/{max_retries}")
            time.sleep(5 * (attempt + 1))
        except (URLError, ConnectionError, TimeoutError) as e:
            logger.warning(f"  -> Connection error: {e}, attempt {attempt+1}/{max_retries}")
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            logger.warning(f"  -> Error: {e}, attempt {attempt+1}/{max_retries}")
            time.sleep(5 * (attempt + 1))
    return False

def check_url_exists(url):
    """Quick check if URL exists."""
    try:
        req = urlopen(url, timeout=10)
        return True
    except:
        return False

def extract_zip(zip_path, extract_dir):
    """Extract a zip file to a directory."""
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(str(extract_dir))
    return extract_dir

def sha256_file(path):
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# ——— Main download logic ———
def build_filename(gcm, ssp, period):
    """Build WorldClim CMIP6 filename."""
    # Pattern: wc2.1_2.5m_bioc_GCM_SSP_period.zip
    # GCM name might need adjustments
    gcm_name = gcm
    return f"wc2.1_2.5m_bioc_{gcm_name}_{ssp}_{period}.zip"

def build_url(gcm, ssp, period):
    """Build download URL."""
    fname = build_filename(gcm, ssp, period)
    return f"{BASE_URL}/{fname}"

def main():
    logger.info("=" * 60)
    logger.info("CMIP6 Future Climate Download — Experiment 4")
    logger.info(f"Target: {len(GCMS)} GCMs × {len(SSPS)} SSPs × {len(PERIODS)} periods = {len(GCMS)*len(SSPS)*len(PERIODS)} scenarios")
    logger.info("=" * 60)

    # ——— First, test connectivity ———
    test_url = f"{BASE_URL}/"
    logger.info(f"Testing connectivity to {test_url}...")
    if not check_url_exists(test_url):
        logger.warning("Cannot reach WorldClim server. May need VPN or alternative approach.")
        logger.info("Will try individual file downloads anyway...")

    # ——— Download each scenario ———
    registry_rows = []
    total = len(GCMS) * len(SSPS) * len(PERIODS)
    completed = 0
    failed = []

    for gcm in GCMS:
        for ssp in SSPS:
            for period in PERIODS:
                fname = build_filename(gcm, ssp, period)
                url = build_url(gcm, ssp, period)

                # Directory structure: GCM/SSP/period/
                scenario_dir = RAW_DIR / gcm / ssp / period
                scenario_dir.mkdir(parents=True, exist_ok=True)
                zip_path = scenario_dir / fname

                logger.info(f"\n[{completed+1}/{total}] {gcm} | {ssp} | {period}")

                if zip_path.exists():
                    # Verify existing file
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            names = zf.namelist()
                        logger.info(f"  -> Already exists ({os.path.getsize(zip_path)/(1024*1024):.1f} MB), {len(names)} files inside")
                        completed += 1
                    except (zipfile.BadZipFile, Exception):
                        logger.warning(f"  -> Corrupt zip, re-downloading...")
                        os.remove(zip_path)

                if not zip_path.exists():
                    success = download_file(url, zip_path)
                    if not success:
                        # Try alternative: some periods use different naming
                        failed.append((gcm, ssp, period, url))
                        logger.warning(f"  -> FAILED to download")
                        continue
                    completed += 1

                # Extract key BIO variables
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        all_names = zf.namelist()
                        # Find our needed BIO files
                        bio_files_found = {}
                        for name in all_names:
                            basename = os.path.basename(name).replace('.tif', '')
                            # WorldClim files are like: wc2.1_2.5m_bioc_GCM_SSP_period_bio02.tif
                            for bio_var in NEEDED_BIOS:
                                if f"_{bio_var}" in basename or basename.endswith(bio_var):
                                    bio_files_found[bio_var] = name

                        # Extract needed BIOs
                        for bio_var in NEEDED_BIOS:
                            if bio_var in bio_files_found:
                                zf.extract(bio_files_found[bio_var], str(scenario_dir))
                    logger.info(f"  -> Extracted {len(bio_files_found)} BIO variables: {sorted(bio_files_found.keys())}")
                except Exception as e:
                    logger.error(f"  -> Extraction error: {e}")
                    continue

                # Record in registry
                registry_rows.append({
                    "gcm": gcm,
                    "ssp": ssp,
                    "period": period,
                    "source_file": fname,
                    "format": "GeoTIFF",
                    "downloaded": True,
                    "status": "complete",
                })

    # ——— Also try extra periods ———
    logger.info("\n--- Optional: trying extra periods ---")
    for gcm in GCMS[:1]:  # Just test with first GCM
        for ssp in ["ssp126", "ssp585"]:
            for period in PERIODS_EXTRA:
                fname = build_filename(gcm, ssp, period)
                url = build_url(gcm, ssp, period)
                scenario_dir = RAW_DIR / gcm / ssp / period
                scenario_dir.mkdir(parents=True, exist_ok=True)
                zip_path = scenario_dir / fname
                if not zip_path.exists():
                    success = download_file(url, zip_path)
                    if success:
                        logger.info(f"  -> Extra period {period} available for {gcm}")
                        registry_rows.append({
                            "gcm": gcm, "ssp": ssp, "period": period,
                            "source_file": fname, "format": "GeoTIFF",
                            "downloaded": True, "status": "supplementary",
                        })
                    else:
                        logger.info(f"  -> Extra period {period} NOT available")

    # ——— Summary ———
    logger.info("\n" + "=" * 60)
    logger.info(f"Download Summary:")
    logger.info(f"  Completed: {completed}/{total}")
    logger.info(f"  Failed: {len(failed)}")
    if failed:
        logger.info("  Failed downloads:")
        for g, s, p, u in failed:
            logger.info(f"    {g} | {s} | {p} -> {u}")

    # ——— Save registry ———
    registry_path = REGISTRY_DIR / "future_climate_download_registry.csv"
    fieldnames = ["gcm", "ssp", "period", "source_file", "format", "downloaded", "status"]
    with open(registry_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(registry_rows)
    logger.info(f"\nDownload registry saved to {registry_path}")

    # ——— Stop if insufficient data ———
    if completed < 10:  # Need at least ~10 of 40 scenarios
        stop_path = EXP4_DIR / "01_input_qc" / "STOP_A_CMIP6_NOT_READY.md"
        with open(stop_path, "w", encoding="utf-8") as f:
            f.write(f"# STOP_A_CMIP6_NOT_READY\n\nOnly {completed}/{total} CMIP6 files downloaded.\n"
                    f"Cannot proceed with formal projections.\n\nFailed:\n")
            for g, s, p, u in failed:
                f.write(f"- {g} | {s} | {p}\n")
        logger.critical(f"STOP_A: Insufficient CMIP6 data ({completed}/{total})")
        sys.exit(1)

    return completed, failed, registry_rows

if __name__ == "__main__":
    main()
