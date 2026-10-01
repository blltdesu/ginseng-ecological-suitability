#!/usr/bin/env python3
"""
Experiment 4 — GCM Summary Map Generation
Computes GCM-mean and GCM-median ensemble suitability maps per SSP×period.
Also generates mean binary, mean change, and GCM agreement maps.
"""
import os, sys, logging
from pathlib import Path
import rasterio
import numpy as np

EXP4_DIR = Path(r"E:\人参种在哪\实验4")
INPUT_DIR = EXP4_DIR / "00_input_from_experiment3"
ENSEMBLE_DIR = EXP4_DIR / "08_future_predictions_ensemble"
SUMMARY_DIR = ENSEMBLE_DIR / "GCM_summary"
LOG_DIR = EXP4_DIR / "logs"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(str(LOG_DIR / "gcm_summary.log"), encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gcm_summary")

GCMS = ["ACCESS-CM2", "BCC-CSM2-MR", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]

import json
with open(INPUT_DIR / "ensemble_threshold.json", "r") as f:
    ENSEMBLE_THRESHOLD = json.load(f)["ensemble_threshold"]

# Load reference
MASK_PATH = INPUT_DIR / "common_valid_mask.tif"
CURRENT_BINARY = INPUT_DIR / "current_binary_suitability.tif"
with rasterio.open(str(MASK_PATH)) as msrc:
    valid_mask = msrc.read(1) > 0
    ref_profile = msrc.profile.copy()
    ref_shape = msrc.shape

with rasterio.open(str(CURRENT_BINARY)) as s:
    current_binary = (s.read(1) > 0.5) & valid_mask

def main():
    logger.info("=== GCM Summary Map Generation ===")

    for ssp in SSPS:
        for period in PERIODS:
            logger.info(f"\n{ssp} | {period}")

            # Collect all available GCM ensembles for this SSP×period
            gcm_rasters = []
            available_gcms = []
            for gcm in GCMS:
                ens_path = ENSEMBLE_DIR / gcm / ssp / period / "ensemble_suitability.tif"
                if ens_path.exists():
                    with rasterio.open(str(ens_path)) as s:
                        gcm_rasters.append(s.read(1).astype(np.float32))
                    available_gcms.append(gcm)
                else:
                    logger.warning(f"  {gcm} — not available, skipping")

            n_gcms = len(gcm_rasters)
            if n_gcms == 0:
                logger.warning(f"  No GCMs available for {ssp}/{period} — skipping")
                continue

            logger.info(f"  Available GCMs: {n_gcms}/5 ({', '.join(available_gcms)})")

            # Stack and compute statistics
            stack = np.stack(gcm_rasters, axis=0)  # (n_gcms, rows, cols)

            # Mean suitability
            mean_suit = np.nanmean(stack, axis=0)
            mean_path = SUMMARY_DIR / f"{ssp}_{period}_mean.tif"
            profile = ref_profile.copy()
            profile.pop('shape', None)
            profile.update(dtype=np.float32, compress='lzw', nodata=-3.4e38)
            with rasterio.open(str(mean_path), 'w', **profile) as dst:
                dst.write(mean_suit.astype(np.float32), 1)
            logger.info(f"  Mean: saved, range=[{np.nanmin(mean_suit):.4f}, {np.nanmax(mean_suit):.4f}]")

            # Median suitability
            median_suit = np.nanmedian(stack, axis=0)
            median_path = SUMMARY_DIR / f"{ssp}_{period}_median.tif"
            with rasterio.open(str(median_path), 'w', **profile) as dst:
                dst.write(median_suit.astype(np.float32), 1)
            logger.info(f"  Median: saved, range=[{np.nanmin(median_suit):.4f}, {np.nanmax(median_suit):.4f}]")

            # Std deviation (inter-GCM spread)
            std_suit = np.nanstd(stack, axis=0)
            std_path = SUMMARY_DIR / f"{ssp}_{period}_std.tif"
            with rasterio.open(str(std_path), 'w', **profile) as dst:
                dst.write(std_suit.astype(np.float32), 1)
            logger.info(f"  Std: saved, range=[{np.nanmin(std_suit):.4f}, {np.nanmax(std_suit):.4f}]")

            # GCM agreement on binary (how many GCMs agree pixel is suitable)
            binary_stack = stack >= ENSEMBLE_THRESHOLD
            agreement = np.sum(binary_stack, axis=0).astype(np.float32)
            # Mask: only where at least half of available GCMs have data
            valid_data_mask = np.sum(~np.isnan(stack), axis=0) >= max(1, n_gcms // 2)
            agreement[~valid_data_mask] = np.nan

            agree_path = SUMMARY_DIR / f"{ssp}_{period}_agreement.tif"
            with rasterio.open(str(agree_path), 'w', **profile) as dst:
                dst.write(agreement.astype(np.float32), 1)
            logger.info(f"  Agreement: {n_gcms}-GCM, range=[{np.nanmin(agreement):.0f}, {np.nanmax(agreement):.0f}]")

            # Mean binary (majority vote: >= half of GCMs agree on suitable)
            mean_binary = (agreement >= max(1, n_gcms // 2)).astype(np.uint8)
            bin_path = SUMMARY_DIR / f"{ssp}_{period}_majority_binary.tif"
            profile2 = ref_profile.copy()
            profile2.pop('shape', None)
            profile2.update(dtype=np.uint8, compress='lzw', nodata=255)
            bin_out = np.where(valid_mask, mean_binary, 255)
            with rasterio.open(str(bin_path), 'w', **profile2) as dst:
                dst.write(bin_out, 1)
            n_suitable = mean_binary.sum()
            logger.info(f"  Majority binary: {n_suitable} suitable pixels")

            # Mean change map (majority vote vs current)
            change_map = np.full(ref_shape, 255, dtype=np.uint8)
            change_map[current_binary & mean_binary] = 1    # stable
            change_map[current_binary & ~mean_binary] = 2   # loss
            change_map[~current_binary & mean_binary] = 3   # gain
            change_map[~current_binary & ~mean_binary] = 0  # persistent unsuitable

            change_path = SUMMARY_DIR / f"{ssp}_{period}_majority_change.tif"
            with rasterio.open(str(change_path), 'w', **profile2) as dst:
                dst.write(change_map, 1)
            n_stable = (change_map == 1).sum()
            n_loss = (change_map == 2).sum()
            n_gain = (change_map == 3).sum()
            logger.info(f"  Change: stable={n_stable}, loss={n_loss}, gain={n_gain}")

    logger.info("\n=== GCM Summary Maps Complete ===")

if __name__ == "__main__":
    main()
