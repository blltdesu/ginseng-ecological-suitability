#!/usr/bin/env python3
"""
Experiment 5 - Script 01b: Percentile Computation (memory-mapped)
==================================================================
Computes median, P10, P90, IQR using numpy memmap for efficient
out-of-core processing of 36 global rasters.
"""

import os
from pathlib import Path
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

EXP5_DIR = Path(r"E:\人参种在哪\实验5")
INPUT_DIR = EXP5_DIR / "00_input_from_experiment4"
OUT_AGREEMENT = EXP5_DIR / "04_agreement"
OUT_UNCERTAINTY = EXP5_DIR / "05_uncertainty_components"

# Temp directory for memmap file
MEMORY_MAP_PATH = EXP5_DIR / "03_prediction_cube" / "ensemble_stack.dat"


def main():
    print("=" * 60)
    print("Percentile Computation (memory-mapped)")
    print("=" * 60)

    # Load registry
    reg_path = EXP5_DIR / "02_scenario_registry" / "scenario_availability.csv"
    df = pd.read_csv(reg_path)
    available = df[df["ensemble_available"] == True]
    n_scenarios = len(available)
    print(f"Scenarios: {n_scenarios}")

    # Get profile
    ref_path = INPUT_DIR / "current_baseline" / "reference_grid_template.tif"
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        n_rows, n_cols = ref.shape
    profile.update(dtype="float32", nodata=np.nan)

    shape = (n_scenarios, n_rows, n_cols)
    print(f"Stack shape: {shape} ({np.prod(shape)*4/1024**3:.1f} GB)")

    # Step 1: Write all scenarios to a single memory-mapped file
    print("\n[Step 1] Writing scenarios to memmap file...")
    mmap = np.memmap(MEMORY_MAP_PATH, dtype="float32", mode="w+", shape=shape)

    nodata_val = None
    for i, (_, row) in enumerate(available.iterrows()):
        print(f"  Scenario {i+1}/{n_scenarios}: {row['ensemble_path']}", flush=True)
        with rasterio.open(row["ensemble_path"]) as src:
            data = src.read(1).astype(np.float32)
            if nodata_val is None:
                nodata_val = src.nodata
            data[data == nodata_val] = np.nan
            mmap[i] = data

    mmap.flush()
    print(f"  Memmap written: {MEMORY_MAP_PATH}")

    # Step 2: Compute percentiles in blocks
    print(f"\n[Step 2] Computing percentiles in blocks...")

    iqr_path = OUT_UNCERTAINTY / "scenario_iqr.tif"
    p10_path = OUT_UNCERTAINTY / "scenario_p10.tif"
    p90_path = OUT_UNCERTAINTY / "scenario_p90.tif"
    median_path = OUT_AGREEMENT / "future_suitability_median_all_scenarios.tif"

    iqr_writer = rasterio.open(iqr_path, "w", **profile)
    p10_writer = rasterio.open(p10_path, "w", **profile)
    p90_writer = rasterio.open(p90_path, "w", **profile)
    median_writer = rasterio.open(median_path, "w", **profile)

    # Process in row blocks to manage memory during percentile computation
    BLK = 200  # 200 rows: numpy percentile reads ~200*8640*36*4 = 249 MB
    n_blocks = (n_rows + BLK - 1) // BLK

    for blk_idx in range(n_blocks):
        blk_start = blk_idx * BLK
        blk_end = min(blk_start + BLK, n_rows)
        blk_h = blk_end - blk_start
        print(f"  Block {blk_idx+1}/{n_blocks} [{blk_start}:{blk_end}]...", end=" ", flush=True)

        # Read block from memmap (much faster than reading 36 individual files)
        block_stack = mmap[:, blk_start:blk_end, :].copy()  # Copy to RAM for percentile computation

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            blk_p10 = np.nanpercentile(block_stack, 10, axis=0).astype(np.float32)
            blk_p90 = np.nanpercentile(block_stack, 90, axis=0).astype(np.float32)
            blk_p25 = np.nanpercentile(block_stack, 25, axis=0).astype(np.float32)
            blk_p75 = np.nanpercentile(block_stack, 75, axis=0).astype(np.float32)
            blk_med = np.nanmedian(block_stack, axis=0).astype(np.float32)
        blk_iqr = blk_p75 - blk_p25

        iqr_writer.write(blk_iqr, 1, window=Window(0, blk_start, n_cols, blk_h))
        p10_writer.write(blk_p10, 1, window=Window(0, blk_start, n_cols, blk_h))
        p90_writer.write(blk_p90, 1, window=Window(0, blk_start, n_cols, blk_h))
        median_writer.write(blk_med, 1, window=Window(0, blk_start, n_cols, blk_h))

        del block_stack, blk_p10, blk_p90, blk_p25, blk_p75, blk_med, blk_iqr
        print("done", flush=True)

    iqr_writer.close(); p10_writer.close(); p90_writer.close(); median_writer.close()

    # Clean up memmap
    print("\n[Step 3] Cleaning up...")
    del mmap
    os.remove(MEMORY_MAP_PATH)
    print(f"  Removed {MEMORY_MAP_PATH}")

    print("\nPercentile computation complete!")


if __name__ == "__main__":
    main()
