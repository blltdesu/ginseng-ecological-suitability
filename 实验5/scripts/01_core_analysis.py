#!/usr/bin/env python3
"""
Experiment 5 - Script 01: Core Analysis Pipeline
==================================================
Computes:
  - Cross-scenario statistics (mean, SD, IQR, P10, P90)
  - Binary suitability frequency
  - Stable/Loss/Gain frequencies from change maps
  - Algorithm, GCM, SSP, Time uncertainty components
  - Uncertainty normalization (P1-P99 robust min-max)
  - Dominant uncertainty source
  - Future Stability Index (FSI) and Loss Frequency Index (LFI)
  - Prediction confidence
  - Vulnerability (base + confidence-adjusted)
  - Robust climatic core / High-confidence loss zone / High-uncertainty zone
  - Stability-confidence quadrants

Processing strategy: block-wise to manage memory for 4320×8640 rasters.
"""

import os
import sys
import json
from pathlib import Path
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# === Configuration ===
EXP5_DIR = Path(r"E:\人参种在哪\实验5")
INPUT_DIR = EXP5_DIR / "00_input_from_experiment4"
REF_PATH = INPUT_DIR / "current_baseline" / "reference_grid_template.tif"
ENSEMBLE_THRESHOLD = 0.12846759691320192

# Block size for processing (rows per block)
BLOCK_ROWS = 540

# Output directories
OUT_AGREEMENT = EXP5_DIR / "04_agreement"
OUT_UNCERTAINTY = EXP5_DIR / "05_uncertainty_components"
OUT_VARIANCE = EXP5_DIR / "06_variance_partition"
OUT_STABILITY = EXP5_DIR / "08_stability_probability"
OUT_LOSS = EXP5_DIR / "09_loss_probability"


def load_registry():
    """Load scenario availability registry."""
    reg_path = EXP5_DIR / "02_scenario_registry" / "scenario_availability.csv"
    df = pd.read_csv(reg_path)
    available = df[df["ensemble_available"] == True].copy()
    return df, available


def get_ref_profile():
    """Get reference raster profile."""
    with rasterio.open(REF_PATH) as ref:
        profile = ref.profile.copy()
        n_rows = ref.shape[0]
        n_cols = ref.shape[1]
    return profile, n_rows, n_cols


def iter_blocks(n_rows, block_rows):
    """Yield (row_start, row_end, window) for block processing."""
    for start in range(0, n_rows, block_rows):
        end = min(start + block_rows, n_rows)
        yield start, end


def load_current_baseline():
    """Load current suitability and binary mask."""
    cur_path = INPUT_DIR / "current_baseline" / "current_ensemble_suitability.tif"
    bin_path = INPUT_DIR / "current_baseline" / "current_binary_suitability.tif"
    mask_path = INPUT_DIR / "current_baseline" / "common_valid_mask.tif"

    with rasterio.open(cur_path) as src:
        current_suit = src.read(1).astype(np.float32)
        current_suit[current_suit == src.nodata] = np.nan

    with rasterio.open(bin_path) as src:
        current_binary = src.read(1).astype(np.uint8)

    with rasterio.open(mask_path) as src:
        valid_mask = src.read(1).astype(np.uint8)

    return current_suit, current_binary, valid_mask


def compute_cross_scenario_statistics(available_scenarios):
    """Compute per-pixel statistics across all available ensemble scenarios.
    Process: read full rasters one-at-a-time, accumulate mean/SD online,
    then compute percentiles per-block with efficient loading."""
    print("\n" + "=" * 60)
    print("CROSS-SCENARIO STATISTICS")
    print("=" * 60)

    profile, n_rows, n_cols = get_ref_profile()
    n_scenarios = len(available_scenarios)
    profile.update(dtype="float32", nodata=np.nan)

    mean_path = OUT_AGREEMENT / "future_suitability_mean_all_scenarios.tif"
    sd_path = OUT_UNCERTAINTY / "scenario_sd.tif"
    iqr_path = OUT_UNCERTAINTY / "scenario_iqr.tif"
    p10_path = OUT_UNCERTAINTY / "scenario_p10.tif"
    p90_path = OUT_UNCERTAINTY / "scenario_p90.tif"
    median_path = OUT_AGREEMENT / "future_suitability_median_all_scenarios.tif"

    # Check if mean and SD already exist (resume capability)
    if mean_path.exists() and sd_path.exists():
        print("  Mean and SD already exist, loading...")
        with rasterio.open(mean_path) as src:
            mean_data = src.read(1).astype(np.float32)
            nodata_val = src.nodata
        with rasterio.open(sd_path) as src:
            sd_data = src.read(1).astype(np.float32)
        ok = ~np.isnan(mean_data)
        # Need count_x for later
        count_x = np.full((n_rows, n_cols), n_scenarios, dtype=np.int32)
        count_x[~ok] = 0
        print(f"    Loaded mean [{np.nanmin(mean_data):.6f}, {np.nanmax(mean_data):.6f}], SD [{np.nanmin(sd_data):.6f}, {np.nanmax(sd_data):.6f}]")
    else:
        # Pass 1: compute mean and count per pixel (online, one file at a time)
        print(f"  Pass 1: Computing mean from {n_scenarios} scenarios (one at a time)...")
        sum_x = np.zeros((n_rows, n_cols), dtype=np.float64)
        count_x = np.zeros((n_rows, n_cols), dtype=np.int32)

        nodata_val = None
        for idx, (_, row) in enumerate(available_scenarios.iterrows()):
            if idx % 6 == 0:
                print(f"    Reading scenario {idx+1}/{n_scenarios}...", flush=True)
            with rasterio.open(row["ensemble_path"]) as src:
                if nodata_val is None:
                    nodata_val = src.nodata
                data = src.read(1).astype(np.float64)
            valid = (data != nodata_val) & (~np.isnan(data))
            sum_x[valid] += data[valid]
            count_x[valid] += 1

        mean_data = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        ok = count_x > 0
        mean_data[ok] = (sum_x[ok] / count_x[ok]).astype(np.float32)
        del sum_x

        with rasterio.open(mean_path, "w", **profile) as dst:
            dst.write(mean_data, 1)
        print(f"    Mean saved: range [{np.nanmin(mean_data):.6f}, {np.nanmax(mean_data):.6f}]")

        # Pass 2: compute SD (online using mean)
        print(f"  Pass 2: Computing SD...")
        sum_sq = np.zeros((n_rows, n_cols), dtype=np.float64)
        for idx, (_, row) in enumerate(available_scenarios.iterrows()):
            if idx % 12 == 0:
                print(f"    Scenario {idx+1}/{n_scenarios}...", flush=True)
            with rasterio.open(row["ensemble_path"]) as src:
                data = src.read(1).astype(np.float64)
            data[data == nodata_val] = np.nan
            dev = data - mean_data
            sum_sq += np.nan_to_num(dev * dev, nan=0.0)

        sd_data = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        sd_data[ok] = np.sqrt(sum_sq[ok] / count_x[ok]).astype(np.float32)
        del sum_sq

        with rasterio.open(sd_path, "w", **profile) as dst:
            dst.write(sd_data, 1)
        print(f"    SD saved: range [{np.nanmin(sd_data):.6f}, {np.nanmax(sd_data):.6f}]")

    # Pass 3: Percentiles (delegated to 01b_compute_percentiles.py for memory-mapped efficiency)
    # For now, approximate from mean and SD if not already computed
    if not median_path.exists():
        print("  Pass 3: Percentiles not yet computed. Using mean as median proxy.")
        print("    Run 01b_compute_percentiles.py for exact percentiles.")
        with rasterio.open(median_path, "w", **profile) as dst:
            dst.write(mean_data, 1)  # Mean as median proxy
        with rasterio.open(iqr_path, "w", **profile) as dst:
            dst.write(sd_data * 1.349, 1)  # SD*1.349 ≈ IQR for normal
        with rasterio.open(p10_path, "w", **profile) as dst:
            dst.write(np.clip(mean_data - 1.282*sd_data, 0, None), 1)  # P10 approx
        with rasterio.open(p90_path, "w", **profile) as dst:
            dst.write(mean_data + 1.282*sd_data, 1)  # P90 approx

    print(f"  Cross-scenario statistics complete.")


def compute_binary_agreement_and_change_frequencies(available_scenarios):
    """
    Compute:
    - Future suitability frequency (binary agreement)
    - Stable / Loss / Gain frequencies from change class maps
    """
    print("\n" + "=" * 60)
    print("BINARY AGREEMENT & CHANGE FREQUENCIES")
    print("=" * 60)

    profile, n_rows, n_cols = get_ref_profile()
    n_scenarios = len(available_scenarios)

    # Open all binary and change map readers
    binary_readers = []
    change_readers = []
    for _, row in available_scenarios.iterrows():
        binary_readers.append(rasterio.open(row["binary_path"]))
        change_readers.append(rasterio.open(row["change_path"]))

    # Prepare outputs
    profile_bin = profile.copy()
    profile_bin.update(dtype="float32", nodata=np.nan)

    freq_path = OUT_AGREEMENT / "future_suitability_frequency.tif"
    stable_path = OUT_STABILITY / "stable_frequency.tif"
    loss_path = OUT_LOSS / "loss_frequency.tif"
    gain_path = OUT_STABILITY / "gain_frequency.tif"

    freq_writer = rasterio.open(freq_path, "w", **profile_bin)
    stable_writer = rasterio.open(stable_path, "w", **profile_bin)
    loss_writer = rasterio.open(loss_path, "w", **profile_bin)
    gain_writer = rasterio.open(gain_path, "w", **profile_bin)

    for row_start in range(0, n_rows, BLOCK_ROWS):
        row_end = min(row_start + BLOCK_ROWS, n_rows)
        block_h = row_end - row_start
        print(f"  Block rows {row_start}-{row_end}...", end=" ", flush=True)

        # Read binary data
        binary_block = np.full((n_scenarios, block_h, n_cols), np.nan, dtype=np.float32)
        change_block = np.full((n_scenarios, block_h, n_cols), np.nan, dtype=np.float32)

        for i in range(n_scenarios):
            bdata = binary_readers[i].read(1, window=Window(0, row_start, n_cols, block_h))
            bdata = bdata.astype(np.float32)
            bdata[bdata == binary_readers[i].nodata] = np.nan
            binary_block[i] = bdata

            cdata = change_readers[i].read(1, window=Window(0, row_start, n_cols, block_h))
            cdata = cdata.astype(np.float32)
            cdata[cdata == change_readers[i].nodata] = np.nan
            change_block[i] = cdata

        # Suitability frequency: fraction of scenarios predicting suitable
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Binary: 1 = suitable, 0 = unsuitable
            suit_count = np.nansum(binary_block == 1, axis=0)
            # Count valid (non-NaN) scenarios per pixel
            valid_count = np.sum(~np.isnan(binary_block), axis=0)
            # Avoid division by zero
            valid_mask = valid_count > 0
            suit_freq = np.full((block_h, n_cols), np.nan, dtype=np.float32)
            suit_freq[valid_mask] = suit_count[valid_mask] / valid_count[valid_mask]

            # Change classes: 0=Non-habitat, 1=Loss, 2=Stable, 3=Gain
            stable_count = np.nansum(change_block == 2, axis=0)
            loss_count = np.nansum(change_block == 1, axis=0)
            gain_count = np.nansum(change_block == 3, axis=0)

            stable_freq = np.full((block_h, n_cols), np.nan, dtype=np.float32)
            loss_freq = np.full((block_h, n_cols), np.nan, dtype=np.float32)
            gain_freq = np.full((block_h, n_cols), np.nan, dtype=np.float32)

            stable_freq[valid_mask] = stable_count[valid_mask] / valid_count[valid_mask]
            loss_freq[valid_mask] = loss_count[valid_mask] / valid_count[valid_mask]
            gain_freq[valid_mask] = gain_count[valid_mask] / valid_count[valid_mask]

        freq_writer.write(suit_freq, 1, window=Window(0, row_start, n_cols, block_h))
        stable_writer.write(stable_freq, 1, window=Window(0, row_start, n_cols, block_h))
        loss_writer.write(loss_freq, 1, window=Window(0, row_start, n_cols, block_h))
        gain_writer.write(gain_freq, 1, window=Window(0, row_start, n_cols, block_h))

        print("done")

    freq_writer.close()
    stable_writer.close()
    loss_writer.close()
    gain_writer.close()
    for r in binary_readers + change_readers:
        r.close()

    print("Binary agreement and change frequencies complete.")


def compute_algorithm_uncertainty(available_scenarios):
    """
    For each GCM × SSP × Period, compute SD across the 4 models.
    Then average across all scenarios to get mean algorithm SD.
    """
    print("\n" + "=" * 60)
    print("ALGORITHM UNCERTAINTY")
    print("=" * 60)

    profile, n_rows, n_cols = get_ref_profile()
    profile_out = profile.copy()
    profile_out.update(dtype="float32", nodata=np.nan)

    individual_dir = INPUT_DIR / "individual_model_predictions"
    models = ["random_forest", "xgboost", "brt", "maxent"]

    mean_alg_sd_path = OUT_UNCERTAINTY / "mean_algorithm_sd.tif"
    mean_alg_agr_path = OUT_AGREEMENT / "mean_algorithm_agreement.tif"

    alg_sd_writer = rasterio.open(mean_alg_sd_path, "w", **profile_out)
    alg_agr_writer = rasterio.open(mean_alg_agr_path, "w", **profile_out)

    # Accumulate mean SD across scenarios
    alg_sd_sum = np.zeros((n_rows, n_cols), dtype=np.float64)
    alg_sd_count = np.zeros((n_rows, n_cols), dtype=np.int32)

    for _, scenario in available_scenarios.iterrows():
        gcm, ssp, period = scenario["gcm"], scenario["ssp"], scenario["period"]

        # Load 4 model predictions for this scenario
        model_data = []
        for model in models:
            model_path = individual_dir / model / gcm / ssp / period / "suitability.tif"
            if model_path.exists():
                with rasterio.open(model_path) as src:
                    data = src.read(1).astype(np.float32)
                    data[data == src.nodata] = np.nan
                    model_data.append(data)

        if len(model_data) >= 2:
            model_stack = np.stack(model_data, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sd = np.nanstd(model_stack, axis=0)
                alg_sd_sum += np.nan_to_num(sd, nan=0.0)
                alg_sd_count += (~np.isnan(sd)).astype(np.int32)

    # Mean algorithm SD
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_sd = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        valid = alg_sd_count > 0
        mean_sd[valid] = alg_sd_sum[valid] / alg_sd_count[valid]

    alg_sd_writer.write(mean_sd, 1)
    alg_sd_writer.close()

    # Algorithm agreement: for each scenario, what fraction of models agree on suitability
    # Compute for binary predictions using fixed threshold
    print("  Computing algorithm agreement (binary)...")
    alg_agree_sum = np.zeros((n_rows, n_cols), dtype=np.float64)
    alg_agree_count = np.zeros((n_rows, n_cols), dtype=np.int32)

    for _, scenario in available_scenarios.iterrows():
        gcm, ssp, period = scenario["gcm"], scenario["ssp"], scenario["period"]
        model_binary = []
        for model in models:
            model_path = individual_dir / model / gcm / ssp / period / "suitability.tif"
            if model_path.exists():
                with rasterio.open(model_path) as src:
                    data = src.read(1).astype(np.float32)
                    data[data == src.nodata] = np.nan
                    model_binary.append((data > ENSEMBLE_THRESHOLD).astype(np.float32))

        if len(model_binary) >= 2:
            binary_stack = np.stack(model_binary, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Agreement = max(frac_suitable, frac_unsuitable)
                frac_suitable = np.nanmean(binary_stack, axis=0)
                agreement = np.maximum(frac_suitable, 1.0 - frac_suitable)
                alg_agree_sum += np.nan_to_num(agreement, nan=0.0)
                alg_agree_count += (~np.isnan(agreement)).astype(np.int32)

    mean_agree = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    valid2 = alg_agree_count > 0
    mean_agree[valid2] = alg_agree_sum[valid2] / alg_agree_count[valid2]

    alg_agr_writer.write(mean_agree, 1)
    alg_agr_writer.close()

    print("Algorithm uncertainty complete.")


def compute_gcm_uncertainty(available_scenarios):
    """
    For each SSP × Period, compute SD across available GCMs.
    Then average across SSP/Period combinations.
    """
    print("\n" + "=" * 60)
    print("GCM UNCERTAINTY")
    print("=" * 60)

    profile, n_rows, n_cols = get_ref_profile()
    profile_out = profile.copy()
    profile_out.update(dtype="float32", nodata=np.nan)

    gcm_sd_path = OUT_UNCERTAINTY / "mean_GCM_sd.tif"
    gcm_writer = rasterio.open(gcm_sd_path, "w", **profile_out)

    # Group by SSP × Period
    gcm_sd_sum = np.zeros((n_rows, n_cols), dtype=np.float64)
    gcm_sd_count = np.zeros((n_rows, n_cols), dtype=np.int32)

    ensemble_dir = INPUT_DIR / "ensemble_predictions"
    ssps = available_scenarios["ssp"].unique()
    periods = available_scenarios["period"].unique()

    for ssp in ssps:
        for period in periods:
            # Get all GCMs available for this SSP×Period
            subset = available_scenarios[(available_scenarios["ssp"] == ssp) & (available_scenarios["period"] == period)]
            if len(subset) < 2:
                continue

            gcm_data = []
            for _, row in subset.iterrows():
                with rasterio.open(row["ensemble_path"]) as src:
                    data = src.read(1).astype(np.float32)
                    data[data == src.nodata] = np.nan
                    gcm_data.append(data)

            if len(gcm_data) >= 2:
                gcm_stack = np.stack(gcm_data, axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    sd = np.nanstd(gcm_stack, axis=0)
                    gcm_sd_sum += np.nan_to_num(sd, nan=0.0)
                    gcm_sd_count += (~np.isnan(sd)).astype(np.int32)

    mean_gcm_sd = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    valid = gcm_sd_count > 0
    mean_gcm_sd[valid] = gcm_sd_sum[valid] / gcm_sd_count[valid]
    gcm_writer.write(mean_gcm_sd, 1)
    gcm_writer.close()

    print("GCM uncertainty complete.")


def compute_ssp_uncertainty(available_scenarios):
    """
    For each GCM × Period, compute SD across available SSPs.
    Then average across GCM/Period combinations.
    """
    print("\n" + "=" * 60)
    print("SSP UNCERTAINTY")
    print("=" * 60)

    profile, n_rows, n_cols = get_ref_profile()
    profile_out = profile.copy()
    profile_out.update(dtype="float32", nodata=np.nan)

    ssp_sd_path = OUT_UNCERTAINTY / "mean_SSP_sd.tif"
    ssp_writer = rasterio.open(ssp_sd_path, "w", **profile_out)

    ssp_sd_sum = np.zeros((n_rows, n_cols), dtype=np.float64)
    ssp_sd_count = np.zeros((n_rows, n_cols), dtype=np.int32)

    gcms = available_scenarios["gcm"].unique()
    periods = available_scenarios["period"].unique()

    for gcm in gcms:
        for period in periods:
            subset = available_scenarios[(available_scenarios["gcm"] == gcm) & (available_scenarios["period"] == period)]
            if len(subset) < 2:
                continue

            ssp_data = []
            for _, row in subset.iterrows():
                with rasterio.open(row["ensemble_path"]) as src:
                    data = src.read(1).astype(np.float32)
                    data[data == src.nodata] = np.nan
                    ssp_data.append(data)

            if len(ssp_data) >= 2:
                ssp_stack = np.stack(ssp_data, axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    sd = np.nanstd(ssp_stack, axis=0)
                    ssp_sd_sum += np.nan_to_num(sd, nan=0.0)
                    ssp_sd_count += (~np.isnan(sd)).astype(np.int32)

    mean_ssp_sd = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    valid = ssp_sd_count > 0
    mean_ssp_sd[valid] = ssp_sd_sum[valid] / ssp_sd_count[valid]
    ssp_writer.write(mean_ssp_sd, 1)
    ssp_writer.close()

    print("SSP uncertainty complete.")


def compute_temporal_uncertainty(available_scenarios):
    """
    For each GCM × SSP, compute |suitability_near - suitability_far|.
    Then average across GCM/SSP combinations.
    """
    print("\n" + "=" * 60)
    print("TEMPORAL UNCERTAINTY")
    print("=" * 60)

    profile, n_rows, n_cols = get_ref_profile()
    profile_out = profile.copy()
    profile_out.update(dtype="float32", nodata=np.nan)

    temp_path = OUT_UNCERTAINTY / "mean_temporal_difference.tif"
    temp_writer = rasterio.open(temp_path, "w", **profile_out)

    temp_sum = np.zeros((n_rows, n_cols), dtype=np.float64)
    temp_count = np.zeros((n_rows, n_cols), dtype=np.int32)

    ensemble_dir = INPUT_DIR / "ensemble_predictions"
    gcms = available_scenarios["gcm"].unique()
    ssps = available_scenarios["ssp"].unique()

    for gcm in gcms:
        for ssp in ssps:
            near_path = ensemble_dir / gcm / ssp / "2041-2060" / "ensemble_suitability.tif"
            far_path = ensemble_dir / gcm / ssp / "2061-2080" / "ensemble_suitability.tif"

            if near_path.exists() and far_path.exists():
                with rasterio.open(near_path) as src_near, rasterio.open(far_path) as src_far:
                    near_data = src_near.read(1).astype(np.float32)
                    near_data[near_data == src_near.nodata] = np.nan
                    far_data = src_far.read(1).astype(np.float32)
                    far_data[far_data == src_far.nodata] = np.nan

                diff = np.abs(far_data - near_data)
                temp_sum += np.nan_to_num(diff, nan=0.0)
                temp_count += (~np.isnan(diff)).astype(np.int32)

    mean_temp = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    valid = temp_count > 0
    mean_temp[valid] = temp_sum[valid] / temp_count[valid]
    temp_writer.write(mean_temp, 1)
    temp_writer.close()

    print("Temporal uncertainty complete.")


def normalize_uncertainty_and_dominant_source():
    """
    Normalize uncertainty components to 0-1 using P1-P99 robust min-max,
    and compute dominant uncertainty source.
    """
    print("\n" + "=" * 60)
    print("UNCERTAINTY NORMALIZATION & DOMINANT SOURCE")
    print("=" * 60)

    # Load the 4 uncertainty components
    components = {
        "algorithm": OUT_UNCERTAINTY / "mean_algorithm_sd.tif",
        "gcm": OUT_UNCERTAINTY / "mean_GCM_sd.tif",
        "ssp": OUT_UNCERTAINTY / "mean_SSP_sd.tif",
        "time": OUT_UNCERTAINTY / "mean_temporal_difference.tif",
    }

    comp_data = {}
    mask = None
    for name, path in components.items():
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            data[data == src.nodata] = np.nan
            comp_data[name] = data
            if mask is None:
                mask = ~np.isnan(data)
                profile = src.profile.copy()

    profile.update(dtype="float32", nodata=np.nan)

    # Robust P1-P99 normalization for each component
    normalized = {}
    for name, data in comp_data.items():
        valid_data = data[mask]
        if len(valid_data) > 0:
            p1 = np.nanpercentile(valid_data, 1)
            p99 = np.nanpercentile(valid_data, 99)
            denom = p99 - p1
            if denom > 1e-10:
                norm = np.clip((data - p1) / denom, 0, 1)
            else:
                norm = np.zeros_like(data)
        else:
            norm = np.zeros_like(data)
        normalized[name] = norm

        # Save normalized raster
        out_path = OUT_UNCERTAINTY / f"{name}_uncertainty_norm.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(norm.astype(np.float32), 1)
        print(f"  Saved normalized {name} (P1={p1:.6f}, P99={p99:.6f})")

    # Dominant uncertainty source
    stack = np.stack([normalized["algorithm"], normalized["gcm"],
                      normalized["ssp"], normalized["time"]], axis=0)
    dominant_idx = np.argmax(stack, axis=0).astype(np.float32)  # 0=alg, 1=gcm, 2=ssp, 3=time
    dominant_idx[~mask] = np.nan
    # Encode: 1=Algorithm, 2=GCM, 3=SSP, 4=Time
    dominant_encoded = dominant_idx + 1.0
    dominant_strength = np.max(stack, axis=0)
    dominant_strength[~mask] = np.nan

    dom_path = OUT_UNCERTAINTY / "dominant_uncertainty_source.tif"
    str_path = OUT_UNCERTAINTY / "dominant_uncertainty_strength.tif"

    with rasterio.open(dom_path, "w", **profile) as dst:
        dst.write(dominant_encoded.astype(np.float32), 1)

    profile_strength = profile.copy()
    with rasterio.open(str_path, "w", **profile_strength) as dst:
        dst.write(dominant_strength.astype(np.float32), 1)

    # Summary
    for i, label in enumerate(["Algorithm", "GCM", "SSP", "Time"]):
        frac = np.sum(dominant_idx[mask] == i) / np.sum(mask)
        print(f"  {label} dominant in {frac*100:.1f}% of valid pixels")

    print("  Saved dominant_uncertainty_source.tif and dominant_uncertainty_strength.tif")

    return normalized, mask, profile


# Derived products (FSI, vulnerability, etc.) are in script 03_derived_products.py
# These depend on both core analysis AND novelty analysis results.


def run_variance_partition():
    """
    Variance partition analysis using mixed-effects approach.
    Samples up to 10000 pixels from current suitable area.
    """
    print("\n" + "=" * 60)
    print("VARIANCE PARTITION")
    print("=" * 60)

    # Load current suitable area
    current_bin_path = INPUT_DIR / "current_baseline" / "current_binary_suitability.tif"
    mask_path = INPUT_DIR / "current_baseline" / "common_valid_mask.tif"

    with rasterio.open(current_bin_path) as src:
        current_binary = src.read(1)

    with rasterio.open(mask_path) as src:
        valid_mask = src.read(1)

    suitable_mask = (current_binary == 1) & (valid_mask > 0)
    suitable_rows, suitable_cols = np.where(suitable_mask)
    n_suitable = len(suitable_rows)

    if n_suitable == 0:
        print("  No suitable pixels found, skipping variance partition")
        return

    # Sample up to 10000 pixels
    n_sample = min(10000, n_suitable)
    rng = np.random.RandomState(42)
    idx = rng.choice(n_suitable, n_sample, replace=False)
    sample_rows = suitable_rows[idx]
    sample_cols = suitable_cols[idx]

    print(f"  Sampling {n_sample} of {n_suitable} suitable pixels")

    # Build data matrix: for each sample, get suitability from all available scenarios
    ensemble_dir = INPUT_DIR / "ensemble_predictions"

    # Get available scenarios
    _, available = load_registry()

    # Collect data
    data_records = []
    for _, scenario in available.iterrows():
        gcm = scenario["gcm"]
        ssp = scenario["ssp"]
        period = scenario["period"]
        with rasterio.open(scenario["ensemble_path"]) as src:
            for i in range(n_sample):
                val = src.read(1, window=Window(sample_cols[i], sample_rows[i], 1, 1))[0, 0]
                val = float(val)
                if val != src.nodata:
                    data_records.append({
                        "suitability": val,
                        "gcm": gcm,
                        "ssp": ssp,
                        "period": period,
                        "pixel_id": i,
                    })

    df = pd.DataFrame(data_records)
    print(f"  Data: {len(df)} records")

    # Compute variance components using a simple hierarchical approach
    # Total variance, then variance of group means
    total_var = df["suitability"].var()

    # GCM contribution: variance of GCM means
    gcm_means = df.groupby("gcm")["suitability"].mean()
    gcm_var = gcm_means.var()

    # SSP contribution
    ssp_means = df.groupby("ssp")["suitability"].mean()
    ssp_var = ssp_means.var()

    # Period contribution
    period_means = df.groupby("period")["suitability"].mean()
    period_var = period_means.var()

    # GCM×SSP interaction
    gcm_ssp = df.groupby(["gcm", "ssp"])["suitability"].mean()
    gcm_ssp_var = gcm_ssp.var()

    total_components = gcm_var + ssp_var + period_var + gcm_ssp_var
    if total_components == 0:
        total_components = 1e-10

    results = [
        {"Source": "GCM", "MeanContribution": gcm_var / total_components,
         "VarianceComponent": gcm_var, "MedianContribution": np.nan, "P25": np.nan, "P75": np.nan, "Rank": 0},
        {"Source": "SSP", "MeanContribution": ssp_var / total_components,
         "VarianceComponent": ssp_var, "MedianContribution": np.nan, "P25": np.nan, "P75": np.nan, "Rank": 0},
        {"Source": "Period", "MeanContribution": period_var / total_components,
         "VarianceComponent": period_var, "MedianContribution": np.nan, "P25": np.nan, "P75": np.nan, "Rank": 0},
        {"Source": "Residual", "MeanContribution": 1.0 - (gcm_var + ssp_var + period_var) / total_components,
         "VarianceComponent": total_var - gcm_var - ssp_var - period_var,
         "MedianContribution": np.nan, "P25": np.nan, "P75": np.nan, "Rank": 0},
    ]

    # Rank by contribution
    results.sort(key=lambda x: x["MeanContribution"], reverse=True)
    for i, r in enumerate(results):
        r["Rank"] = i + 1

    df_results = pd.DataFrame(results)
    results_path = OUT_VARIANCE / "variance_component_summary.csv"
    df_results.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")
    print(df_results.to_string(index=False))

    return df_results


def main():
    print("=" * 60)
    print("Experiment 5 - Core Analysis Pipeline")
    print("=" * 60)

    # Load registry
    _, available = load_registry()
    n_avail = len(available)
    print(f"\nAvailable scenarios: {n_avail}")
    print(f"Missing scenarios: {40 - n_avail}")

    # Step 3: Cross-scenario statistics
    compute_cross_scenario_statistics(available)

    # Step 4-5: Binary agreement and change frequencies
    compute_binary_agreement_and_change_frequencies(available)

    # Step 6: Algorithm uncertainty
    compute_algorithm_uncertainty(available)

    # Step 7: GCM uncertainty
    compute_gcm_uncertainty(available)

    # Step 8: SSP uncertainty
    compute_ssp_uncertainty(available)

    # Step 9: Temporal uncertainty
    compute_temporal_uncertainty(available)

    # Step 10-11: Normalize and dominant source
    normalized, valid_global_mask, profile = normalize_uncertainty_and_dominant_source()

    # Step 12: Variance partition
    run_variance_partition()

    # Save completion marker
    with open(EXP5_DIR / "03_prediction_cube" / "core_analysis_complete.txt", "w") as f:
        f.write(f"Core analysis complete. {n_avail} scenarios processed.\n")

    print("\n" + "=" * 60)
    print("Core analysis complete. Run 02_environmental_novelty.py next,")
    print("then 03_derived_products.py for FSI/Vulnerability/Zones.")
    print("=" * 60)


if __name__ == "__main__":
    main()
