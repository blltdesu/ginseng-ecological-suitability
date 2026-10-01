"""Step 7: Process future climate data (WorldClim CMIP6 downscaled).
Each file is a multi-band GeoTIFF containing 19 BIO variables.
Tasks: inspect structure, split bands, align to reference grid, build registry."""
import os
import csv
import json
import re
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from datetime import datetime

FUTURE_SRC = os.path.join(os.path.expanduser("~"), "ginseng_sdm", "04_climate_future")
OUT_DIR = r"E:\人参种在哪\00_统一数据预处理\05_future_climate"
ALIGN_DIR = os.path.join(OUT_DIR, "aligned")
REF_META_PATH = r"E:\人参种在哪\00_统一数据预处理\09_reference_grid\reference_grid_metadata.json"
LOG_PATH = r"E:\人参种在哪\00_统一数据预处理\logs\climate_future.log"
QC_DIR = r"E:\人参种在哪\00_统一数据预处理\11_qc"

os.makedirs(ALIGN_DIR, exist_ok=True)
os.makedirs(QC_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

log_lines = []
log_lines.append(f"[{datetime.now().isoformat()}] FUTURE CLIMATE PROCESSING STARTED\n")


def log(msg):
    print(msg)
    log_lines.append(f"[{datetime.now().isoformat()}] {msg}\n")


def discover_files():
    """Discover all future climate TIFF files."""
    files = []
    for root, dirs, filenames in os.walk(FUTURE_SRC):
        for f in filenames:
            if f.endswith(".tif") and "bioc" in f.lower():
                files.append(os.path.join(root, f))
    files.sort()
    log(f"Found {len(files)} future climate TIFF files")
    return files


def parse_filename(fpath):
    """Parse GCM, SSP, and period from filename.
    e.g., wc2.1_2.5m_bioc_ACCESS-CM2_ssp126_2041-2060.tif
    """
    fname = os.path.basename(fpath)
    m = re.match(r"wc2\.1_2\.5m_bioc_([A-Za-z0-9\-]+)_(ssp\d+)_(\d{4}-\d{4})\.tif", fname)
    if m:
        return {
            "gcm": m.group(1),
            "ssp": m.group(2),
            "period": m.group(3),
        }
    return None


def inspect_file(fpath):
    """Inspect a future climate file to understand band structure."""
    with rasterio.open(fpath) as src:
        n_bands = src.count
        crs = str(src.crs)
        shape = src.shape
        resolution = src.res
        band_descs = [src.descriptions[i] for i in range(n_bands)]
        return {
            "n_bands": n_bands,
            "crs": crs,
            "shape": shape,
            "resolution": resolution,
            "band_descs": band_descs,
        }


def process_future_file(fpath, info, ref_meta):
    """Split multi-band future climate file into individual BIO rasters,
    align to reference grid, and save."""
    parsed = parse_filename(fpath)
    if not parsed:
        log(f"  ERROR: Cannot parse filename: {fpath}")
        return None

    gcm = parsed["gcm"]
    ssp = parsed["ssp"]
    period = parsed["period"]

    gcm_ssp_dir = os.path.join(ALIGN_DIR, gcm, ssp, period)
    os.makedirs(gcm_ssp_dir, exist_ok=True)

    with rasterio.open(fpath) as src:
        n_bands = src.count
        band_descs = src.descriptions
        src_crs = str(src.crs)
        src_transform = src.transform

        # Detect BIO mapping from band descriptions
        bio_indices = {}  # bio01 -> band_index (0-based)
        for i in range(n_bands):
            desc = band_descs[i] if i < len(band_descs) else ""
            if not desc:
                # Try positional mapping: band 1 = bio01, etc.
                band_num = i + 1
                bio_indices[f"bio{band_num:02d}"] = i
            else:
                # e.g., "wc2.1_2.5m_bioc_ACCESS-CM2_ssp126_2041-2060_1" -> band 1
                # Also handle "bio01", "BIO_1", "wc2.1_2.5m_bio_1", etc.
                m = re.search(r"(?:bio|BIO)[_\s]*(\d+)", desc)
                if m:
                    bio_num = int(m.group(1))
                else:
                    # Try extracting the trailing number (e.g., "..._1", "..._19")
                    m2 = re.search(r"_(\d+)$", desc)
                    if m2:
                        bio_num = int(m2.group(1))
                    else:
                        # Fallback: positional (band 1 = bio01, etc.)
                        bio_num = i + 1
                if 1 <= bio_num <= 19:
                    bio_indices[f"bio{bio_num:02d}"] = i

        log(f"  {gcm}/{ssp}/{period}: {n_bands} bands, "
            f"mapped {len(bio_indices)} BIO variables")

        # Reference transform
        ref_transform = rasterio.transform.from_origin(
            ref_meta["bounds"]["left"],
            ref_meta["bounds"]["top"],
            ref_meta["resolution"][0],
            abs(ref_meta["resolution"][1]),
        )
        ref_width = ref_meta["width"]
        ref_height = ref_meta["height"]
        ref_crs = ref_meta["crs"]

        saved_files = []
        for bio_name in [f"bio{b:02d}" for b in range(1, 20)]:
            if bio_name not in bio_indices:
                continue

            band_idx = bio_indices[bio_name]
            data = src.read(band_idx + 1, masked=True).astype(np.float32)

            # Align to reference
            dst_data = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
            reproject(
                source=data.filled(np.nan),
                destination=dst_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )

            out_path = os.path.join(gcm_ssp_dir, f"{bio_name}.tif")
            with rasterio.open(
                out_path, "w",
                driver="GTiff",
                height=ref_height,
                width=ref_width,
                count=1,
                dtype="float32",
                crs=ref_crs,
                transform=ref_transform,
                nodata=-9999,
                compress="lzw",
            ) as dst:
                dst.write(np.nan_to_num(dst_data, nan=-9999).astype(np.float32), 1)

            saved_files.append(bio_name)

        return {
            "gcm": gcm,
            "ssp": ssp,
            "period": period,
            "n_bands": n_bands,
            "n_bio_mapped": len(saved_files),
            "saved": saved_files,
        }


def build_registry(results):
    """Build the future climate registry CSV."""
    reg_path = os.path.join(OUT_DIR, "future_climate_registry.csv")

    records = []
    for r in results:
        if r is None:
            continue
        records.append({
            "gcm": r["gcm"],
            "ssp": r["ssp"],
            "period": r["period"],
            "source_file": f"wc2.1_2.5m_bioc_{r['gcm']}_{r['ssp']}_{r['period']}.tif",
            "bands": r["n_bands"],
            "recognized_bioclim_count": r["n_bio_mapped"],
            "aligned_dir": os.path.join(ALIGN_DIR, r["gcm"], r["ssp"], r["period"]),
            "status": "aligned" if r["n_bio_mapped"] == 19 else "warning",
        })

    with open(reg_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    log(f"\nFuture climate registry: {reg_path} ({len(records)} entries)")
    return records


def build_completeness_matrix(results):
    """Build future climate completeness matrix (GCM×SSP×period × BIO)."""
    # Build matrix
    gcms = sorted(set(r["gcm"] for r in results if r))
    ssps = sorted(set(r["ssp"] for r in results if r))
    periods = sorted(set(r["period"] for r in results if r))
    bio_vars = [f"bio{b:02d}" for b in range(1, 20)]

    matrix_records = []
    for r in results:
        if r is None:
            continue
        for bio in bio_vars:
            matrix_records.append({
                "gcm": r["gcm"],
                "ssp": r["ssp"],
                "period": r["period"],
                "variable": bio,
                "available": 1 if bio in r["saved"] else 0,
            })

    csv_path = os.path.join(QC_DIR, "future_climate_completeness_matrix.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["gcm", "ssp", "period", "variable", "available"])
        writer.writeheader()
        writer.writerows(matrix_records)

    # Summary
    available_count = sum(1 for m in matrix_records if m["available"] == 1)
    total_count = len(matrix_records)
    log(f"Completeness matrix: {available_count}/{total_count} BIO variables available")

    # Quick summary per combination
    from collections import defaultdict
    combo_counts = defaultdict(int)
    for m in matrix_records:
        key = f"{m['gcm']}/{m['ssp']}/{m['period']}"
        combo_counts[key] += m["available"]

    log("BIO count per GCM/SSP/period:")
    for key, count in sorted(combo_counts.items()):
        log(f"  {key}: {count}/19")

    return csv_path


def main():
    log("=" * 60)
    log("FUTURE CLIMATE PROCESSING")
    log("=" * 60)

    # Load reference metadata
    with open(REF_META_PATH) as f:
        ref_meta = json.load(f)
    log(f"Reference grid: {ref_meta['width']}x{ref_meta['height']}, CRS={ref_meta['crs']}")

    # Discover files
    files = discover_files()

    if not files:
        log("WARNING: No future climate files found!")
        return

    # Inspect first file
    first_info = inspect_file(files[0])
    log(f"First file inspection: {first_info['n_bands']} bands, "
        f"CRS={first_info['crs']}, shape={first_info['shape']}")
    log(f"Band descriptions (first 5): {first_info['band_descs'][:5]}")

    # Process all files
    results = []
    for fpath in files:
        parsed = parse_filename(fpath)
        if parsed:
            log(f"\nProcessing: {parsed['gcm']} / {parsed['ssp']} / {parsed['period']}")
        result = process_future_file(fpath, first_info, ref_meta)
        if result:
            results.append(result)

    # Build registry
    registry = build_registry(results)

    # Build completeness matrix
    matrix_path = build_completeness_matrix(results)

    # Build Experiment 4 handoff
    import shutil
    to_exp4 = os.path.join(r"E:\人参种在哪\00_统一数据预处理\14_handoff\to_experiment_4")
    os.makedirs(to_exp4, exist_ok=True)

    # Copy registry
    reg_src = os.path.join(OUT_DIR, "future_climate_registry.csv")
    if os.path.exists(reg_src):
        shutil.copy2(reg_src, os.path.join(to_exp4, "future_climate_registry.csv"))

    # Copy completeness matrix
    if os.path.exists(matrix_path):
        shutil.copy2(matrix_path, os.path.join(to_exp4, "future_climate_completeness_matrix.csv"))

    # Copy aligned future climate (symlink-like, just copy the directory)
    aligned_dst = os.path.join(to_exp4, "aligned_future_climate")
    if os.path.exists(aligned_dst):
        shutil.rmtree(aligned_dst)
    shutil.copytree(ALIGN_DIR, aligned_dst)
    log(f"\nExperiment 4 handoff: {to_exp4}")

    # Save P4 figure data
    import pandas as pd
    fig_data_dir = r"E:\人参种在哪\00_统一数据预处理\13_figure_data"
    matrix_df = pd.DataFrame([
        {"gcm": m["gcm"], "ssp": m["ssp"], "period": m["period"],
         "variable": m["variable"], "available": m["available"]}
        for m in ([] if not results else [
            {"gcm": r["gcm"], "ssp": r["ssp"], "period": r["period"],
             "variable": bio, "available": 1 if bio in r.get("saved", []) else 0}
            for r in results if r for bio in [f"bio{b:02d}" for b in range(1, 20)]
        ])
    ])
    if len(matrix_df) > 0:
        matrix_df.to_csv(os.path.join(fig_data_dir, "P4_future_climate_completeness.csv"),
                         index=False, encoding="utf-8-sig")

    # Log
    log_lines.append(f"[{datetime.now().isoformat()}] FUTURE CLIMATE PROCESSING COMPLETED\n")
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    # Summary
    total_bios = sum(r["n_bio_mapped"] for r in results if r)
    log(f"\n{'='*50}")
    log(f"FUTURE CLIMATE PROCESSING COMPLETE")
    log(f"Files processed: {len(files)}")
    log(f"Total BIO variables aligned: {total_bios}")
    log(f"Expected: {len(files) * 19} = {total_bios == len(files) * 19}")
    log(f"Output: {ALIGN_DIR}")


if __name__ == "__main__":
    main()
