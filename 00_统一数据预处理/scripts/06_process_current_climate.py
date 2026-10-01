"""Step 6: Process current climate data (WorldClim BIO1-BIO19)."""
import os
import json
import shutil
import numpy as np
import rasterio
from rasterio.crs import CRS
from datetime import datetime

RAW_CLIMATE = r"E:\人参种在哪\数据\气候数据\wc2.1_2.5m_bio"
OUT_DIR = r"E:\人参种在哪\00_统一数据预处理\04_current_climate"
REF_DIR = r"E:\人参种在哪\00_统一数据预处理\09_reference_grid"
LOG_PATH = r"E:\人参种在哪\00_统一数据预处理\logs\climate_current.log"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

log_lines = []
log_lines.append(f"[{datetime.now().isoformat()}] CURRENT CLIMATE PROCESSING STARTED\n")


def log(msg):
    print(msg)
    log_lines.append(f"[{datetime.now().isoformat()}] {msg}\n")


def main():
    log("=" * 50)
    log("CURRENT CLIMATE PROCESSING")

    # Find all BIO TIFs
    tif_files = sorted([f for f in os.listdir(RAW_CLIMATE) if f.endswith(".tif")])
    log(f"Found {len(tif_files)} BIO TIFF files")

    # Map to bio names
    import re
    bio_map = {}
    for f in tif_files:
        m = re.search(r"bio_(\d+)", f)
        if m:
            num = int(m.group(1))
            bio_map[f"bio{num:02d}"] = f

    log(f"Mapped {len(bio_map)} BIO variables")

    # Read first file for reference metadata
    first_f = tif_files[0]
    first_path = os.path.join(RAW_CLIMATE, first_f)
    with rasterio.open(first_path) as src:
        ref_crs = str(src.crs)
        ref_shape = src.shape
        ref_transform = src.transform
        ref_bounds = src.bounds
        ref_resolution = src.res
        ref_nodata = src.nodata

    log(f"Reference metadata from {first_f}:")
    log(f"  CRS: {ref_crs}")
    log(f"  Shape: {ref_shape}")
    log(f"  Resolution: {ref_resolution}")
    log(f"  Bounds: {ref_bounds}")
    log(f"  Nodata: {ref_nodata}")

    # This IS the master reference grid
    # Save reference grid template
    ref_meta = {
        "crs": ref_crs,
        "width": ref_shape[1],
        "height": ref_shape[0],
        "resolution": [float(ref_resolution[0]), float(ref_resolution[1])],
        "bounds": {
            "left": float(ref_bounds.left),
            "bottom": float(ref_bounds.bottom),
            "right": float(ref_bounds.right),
            "top": float(ref_bounds.top),
        },
        "transform": list(ref_transform)[:6],
        "nodata": float(ref_nodata) if ref_nodata is not None else -9999,
    }
    with open(os.path.join(REF_DIR, "reference_grid_metadata.json"), "w") as f:
        json.dump(ref_meta, f, indent=2)

    # Create template raster
    kwargs = {
        "driver": "GTiff",
        "height": ref_shape[0],
        "width": ref_shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": ref_crs,
        "transform": ref_transform,
        "nodata": 255,
    }
    with rasterio.open(os.path.join(REF_DIR, "reference_grid_template.tif"), "w", **kwargs) as dst:
        dst.write(np.ones(ref_shape, dtype="uint8"), 1)

    log("Reference grid template saved")

    # Copy/rename BIO files to working directory
    for bio_name, src_file in sorted(bio_map.items()):
        src_path = os.path.join(RAW_CLIMATE, src_file)
        dst_path = os.path.join(OUT_DIR, f"{bio_name}_aligned.tif")
        shutil.copy2(src_path, dst_path)
        log(f"  {bio_name}: copied (IS reference grid)")

    # QC: Check all 19 variables
    log("\n--- QC Check ---")
    qc_records = []
    for bio_name in sorted(bio_map.keys()):
        tif_path = os.path.join(OUT_DIR, f"{bio_name}_aligned.tif")
        if not os.path.exists(tif_path):
            log(f"  WARNING: {bio_name} missing!")
            continue
        with rasterio.open(tif_path) as src:
            data = src.read(1, masked=True)
            qc_records.append({
                "variable": bio_name,
                "min": float(data.min()),
                "q01": float(np.percentile(data.compressed(), 1)),
                "median": float(np.ma.median(data)),
                "mean": float(data.mean()),
                "q99": float(np.percentile(data.compressed(), 99)),
                "max": float(data.max()),
                "nodata_pct": round(float((1 - data.count() / data.size) * 100), 2),
                "crs": str(src.crs),
                "resolution": str(src.res),
                "rows": src.shape[0],
                "cols": src.shape[1],
            })

    # Save QC
    import csv
    qc_path = r"E:\人参种在哪\00_统一数据预处理\11_qc\current_climate_qc.csv"
    os.makedirs(os.path.dirname(qc_path), exist_ok=True)
    if qc_records:
        with open(qc_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=qc_records[0].keys())
            writer.writeheader()
            writer.writerows(qc_records)
        log(f"QC saved to: {qc_path}")

    # Log
    log_lines.append(f"[{datetime.now().isoformat()}] CURRENT CLIMATE PROCESSING COMPLETED\n")
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    print(f"\nCurrent climate processing complete. Output: {OUT_DIR}")
    print(f"Reference grid: {REF_DIR}")


if __name__ == "__main__":
    main()
