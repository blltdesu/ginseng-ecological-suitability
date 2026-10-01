"""Step 9: Process terrain data (SRTM-derived indices)."""
import os
import csv
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from datetime import datetime

RAW_TERRAIN = r"E:\人参种在哪\数据\地形数据\SRTM_topo_indices_EastAsia.tif"
OUT_DIR = r"E:\人参种在哪\00_统一数据预处理\07_terrain"
ALIGN_DIR = os.path.join(OUT_DIR, "aligned")
REF_META_PATH = r"E:\人参种在哪\00_统一数据预处理\09_reference_grid\reference_grid_metadata.json"
QC_DIR = r"E:\人参种在哪\00_统一数据预处理\11_qc"
LOG_PATH = r"E:\人参种在哪\00_统一数据预处理\logs\terrain.log"

os.makedirs(ALIGN_DIR, exist_ok=True)
os.makedirs(QC_DIR, exist_ok=True)


def main():
    print("=" * 50)
    print("TERRAIN PROCESSING")

    # Load reference metadata
    with open(REF_META_PATH) as f:
        ref_meta = json.load(f)
    print(f"Reference grid: {ref_meta['width']}x{ref_meta['height']}")

    # Open terrain file
    with rasterio.open(RAW_TERRAIN) as src:
        band_descs = [src.descriptions[i] for i in range(src.count)]
        n_bands = src.count
        src_crs = str(src.crs)
        src_transform = src.transform
        print(f"Bands: {band_descs}")
        print(f"CRS: {src_crs}, Shape: {src.shape}")

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

        qc_records = []

        for i in range(n_bands):
            band_name = band_descs[i] if i < len(band_descs) and band_descs[i] else f"band_{i+1}"
            var_out = band_name.lower()  # elevation, slope, northness, eastness

            print(f"\nProcessing band {i+1}: {band_name}")

            data = src.read(i + 1, masked=True).astype(np.float32)

            # Align to reference grid
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
            dst_masked = np.ma.masked_invalid(dst_data)

            # Save aligned GeoTIFF
            out_path = os.path.join(ALIGN_DIR, f"{var_out}.tif")
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
                dst.write(dst_masked.filled(-9999).astype(np.float32), 1)

            print(f"  Saved: {out_path}")

            # QC
            valid = dst_masked.compressed()
            if len(valid) > 0:
                qc_records.append({
                    "variable": var_out,
                    "min": round(float(valid.min()), 4),
                    "q01": round(float(np.percentile(valid, 1)), 4),
                    "median": round(float(np.ma.median(dst_masked)), 4),
                    "mean": round(float(dst_masked.mean()), 4),
                    "q99": round(float(np.percentile(valid, 99)), 4),
                    "max": round(float(valid.max()), 4),
                    "nodata_pct": round(float((1 - dst_masked.count() / dst_masked.size) * 100), 2),
                    "range_ok": "PASS" if valid.min() > -1000 and valid.max() < 10000 else "WARN",
                })
                print(f"  Range: [{valid.min():.2f}, {valid.max():.2f}]")

    # Save QC
    qc_path = os.path.join(QC_DIR, "terrain_qc.csv")
    with open(qc_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=qc_records[0].keys())
        writer.writeheader()
        writer.writerows(qc_records)
    print(f"\nQC saved: {qc_path}")

    # Log
    log_path = LOG_PATH
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] TERRAIN PROCESSING COMPLETED\n")
        for qc in qc_records:
            f.write(f"  {qc['variable']}: range [{qc['min']}, {qc['max']}]\n")

    print(f"\nTerrain processing complete. Output: {ALIGN_DIR}")


if __name__ == "__main__":
    main()
