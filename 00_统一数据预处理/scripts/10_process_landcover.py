"""Step 10: Process land cover data (MODIS MCD12Q1)."""
import os
import csv
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from datetime import datetime

RAW_LC = r"E:\人参种在哪\数据\土地覆盖\MODIS_MCD12Q1_LC_Type1_mode_2019_2024_EastAsia.tif"
OUT_DIR = r"E:\人参种在哪\00_统一数据预处理\08_landcover"
OUT_PATH = os.path.join(OUT_DIR, "landcover_aligned.tif")
DICT_PATH = os.path.join(OUT_DIR, "landcover_class_dictionary.csv")
REF_META_PATH = r"E:\人参种在哪\00_统一数据预处理\09_reference_grid\reference_grid_metadata.json"
LOG_PATH = r"E:\人参种在哪\00_统一数据预处理\logs\landcover.log"

os.makedirs(OUT_DIR, exist_ok=True)

# MODIS MCD12Q1 IGBP classification
IGBP_CLASSES = {
    0: "Water",
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forest",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up",
    14: "Cropland/Natural Vegetation Mosaic",
    15: "Snow and Ice",
    16: "Barren or Sparsely Vegetated",
    17: "Water Bodies",  # Alternative
    255: "Unclassified",
}


def main():
    print("=" * 50)
    print("LANDCOVER PROCESSING")

    # Load reference metadata
    with open(REF_META_PATH) as f:
        ref_meta = json.load(f)

    with rasterio.open(RAW_LC) as src:
        data = src.read(1, masked=True)
        src_crs = str(src.crs)
        src_transform = src.transform
        print(f"Source: shape={src.shape}, CRS={src_crs}, dtype={src.dtypes[0]}")

        # Get unique classes
        unique_classes = np.unique(data.compressed())
        print(f"Classes found: {sorted(unique_classes)}")

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

        # Align - MUST use nearest neighbor for categorical data
        dst_data = np.full((ref_height, ref_width), 255, dtype=np.uint8)
        reproject(
            source=data.filled(255).astype(np.uint8),
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,  # NEAREST for categorical!
        )

        # Save aligned
        with rasterio.open(
            OUT_PATH, "w",
            driver="GTiff",
            height=ref_height,
            width=ref_width,
            count=1,
            dtype="uint8",
            crs=ref_crs,
            transform=ref_transform,
            nodata=255,
            compress="lzw",
        ) as dst:
            dst.write(dst_data, 1)

        print(f"Saved: {OUT_PATH}")

    # Write class dictionary
    with open(DICT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_code", "class_name", "source_scheme"])
        for code, name in sorted(IGBP_CLASSES.items()):
            writer.writerow([code, name, "MCD12Q1 IGBP (LC_Type1)"])

    print(f"Class dictionary saved: {DICT_PATH}")

    # Log
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] LANDCOVER PROCESSING COMPLETED\n")
        f.write(f"  Resampling: nearest neighbor (categorical)\n")
        f.write(f"  Classes: {sorted(unique_classes)}\n")

    print(f"\nLandcover processing complete. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
