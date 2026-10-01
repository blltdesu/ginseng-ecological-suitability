# -*- coding: utf-8 -*-
"""Prepare Fig1D drawing material package.

Outputs to E:/人参种在哪/绘图/Fig1D_绘图素材包:
  - Fig1D_current_ensemble_suitability.tif  (East Asia crop, continuous 0-1)
  - Fig1D_intermodel_sd.tif                 (East Asia crop, inset raster)
  - Fig1D_occurrence_10km.csv               (lon/lat of 252 thinned records)
  - Fig1D_adm0.gpkg / Fig1D_adm1.gpkg       (boundaries clipped to East Asia extent)
  - Fig1D_threshold.json                    (ensemble binary threshold)
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(r"E:\人参种在哪\实验1")
OUT = Path(r"E:\人参种在哪\绘图\Fig1D_绘图素材包")
OUT.mkdir(parents=True, exist_ok=True)

# East Asia study extent (same as main-text map used in the Experiment 1 figures)
EXTENT = (95.0, 20.0, 150.0, 55.0)  # (west, south, east, north)


def crop_raster(src_path, dst_path, extent):
    west, south, east, north = extent
    with rasterio.open(src_path) as src:
        win = from_bounds(west, south, east, north, src.transform)
        win = win.round_offsets().round_lengths()
        data = src.read(1, window=win)
        transform = src.window_transform(win)
        profile = src.profile.copy()
        profile.update(
            height=data.shape[0], width=data.shape[1],
            transform=transform, compress="lzw",
        )
        nodata = src.nodata
        if nodata is not None:
            data = np.where(np.isfinite(data), data, nodata).astype(profile["dtype"])
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data, 1)
    vals = data[np.isfinite(data)] if nodata is None else data[(data != nodata) & np.isfinite(data)]
    print(f"  {dst_path.name}: {data.shape[1]}x{data.shape[0]} px, "
          f"range [{vals.min():.4f}, {vals.max():.4f}], nodata={nodata}")


print("=== Fig1D material preparation ===")

# 1. Main ensemble suitability raster (crop to East Asia)
print("Cropping ensemble suitability raster...")
crop_raster(
    ROOT / "09_current_prediction/current_ensemble_suitability.tif",
    OUT / "Fig1D_current_ensemble_suitability.tif",
    EXTENT,
)

# 2. Inter-model SD raster (inset)
print("Cropping inter-model SD raster...")
crop_raster(
    ROOT / "10_uncertainty/current_intermodel_sd.tif",
    OUT / "Fig1D_intermodel_sd.tif",
    EXTENT,
)

# 3. Occurrence points (10 km thinned) -> simple lon/lat CSV
print("Preparing occurrence points...")
occ = pd.read_csv(ROOT / "01_input/occurrence_thin_10km.csv")
occ_out = occ[["decimalLongitude", "decimalLatitude"]].rename(
    columns={"decimalLongitude": "longitude", "decimalLatitude": "latitude"}
)
occ_out.to_csv(OUT / "Fig1D_occurrence_10km.csv", index=False)
in_extent = occ_out[
    (occ_out.longitude >= EXTENT[0]) & (occ_out.longitude <= EXTENT[2])
    & (occ_out.latitude >= EXTENT[1]) & (occ_out.latitude <= EXTENT[3])
]
print(f"  Fig1D_occurrence_10km.csv: {len(occ_out)} records "
      f"({len(in_extent)} inside East Asia extent)")

# 4. Boundaries clipped to East Asia extent
from shapely.geometry import box
bbox = box(*EXTENT)
for level in ("adm0", "adm1"):
    print(f"Clipping {level} boundaries...")
    gdf = gpd.read_file(ROOT / f"01_input/study_context_{level}.gpkg", bbox=bbox)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    keep = [c for c in gdf.columns if c == "geometry" or gdf[c].dtype == object][:6]
    if "geometry" not in keep:
        keep.append("geometry")
    gdf = gdf[keep]
    gdf.to_file(OUT / f"Fig1D_{level}.gpkg", driver="GPKG")
    print(f"  Fig1D_{level}.gpkg: {len(gdf)} features")

# 5. Ensemble threshold
with open(ROOT / "09_current_prediction/ensemble_threshold.json") as f:
    thr = json.load(f)
with open(OUT / "Fig1D_threshold.json", "w") as f:
    json.dump({"ensemble_threshold": thr["ensemble_threshold"]}, f, indent=2)
print(f"  Fig1D_threshold.json: threshold = {thr['ensemble_threshold']:.5f}")

print("=== Done ===")
