# -*- coding: utf-8 -*-
"""
Prepare Fig3A material package.

Derives the Fig3A drawing materials from Experiment 4 outputs:
  - 8 GCM-mean future ensemble suitability rasters (4 SSP x 2 Period),
    cropped to the shared East Asia window (95E-150E, 20N-55N) for the
    main-text map, plus the full global rasters copied for supplementary use.
  - ADM0 / ADM1 boundary layers (already clipped to the map extent),
    copied from the Fig2F material package (identical to Fig1D).

Output folder: E:\\人参种在哪\\绘图\\Fig3A_绘图素材包
"""
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

sys.stdout.reconfigure(encoding="utf-8")

GCM_SUMMARY = Path(r"E:\人参种在哪\实验4\08_future_predictions_ensemble\GCM_summary")
BOUNDARY_SRC = Path(r"E:\人参种在哪\绘图\Fig2F_绘图素材包")
OUT = Path(r"E:\人参种在哪\绘图\Fig3A_绘图素材包")

SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]

# Shared East Asia window, identical to Fig1D / Fig2F
EA_BOUNDS = (95.0, 20.0, 150.0, 55.0)  # (left, bottom, right, top)


def crop_mean_rasters():
    (OUT / "global_supplement").mkdir(parents=True, exist_ok=True)
    # land mask from ADM0 polygons over the East Asia window; the GCM-mean
    # rasters encode ocean as 0.0, so ocean must be set to NaN explicitly
    adm0 = gpd.read_file(BOUNDARY_SRC / "Fig2F_adm0.gpkg")
    ref = GCM_SUMMARY / f"{SSPS[0]}_{PERIODS[0]}_mean.tif"
    with rasterio.open(ref) as s:
        win = from_bounds(*EA_BOUNDS, transform=s.transform).round_offsets().round_lengths()
        ea_transform = s.window_transform(win)
        ea_shape = (int(win.height), int(win.width))
    land = geometry_mask(
        adm0.geometry, out_shape=ea_shape, transform=ea_transform, invert=True
    )
    print(f"land pixels in window: {land.sum()} / {land.size}")

    for ssp in SSPS:
        for period in PERIODS:
            src = GCM_SUMMARY / f"{ssp}_{period}_mean.tif"
            dst = OUT / f"Fig3A_{ssp}_{period}_mean.tif"
            with rasterio.open(src) as s:
                win = from_bounds(*EA_BOUNDS, transform=s.transform)
                win = win.round_offsets().round_lengths()
                data = s.read(1, window=win)
                transform = s.window_transform(win)
                profile = s.profile.copy()
                profile.update(
                    height=data.shape[0],
                    width=data.shape[1],
                    transform=transform,
                    compress="deflate",
                    nodata=np.nan,
                )
                nd = s.nodata
                if nd is not None:
                    data = np.where(data == nd, np.nan, data)
                data = np.where(land, data, np.nan).astype("float32")
                with rasterio.open(dst, "w", **profile) as d:
                    d.write(data, 1)
            # full global copy for supplementary figure
            shutil.copy2(src, OUT / "global_supplement" / f"Fig3A_{ssp}_{period}_mean_global.tif")
            valid = np.isfinite(data)
            print(f"{dst.name}: {data.shape}, "
                  f"valid={valid.sum()}, mean={np.nanmean(data):.4f}, "
                  f"min={np.nanmin(data):.4f}, max={np.nanmax(data):.4f}")


def copy_boundaries():
    for level in ["adm0", "adm1"]:
        src = BOUNDARY_SRC / f"Fig2F_{level}.gpkg"
        dst = OUT / f"Fig3A_{level}.gpkg"
        shutil.copy2(src, dst)
        print(f"{dst.name} copied from Fig2F package")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    crop_mean_rasters()
    copy_boundaries()
    print("Fig3A material package prepared.")
