# -*- coding: utf-8 -*-
"""
Prepare Fig3D material package (relative centroid shift).

Recomputes current and future suitable-range centroids INSIDE the East
Asia study window (95E-150E, 20N-55N), because the Experiment 4 global
centroids (e.g. 16.92E, 0.88N, Gulf of Guinea) are artifacts of the
global prediction domain and cannot enter the main-text East Asia maps.

Method (fixed for the whole paper):
  - binary centroid: pixel set = ensemble suitability >= 0.1285 (fixed
    TSS threshold), weighted by cos(latitude) (equal-area weighting),
    computed only over pixels inside the East Asia window.
  - a continuous suitability-weighted centroid (weight = suitability x
    cos(lat)) is also computed as a sensitivity column.
Distances: haversine (km); bearings: initial great-circle bearing (deg).

Outputs in E:\\人参种在哪\\绘图\\Fig3D_绘图素材包:
  Fig3D_centroid_by_scenario_ea.csv  (37 rows, GCM-level)
  Fig3D_centroid_migration_ea.csv    (8 rows, SSP x Period means)
  Fig3D_adm0.gpkg / Fig3D_adm1.gpkg  (boundaries for the map)
"""
import csv
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

sys.stdout.reconfigure(encoding="utf-8")

ENS_DIR = Path(r"E:\人参种在哪\实验4\08_future_predictions_ensemble")
CHANGE_DIR = Path(r"E:\人参种在哪\实验4\09_suitability_change")
BOUNDARY_SRC = Path(r"E:\人参种在哪\绘图\Fig2F_绘图素材包")
OUT = Path(r"E:\人参种在哪\绘图\Fig3D_绘图素材包")

GCMS = ["ACCESS-CM2", "BCC-CSM2-MR", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]
SSP_LABEL = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5",
             "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
THRESHOLD = 0.12846759691320192
EA_BOUNDS = (95.0, 20.0, 150.0, 55.0)  # left, bottom, right, top
NODATA = -3.4e38


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def cardinal(b):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((b + 11.25) / 22.5) % 16]


def weighted_centroid(lons_grid, lats_grid, weights):
    w = np.where(weights > 0, weights, 0.0)
    total = w.sum()
    if total <= 0:
        return None, None
    area_w = np.cos(np.radians(lats_grid))
    ww = w * area_w
    return float((ww * lons_grid).sum() / ww.sum()), float((ww * lats_grid).sum() / ww.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # --- current binary mask inside the EA window (from any change_class map)
    ref_change = next(CHANGE_DIR.glob("ACCESS-CM2/ssp126/2041-2060/change_class.tif"))
    with rasterio.open(ref_change) as s:
        win = from_bounds(*EA_BOUNDS, transform=s.transform).round_offsets().round_lengths()
        d = s.read(1, window=win)
        transform = s.window_transform(win)
    cur_binary = (d == 1) | (d == 2)
    h, w = cur_binary.shape
    rows, cols = np.mgrid[0:h, 0:w]
    lons = transform.c + (cols + 0.5) * transform.a
    lats = transform.f + (rows + 0.5) * transform.e
    print(f"EA window: {h}x{w}, current suitable px = {cur_binary.sum()}")

    cur_bin_lon, cur_bin_lat = weighted_centroid(lons, lats, cur_binary.astype(float))
    print(f"current binary centroid (EA): {cur_bin_lon:.4f}E, {cur_bin_lat:.4f}N")

    # --- per-scenario future centroids
    scen_rows = []
    for gcm in GCMS:
        for ssp in SSPS:
            for period in PERIODS:
                p = ENS_DIR / gcm / ssp / period / "ensemble_suitability.tif"
                if not p.exists():
                    continue
                with rasterio.open(p) as s:
                    win2 = from_bounds(*EA_BOUNDS, transform=s.transform).round_offsets().round_lengths()
                    fut = s.read(1, window=win2)
                if (win2.row_off, win2.col_off, win2.height, win2.width) != \
                   (win.row_off, win.col_off, win.height, win.width):
                    raise RuntimeError(f"window mismatch: {p}")
                fut = np.where(fut < -1e30, np.nan, fut)
                fut_binary = np.nan_to_num(fut, nan=0.0) >= THRESHOLD
                blon, blat = weighted_centroid(lons, lats, fut_binary.astype(float))
                clon, clat = weighted_centroid(lons, lats, np.nan_to_num(fut, nan=0.0))
                dist = haversine_km(cur_bin_lon, cur_bin_lat, blon, blat)
                bear = bearing_deg(cur_bin_lon, cur_bin_lat, blon, blat)
                scen_rows.append({
                    "gcm": gcm, "ssp": ssp, "ssp_label": SSP_LABEL[ssp], "period": period,
                    "current_lon": round(cur_bin_lon, 5), "current_lat": round(cur_bin_lat, 5),
                    "future_lon_binary": round(blon, 5), "future_lat_binary": round(blat, 5),
                    "future_lon_continuous": round(clon, 5), "future_lat_continuous": round(clat, 5),
                    "distance_km": round(dist, 2), "bearing_deg": round(bear, 1),
                    "direction": cardinal(bear),
                })

    with open(OUT / "Fig3D_centroid_by_scenario_ea.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(scen_rows[0].keys()))
        wr.writeheader()
        wr.writerows(scen_rows)

    # --- SSP x Period means of the binary centroid
    agg = []
    for ssp in SSPS:
        for period in PERIODS:
            sub = [r for r in scen_rows if r["ssp"] == ssp and r["period"] == period]
            if not sub:
                continue
            mlon = np.mean([r["future_lon_binary"] for r in sub])
            mlat = np.mean([r["future_lat_binary"] for r in sub])
            dists = [r["distance_km"] for r in sub]
            dist = haversine_km(cur_bin_lon, cur_bin_lat, mlon, mlat)
            bear = bearing_deg(cur_bin_lon, cur_bin_lat, mlon, mlat)
            agg.append({
                "ssp": ssp, "ssp_label": SSP_LABEL[ssp], "period": period,
                "n_gcms": len(sub),
                "current_lon": round(cur_bin_lon, 5), "current_lat": round(cur_bin_lat, 5),
                "future_lon_mean": round(mlon, 5), "future_lat_mean": round(mlat, 5),
                "distance_km": round(dist, 2), "bearing_deg": round(bear, 1),
                "direction": cardinal(bear),
                "gcm_distance_min": round(min(dists), 2), "gcm_distance_max": round(max(dists), 2),
            })

    with open(OUT / "Fig3D_centroid_migration_ea.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        wr.writeheader()
        wr.writerows(agg)

    for r in agg:
        print(r)

    for level in ["adm0", "adm1"]:
        shutil.copy2(BOUNDARY_SRC / f"Fig2F_{level}.gpkg", OUT / f"Fig3D_{level}.gpkg")
    print("boundaries copied; Fig3D package prepared.")


if __name__ == "__main__":
    main()
