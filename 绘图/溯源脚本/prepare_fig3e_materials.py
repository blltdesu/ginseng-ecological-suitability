# -*- coding: utf-8 -*-
"""
Prepare Fig3E material package (landscape structure change).

Builds tidy tables from the Experiment 4 landscape metrics
(13_landscape_structure/landscape_metrics_by_scenario.csv, 36 GCM-level
scenarios, domain = 100-150E / 25-55N crop):
  - Fig3E_landscape_by_gcm.csv   : GCM-level values for the 3 key metrics
  - Fig3E_landscape_summary.csv  : SSP x Period means / SD / min / max
  - Fig3E_landscape_current_baseline.csv : current-landscape reference,
    recomputed with the IDENTICAL method (same crop, 8-neighbor labeling,
    40N reference pixel area, patches < 3 px skipped) so the "not
    fragmentation" comparison has a like-for-like baseline.

Output folder: E:\\人参种在哪\\绘图\\Fig3E_绘图素材包
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(r"E:\人参种在哪\实验4\13_landscape_structure\landscape_metrics_by_scenario.csv")
CHANGE = Path(r"E:\人参种在哪\实验4\09_suitability_change\ACCESS-CM2\ssp126\2041-2060\change_class.tif")
OUT = Path(r"E:\人参种在哪\绘图\Fig3E_绘图素材包")

SSP_LABEL = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5",
             "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
METRICS = ["patch_number", "mean_patch_area_km2", "largest_patch_area_km2"]


def pixel_area_km2(lat, res_deg=0.041666666666666664):
    R = 6371.0
    lat_rad = math.radians(lat)
    dx = res_deg * (math.pi / 180) * R * math.cos(lat_rad)
    dy = res_deg * (math.pi / 180) * R
    return dx * dy


def current_baseline():
    """Replicate the Experiment 4 landscape method on the current binary mask."""
    with rasterio.open(CHANGE) as s:
        d = s.read(1)
        transform = s.transform
    cur = ((d == 1) | (d == 2)).astype(np.uint8)
    col_min = int((100 - transform[2]) / transform[0])
    col_max = int((150 - transform[2]) / transform[0])
    row_min = int((90 - 55) / -transform[4]) if transform[4] < 0 else None
    # rows: latitude 55 -> 25 (north-up grid)
    row_min = int((transform[3] - 55) / (-transform[4]))
    row_max = int((transform[3] - 25) / (-transform[4]))
    crop = cur[row_min:row_max, col_min:col_max]
    labels, n = ndimage.label(crop, structure=np.ones((3, 3)))
    areas = []
    for p in range(1, min(n + 1, 1000)):
        npx = (labels == p).sum()
        if npx < 3:
            continue
        areas.append(npx * pixel_area_km2(40))
    edges = int((np.abs(ndimage.sobel(crop.astype(float))) > 0).sum())
    total_area = crop.sum() * pixel_area_km2(40)
    return {
        "domain": "100-150E, 25-55N",
        "patch_number": n,
        "mean_patch_area_km2": round(float(np.mean(areas)), 2),
        "largest_patch_area_km2": round(float(np.max(areas)), 2),
        "largest_patch_index_pct": round(float(np.max(areas)) / total_area * 100, 2),
        "suitable_area_km2_domain": round(float(total_area), 2),
        "edge_pixels": edges,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with open(SRC, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    gcm_rows = []
    for r in rows:
        gcm_rows.append({
            "gcm": r["gcm"], "ssp": r["ssp"], "ssp_label": SSP_LABEL[r["ssp"]],
            "period": r["period"],
            "patch_number": int(r["patch_number"]),
            "mean_patch_area_km2": float(r["mean_patch_area_km2"]),
            "largest_patch_area_km2": float(r["largest_patch_area_km2"]),
            "largest_patch_index_pct": float(r["largest_patch_index_pct"]),
        })
    with open(OUT / "Fig3E_landscape_by_gcm.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gcm_rows[0].keys()))
        w.writeheader()
        w.writerows(gcm_rows)

    summary = []
    groups = {}
    for r in gcm_rows:
        groups.setdefault((r["ssp"], r["period"]), []).append(r)
    for (ssp, period), sub in sorted(groups.items()):
        row = {"ssp": ssp, "ssp_label": SSP_LABEL[ssp], "period": period,
               "n_gcms": len(sub)}
        for m in METRICS:
            vals = [s[m] for s in sub]
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
            row[f"{m}_mean"] = round(mean, 2)
            row[f"{m}_sd"] = round(sd, 2)
            row[f"{m}_min"] = round(min(vals), 2)
            row[f"{m}_max"] = round(max(vals), 2)
        summary.append(row)
    with open(OUT / "Fig3E_landscape_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    base = current_baseline()
    with open(OUT / "Fig3E_landscape_current_baseline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(base.keys()))
        w.writeheader()
        w.writerow(base)

    print("current baseline:", base)
    for s in summary:
        print(s["ssp"], s["period"], "n=", s["n_gcms"],
              "patches=", s["patch_number_mean"],
              "mean_area=", s["mean_patch_area_km2_mean"],
              "largest=", s["largest_patch_area_km2_mean"])


if __name__ == "__main__":
    main()
