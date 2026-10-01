# -*- coding: utf-8 -*-
"""
Prepare Fig3B material package (Stable-Loss-Gain change classification).

Rebuilds correct across-GCM majority-vote change maps from the 37
per-scenario change_class.tif rasters of Experiment 4 (the GCM_summary
*_majority_change.tif files produced by 05b_gcm_summary_maps.py are
defective: a bitwise-NOT on a uint8 array left almost all pixels at the
255 nodata value, and no Stable/Loss/Gain classes survived).

Classes (uint8): 0 = persistent unsuitable, 1 = Stable, 2 = Loss,
3 = Gain, 255 = NoData (ocean / outside modeling domain).

Also crops the Experiment 5 gain-frequency raster to the East Asia
window and computes per-combo Stable/Loss/Gain proportions (within the
window, relative to current suitable pixels) for map annotation.

Output folder: E:\\人参种在哪\\绘图\\Fig3B_绘图素材包
"""
import csv
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

sys.stdout.reconfigure(encoding="utf-8")

CHANGE_DIR = Path(r"E:\人参种在哪\实验4\09_suitability_change")
GAIN_FREQ = Path(r"E:\人参种在哪\实验5\08_stability_probability\gain_frequency.tif")
BOUNDARY_SRC = Path(r"E:\人参种在哪\绘图\Fig2F_绘图素材包")
OUT = Path(r"E:\人参种在哪\绘图\Fig3B_绘图素材包")

GCMS = ["ACCESS-CM2", "BCC-CSM2-MR", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]

# Shared East Asia window, identical to Fig1D / Fig2F / Fig3A
EA_BOUNDS = (95.0, 20.0, 150.0, 55.0)  # (left, bottom, right, top)


def rebuild_majority_change():
    rows_out = []
    for ssp in SSPS:
        for period in PERIODS:
            paths = [CHANGE_DIR / g / ssp / period / "change_class.tif" for g in GCMS]
            paths = [p for p in paths if p.exists()]
            cur_stack, fut_stack, valid_stack = [], [], []
            for p in paths:
                with rasterio.open(p) as s:
                    d = s.read(1)
                    ref_profile = s.profile
                    valid = d != 255
                    cur = valid & ((d == 1) | (d == 2))   # currently suitable
                    fut = valid & ((d == 1) | (d == 3))   # future suitable
                    cur_stack.append(cur)
                    fut_stack.append(fut)
                    valid_stack.append(valid)
            cur_stack = np.stack(cur_stack)
            fut_stack = np.stack(fut_stack)
            any_valid = np.stack(valid_stack).any(axis=0)
            # current suitability is fixed (same threshold) — verify consistency
            if not np.all(cur_stack == cur_stack[0]):
                print(f"WARNING: current mask differs across GCMs for {ssp} {period}")
            current = cur_stack[0]
            n = len(paths)
            fut_frac = fut_stack.sum(axis=0) / n
            fut_majority = fut_frac >= 0.5

            change = np.full(current.shape, 255, dtype=np.uint8)
            change[current & fut_majority] = 1
            change[current & ~fut_majority] = 2
            change[~current & fut_majority] = 3
            change[~current & ~fut_majority] = 0
            change[~any_valid] = 255  # ocean / outside modeling domain

            # crop to East Asia window
            win = from_bounds(*EA_BOUNDS, transform=ref_profile["transform"])
            win = win.round_offsets().round_lengths()
            r0, c0 = int(win.row_off), int(win.col_off)
            r1, c1 = r0 + int(win.height), c0 + int(win.width)
            crop = change[r0:r1, c0:c1]
            cur_crop = current[r0:r1, c0:c1]
            profile = ref_profile.copy()
            profile.pop("shape", None)
            profile.update(
                height=crop.shape[0], width=crop.shape[1],
                transform=ref_profile["transform"] * rasterio.Affine.translation(c0, r0),
                dtype="uint8", compress="deflate", nodata=255,
            )
            dst = OUT / f"Fig3B_{ssp}_{period}_majority_change.tif"
            with rasterio.open(dst, "w", **profile) as d:
                d.write(crop, 1)

            # proportions within the EA window
            n_stable = int((crop == 1).sum())
            n_loss = int((crop == 2).sum())
            n_gain = int((crop == 3).sum())
            n_cur = int(cur_crop.sum())
            n_pers = int((crop == 0).sum())
            rows_out.append({
                "ssp": ssp, "period": period, "n_gcms": n,
                "current_suitable_px": n_cur,
                "stable_px": n_stable, "loss_px": n_loss, "gain_px": n_gain,
                "persistent_unsuitable_px": n_pers,
                "stable_pct_of_current": round(100 * n_stable / n_cur, 2),
                "loss_pct_of_current": round(100 * n_loss / n_cur, 2),
                "gain_pct_of_current": round(100 * n_gain / n_cur, 4),
            })
            print(f"{dst.name}: GCMs={n} stable={n_stable} loss={n_loss} gain={n_gain}")

    with open(OUT / "Fig3B_change_proportions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)


def crop_gain_frequency():
    with rasterio.open(GAIN_FREQ) as s:
        win = from_bounds(*EA_BOUNDS, transform=s.transform)
        win = win.round_offsets().round_lengths()
        d = s.read(1, window=win)
        profile = s.profile.copy()
        profile.update(
            height=d.shape[0], width=d.shape[1],
            transform=s.window_transform(win), compress="deflate", nodata=np.nan,
        )
        with rasterio.open(OUT / "Fig3B_gain_frequency.tif", "w", **profile) as o:
            o.write(d, 1)
    nz = d[np.isfinite(d) & (d > 0)]
    print(f"Fig3B_gain_frequency.tif: nonzero px={nz.size}, max={nz.max() if nz.size else 0:.4f}")


def copy_boundaries():
    for level in ["adm0", "adm1"]:
        shutil.copy2(BOUNDARY_SRC / f"Fig2F_{level}.gpkg", OUT / f"Fig3B_{level}.gpkg")
    print("boundaries copied")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rebuild_majority_change()
    crop_gain_frequency()
    copy_boundaries()
    print("Fig3B material package prepared.")
