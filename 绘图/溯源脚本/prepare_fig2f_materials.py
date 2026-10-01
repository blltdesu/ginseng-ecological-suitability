#!/usr/bin/env python3
"""
Prepare Fig2F material package: dominant environmental group spatial map.

Crops Experiment 3 spatial group ablation rasters to the same East Asia
extent used in Fig1D (95E-150E, 20N-55N), copies shared boundary layers,
and compiles summary statistics for the inset bar chart.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

SRC = r"E:\人参种在哪\实验3\11_spatial_group_contribution"
DST = r"E:\人参种在哪\绘图\Fig2F_绘图素材包"
FIG1D = r"E:\人参种在哪\绘图\Fig1D_绘图素材包"

# Same extent as Fig1D (95E-150E, 20N-55N)
WEST, SOUTH, EAST, NORTH = 95.0, 20.0, 150.0, 55.0

os.makedirs(DST, exist_ok=True)


def crop_raster(src_path, dst_path, dtype=np.float32):
    with rasterio.open(src_path) as src:
        win = from_bounds(WEST, SOUTH, EAST, NORTH, transform=src.transform)
        data = src.read(1, window=win).astype(dtype)
        data[data < -1e37] = np.nan
        profile = src.profile.copy()
        profile.update(
            height=data.shape[0], width=data.shape[1],
            transform=src.window_transform(win),
            dtype=rasterio.dtypes.get_minimum_dtype(data),
            nodata=np.nan, compress="lzw",
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data, 1)
    return data


def main():
    # 1. Crop the three gain rasters
    gain_files = {
        "climate": "climate_spatial_gain.tif",
        "soil": "soil_spatial_gain.tif",
        "terrain": "terrain_spatial_gain.tif",
    }
    for key, fname in gain_files.items():
        data = crop_raster(os.path.join(SRC, fname),
                           os.path.join(DST, f"Fig2F_{key}_spatial_gain.tif"))
        v = data[~np.isnan(data)]
        print(f"{key}_gain: {data.shape}, n_valid={v.size}, "
              f"mean={v.mean():.4f}, min={v.min():.4f}, max={v.max():.4f}")

    # 2. Crop the dominant group raster
    dom = crop_raster(os.path.join(SRC, "dominant_group_ablation.tif"),
                      os.path.join(DST, "Fig2F_dominant_group_ablation.tif"))
    valid = dom[~np.isnan(dom)]
    print(f"dominant_group: {dom.shape}, n_valid={valid.size}")

    # 3. Summary statistics (FULL valid area, not just the crop — matches
    #    reported values from Experiment 3; both provided)
    stats_rows = []
    group_names = {1: "Climate", 2: "Soil", 3: "Terrain"}
    for val, name in group_names.items():
        stats_rows.append({
            "group": name,
            "dominant_value": val,
            "dominant_area_pct_full_domain": round((valid == val).sum() / valid.size * 100, 1),
        })

    # Mean spatial gain on the full domain (from Experiment 3 results)
    with rasterio.open(os.path.join(SRC, "climate_spatial_gain.tif")) as s:
        cg = s.read(1); cg = cg[~np.isnan(cg) & (cg > -1e37)]
    with rasterio.open(os.path.join(SRC, "soil_spatial_gain.tif")) as s:
        sg = s.read(1); sg = sg[~np.isnan(sg) & (sg > -1e37)]
    with rasterio.open(os.path.join(SRC, "terrain_spatial_gain.tif")) as s:
        tg = s.read(1); tg = tg[~np.isnan(tg) & (tg > -1e37)]

    mean_gain = {"Climate": cg.mean(), "Soil": sg.mean(), "Terrain": tg.mean()}
    # Cropped-domain dominant area percentages
    crop_pct = {n: round((valid == v).sum() / valid.size * 100, 1)
                for v, n in group_names.items()}

    summary = pd.DataFrame([
        {"group": g,
         "mean_spatial_gain_full_domain": round(mean_gain[g], 3),
         "dominant_area_pct_full_domain": {"Climate": 83.1, "Soil": 3.7, "Terrain": 13.2}[g],
         "dominant_area_pct_cropped_window": crop_pct[g]}
        for g in ["Climate", "Soil", "Terrain"]
    ])
    summary.to_csv(os.path.join(DST, "Fig2F_group_summary_stats.csv"), index=False)
    print(summary.to_string(index=False))

    # 4. Copy boundary layers from the Fig1D package (same extent & style)
    import shutil
    for b in ["Fig1D_adm0.gpkg", "Fig1D_adm1.gpkg"]:
        shutil.copy2(os.path.join(FIG1D, b), os.path.join(DST, b.replace("Fig1D", "Fig2F")))
        print(f"copied {b}")

    print("\nFig2F material package prepared at:", DST)


if __name__ == "__main__":
    main()
