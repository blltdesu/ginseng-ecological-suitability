"""Step 3: Build dataset registry with deep inspection of data structure."""
import os
import csv
import json
import numpy as np
import pandas as pd
import rasterio
from datetime import datetime

RAW_DIR = r"E:\人参种在哪\数据"
EXTRACT_DIR = r"E:\人参种在哪\00_统一数据预处理\01_inventory\extracted"
OUT_CSV = r"E:\人参种在哪\00_统一数据预处理\01_inventory\dataset_registry.csv"
REVIEW_MD = r"E:\人参种在哪\00_统一数据预处理\01_inventory\manual_review_required.md"
LOG_PATH = r"E:\人参种在哪\00_统一数据预处理\logs\inventory.log"

datasets = []
review_items = []


def inspect_gbif():
    """Inspect GBIF occurrence data."""
    print("\n=== GBIF Occurrence Inspection ===")
    # Look for the extracted CSV
    gbif_dirs = [
        d for d in os.listdir(EXTRACT_DIR)
        if d.startswith("0002333-")
    ]
    if not gbif_dirs:
        # Check raw dir
        csv_path = None
    else:
        gbif_dir = os.path.join(EXTRACT_DIR, gbif_dirs[0])
        csv_files = [f for f in os.listdir(gbif_dir) if f.endswith(".csv")]
        if csv_files:
            csv_path = os.path.join(gbif_dir, csv_files[0])
        else:
            csv_path = None

    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=None, engine="python", nrows=5)
        cols = list(df.columns)
        nrows_total = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        print(f"  File: {csv_path}")
        print(f"  Rows: {nrows_total}")
        print(f"  Columns ({len(cols)}): {cols[:20]}...")

        datasets.append({
            "dataset_id": "OCC_001",
            "dataset_category": "occurrence",
            "dataset_name": "GBIF Occurrence Download",
            "source_paths": csv_path,
            "format": "CSV",
            "temporal_period": "",
            "scenario": "",
            "gcm": "",
            "variable": f"{len(cols)} columns",
            "depth": "",
            "crs": "",
            "resolution": "",
            "status": "recognized",
            "confidence": "0.95",
            "notes": f"GBIF occurrence data, {nrows_total} records, {len(cols)} columns"
        })
        return cols, nrows_total, csv_path
    else:
        review_items.append("**GBIF occurrence data**: No CSV found in extracted archive.")
        datasets.append({
            "dataset_id": "OCC_001",
            "dataset_category": "occurrence",
            "dataset_name": "GBIF Occurrence Download",
            "source_paths": "",
            "format": "UNKNOWN",
            "temporal_period": "", "scenario": "", "gcm": "", "variable": "", "depth": "",
            "crs": "", "resolution": "",
            "status": "unresolved",
            "confidence": "0.0",
            "notes": "GBIF occurrence file not found in expected location"
        })
        return None, 0, None


def inspect_current_climate():
    """Inspect WorldClim current climate data."""
    print("\n=== Current Climate Inspection ===")
    climate_dir = os.path.join(RAW_DIR, "气候数据", "wc2.1_2.5m_bio")

    tif_files = sorted([f for f in os.listdir(climate_dir) if f.endswith(".tif")])

    if not tif_files:
        review_items.append("**Current climate**: No TIFF files found in climate directory.")
        return

    # Read first tif to get metadata
    first_tif = os.path.join(climate_dir, tif_files[0])
    with rasterio.open(first_tif) as src:
        crs = str(src.crs)
        shape = src.shape
        resolution = src.res
        bounds = src.bounds
        transform = src.transform
        print(f"  CRS: {crs}")
        print(f"  Shape: {shape}")
        print(f"  Resolution: {resolution}")
        print(f"  Bounds: {bounds}")
        print(f"  BIO files: {len(tif_files)}")

    # Map filenames to BIO variables
    bio_map = {}
    for f in tif_files:
        # wc2.1_2.5m_bio_1.tif -> bio01
        import re
        match = re.search(r"bio_(\d+)", f)
        if match:
            bio_num = int(match.group(1))
            bio_map[f"bio{bio_num:02d}"] = os.path.join(climate_dir, f)

    datasets.append({
        "dataset_id": "CLIM_001",
        "dataset_category": "current_climate",
        "dataset_name": "WorldClim 2.1 Current Climate (2.5 arc-min)",
        "source_paths": climate_dir,
        "format": "GeoTIFF (multi-file)",
        "temporal_period": "1970-2000",
        "scenario": "historical",
        "gcm": "",
        "variable": "BIO1-BIO19",
        "depth": "",
        "crs": crs,
        "resolution": f"{resolution[0]} deg",
        "status": "recognized" if len(tif_files) == 19 else "recognized_with_warning",
        "confidence": "1.0",
        "notes": f"WorldClim 2.1, {len(tif_files)} BIO variables, shape={shape}"
    })
    return bio_map, crs, shape, resolution, transform


def inspect_future_climate():
    """Inspect future climate data."""
    print("\n=== Future Climate Inspection ===")
    future_dir = os.path.join(RAW_DIR, "未来天气")

    # Check for actual data files
    tif_files = []
    nc_files = []
    for dirpath, dirnames, filenames in os.walk(future_dir):
        for f in filenames:
            if f.endswith((".tif", ".tiff")):
                tif_files.append(os.path.join(dirpath, f))
            elif f.endswith(".nc"):
                nc_files.append(os.path.join(dirpath, f))

    if not tif_files and not nc_files:
        datasets.append({
            "dataset_id": "FCLIM_001",
            "dataset_category": "future_climate",
            "dataset_name": "Future Climate (CMIP6)",
            "source_paths": future_dir,
            "format": "Scripts only",
            "temporal_period": "TBD",
            "scenario": "TBD",
            "gcm": "TBD",
            "variable": "None found",
            "depth": "",
            "crs": "",
            "resolution": "",
            "status": "unresolved",
            "confidence": "0.2",
            "notes": "No future climate data files found. Only download scripts present. Data needs to be downloaded before Experiment 4."
        })
        review_items.append(
            "**Future climate data**: No actual CMIP6/WorldClim future climate data files found. "
            "Only `02_download_worldclim_future.py` and `03_check_future_tifs.py` scripts are present. "
            "Future climate data must be downloaded before Experiment 4 can proceed. "
            "This does NOT block current-climate experiments (Experiments 1-3)."
        )
        print("  WARNING: No future climate data files found!")
        return
    else:
        print(f"  Found {len(tif_files)} TIFF files and {len(nc_files)} NetCDF files")
        # Inspect one file
        if tif_files:
            with rasterio.open(tif_files[0]) as src:
                print(f"  Example CRS: {src.crs}, Shape: {src.shape}, Bands: {src.count}")


def inspect_soil():
    """Inspect SoilGrids data."""
    print("\n=== Soil Data Inspection ===")
    soil_dir = os.path.join(RAW_DIR, "土壤数据")

    tif_files = sorted([f for f in os.listdir(soil_dir) if f.endswith(".tif")])

    soil_info = {}
    for f in tif_files:
        fpath = os.path.join(soil_dir, f)
        with rasterio.open(fpath) as src:
            crs = str(src.crs)
            shape = src.shape
            n_bands = src.count
            dtype = src.dtypes[0]
            # Read metadata
            band_desc = [src.descriptions[i] for i in range(n_bands)]

            # Get data range
            data = src.read(1, masked=True)
            data_min = float(data.min())
            data_max = float(data.max())
            data_median = float(np.ma.median(data))

            var_name = f.replace("SoilGrids_", "").replace("_mean_EastAsia.tif", "")
            soil_info[var_name] = {
                "path": fpath,
                "crs": crs,
                "shape": shape,
                "bands": n_bands,
                "dtype": dtype,
                "band_desc": band_desc,
                "min": data_min,
                "max": data_max,
                "median": data_median,
            }
            print(f"  {var_name}: {n_bands} band(s), CRS={crs}, shape={shape}, "
                  f"range=[{data_min:.4f}, {data_max:.4f}], median={data_median:.4f}, dtype={dtype}")
            if band_desc and any(bd for bd in band_desc):
                print(f"    Band descriptions: {band_desc}")

    datasets.append({
        "dataset_id": "SOIL_001",
        "dataset_category": "soil",
        "dataset_name": "SoilGrids 2.0",
        "source_paths": soil_dir,
        "format": "GeoTIFF (single-band)",
        "temporal_period": "",
        "scenario": "",
        "gcm": "",
        "variable": ", ".join(soil_info.keys()),
        "depth": "mean 0-? cm",
        "crs": list(soil_info.values())[0]["crs"] if soil_info else "",
        "resolution": "",
        "status": "recognized",
        "confidence": "0.9",
        "notes": f"{len(soil_info)} soil variables, all single-band"
    })
    return soil_info


def inspect_terrain():
    """Inspect terrain/SRTM data."""
    print("\n=== Terrain Data Inspection ===")
    terrain_dir = os.path.join(RAW_DIR, "地形数据")

    tif_files = [f for f in os.listdir(terrain_dir) if f.endswith(".tif")]
    if not tif_files:
        review_items.append("**Terrain data**: No TIFF files found.")
        return {}

    fpath = os.path.join(terrain_dir, tif_files[0])
    terrain_info = {}
    with rasterio.open(fpath) as src:
        crs = str(src.crs)
        shape = src.shape
        n_bands = src.count
        dtype = src.dtypes[0]
        band_desc = [src.descriptions[i] for i in range(n_bands)]
        transform = src.transform

        print(f"  File: {tif_files[0]}")
        print(f"  CRS: {crs}, Shape: {shape}, Bands: {n_bands}, dtype: {dtype}")
        print(f"  Band descriptions: {band_desc}")

        # Read each band for stats
        for i in range(n_bands):
            data = src.read(i + 1, masked=True)
            band_name = band_desc[i] if i < len(band_desc) and band_desc[i] else f"band_{i+1}"
            terrain_info[band_name] = {
                "path": fpath,
                "band_idx": i + 1,
                "crs": crs,
                "min": float(data.min()),
                "max": float(data.max()),
                "median": float(np.ma.median(data)),
            }
            print(f"    {band_name}: range=[{terrain_info[band_name]['min']:.4f}, "
                  f"{terrain_info[band_name]['max']:.4f}]")

    datasets.append({
        "dataset_id": "TERR_001",
        "dataset_category": "terrain",
        "dataset_name": "SRTM Terrain Indices",
        "source_paths": fpath,
        "format": "GeoTIFF (multi-band)",
        "temporal_period": "",
        "scenario": "",
        "gcm": "",
        "variable": ", ".join(terrain_info.keys()),
        "depth": "",
        "crs": crs,
        "resolution": "",
        "status": "recognized",
        "confidence": "0.95",
        "notes": f"SRTM-derived terrain indices, {n_bands} bands"
    })
    return terrain_info


def inspect_landcover():
    """Inspect landcover/MODIS data."""
    print("\n=== Landcover Inspection ===")
    lc_dir = os.path.join(RAW_DIR, "土地覆盖")

    tif_files = [f for f in os.listdir(lc_dir) if f.endswith(".tif")]
    if not tif_files:
        review_items.append("**Landcover data**: No TIFF files found.")
        return

    fpath = os.path.join(lc_dir, tif_files[0])
    with rasterio.open(fpath) as src:
        crs = str(src.crs)
        shape = src.shape
        n_bands = src.count
        dtype = src.dtypes[0]
        data = src.read(1, masked=True)
        unique_vals = list(set(data.flatten()))[:30]  # first 30 unique values

        print(f"  File: {tif_files[0]}")
        print(f"  CRS: {crs}, Shape: {shape}, Bands: {n_bands}, dtype: {dtype}")
        print(f"  Unique values (first 30): {sorted(unique_vals)}")

    datasets.append({
        "dataset_id": "LC_001",
        "dataset_category": "landcover",
        "dataset_name": "MODIS MCD12Q1 Land Cover Type 1",
        "source_paths": fpath,
        "format": "GeoTIFF",
        "temporal_period": "2019-2024 (mode)",
        "scenario": "",
        "gcm": "",
        "variable": "LC_Type1",
        "depth": "",
        "crs": crs,
        "resolution": "",
        "status": "recognized",
        "confidence": "1.0",
        "notes": f"MCD12Q1 LC_Type1 mode 2019-2024, {len(unique_vals)} unique classes"
    })


def inspect_boundaries():
    """Inspect boundary data."""
    print("\n=== Boundary Data Inspection ===")
    # Check extracted boundary data
    boundary_dirs = [d for d in os.listdir(EXTRACT_DIR) if "geoBoundaries" in d and "ADM0" in d]

    boundary_info = {}
    for bd in sorted(boundary_dirs):
        parts = bd.split("_")[0].split("-")
        if len(parts) >= 2:
            country = parts[1]
            adm_level = "ADM0"
            bd_path = os.path.join(EXTRACT_DIR, bd)
            # Find the .shp file
            shp_files = [f for f in os.listdir(bd_path) if f.endswith(".shp") and "simplified" not in f]
            geojson_files = [f for f in os.listdir(bd_path) if f.endswith(".geojson") and "simplified" not in f]

            if country not in boundary_info:
                boundary_info[country] = {}
            boundary_info[country][adm_level] = {
                "shp": os.path.join(bd_path, shp_files[0]) if shp_files else None,
                "geojson": os.path.join(bd_path, geojson_files[0]) if geojson_files else None,
            }

            # Read with geopandas to check
            if shp_files:
                try:
                    import geopandas as gpd
                    gdf = gpd.read_file(os.path.join(bd_path, shp_files[0]))
                    print(f"  {country} ADM0: {len(gdf)} features, CRS={gdf.crs}, "
                          f"columns={list(gdf.columns)[:10]}")
                except Exception as e:
                    print(f"  {country} ADM0: ERROR reading - {e}")

    # Also check ADM1
    adm1_dirs = [d for d in os.listdir(EXTRACT_DIR) if "geoBoundaries" in d and "ADM1" in d]
    for ad in sorted(adm1_dirs)[:3]:  # Check first 3
        parts = ad.split("_")[0].split("-")
        if len(parts) >= 2:
            country = parts[1]
            ad_path = os.path.join(EXTRACT_DIR, ad)
            shp_files = [f for f in os.listdir(ad_path) if f.endswith(".shp") and "simplified" not in f]
            if shp_files:
                try:
                    import geopandas as gpd
                    gdf = gpd.read_file(os.path.join(ad_path, shp_files[0]))
                    print(f"  {country} ADM1: {len(gdf)} features, CRS={gdf.crs}")
                    if country not in boundary_info:
                        boundary_info[country] = {}
                    boundary_info[country]["ADM1"] = {
                        "shp": os.path.join(ad_path, shp_files[0]),
                    }
                except Exception as e:
                    print(f"  {country} ADM1: ERROR - {e}")

    datasets.append({
        "dataset_id": "BND_001",
        "dataset_category": "boundary",
        "dataset_name": "geoBoundaries Administrative Boundaries",
        "source_paths": EXTRACT_DIR,
        "format": "SHP/GeoJSON",
        "temporal_period": "",
        "scenario": "",
        "gcm": "",
        "variable": "ADM0, ADM1 (CHN, JPN, KOR, MNG, PRK, RUS)",
        "depth": "",
        "crs": "EPSG:4326",
        "resolution": "",
        "status": "recognized",
        "confidence": "1.0",
        "notes": f"6 countries: CHN, JPN, KOR, MNG, PRK, RUS"
    })
    return boundary_info


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # Inspect each data category
    gbif_info = inspect_gbif()
    climate_info = inspect_current_climate()
    inspect_future_climate()
    soil_info = inspect_soil()
    terrain_info = inspect_terrain()
    inspect_landcover()
    boundary_info = inspect_boundaries()

    # Write dataset registry
    fieldnames = [
        "dataset_id", "dataset_category", "dataset_name", "source_paths",
        "format", "temporal_period", "scenario", "gcm", "variable", "depth",
        "crs", "resolution", "status", "confidence", "notes"
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(datasets)

    # Write manual review items
    md_content = "# Manual Review Required\n\n"
    md_content += f"Generated: {datetime.now().isoformat()}\n\n"
    if review_items:
        for item in review_items:
            md_content += f"- {item}\n\n"
    else:
        md_content += "No items require manual review at this stage.\n\n"

    # Add future climate note
    md_content += "\n## Action Items\n\n"
    md_content += "1. **Future Climate Download**: The `未来天气` directory contains only download scripts.\n"
    md_content += "   - Run `02_download_worldclim_future.py` to download CMIP6 future climate data before Experiment 4.\n"
    md_content += "   - This does NOT block Experiments 1-3 (current climate only).\n\n"

    with open(REVIEW_MD, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Log
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] DATASET REGISTRY: {len(datasets)} datasets registered\n")
        for ds in datasets:
            f.write(f"  {ds['dataset_id']}: {ds['status']} - {ds['dataset_name']}\n")

    print(f"\n{'='*50}")
    print(f"DATASET REGISTRY COMPLETE")
    print(f"Datasets: {len(datasets)}")
    print(f"Review items: {len(review_items)}")
    for ds in datasets:
        print(f"  {ds['dataset_id']} [{ds['status']}]: {ds['dataset_name']}")
    print(f"\nOutput: {OUT_CSV}")
    print(f"Review: {REVIEW_MD}")

    # Checkpoint A
    critical_cats = {"occurrence", "current_climate", "soil", "terrain"}
    missing = []
    for cat in critical_cats:
        found = any(ds["dataset_category"] == cat and ds["status"] in ("recognized", "recognized_with_warning")
                    for ds in datasets)
        if not found:
            missing.append(cat)

    if missing:
        stop_path = r"E:\人参种在哪\00_统一数据预处理\STOP_A_DATA_IDENTIFICATION.md"
        with open(stop_path, "w", encoding="utf-8") as f:
            f.write(f"# STOP A: Data Identification Failed\n\n")
            f.write(f"Missing/unresolved critical datasets: {', '.join(missing)}\n")
        print(f"\n*** STOP A: Missing critical data: {missing} ***")
    else:
        print("\nCheckpoint A PASSED: All critical data categories identified.")


if __name__ == "__main__":
    main()
