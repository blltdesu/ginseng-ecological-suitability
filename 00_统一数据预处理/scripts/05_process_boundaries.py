"""Step 5: Process administrative boundaries."""
import os
import pandas as pd
import geopandas as gpd
from datetime import datetime

EXTRACT_DIR = r"E:\人参种在哪\00_统一数据预处理\01_inventory\extracted"
OUT_DIR = r"E:\人参种在哪\00_统一数据预处理\03_boundaries"
LOG_PATH = r"E:\人参种在哪\00_统一数据预处理\logs\inventory.log"

os.makedirs(OUT_DIR, exist_ok=True)

COUNTRY_CODES = ["CHN", "JPN", "KOR", "MNG", "PRK", "RUS"]
COUNTRY_NAMES = {
    "CHN": "China", "JPN": "Japan", "KOR": "South Korea",
    "MNG": "Mongolia", "PRK": "North Korea", "RUS": "Russia"
}


def main():
    log_lines = []
    log_lines.append(f"[{datetime.now().isoformat()}] BOUNDARY PROCESSING STARTED\n")
    print("=" * 50)
    print("BOUNDARY PROCESSING")

    # === ADM0 ===
    adm0_parts = []
    for code in COUNTRY_CODES:
        # Find the extracted directory
        candidates = [d for d in os.listdir(EXTRACT_DIR)
                      if f"geoBoundaries-{code}-ADM0" in d and "simplified" not in d]
        # Actually the extracted dir names include hash suffix
        dirs = [d for d in os.listdir(EXTRACT_DIR)
                if d.startswith(f"geoBoundaries-{code}-ADM0-all_")]
        if not dirs:
            print(f"  WARNING: No ADM0 data for {code}")
            continue

        shp_dir = os.path.join(EXTRACT_DIR, dirs[0])
        shp_files = [f for f in os.listdir(shp_dir) if f.endswith(".shp") and "_simplified" not in f]
        if not shp_files:
            # Try GeoJSON
            geojson_files = [f for f in os.listdir(shp_dir) if f.endswith(".geojson") and "_simplified" not in f]
            if geojson_files:
                gdf = gpd.read_file(os.path.join(shp_dir, geojson_files[0]))
            else:
                continue
        else:
            gdf = gpd.read_file(os.path.join(shp_dir, shp_files[0]))

        # Standardize columns
        gdf["country"] = COUNTRY_NAMES.get(code, code)
        gdf["iso_code"] = code
        # Keep original name
        if "shapeName" in gdf.columns:
            gdf["original_name"] = gdf["shapeName"]
        print(f"  {code}: {len(gdf)} features, CRS={gdf.crs}")
        adm0_parts.append(gdf)

    if adm0_parts:
        adm0 = gpd.GeoDataFrame(
            pd.concat(adm0_parts, ignore_index=True),
            crs=adm0_parts[0].crs
        )

        # Convert to EPSG:4326
        if adm0.crs and adm0.crs.to_string() != "EPSG:4326":
            adm0 = adm0.to_crs("EPSG:4326")

        # Fix invalid geometries
        invalid = ~adm0.is_valid
        if invalid.any():
            print(f"  Fixing {invalid.sum()} invalid geometries...")
            adm0.loc[invalid, "geometry"] = adm0.loc[invalid, "geometry"].buffer(0)

        out_path = os.path.join(OUT_DIR, "study_context_adm0.gpkg")
        adm0.to_file(out_path, driver="GPKG")
        print(f"  ADM0 saved: {out_path} ({len(adm0)} features)")
        log_lines.append(f"  ADM0: {len(adm0)} countries\n")

    # === ADM1 ===
    adm1_parts = []
    for code in COUNTRY_CODES:
        dirs = [d for d in os.listdir(EXTRACT_DIR)
                if d.startswith(f"geoBoundaries-{code}-ADM1-all_")]
        if not dirs:
            print(f"  WARNING: No ADM1 data for {code}")
            continue

        shp_dir = os.path.join(EXTRACT_DIR, dirs[0])
        shp_files = [f for f in os.listdir(shp_dir) if f.endswith(".shp") and "_simplified" not in f]
        if shp_files:
            gdf = gpd.read_file(os.path.join(shp_dir, shp_files[0]))
        else:
            geojson_files = [f for f in os.listdir(shp_dir) if f.endswith(".geojson") and "_simplified" not in f]
            if geojson_files:
                gdf = gpd.read_file(os.path.join(shp_dir, geojson_files[0]))
            else:
                continue

        gdf["country"] = COUNTRY_NAMES.get(code, code)
        gdf["iso_code"] = code
        if "shapeName" in gdf.columns:
            gdf["adm1_name"] = gdf["shapeName"]
        elif "shapeGroup" in gdf.columns:
            gdf["adm1_name"] = gdf["shapeGroup"]
        print(f"  {code} ADM1: {len(gdf)} features")
        adm1_parts.append(gdf)

    if adm1_parts:
        adm1 = gpd.GeoDataFrame(
            pd.concat(adm1_parts, ignore_index=True),
            crs=adm1_parts[0].crs
        )

        if adm1.crs and adm1.crs.to_string() != "EPSG:4326":
            adm1 = adm1.to_crs("EPSG:4326")

        invalid = ~adm1.is_valid
        if invalid.any():
            print(f"  Fixing {invalid.sum()} invalid ADM1 geometries...")
            adm1.loc[invalid, "geometry"] = adm1.loc[invalid, "geometry"].buffer(0)

        out_path = os.path.join(OUT_DIR, "study_context_adm1.gpkg")
        adm1.to_file(out_path, driver="GPKG")
        print(f"  ADM1 saved: {out_path} ({len(adm1)} features)")
        log_lines.append(f"  ADM1: {len(adm1)} subdivisions\n")

    # Write log
    log_lines.append(f"[{datetime.now().isoformat()}] BOUNDARY PROCESSING COMPLETED\n")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.writelines(log_lines)

    print(f"\nBoundary processing complete. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
