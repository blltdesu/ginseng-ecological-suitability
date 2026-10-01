"""
Step 01: Build ecological accessible area M.
Creates M buffers at 200, 300, 500 km around occurrence points.
"""
import os, sys, json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
import rasterio
from rasterio.features import geometry_mask, rasterize
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

ROOT = Path(r"E:\人参种在哪\实验1")
INPUT = ROOT / "01_input"
OUT = ROOT / "02_accessible_area_M"
QC = ROOT / "16_qc"
os.makedirs(OUT, exist_ok=True)
os.makedirs(QC, exist_ok=True)

BUFFERS = [200, 300, 500]  # km
MAIN = 300

# Load occurrence data
occ = pd.read_csv(INPUT / "occurrence_thin_10km.csv")
lon_col = 'decimalLongitude' if 'decimalLongitude' in occ.columns else 'longitude'
lat_col = 'decimalLatitude' if 'decimalLatitude' in occ.columns else 'latitude'
log.info(f"Loaded {len(occ)} occurrence records (10 km thinning)")

# Load study area boundaries
adm0 = gpd.read_file(INPUT / "study_context_adm0.gpkg")
adm1 = gpd.read_file(INPUT / "study_context_adm1.gpkg")
log.info(f"Loaded ADM0: {len(adm0)} countries, ADM1: {len(adm1)} regions")

# Create occurrence GeoDataFrame
from shapely.geometry import Point
occ_gdf = gpd.GeoDataFrame(
    occ, geometry=gpd.points_from_xy(occ[lon_col], occ[lat_col]), crs="EPSG:4326"
)

# Load reference grid for valid mask
with rasterio.open(INPUT / "common_valid_mask.tif") as src:
    valid_mask = src.read(1)
    ref_transform = src.transform
    ref_crs = src.crs
    ref_shape = src.shape

# Build M area for each buffer size
qc_rows = []
for buf_km in BUFFERS:
    log.info(f"Building M area: {buf_km} km buffer...")

    # Project to equidistant for buffer
    # Use custom AEQD centered on occurrence centroid
    centroid = occ_gdf.geometry.unary_union.centroid
    aeqd_crs = f"+proj=aeqd +lat_0={centroid.y} +lon_0={centroid.x} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

    occ_proj = occ_gdf.to_crs(aeqd_crs)
    buf_m = buf_km * 1000

    # Buffer and dissolve
    buffered = occ_proj.geometry.buffer(buf_m)
    dissolved = unary_union(buffered)

    # Convert back to WGS84
    M_poly = gpd.GeoDataFrame(geometry=[dissolved], crs=aeqd_crs).to_crs("EPSG:4326")

    # Intersect with land area (ADM0)
    M_land = M_poly.geometry.iloc[0].intersection(adm0.geometry.unary_union)

    # Intersect with valid mask area
    # Get the valid mask polygon (bounding box of valid pixels)
    valid_ys, valid_xs = np.where(valid_mask == 1)
    if len(valid_ys) > 0:
        from rasterio.transform import xy
        # Create valid area as union of valid pixel polygons (approximate with convex hull)
        from scipy.spatial import ConvexHull
        valid_coords = np.array([
            xy(ref_transform, y, x) for x, y in zip(valid_xs[::100], valid_ys[::100])
        ])
        hull = ConvexHull(valid_coords)
        valid_poly = gpd.GeoDataFrame(
            geometry=[box(
                valid_coords[:,0].min(), valid_coords[:,1].min(),
                valid_coords[:,0].max(), valid_coords[:,1].max()
            )], crs="EPSG:4326"
        ).geometry.iloc[0]

        M_final = M_land.intersection(valid_poly)
    else:
        M_final = M_land

    # Handle MultiPolygon
    if M_final.geom_type == 'GeometryCollection':
        M_final = unary_union([g for g in M_final.geoms if g.geom_type == 'Polygon' or g.geom_type == 'MultiPolygon'])

    M_gdf = gpd.GeoDataFrame(geometry=[M_final], crs="EPSG:4326")

    # Save vector
    gpkg_path = OUT / f"M_{buf_km}km.gpkg"
    M_gdf.to_file(gpkg_path, driver="GPKG")
    log.info(f"  Saved: {gpkg_path}")

    # Create mask raster
    with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
        mask_array = rasterize(
            [(geom, 1) for geom in M_gdf.geometry],
            out_shape=ref.shape,
            transform=ref.transform,
            dtype='uint8',
            fill=0,
            all_touched=True
        )

        mask_path = OUT / f"M_{buf_km}km_mask.tif"
        profile = ref.profile.copy()
        profile.update(dtype='uint8', nodata=0, compress='lzw')
        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(mask_array, 1)
    log.info(f"  Saved mask: {mask_path}")

    # QC: check occurrence containment
    occ_in = occ_gdf.geometry.within(M_final).sum()
    occ_out = (~occ_gdf.geometry.within(M_final)).sum()

    # Area calculation in equal-area projection
    M_proj = M_gdf.to_crs("+proj=eck4 +datum=WGS84 +units=m +no_defs")
    area_km2 = M_proj.geometry.area.sum() / 1e6

    qc_rows.append({
        'buffer_km': buf_km,
        'area_km2': round(area_km2, 2),
        'n_occurrence_inside': occ_in,
        'n_occurrence_outside': occ_out,
        'valid_pixel_count': int(mask_array.sum())
    })

    log.info(f"  Area: {area_km2:.0f} km², Occurrence: {occ_in}/{len(occ_gdf)} inside")

# Save QC
qc_df = pd.DataFrame(qc_rows)
qc_df.to_csv(QC / "M_area_qc.csv", index=False)
log.info(f"\nM area QC saved to {QC / 'M_area_qc.csv'}")

# Check all occurrences in main M
main_qc = qc_df[qc_df['buffer_km'] == MAIN].iloc[0]
if main_qc['n_occurrence_outside'] > 0:
    log.warning(f"WARNING: {main_qc['n_occurrence_outside']} occurrences outside main M area!")

log.info("\nM area construction: COMPLETE")
