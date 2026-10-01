#!/usr/bin/env python3
"""
Experiment 4 — Steps 17-30: Analysis Pipeline
Area statistics, centroid shift, elevation shift, landscape metrics,
SSP comparison, sensitivity analysis, and handoff generation.
"""
import os, sys, csv, json, logging, time, math
from pathlib import Path
import rasterio
import numpy as np
from collections import defaultdict

# Fix stdout encoding for Windows GBK issues
if sys.stdout.encoding and 'gbk' in sys.stdout.encoding.lower():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ——— Config ———
EXP4_DIR = Path(r"E:\人参种在哪\实验4")
INPUT_DIR = EXP4_DIR / "00_input_from_experiment3"
ENSEMBLE_DIR = EXP4_DIR / "08_future_predictions_ensemble"
CHANGE_DIR = EXP4_DIR / "09_suitability_change"
AREA_DIR = EXP4_DIR / "10_area_statistics"
CENTROID_DIR = EXP4_DIR / "11_centroid_shift"
ELEV_DIR = EXP4_DIR / "12_elevation_shift"
LANDSCAPE_DIR = EXP4_DIR / "13_landscape_structure"
COMPARE_DIR = EXP4_DIR / "14_scenario_comparison"
SENS_DIR = EXP4_DIR / "15_sensitivity"
TABLES_DIR = EXP4_DIR / "18_tables"
QC_DIR = EXP4_DIR / "19_qc"
LOG_DIR = EXP4_DIR / "logs"

for d in [AREA_DIR, CENTROID_DIR, ELEV_DIR, LANDSCAPE_DIR, COMPARE_DIR,
          SENS_DIR, TABLES_DIR, QC_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "analysis.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("analysis")

GCMS = ["ACCESS-CM2", "BCC-CSM2-MR", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]

# ——— Helpers ———
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def bearing_deg(lon1, lat1, lon2, lat2):
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon)
    brg = math.degrees(math.atan2(y, x))
    return (brg + 360) % 360

def bearing_to_cardinal(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[round(deg / 22.5) % 16]

def pixel_area_km2(lat, res_deg=0.041666666666666664):
    """Approximate pixel area in km² at given latitude (WGS84)."""
    R = 6371.0
    lat_rad = math.radians(lat)
    # width at latitude
    dx = res_deg * (math.pi / 180) * R * math.cos(lat_rad)
    dy = res_deg * (math.pi / 180) * R
    return dx * dy

# Load reference grid and mask
MASK_PATH = INPUT_DIR / "common_valid_mask.tif"
REF_PATH = INPUT_DIR / "reference_grid_template.tif"
with rasterio.open(str(MASK_PATH)) as msrc:
    valid_mask = msrc.read(1) > 0
    ref_profile = msrc.profile
    ref_shape = msrc.shape
with rasterio.open(str(REF_PATH)) as ref:
    ref_bounds = ref.bounds

# Get lat/lon for each pixel
rows, cols = np.where(valid_mask)
lons = ref_bounds.left + (cols + 0.5) * ref_profile["transform"][0]
lats = ref_bounds.top + (rows + 0.5) * ref_profile["transform"][4]
logger.info(f"Valid pixels: {len(rows)}, Lon range: [{lons.min():.1f}, {lons.max():.1f}], Lat range: [{lats.min():.1f}, {lats.max():.1f}]")

# Pre-compute pixel areas for area-weighting
pixel_areas = np.array([pixel_area_km2(lat) for lat in lats])

# Load elevation
elevation_data = None
for elev_path_candidate in [
    Path(r"E:\人参种在哪\实验3\00_input_from_experiment2\predictors\elevation.tif"),
    Path(r"E:\人参种在哪\实验1\03_current_environment_processed\elevation.tif"),
    Path(r"E:\人参种在哪\实验3\03_current_environment_processed\elevation.tif"),
    Path(r"E:\人参种在哪\实验4\00_input_from_experiment3\elevation.tif"),
]:
    if elev_path_candidate.exists():
        try:
            with rasterio.open(str(elev_path_candidate)) as esrc:
                elev_raw = esrc.read(1, masked=True)
                if elev_raw.shape != valid_mask.shape:
                    from rasterio.warp import reproject, Resampling
                    elev_aligned = np.full(valid_mask.shape, np.nan)
                    reproject(source=elev_raw.filled(np.nan), destination=elev_aligned,
                             src_transform=esrc.transform, src_crs=esrc.crs,
                             dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                             resampling=Resampling.bilinear)
                    elevation_data = elev_aligned
                else:
                    elevation_data = elev_raw.filled(np.nan)
            logger.info(f"Loaded elevation from {elev_path_candidate}")
            break
        except Exception as e:
            logger.warning(f"Failed to load elevation from {elev_path_candidate}: {e}")

if elevation_data is None:
    logger.warning("No elevation data found — elevation analysis will be skipped")

# Load current binary
with rasterio.open(str(INPUT_DIR / "current_binary_suitability.tif")) as s:
    current_binary = s.read(1) > 0.5

def load_raster(path):
    """Load a raster and return valid-pixel values."""
    with rasterio.open(str(path)) as s:
        return s.read(1)

def main():
    logger.info("=" * 60)
    logger.info("Analysis Pipeline — Experiment 4")
    logger.info("=" * 60)

    # ——— Area Statistics ———
    logger.info("\n=== Area Statistics ===")
    area_rows = []
    centroid_rows = []
    elev_rows = []
    landscape_rows = []

    for gcm in GCMS:
        for ssp in SSPS:
            for period in PERIODS:
                # Load ensemble
                ens_path = ENSEMBLE_DIR / gcm / ssp / period / "ensemble_suitability.tif"
                change_path = CHANGE_DIR / gcm / ssp / period / "change_class.tif"
                if not ens_path.exists() or not change_path.exists():
                    logger.debug(f"  Skipping {gcm}/{ssp}/{period} — data not found")
                    continue

                logger.info(f"  Processing {gcm}/{ssp}/{period}")
                ensemble = load_raster(ens_path)
                change_map = load_raster(change_path)
                future_binary = ensemble >= json.load(open(INPUT_DIR / "ensemble_threshold.json"))["ensemble_threshold"]

                # Area calculation (equal-area using pixel_areas)
                valid_ensemble = ensemble[valid_mask]
                valid_future_bin = future_binary[valid_mask]
                valid_change = change_map[valid_mask]

                current_area = np.sum(pixel_areas[current_binary[valid_mask]])
                future_area = np.sum(pixel_areas[valid_future_bin])
                stable_area = np.sum(pixel_areas[valid_change == 1])
                loss_area = np.sum(pixel_areas[valid_change == 2])
                gain_area = np.sum(pixel_areas[valid_change == 3])

                net_change = future_area - current_area
                area_change_pct = (net_change / current_area * 100) if current_area > 0 else 0
                gain_pct = (gain_area / current_area * 100) if current_area > 0 else 0
                loss_pct = (loss_area / current_area * 100) if current_area > 0 else 0
                persistence_pct = (stable_area / current_area * 100) if current_area > 0 else 0

                area_rows.append({
                    "gcm": gcm, "ssp": ssp, "period": period,
                    "current_area_km2": f"{current_area:.2f}",
                    "future_area_km2": f"{future_area:.2f}",
                    "stable_km2": f"{stable_area:.2f}",
                    "gain_km2": f"{gain_area:.2f}",
                    "loss_km2": f"{loss_area:.2f}",
                    "net_change_km2": f"{net_change:.2f}",
                    "area_change_pct": f"{area_change_pct:.2f}",
                    "gain_pct": f"{gain_pct:.2f}",
                    "loss_pct": f"{loss_pct:.2f}",
                    "persistence_pct": f"{persistence_pct:.2f}",
                })
                logger.info(f"    Area: current={current_area:.0f}km², future={future_area:.0f}km², "
                          f"change={area_change_pct:.1f}%, gain={gain_pct:.1f}%, loss={loss_pct:.1f}%")

                # ——— Centroid Calculation ———
                # Weighted centroid (suitability-weighted)
                weights = valid_ensemble[valid_future_bin]
                w_lons = lons[valid_future_bin]
                w_lats = lats[valid_future_bin]
                if weights.sum() > 0 and len(weights) > 0:
                    cx_weighted = np.average(w_lons, weights=weights)
                    cy_weighted = np.average(w_lats, weights=weights)
                else:
                    cx_weighted, cy_weighted = np.nan, np.nan

                # Binary centroid
                if valid_future_bin.sum() > 0:
                    cx_binary = w_lons.mean()
                    cy_binary = w_lats.mean()
                else:
                    cx_binary, cy_binary = np.nan, np.nan

                centroid_rows.append({
                    "gcm": gcm, "ssp": ssp, "period": period,
                    "centroid_type": "weighted",
                    "longitude": f"{cx_weighted:.6f}" if not np.isnan(cx_weighted) else "nan",
                    "latitude": f"{cy_weighted:.6f}" if not np.isnan(cy_weighted) else "nan",
                })
                centroid_rows.append({
                    "gcm": gcm, "ssp": ssp, "period": period,
                    "centroid_type": "binary",
                    "longitude": f"{cx_binary:.6f}" if not np.isnan(cx_binary) else "nan",
                    "latitude": f"{cy_binary:.6f}" if not np.isnan(cy_binary) else "nan",
                })

                # ——— Elevation Distribution ———
                if elevation_data is not None:
                    # Compute for each class
                    for class_name, class_mask in [
                        ("current_suitable", current_binary[valid_mask]),
                        ("future_suitable", future_binary[valid_mask]),
                        ("stable", valid_change == 1),
                        ("gain", valid_change == 3),
                        ("loss", valid_change == 2),
                    ]:
                        elev_vals = elevation_data[valid_mask][class_mask]
                        elev_vals = elev_vals[~np.isnan(elev_vals)]
                        if len(elev_vals) > 0:
                            elev_rows.append({
                                "gcm": gcm, "ssp": ssp, "period": period,
                                "class": class_name,
                                "mean_elevation_m": f"{np.mean(elev_vals):.2f}",
                                "median_elevation_m": f"{np.median(elev_vals):.2f}",
                                "p25_m": f"{np.percentile(elev_vals, 25):.2f}",
                                "p75_m": f"{np.percentile(elev_vals, 75):.2f}",
                                "p10_m": f"{np.percentile(elev_vals, 10):.2f}",
                                "p90_m": f"{np.percentile(elev_vals, 90):.2f}",
                            })

                # ——— Landscape Metrics (cropped to study area for performance) ———
                from scipy import ndimage
                # Crop to study region bounding box to speed up connected-components
                transform = ref_profile["transform"]
                # Study area: lon 100-150, lat 25-55 (East Asia ginseng range)
                col_min = int((100 - transform[2]) / transform[0])
                col_max = int((150 - transform[2]) / transform[0]) + 1
                row_max = int((25 - transform[5]) / transform[4])
                row_min = int((55 - transform[5]) / transform[4]) + 1
                # Clamp to valid range
                col_min = max(0, col_min); col_max = min(ref_shape[1], col_max)
                row_min = max(0, row_min); row_max = min(ref_shape[0], row_max)
                fb_crop = future_binary[row_min:row_max, col_min:col_max]
                crop_rows, crop_cols = fb_crop.shape
                logger.info(f"    Landscape crop: [{row_min}:{row_max}, {col_min}:{col_max}] = {crop_rows}x{crop_cols}")

                patch_labels, n_patches = ndimage.label(fb_crop, structure=np.ones((3,3)))
                if n_patches > 0:
                    patch_areas = []
                    for p in range(1, min(n_patches + 1, 1000)):  # Cap at 1000 patches for speed
                        pmask = patch_labels == p
                        n_pixels = pmask.sum()
                        if n_pixels < 3:  # Skip tiny fragments
                            continue
                        # Approximate area using mid-latitude of crop
                        crop_mid_lat = 25 + (55 - 25) * (crop_rows - (pmask.nonzero()[0].mean() if pmask.any() else 0)) / crop_rows
                        pa = n_pixels * pixel_area_km2(40)  # Use 40°N as fixed reference
                        patch_areas.append(pa)

                    # Edge density (on cropped region)
                    if crop_rows > 0 and crop_cols > 0:
                        edges = ndimage.sobel(fb_crop.astype(float))
                        edge_pixels = (np.abs(edges) > 0).sum()
                    else:
                        edge_pixels = 0

                    landscape_rows.append({
                        "gcm": gcm, "ssp": ssp, "period": period,
                        "patch_number": n_patches,
                        "mean_patch_area_km2": f"{np.mean(patch_areas):.2f}" if patch_areas else "nan",
                        "largest_patch_area_km2": f"{np.max(patch_areas):.2f}" if patch_areas else "nan",
                        "largest_patch_index_pct": f"{np.max(patch_areas)/future_area*100:.2f}" if patch_areas and future_area > 0 else "nan",
                        "patch_density_per_1000km2": f"{n_patches/(future_area/1000):.4f}" if future_area > 0 else "nan",
                        "edge_density": str(edge_pixels),
                    })
                    logger.info(f"    Landscape: patches={n_patches}, mean_area={np.mean(patch_areas):.0f}km², largest={np.max(patch_areas):.0f}km²")
                else:
                    landscape_rows.append({
                        "gcm": gcm, "ssp": ssp, "period": period,
                        "patch_number": 0,
                        "mean_patch_area_km2": "nan", "largest_patch_area_km2": "nan",
                        "largest_patch_index_pct": "nan", "patch_density_per_1000km2": "nan",
                        "edge_density": "0",
                    })

    # ——— Save Area Statistics ———
    with open(AREA_DIR / "scenario_area_statistics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=area_rows[0].keys())
        w.writeheader(); w.writerows(area_rows)
    logger.info(f"Area stats saved ({len(area_rows)} scenarios)")

    # ——— Centroid Migration (from current centroid) ———
    # Current centroid
    cur_w_lons = lons[current_binary[valid_mask]]
    cur_w_lats = lats[current_binary[valid_mask]]
    cur_cx = cur_w_lons.mean()
    cur_cy = cur_w_lats.mean()
    logger.info(f"Current centroid: lon={cur_cx:.4f}, lat={cur_cy:.4f}")

    migration_rows = []
    for cr in centroid_rows:
        if cr["centroid_type"] != "weighted":
            continue
        try:
            cx = float(cr["longitude"])
            cy = float(cr["latitude"])
            dist = haversine_km(cur_cx, cur_cy, cx, cy)
            brg = bearing_deg(cur_cx, cur_cy, cx, cy)
            card = bearing_to_cardinal(brg)
            migration_rows.append({
                "gcm": cr["gcm"], "ssp": cr["ssp"], "period": cr["period"],
                "current_lon": f"{cur_cx:.6f}", "current_lat": f"{cur_cy:.6f}",
                "future_lon": cr["longitude"], "future_lat": cr["latitude"],
                "distance_km": f"{dist:.2f}", "bearing_deg": f"{brg:.1f}",
                "direction": card,
            })
        except (ValueError, TypeError):
            pass

    with open(CENTROID_DIR / "centroid_by_scenario.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=centroid_rows[0].keys())
        w.writeheader(); w.writerows(centroid_rows)

    with open(CENTROID_DIR / "centroid_migration.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=migration_rows[0].keys())
        w.writeheader(); w.writerows(migration_rows)
    logger.info(f"Centroid migration saved ({len(migration_rows)} scenarios)")

    # ——— Save Elevation ———
    if elev_rows:
        with open(ELEV_DIR / "elevation_distribution_by_scenario.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=elev_rows[0].keys())
            w.writeheader(); w.writerows(elev_rows)
        logger.info(f"Elevation stats saved ({len(elev_rows)} entries)")

    # ——— Save Landscape ———
    if landscape_rows:
        with open(LANDSCAPE_DIR / "landscape_metrics_by_scenario.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=landscape_rows[0].keys())
            w.writeheader(); w.writerows(landscape_rows)
        logger.info(f"Landscape metrics saved ({len(landscape_rows)} scenarios)")

    # ——— GCM Summary ———
    logger.info("\n=== GCM Summary ===")
    summary_rows = []
    for ssp in SSPS:
        for period in PERIODS:
            ssp_area_rows = [r for r in area_rows if r["ssp"] == ssp and r["period"] == period]
            if len(ssp_area_rows) < 3:
                continue
            future_areas = [float(r["future_area_km2"]) for r in ssp_area_rows]
            gains = [float(r["gain_pct"]) for r in ssp_area_rows]
            losses = [float(r["loss_pct"]) for r in ssp_area_rows]
            stables = [float(r["persistence_pct"]) for r in ssp_area_rows]
            nets = [float(r["net_change_km2"]) for r in ssp_area_rows]

            summary_rows.append({
                "ssp": ssp, "period": period, "n_gcms": len(ssp_area_rows),
                "future_area_mean_km2": f"{np.mean(future_areas):.2f}",
                "future_area_median_km2": f"{np.median(future_areas):.2f}",
                "future_area_sd_km2": f"{np.std(future_areas):.2f}",
                "future_area_min_km2": f"{np.min(future_areas):.2f}",
                "future_area_max_km2": f"{np.max(future_areas):.2f}",
                "gain_pct_mean": f"{np.mean(gains):.2f}",
                "loss_pct_mean": f"{np.mean(losses):.2f}",
                "persistence_pct_mean": f"{np.mean(stables):.2f}",
                "net_change_mean_km2": f"{np.mean(nets):.2f}",
            })

    with open(AREA_DIR / "GCM_summary_area_statistics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader(); w.writerows(summary_rows)
    logger.info(f"GCM summary saved ({len(summary_rows)} rows)")

    # ——— SSP Comparison & Gradient Test ———
    logger.info("\n=== SSP Gradient Test ===")
    from scipy.stats import spearmanr
    gradient_rows = []
    ssp_order = {"ssp126": 1, "ssp245": 2, "ssp370": 3, "ssp585": 4}

    for gcm in GCMS:
        for period in PERIODS:
            gcm_rows = [r for r in area_rows if r["gcm"] == gcm and r["period"] == period]
            if len(gcm_rows) < 4:
                continue
            gcm_rows.sort(key=lambda r: ssp_order.get(r["ssp"], 0))
            ssp_vals = [ssp_order[r["ssp"]] for r in gcm_rows]
            loss_vals = [float(r["loss_pct"]) for r in gcm_rows]
            gain_vals = [float(r["gain_pct"]) for r in gcm_rows]
            net_vals = [abs(float(r["net_change_km2"])) for r in gcm_rows]

            for metric_name, vals in [("loss_rate", loss_vals), ("gain_rate", gain_vals),
                                       ("net_change_magnitude", net_vals)]:
                if len(set(vals)) > 1:
                    rho, p = spearmanr(ssp_vals, vals)
                else:
                    rho, p = np.nan, np.nan
                gradient_rows.append({
                    "gcm": gcm, "period": period, "metric": metric_name,
                    "spearman_rho": f"{rho:.4f}" if not np.isnan(rho) else "nan",
                    "p_value": f"{p:.4f}" if not np.isnan(p) else "nan",
                })

    if gradient_rows:
        with open(COMPARE_DIR / "SSP_gradient_test.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=gradient_rows[0].keys())
            w.writeheader(); w.writerows(gradient_rows)
        logger.info(f"SSP gradient test saved ({len(gradient_rows)} entries)")
    else:
        logger.info("SSP gradient test: insufficient data (need all 4 SSPs per GCM x period)")

    # ——— Generate tables ———
    logger.info("\n=== Generating Tables ===")
    # Table E4-1: Scenario registry
    with open(TABLES_DIR / "E4_Table1_scenario_registry.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gcm","ssp","period","dynamic_variables","static_variables","projection_status"])
        w.writeheader()
        for gcm in GCMS:
            for ssp in SSPS:
                for period in PERIODS:
                    ens_path = ENSEMBLE_DIR / gcm / ssp / period / "ensemble_suitability.tif"
                    w.writerow({
                        "gcm": gcm, "ssp": ssp, "period": period,
                        "dynamic_variables": "bio02,bio03,bio05,bio15",
                        "static_variables": "clay,elevation,northness,sand",
                        "projection_status": "complete" if ens_path.exists() else "pending",
                    })

    logger.info("\n" + "=" * 60)
    logger.info("Analysis pipeline complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
