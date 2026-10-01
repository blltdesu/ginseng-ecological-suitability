"""Build Fig4B plotting material package (cross-scenario SD)."""
import shutil, json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path

EXP5 = Path(r'E:\人参种在哪\实验5')
INP = EXP5 / '00_input_from_experiment4'
PKG = Path(r'E:\人参种在哪\绘图\Fig4B_绘图素材包')
DATA = PKG / 'data'
NOTES = PKG / 'notes'
DATA.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

SD_PATH = EXP5 / '05_uncertainty_components/scenario_sd.tif'
LON0, LON1, LAT0, LAT1 = 95.0, 150.0, 15.0, 55.0

# --- 1. Global stats ---
with rasterio.open(SD_PATH) as src:
    sd = src.read(1).astype(np.float32)
    nodata = src.nodata
    transform = src.transform
    profile = src.profile.copy()
    if nodata is not None:
        sd[sd == nodata] = np.nan

valid = ~np.isnan(sd)
v = sd[valid]
print('Valid pixels:', int(valid.sum()))
print('Range: %.6f - %.6f' % (v.min(), v.max()))
print('Mean: %.6f  Median: %.6f' % (v.mean(), np.median(v)))
for q in (50, 75, 90, 95, 99, 99.5, 99.9):
    print('P%.1f: %.6f' % (q, np.percentile(v, q)))

# --- 2. Current-suitable stats + overlay source ---
with rasterio.open(INP/'current_baseline/current_binary_suitability.tif') as src:
    cur = src.read(1).astype(np.float32); cur[cur==src.nodata] = np.nan
cs = cur == 1
print('\nOn 167 current-suitable pixels: min=%.6f med=%.6f max=%.6f'
      % (np.nanmin(sd[cs]), np.nanmedian(sd[cs]), np.nanmax(sd[cs])))

# --- 3. Study-window stats ---
r0 = int((LAT1 - transform.f) / transform.e); r1 = int((LAT0 - transform.f) / transform.e)
c0 = int((LON0 - transform.c) / transform.a); c1 = int((LON1 - transform.c) / transform.a)
win = sd[r0:r1, c0:c1]
wv = win[~np.isnan(win)]
print('\nStudy window %s: valid=%d mean=%.6f P99=%.6f max=%.6f'
      % (win.shape, wv.size, wv.mean(), np.percentile(wv, 99), wv.max()))

# --- 4. Copy + crop files ---
copies = [
    (SD_PATH, DATA/'scenario_sd_global.tif'),
    (INP/'current_baseline/current_binary_suitability.tif', DATA/'current_binary_suitability.tif'),
    (INP/'current_baseline/common_valid_mask.tif', DATA/'common_valid_mask.tif'),
    (EXP5/'02_scenario_registry/scenario_availability.csv', DATA/'scenario_availability.csv'),
    (EXP5/'02_scenario_registry/experiment5_params.json', DATA/'experiment5_params.json'),
]
for src, dst in copies:
    shutil.copy2(src, dst)
    print(f'copied: {dst.name} ({dst.stat().st_size/1024/1024:.1f} MB)')

with rasterio.open(SD_PATH) as src:
    w = from_bounds(LON0, LAT0, LON1, LAT1, src.transform).round_offsets().round_lengths()
    sub = src.read(1, window=w)
    cprof = src.profile.copy()
    cprof.update(width=int(w.width), height=int(w.height), transform=src.window_transform(w))
crop_path = DATA/'scenario_sd_study_domain.tif'
with rasterio.open(crop_path, 'w', **cprof) as dst:
    dst.write(sub, 1)
print(f'cropped: {crop_path.name} {sub.shape} ({crop_path.stat().st_size/1024/1024:.1f} MB)')

# --- 5. Statistics JSON ---
sv = sub.astype(np.float32)
if nodata is not None:
    sv[sv == nodata] = np.nan
svv = sv[~np.isnan(sv)]
stats = {
    "raster": "scenario_sd",
    "definition": "per-pixel standard deviation of ensemble suitability across the 36 available future scenarios (5 GCMs x 4 SSPs x 2 periods, 4 missing)",
    "interpretation_caution": "Cross-scenario SD measures spread among scenario predictions (epistemic/disagreement uncertainty). It is NOT a risk metric and NOT the probability of habitat loss.",
    "denominator": {"available_scenarios": 36, "designed_scenarios": 40,
                     "note": "4 scenarios unavailable from CMIP6 were excluded, not interpolated"},
    "grid": {"shape": [4320, 8640], "resolution_deg": 0.04, "crs": "EPSG:4326"},
    "global": {
        "valid_pixels": int(valid.sum()),
        "min": float(v.min()), "max": float(v.max()),
        "mean": float(v.mean()), "median": float(np.median(v)),
        "p50": float(np.percentile(v, 50)), "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)), "p995": float(np.percentile(v, 99.5)),
        "p999": float(np.percentile(v, 99.9)),
    },
    "current_suitable_pixels": {
        "n": int(cs.sum()),
        "sd_min": float(np.nanmin(sd[cs])), "sd_median": float(np.nanmedian(sd[cs])),
        "sd_max": float(np.nanmax(sd[cs])),
    },
    "study_domain_window": {
        "extent_lon": [LON0, LON1], "extent_lat": [LAT0, LAT1],
        "shape": list(sub.shape),
        "valid_pixels": int(svv.size),
        "mean": float(svv.mean()),
        "p99": float(np.percentile(svv, 99)),
        "max": float(svv.max()),
    },
    "display_recommendation": {
        "colorbar_upper_limit": "P99 of displayed domain (truncate display only)",
        "true_max": float(v.max()),
        "note": "keep true max 0.3467 in caption or as a colorbar extend arrow; do not truncate the underlying statistics",
    },
}
with open(DATA/'scenario_sd_statistics.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print('wrote: scenario_sd_statistics.json')
print('DONE')
