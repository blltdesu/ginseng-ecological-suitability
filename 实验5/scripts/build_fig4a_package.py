"""Build Fig4A plotting material package."""
import shutil, json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path

EXP5 = Path(r'E:\人参种在哪\实验5')
INP = EXP5 / '00_input_from_experiment4'
PKG = Path(r'E:\人参种在哪\绘图\Fig4A_绘图素材包')
DATA = PKG / 'data'
NOTES = PKG / 'notes'
DATA.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

# --- 1. Copy global rasters and metadata ---
copies = [
    (EXP5/'04_agreement/future_suitability_frequency.tif', DATA/'future_suitability_frequency_global.tif'),
    (INP/'current_baseline/current_binary_suitability.tif', DATA/'current_binary_suitability.tif'),
    (INP/'current_baseline/common_valid_mask.tif', DATA/'common_valid_mask.tif'),
    (EXP5/'02_scenario_registry/scenario_availability.csv', DATA/'scenario_availability.csv'),
    (EXP5/'02_scenario_registry/experiment5_params.json', DATA/'experiment5_params.json'),
]
for src, dst in copies:
    shutil.copy2(src, dst)
    print(f'copied: {dst.name} ({dst.stat().st_size/1024/1024:.1f} MB)')

# --- 2. Crop study-domain subset (East Asia) ---
LON0, LON1, LAT0, LAT1 = 95.0, 150.0, 15.0, 55.0
src_path = EXP5/'04_agreement/future_suitability_frequency.tif'
with rasterio.open(src_path) as src:
    win = from_bounds(LON0, LAT0, LON1, LAT1, src.transform)
    win = win.round_offsets().round_lengths()
    sub = src.read(1, window=win)
    profile = src.profile.copy()
    profile.update(width=int(win.width), height=int(win.height),
                   transform=src.window_transform(win))
crop_path = DATA/'future_suitability_frequency_study_domain.tif'
with rasterio.open(crop_path, 'w', **profile) as dst:
    dst.write(sub, 1)
print(f'cropped: {crop_path.name} {sub.shape} ({crop_path.stat().st_size/1024/1024:.1f} MB), extent lon[{LON0},{LON1}] lat[{LAT0},{LAT1}]')

# --- 3. Statistics ---
with rasterio.open(src_path) as src:
    freq = src.read(1).astype(np.float32); freq[freq==src.nodata] = np.nan
    transform = src.transform
with rasterio.open(INP/'current_baseline/current_binary_suitability.tif') as src:
    cur = src.read(1).astype(np.float32); cur[cur==src.nodata] = np.nan
cs = cur == 1

valid = ~np.isnan(freq)
stats = {
    "raster": "future_suitability_frequency",
    "definition": "fraction of available future scenarios (out of 36) in which the pixel is predicted suitable (ensemble suitability >= 0.12846759691320192)",
    "denominator": {"available_scenarios": 36, "designed_scenarios": 40,
                     "note": "per-pixel denominator = actual available scenarios; 4 missing scenarios are NOT counted as unsuitable"},
    "grid": {"shape": [4320, 8640], "resolution_deg": 0.04, "crs": "EPSG:4326"},
    "global": {
        "valid_pixels": int(valid.sum()),
        "min": float(np.nanmin(freq)), "max": float(np.nanmax(freq)),
        "mean": float(np.nanmean(freq)), "median": float(np.nanmedian(freq)),
        "pixels_freq_eq_0": int(np.nansum(freq == 0)),
        "pixels_freq_eq_1": int(np.nansum(freq == 1)),
        "pixels_freq_ge_0.5": int(np.nansum(freq >= 0.5)),
        "pixels_freq_ge_0.9": int(np.nansum(freq >= 0.9)),
    },
    "study_domain_window": {
        "extent_lon": [LON0, LON1], "extent_lat": [LAT0, LAT1],
        "basis": "167 currently-suitable pixels span lon 104.67-141.33, lat 29.92-48.25; window adds ~5-10 deg margin",
    },
    "current_suitable_pixels": {
        "n": int(cs.sum()),
        "freq_min": float(np.nanmin(freq[cs])),
        "freq_median": float(np.nanmedian(freq[cs])),
        "freq_max": float(np.nanmax(freq[cs])),
        "note": "verified against binary predictions: only 2 of 36x167 scenario-pixels remain suitable; near-total projected loss of current habitat",
    },
    "study_window_stats": None,
}
subf = freq.copy()
r0 = int((LAT1 - transform.f) / transform.e); r1 = int((LAT0 - transform.f) / transform.e)
c0 = int((LON0 - transform.c) / transform.a); c1 = int((LON1 - transform.c) / transform.a)
win_f = freq[r0:r1, c0:c1]
stats["study_window_stats"] = {
    "shape": list(win_f.shape),
    "valid_pixels": int((~np.isnan(win_f)).sum()),
    "mean": float(np.nanmean(win_f)),
    "pixels_freq_gt_0": int(np.nansum(win_f > 0)),
    "pixels_freq_ge_0.5": int(np.nansum(win_f >= 0.5)),
    "pixels_freq_ge_0.9": int(np.nansum(win_f >= 0.9)),
}
with open(DATA/'frequency_statistics.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print('wrote: frequency_statistics.json')
print('DONE')
