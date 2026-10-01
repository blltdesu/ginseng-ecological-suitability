"""Build Fig4D plotting material package (Prediction Confidence + MESS inset)."""
import shutil, json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path

EXP5 = Path(r'E:\人参种在哪\实验5')
INP = EXP5 / '00_input_from_experiment4'
PKG = Path(r'E:\人参种在哪\绘图\Fig4D_绘图素材包')
DATA = PKG / 'data'
DATA.mkdir(parents=True, exist_ok=True)

CONF_PATH = EXP5 / '10_climate_vulnerability/prediction_confidence.tif'
MESS_PATH = EXP5 / '07_environmental_novelty/MESS_summary.tif'
NOV_PATH = EXP5 / '07_environmental_novelty/novelty_frequency.tif'
CUR_PATH = INP / 'current_baseline/current_binary_suitability.tif'
MASK_PATH = INP / 'current_baseline/common_valid_mask.tif'
LON0, LON1, LAT0, LAT1 = 95.0, 150.0, 15.0, 55.0

def load(p):
    with rasterio.open(p) as src:
        d = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None:
            d[d == nd] = np.nan
        tr = src.transform
        prof = src.profile.copy()
    return d, tr, prof

conf, transform, _ = load(CONF_PATH)
mess, _, _ = load(MESS_PATH)
nov, _, _ = load(NOV_PATH)
cur, _, _ = load(CUR_PATH)
mask, _, _ = load(MASK_PATH)
cs = (cur == 1) & (mask > 0)
mv = mask > 0

print('=== Prediction confidence ===')
cv = conf[~np.isnan(conf)]
print(f'valid={cv.size} min={cv.min():.4f} med={np.median(cv):.4f} max={cv.max():.4f} mean={cv.mean():.4f}')
print(f'  on valid mask: med={np.nanmedian(conf[mv]):.4f} mean={np.nanmean(conf[mv]):.4f}')
print(f'  on 167 suitable: min={np.nanmin(conf[cs]):.4f} med={np.nanmedian(conf[cs]):.4f} max={np.nanmax(conf[cs]):.4f}')
for q in (1, 5, 25, 75, 95, 99):
    print(f'  P{q} (mask): {np.nanpercentile(conf[mv], q):.4f}')

print('\n=== MESS summary ===')
msv = mess[~np.isnan(mess)]
print(f'valid={msv.size} min={msv.min():.4f} med={np.median(msv):.4f} max={msv.max():.4f}')
print(f'  on valid mask: min={np.nanmin(mess[mv]):.4f} med={np.nanmedian(mess[mv]):.4f} max={np.nanmax(mess[mv]):.4f}')
print(f'  on 167 suitable: min={np.nanmin(mess[cs]):.4f} med={np.nanmedian(mess[cs]):.4f} max={np.nanmax(mess[cs]):.4f}')
for q in (1, 5, 25, 75):
    print(f'  P{q} (mask): {np.nanpercentile(mess[mv], q):.4f}')

print('\n=== Novelty frequency (proxy) ===')
nvv = nov[~np.isnan(nov)]
print(f'valid={nvv.size} min={nvv.min():.4f} max={nvv.max():.4f} unique={np.unique(nvv).tolist()[:5]}')

# study window stats
r0 = int((LAT1 - transform.f) / transform.e); r1 = int((LAT0 - transform.f) / transform.e)
c0 = int((LON0 - transform.c) / transform.a); c1 = int((LON1 - transform.c) / transform.a)
cw = conf[r0:r1, c0:c1]; mw = mess[r0:r1, c0:c1]
cwv = cw[~np.isnan(cw)]; mwv = mw[~np.isnan(mw)]
print(f'\nStudy window conf: valid={cwv.size} mean={cwv.mean():.4f} P1={np.percentile(cwv,1):.4f}')
print(f'Study window MESS: valid={mwv.size} mean={mwv.mean():.4f}')

# --- copies ---
copies = [
    (CONF_PATH, DATA/'prediction_confidence_global.tif'),
    (MESS_PATH, DATA/'MESS_summary_global.tif'),
    (NOV_PATH, DATA/'novelty_frequency_global.tif'),
    (CUR_PATH, DATA/'current_binary_suitability.tif'),
    (MASK_PATH, DATA/'common_valid_mask.tif'),
    (EXP5/'02_scenario_registry/scenario_availability.csv', DATA/'scenario_availability.csv'),
]
for src, dst in copies:
    shutil.copy2(src, dst)
    print(f'copied: {dst.name} ({dst.stat().st_size/1024/1024:.1f} MB)')

# --- study-domain crops (confidence main + MESS inset, same window) ---
for src_path, out_name in [(CONF_PATH, 'prediction_confidence_study_domain.tif'),
                            (MESS_PATH, 'MESS_summary_study_domain.tif')]:
    with rasterio.open(src_path) as src:
        w = from_bounds(LON0, LAT0, LON1, LAT1, src.transform).round_offsets().round_lengths()
        sub = src.read(1, window=w)
        cp = src.profile.copy()
        cp.update(width=int(w.width), height=int(w.height), transform=src.window_transform(w))
    with rasterio.open(DATA/out_name, 'w', **cp) as dst:
        dst.write(sub, 1)
    print(f'cropped: {out_name} {sub.shape} ({(DATA/out_name).stat().st_size/1024/1024:.1f} MB)')

# --- statistics JSON ---
stats = {
    "main_layer": "prediction_confidence",
    "inset_layer": "MESS_summary",
    "definition": {
        "prediction_confidence": "1 - P1/P99-normalized cross-scenario SD; 1 = scenarios agree strongly, 0 = maximal disagreement (within observed range)",
        "MESS_summary": "MESS-like multivariate environmental similarity to the training environment; higher = closer to training conditions. NOTE: computed via a conservative proxy because future climate rasters were not available to Experiment 5",
        "novelty_frequency": "fraction of scenarios flagged as environmentally novel; current conservative CV-based proxy yields 0 everywhere - a zero here does NOT prove absence of extrapolation risk",
    },
    "grid": {"shape": [4320, 8640], "resolution_deg": 0.04, "crs": "EPSG:4326"},
    "confidence": {
        "global": {"min": float(cv.min()), "median": float(np.median(cv)),
                    "max": float(cv.max()), "mean": float(cv.mean())},
        "valid_mask": {"median": float(np.nanmedian(conf[mv])), "mean": float(np.nanmean(conf[mv])),
                        "p01": float(np.nanpercentile(conf[mv], 1)),
                        "p99": float(np.nanpercentile(conf[mv], 99))},
        "current_suitable": {"n": int(cs.sum()),
                              "min": float(np.nanmin(conf[cs])),
                              "median": float(np.nanmedian(conf[cs])),
                              "max": float(np.nanmax(conf[cs]))},
    },
    "MESS": {
        "global": {"min": float(msv.min()), "median": float(np.median(msv)), "max": float(msv.max())},
        "valid_mask": {"min": float(np.nanmin(mess[mv])), "median": float(np.nanmedian(mess[mv])),
                        "max": float(np.nanmax(mess[mv]))},
        "current_suitable": {"min": float(np.nanmin(mess[cs])),
                              "median": float(np.nanmedian(mess[cs])),
                              "max": float(np.nanmax(mess[cs]))},
    },
    "novelty_frequency": {"min": float(nvv.min()), "max": float(nvv.max()),
                           "note": "all zeros under conservative proxy; use in caption only, not as a main panel"},
    "study_domain_window": {"extent_lon": [LON0, LON1], "extent_lat": [LAT0, LAT1]},
}
with open(DATA/'confidence_MESS_statistics.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print('wrote: confidence_MESS_statistics.json')
print('DONE')
