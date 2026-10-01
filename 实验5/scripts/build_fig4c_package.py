"""Build Fig4C plotting material package (FSI & LFI, CORRECTED values)."""
import shutil, json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path

EXP5 = Path(r'E:\人参种在哪\实验5')
INP = EXP5 / '00_input_from_experiment4'
PKG = Path(r'E:\人参种在哪\绘图\Fig4C_绘图素材包')
DATA = PKG / 'data'
NOTES = PKG / 'notes'
DATA.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

FSI_PATH = EXP5 / '08_stability_probability/future_stability_index.tif'
LFI_PATH = EXP5 / '09_loss_probability/loss_frequency_current_suitable.tif'
CUR_PATH = INP / 'current_baseline/current_binary_suitability.tif'

def load(p):
    with rasterio.open(p) as src:
        d = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None:
            d[d == nd] = np.nan
        prof = src.profile.copy()
        tr = src.transform
    return d, prof, tr

fsi, prof, transform = load(FSI_PATH)
lfi, _, _ = load(LFI_PATH)
cur, _, _ = load(CUR_PATH)
cs = cur == 1

print('=== CORRECTED FSI/LFI statistics (post 2026-08-08 encoding fix) ===')
fv = fsi[~np.isnan(fsi)]
lv = lfi[~np.isnan(lfi)]
print(f'FSI: valid={fv.size} min={fv.min():.6f} median={np.median(fv):.6f} max={fv.max():.6f} mean={fv.mean():.6f}')
print(f'LFI: valid={lv.size} min={lv.min():.6f} median={np.median(lv):.6f} max={lv.max():.6f} mean={lv.mean():.6f}')
print(f'FSI on 167 suitable: min={np.nanmin(fsi[cs]):.6f} median={np.nanmedian(fsi[cs]):.6f} max={np.nanmax(fsi[cs]):.6f}')
print(f'LFI on 167 suitable: min={np.nanmin(lfi[cs]):.6f} median={np.nanmedian(lfi[cs]):.6f} max={np.nanmax(lfi[cs]):.6f}')
print(f'FSI+LFI=1 check: {np.allclose(np.nansum([fsi[cs], lfi[cs]], axis=0), 1.0, atol=1e-5)}')

# distribution of the 167 pixels
uniq_f, cnt_f = np.unique(fsi[cs], return_counts=True)
print('FSI unique values on suitable:', dict(zip([round(float(u),4) for u in uniq_f], cnt_f.tolist())))

# bounding box of current suitable pixels (panel extent)
rows, cols = np.where(cs)
lons = transform.c + cols * transform.a
lats = transform.f + rows * transform.e
print(f'\nSuitable pixel extent: lon [{lons.min():.2f},{lons.max():.2f}] lat [{lats.min():.2f},{lats.max():.2f}]')

# --- copies ---
copies = [
    (FSI_PATH, DATA/'future_stability_index.tif'),
    (LFI_PATH, DATA/'loss_frequency_current_suitable.tif'),
    (CUR_PATH, DATA/'current_binary_suitability.tif'),
    (EXP5/'08_stability_probability/future_stability_classes.tif', DATA/'future_stability_classes.tif'),
    (EXP5/'10_climate_vulnerability/vulnerability_summary.json', DATA/'vulnerability_summary.json'),
    (EXP5/'02_scenario_registry/scenario_availability.csv', DATA/'scenario_availability.csv'),
]
for src, dst in copies:
    shutil.copy2(src, dst)
    print(f'copied: {dst.name} ({dst.stat().st_size/1024/1024:.1f} MB)')

# --- zoom-window crops around the 167 pixels (main figure extent) ---
MARGIN = 3.0  # degrees
ZLON0, ZLON1 = float(np.floor(lons.min() - MARGIN)), float(np.ceil(lons.max() + MARGIN))
ZLAT0, ZLAT1 = float(np.floor(lats.min() - MARGIN)), float(np.ceil(lats.max() + MARGIN))
print(f'Zoom window: lon [{ZLON0},{ZLON1}] lat [{ZLAT0},{ZLAT1}]')

for src_path, out_name in [(FSI_PATH, 'FSI_zoom_window.tif'), (LFI_PATH, 'LFI_zoom_window.tif')]:
    with rasterio.open(src_path) as src:
        w = from_bounds(ZLON0, ZLAT0, ZLON1, ZLAT1, src.transform).round_offsets().round_lengths()
        sub = src.read(1, window=w)
        zp = src.profile.copy()
        zp.update(width=int(w.width), height=int(w.height), transform=src.window_transform(w))
    with rasterio.open(DATA/out_name, 'w', **zp) as dst:
        dst.write(sub, 1)
    print(f'cropped: {out_name} {sub.shape}')

# --- statistics JSON ---
stats = {
    "panels": ["FSI (left)", "LFI (right)"],
    "definition": {
        "FSI": "Future Stability Index = fraction of the 36 available future scenarios in which the pixel is classified Stable (change class 1) - computed on current-suitable pixels only",
        "LFI": "Loss Frequency Index = fraction of the 36 available future scenarios in which the pixel is classified Loss (change class 2) - computed on current-suitable pixels only",
    },
    "CORRECTION_NOTICE": {
        "date": "2026-08-08",
        "issue": "Experiment 4 change_class.tif encoding verified as 1=Stable, 2=Loss (reverse of the experiment 5 manual). Products recomputed with corrected encoding.",
        "consequence": "Earlier statement 'all 167 pixels FSI >= 0.90, median FSI = 1.000' was WRONG. Corrected: median FSI = 0.000, median LFI = 1.000.",
        "validation": "corrected FSI matches the independent binary-prediction-derived future_suitability_frequency exactly on all 167 pixels; FSI + LFI = 1 holds (max deviation 0.0)",
        "evidence_file": "notes/DATA_INTEGRITY_NOTE.md",
    },
    "statistical_domain_caution": "FSI/LFI are defined ONLY on the 167 currently suitable pixels (0.04 deg). Do not compare directly with Experiment 4's ~44-45% loss / ~55-56% stable figures, which come from a different (larger) statistical domain/mask. Caption must state the domain explicitly.",
    "domain": {
        "n_current_suitable_pixels": int(cs.sum()),
        "denominator_scenarios": 36,
        "suitable_extent_lon": [float(lons.min()), float(lons.max())],
        "suitable_extent_lat": [float(lats.min()), float(lats.max())],
    },
    "FSI_stats_on_suitable": {
        "min": float(np.nanmin(fsi[cs])), "median": float(np.nanmedian(fsi[cs])),
        "max": float(np.nanmax(fsi[cs])), "mean": float(np.nanmean(fsi[cs])),
        "n_eq_0": int(np.nansum(fsi[cs] == 0)), "n_gt_0": int(np.nansum(fsi[cs] > 0)),
    },
    "LFI_stats_on_suitable": {
        "min": float(np.nanmin(lfi[cs])), "median": float(np.nanmedian(lfi[cs])),
        "max": float(np.nanmax(lfi[cs])), "mean": float(np.nanmean(lfi[cs])),
        "n_eq_1": int(np.nansum(lfi[cs] == 1)), "n_lt_1": int(np.nansum(lfi[cs] < 1)),
    },
    "zoom_window": {"extent_lon": [ZLON0, ZLON1], "extent_lat": [ZLAT0, ZLAT1],
                     "margin_deg": MARGIN,
                     "note": "tight window around the 167 suitable pixels for the side-by-side panel figure"},
}
with open(DATA/'FSI_LFI_statistics.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print('wrote: FSI_LFI_statistics.json')

# copy evidence note from Fig4A package
note_src = Path(r'E:\人参种在哪\绘图\Fig4A_绘图素材包\notes\DATA_INTEGRITY_NOTE.md')
if note_src.exists():
    shutil.copy2(note_src, NOTES/'DATA_INTEGRITY_NOTE.md')
    print('copied: notes/DATA_INTEGRITY_NOTE.md')
print('DONE')
