"""Build Fig4E plotting material package (Robust core & high-uncertainty, CORRECTED)."""
import shutil, json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path

EXP5 = Path(r'E:\人参种在哪\实验5')
INP = EXP5 / '00_input_from_experiment4'
PKG = Path(r'E:\人参种在哪\绘图\Fig4E_绘图素材包')
DATA = PKG / 'data'
NOTES = PKG / 'notes'
DATA.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

ROBUST_PATH = EXP5 / '11_robust_core/robust_climatic_core.tif'
HLOSS_PATH = EXP5 / '11_robust_core/high_confidence_loss_zone.tif'
HUNC_PATH = EXP5 / '12_spatial_uncertainty_zones/high_uncertainty_zone.tif'
CUR_PATH = INP / 'current_baseline/current_binary_suitability.tif'
MASK_PATH = INP / 'current_baseline/common_valid_mask.tif'

LON0, LON1, LAT0, LAT1 = 95.0, 150.0, 15.0, 55.0       # study domain (Fig4A/B/D)
ZLON0, ZLON1, ZLAT0, ZLAT1 = 101.0, 145.0, 26.0, 52.0  # suitable-domain zoom (Fig4C)

def load(p):
    with rasterio.open(p) as src:
        d = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None:
            d[d == nd] = np.nan
        tr = src.transform
    return d, tr

robust, transform = load(ROBUST_PATH)
hloss, _ = load(HLOSS_PATH)
hunc, _ = load(HUNC_PATH)
cur, _ = load(CUR_PATH)
mask, _ = load(MASK_PATH)
cs = (cur == 1) & (mask > 0)
mv = mask > 0

print('=== CORRECTED zone statistics (post 2026-08-08 encoding fix) ===')
n_robust = int(np.nansum(robust == 1))
n_hloss = int(np.nansum(hloss == 1))
n_hunc = int(np.nansum(hunc == 1))
print(f'Robust core: {n_robust} px (was 7 pre-correction)')
print(f'High-confidence loss zone: {n_hloss} px (was 0 pre-correction)')
print(f'High-uncertainty zone (global): {n_hunc} px')

# overlap checks
print(f'\nOverlap robust & high-loss: {int(np.nansum((robust==1)&(hloss==1)))}')
print(f'Overlap high-loss & current suitable: {int(np.nansum((hloss==1)&cs))} / {int(cs.sum())}')
print(f'High-unc within valid mask: {int(np.nansum((hunc==1)&mv))}')
print(f'High-unc overlapping 167 suitable px: {int(np.nansum((hunc==1)&cs))}')

# study-window high-uncertainty count
r0 = int((LAT1 - transform.f) / transform.e); r1 = int((LAT0 - transform.f) / transform.e)
c0 = int((LON0 - transform.c) / transform.a); c1 = int((LON1 - transform.c) / transform.a)
hw = hunc[r0:r1, c0:c1]
vw = mask[r0:r1, c0:c1]
print(f'\nHigh-unc in study window: {int(np.nansum(hw==1))} px; valid-mask px in window: {int(np.nansum(vw>0))}')

# --- copies (global rasters) ---
copies = [
    (ROBUST_PATH, DATA/'robust_climatic_core_global.tif'),
    (HLOSS_PATH, DATA/'high_confidence_loss_zone_global.tif'),
    (HUNC_PATH, DATA/'high_uncertainty_zone_global.tif'),
    (CUR_PATH, DATA/'current_binary_suitability.tif'),
    (MASK_PATH, DATA/'common_valid_mask.tif'),
    (EXP5/'10_climate_vulnerability/vulnerability_summary.json', DATA/'vulnerability_summary.json'),
    (EXP5/'02_scenario_registry/scenario_availability.csv', DATA/'scenario_availability.csv'),
]
for src, dst in copies:
    shutil.copy2(src, dst)
    print(f'copied: {dst.name} ({dst.stat().st_size/1024/1024:.1f} MB)')

# --- study-domain crop of high-uncertainty zone (main map) ---
with rasterio.open(HUNC_PATH) as src:
    w = from_bounds(LON0, LAT0, LON1, LAT1, src.transform).round_offsets().round_lengths()
    sub = src.read(1, window=w)
    cp = src.profile.copy()
    cp.update(width=int(w.width), height=int(w.height), transform=src.window_transform(w))
with rasterio.open(DATA/'high_uncertainty_zone_study_domain.tif', 'w', **cp) as dst:
    dst.write(sub, 1)
print(f'cropped: high_uncertainty_zone_study_domain.tif {sub.shape}')

# --- suitable-domain zoom crops (for the inset / detail view) ---
for src_path, out_name in [(ROBUST_PATH, 'robust_climatic_core_zoom.tif'),
                            (HLOSS_PATH, 'high_confidence_loss_zone_zoom.tif'),
                            (CUR_PATH, 'current_binary_suitability_zoom.tif')]:
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
    "CORRECTION_NOTICE": {
        "date": "2026-08-08",
        "issue": "change_class encoding verified as 1=Stable, 2=Loss (reverse of manual); products recomputed",
        "consequence": "robust_climatic_core: 7 -> 0 pixels; high_confidence_loss_zone: 0 -> 124 pixels. The panel story is inverted relative to the original briefing: there is NO robust climatic core, and most of the current habitat is a high-confidence loss zone.",
        "evidence_file": "notes/DATA_INTEGRITY_NOTE.md",
    },
    "layers": {
        "robust_climatic_core": {
            "definition": "current suitable AND suitability >= median AND FSI >= 0.90 AND scenario SD <= P25 AND novelty frequency <= 0.10",
            "n_pixels": n_robust,
            "note": "0 after correction because FSI >= 0.90 fails on every suitable pixel (median FSI = 0)",
        },
        "high_confidence_loss_zone": {
            "definition": "current suitable AND LFI >= 0.75 AND prediction confidence >= 0.75",
            "n_pixels": n_hloss,
            "pct_of_current_suitable": n_hloss / int(cs.sum()) * 100,
        },
        "high_uncertainty_zone": {
            "definition": "scenario SD >= P75 (of current-suitable SD) OR novelty frequency >= 0.50; computed over the full valid mask - a GLOBAL-domain product, far larger than the 167-pixel FSI domain",
            "n_pixels_global": n_hunc,
            "n_pixels_valid_mask": int(np.nansum((hunc == 1) & mv)),
            "n_pixels_study_window": int(np.nansum(hw == 1)),
            "n_overlap_current_suitable": int(np.nansum((hunc == 1) & cs)),
        },
        "current_binary_suitability": {"n_pixels": int(cs.sum())},
    },
    "spatial_domains_caution": "High-uncertainty zone covers the full valid analysis domain (millions of pixels); robust core and high-confidence loss are defined only on the 167 currently suitable pixels. The caption must state each layer's spatial definition explicitly.",
    "extents": {
        "study_domain": {"lon": [LON0, LON1], "lat": [LAT0, LAT1]},
        "suitable_zoom": {"lon": [ZLON0, ZLON1], "lat": [ZLAT0, ZLAT1]},
    },
}
with open(DATA/'zones_statistics.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print('wrote: zones_statistics.json')

note_src = Path(r'E:\人参种在哪\绘图\Fig4A_绘图素材包\notes\DATA_INTEGRITY_NOTE.md')
if note_src.exists():
    shutil.copy2(note_src, NOTES/'DATA_INTEGRITY_NOTE.md')
    print('copied: notes/DATA_INTEGRITY_NOTE.md')
print('DONE')
