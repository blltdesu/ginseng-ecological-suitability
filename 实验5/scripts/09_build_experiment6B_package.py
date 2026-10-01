"""Build E:\\人参种在哪\\实验6B transfer package with corrected products."""
import shutil, json, hashlib
from pathlib import Path
from datetime import datetime

EXP5 = Path(r'E:\人参种在哪\实验5')
INP = EXP5 / '00_input_from_experiment4'
OLD_HANDOFF = Path(r'E:\人参种在哪\实验6\00_input_from_experiment5')
EXP6B = Path(r'E:\人参种在哪\实验6B')
DST = EXP6B / '00_input_from_experiment5'

def sha256_file(fp):
    h = hashlib.sha256()
    with open(fp, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

# --- directory structure ---
for sub in ['current', 'future_stability', 'uncertainty', 'vulnerability',
            'novelty', 'landcover', 'reference', 'experiment5_results']:
    (DST / sub).mkdir(parents=True, exist_ok=True)

# --- file transfers (source, dest_subdir) ---
transfers = [
    # current baseline (from experiment 4 input)
    (INP/'current_baseline/current_ensemble_suitability.tif', 'current'),
    (INP/'current_baseline/current_binary_suitability.tif', 'current'),
    (INP/'current_baseline/ensemble_threshold.json', 'current'),
    # CORRECTED future stability products
    (EXP5/'08_stability_probability/future_stability_index.tif', 'future_stability'),
    (EXP5/'08_stability_probability/future_stability_classes.tif', 'future_stability'),
    (EXP5/'09_loss_probability/loss_frequency_current_suitable.tif', 'future_stability'),
    # uncertainty (not affected by encoding fix)
    (EXP5/'05_uncertainty_components/scenario_sd.tif', 'uncertainty'),
    (EXP5/'10_climate_vulnerability/prediction_confidence.tif', 'uncertainty'),
    (EXP5/'12_spatial_uncertainty_zones/high_uncertainty_zone.tif', 'uncertainty'),
    # CORRECTED vulnerability products
    (EXP5/'10_climate_vulnerability/vulnerability_base.tif', 'vulnerability'),
    (EXP5/'10_climate_vulnerability/vulnerability_classes.tif', 'vulnerability'),
    (EXP5/'10_climate_vulnerability/vulnerability_confidence_adjusted.tif', 'vulnerability'),
    (EXP5/'11_robust_core/robust_climatic_core.tif', 'vulnerability'),
    (EXP5/'11_robust_core/high_confidence_loss_zone.tif', 'vulnerability'),
    (EXP5/'12_spatial_uncertainty_zones/stability_confidence_quadrants.tif', 'vulnerability'),
    # novelty (not affected)
    (EXP5/'07_environmental_novelty/novelty_frequency.tif', 'novelty'),
    (EXP5/'07_environmental_novelty/MESS_summary.tif', 'novelty'),
    # landcover (from previous handoff, originally from unified preprocessing)
    (OLD_HANDOFF/'landcover/landcover_aligned.tif', 'landcover'),
    (OLD_HANDOFF/'landcover/landcover_class_dictionary.csv', 'landcover'),
    # reference
    (INP/'current_baseline/reference_grid_template.tif', 'reference'),
    (INP/'current_baseline/common_valid_mask.tif', 'reference'),
    (EXP5/'02_scenario_registry/scenario_availability.csv', 'reference'),
    (EXP5/'02_scenario_registry/experiment5_params.json', 'reference'),
    # results documentation
    (EXP5/'10_climate_vulnerability/vulnerability_summary.json', 'experiment5_results'),
    (Path(r'E:\人参种在哪\绘图\Fig4A_绘图素材包\notes\DATA_INTEGRITY_NOTE.md'), 'experiment5_results'),
]

manifest_rows = []
n_ok = 0
for src, sub in transfers:
    dst = DST / sub / src.name
    if not src.exists():
        print(f'MISSING: {src}')
        continue
    shutil.copy2(src, dst)
    h = sha256_file(dst)
    manifest_rows.append((str(dst.relative_to(DST)), h, dst.stat().st_size))
    n_ok += 1
    print(f'OK: {sub}/{src.name} ({dst.stat().st_size/1024/1024:.1f} MB)')

# --- operation manual at 实验6B root ---
manual_src = EXP5 / '实验6_长期稳定种植候选区识别_操作手册.md'
manual_dst = EXP6B / manual_src.name
shutil.copy2(manual_src, manual_dst)
print(f'\nManual copied: {manual_dst.name}')

# --- HANDOFF document ---
handoff_md = f"""# HANDOFF FROM EXPERIMENT 5 (CORRECTED)

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Target**: Experiment 6 (长期稳定种植候选区识别)
**Status**: CORRECTED HANDOFF — supersedes `实验6\\00_input_from_experiment5`

## Critical correction applied

The Experiment 4 `change_class.tif` encoding was verified to be
**1=Stable, 2=Loss, 3=Gain** (0=Non-habitat) — classes 1/2 are reversed
relative to the Experiment 5 operation manual. All change-map-derived
products in this handoff were recomputed with the corrected encoding and
independently validated against the binary-prediction-derived
`future_suitability_frequency` product (exact agreement on the 167
currently suitable pixels).

Recomputed products (2026-08-08):
- stable_frequency / loss_frequency / gain_frequency
- future_stability_index(.tif, classes.tif)
- loss_frequency_current_suitable.tif
- vulnerability_base / vulnerability_confidence_adjusted / vulnerability_classes
- robust_climatic_core / high_confidence_loss_zone
- stability_confidence_quadrants
- vulnerability_summary.json

Unaffected products (carried over unchanged):
- scenario_sd, prediction_confidence, high_uncertainty_zone
- novelty_frequency, MESS_summary
- current baseline, landcover, reference grids

## Corrected key results (167 currently suitable pixels)

| Metric | Corrected value | (Previous wrong value) |
|---|---|---|
| FSI median | 0.0000 (all 167 px "highly unstable", FSI<0.25) | 1.0000 |
| LFI median | 1.0000 (min 0.9722) | 0.0000 |
| Robust climatic core | 0 pixels | 7 pixels |
| High-confidence loss zone | 124 pixels (74.3% of suitable) | 0 pixels |
| Vulnerability (conf-adj) median | 0.1355 | 0.0000 |
| Prediction confidence median | 0.9063 | 0.9063 (unchanged) |

Interpretation: across the 36 available future scenarios, the current
ginseng habitat is projected to be lost with very high cross-scenario
consistency (only 2 of 36×167 scenario-pixels remain suitable). Note this
refers to the 0.04° pixels currently classified as suitable; Experiment 4
reported ~55–56% stability over a different statistical domain — this
discrepancy is exactly what Experiment 6 Step 2 (cross-experiment
consistency audit) must resolve before final zoning.

## Directory structure

- current/ — current ensemble suitability, binary, threshold
- future_stability/ — FSI, FSI classes, LFI (CORRECTED)
- uncertainty/ — scenario SD, prediction confidence, high-uncertainty zone
- vulnerability/ — vulnerability products, robust core, loss zone, quadrants (CORRECTED)
- novelty/ — novelty frequency, MESS summary
- landcover/ — MODIS-aligned landcover + class dictionary
- reference/ — grid template, valid mask, scenario registry, params
- experiment5_results/ — summary JSON + data integrity note

## QC

- SHA256 manifest: sha256_manifest.csv (this package, {len(manifest_rows)} files)
- All rasters: EPSG:4326, 0.04°, 4320×8640, aligned to reference_grid_template.tif
- FSI + LFI = 1 on all 167 suitable pixels (max deviation 0.0)
- dominant_uncertainty_source.tif and variance_component_summary.csv remain
  unavailable (Experiment 5 warning items; not blocking per manual section 3)
- Boundary (ADM) shapefiles were never available; Experiment 6 admin
  statistics will need an external source.

See experiment5_results/DATA_INTEGRITY_NOTE.md for the full evidence chain
of the encoding correction.
"""
(DST / 'HANDOFF_FROM_EXPERIMENT5.md').write_text(handoff_md, encoding='utf-8')
manifest_rows.append(('HANDOFF_FROM_EXPERIMENT5.md',
                      sha256_file(DST / 'HANDOFF_FROM_EXPERIMENT5.md'),
                      (DST / 'HANDOFF_FROM_EXPERIMENT5.md').stat().st_size))

# --- DATA DICTIONARY ---
dd = """# DATA DICTIONARY — EXPERIMENT 5 HANDOFF (CORRECTED, to Experiment 6B)

All rasters: GeoTIFF, EPSG:4326, 0.04° (4320×8640), float32 with NaN NoData
unless noted. Aligned to reference/reference_grid_template.tif.

| File | Description | Values |
|---|---|---|
| current/current_ensemble_suitability.tif | Current 4-model ensemble suitability | 0–1 |
| current/current_binary_suitability.tif | Current binary suitable (167 px = 1) | 0/1 |
| current/ensemble_threshold.json | Binarization threshold 0.12846759 | — |
| future_stability/future_stability_index.tif | FSI = stable frequency on current suitable (CORRECTED) | 0–1; median 0 |
| future_stability/future_stability_classes.tif | FSI classes 1–5 (CORRECTED; all 167 px = class 5) | 1–5 |
| future_stability/loss_frequency_current_suitable.tif | LFI = loss frequency on current suitable (CORRECTED) | 0–1; median 1 |
| uncertainty/scenario_sd.tif | Cross-scenario SD of ensemble suitability | ≥0 |
| uncertainty/prediction_confidence.tif | 1 − normalized scenario SD | 0–1 |
| uncertainty/high_uncertainty_zone.tif | SD ≥ P75 or novelty ≥ 0.5 (uint8, 255=NoData) | 0/1 |
| novelty/novelty_frequency.tif | Fraction of scenarios with novelty flags (proxy) | 0–1 (all 0) |
| novelty/MESS_summary.tif | MESS-like similarity to training environment | ~0.03–1.0 |
| vulnerability/vulnerability_base.tif | CurrentSuitability × LFI (CORRECTED) | 0–1 |
| vulnerability/vulnerability_confidence_adjusted.tif | CurrentSuitability × LFI × Confidence (CORRECTED) | 0–1 |
| vulnerability/vulnerability_classes.tif | Vulnerability classes 1–5 by percentiles (CORRECTED) | 1–5 |
| vulnerability/robust_climatic_core.tif | Robust core mask (CORRECTED; 0 pixels) | 0/1, 255=NoData |
| vulnerability/high_confidence_loss_zone.tif | LFI≥0.75 & Conf≥0.75 (CORRECTED; 124 px) | 0/1, 255=NoData |
| vulnerability/stability_confidence_quadrants.tif | Stability–confidence quadrants I–IV (CORRECTED) | 1–4 |
| landcover/landcover_aligned.tif | MODIS landcover, aligned (categorical) | class codes |
| landcover/landcover_class_dictionary.csv | Landcover class names | — |
| reference/reference_grid_template.tif | Grid definition template | — |
| reference/common_valid_mask.tif | Valid analysis domain | 0/1 |
| reference/scenario_availability.csv | 40-scenario registry (36 available) | — |
| reference/experiment5_params.json | Experiment 5 parameters | — |
| experiment5_results/vulnerability_summary.json | Corrected summary statistics | — |
| experiment5_results/DATA_INTEGRITY_NOTE.md | Encoding correction evidence | — |

Change-class encoding used for all CORRECTED products: 0=Non-habitat,
1=Stable, 2=Loss, 3=Gain (verified against binary predictions 2026-08-08).
"""
(DST / 'DATA_DICTIONARY_EXPERIMENT5.md').write_text(dd, encoding='utf-8')
manifest_rows.append(('DATA_DICTIONARY_EXPERIMENT5.md',
                      sha256_file(DST / 'DATA_DICTIONARY_EXPERIMENT5.md'),
                      (DST / 'DATA_DICTIONARY_EXPERIMENT5.md').stat().st_size))

# --- SHA256 manifest ---
with open(DST / 'sha256_manifest.csv', 'w', encoding='utf-8') as f:
    f.write('file,sha256,size_bytes\n')
    for rel, h, sz in manifest_rows:
        f.write(f'{rel},{h},{sz}\n')
print(f'\nsha256_manifest.csv: {len(manifest_rows)} files')

print(f'\n=== 实验6B package complete: {n_ok}/{len(transfers)} data files transferred ===')
print(f'Target: {EXP6B}')
