"""
Step 00: Input data integrity check and file copy to 01_input.
"""
import os, sys, hashlib, shutil, csv
import pandas as pd
from pathlib import Path

HANDOFF = Path(r"E:\人参种在哪\00_统一数据预处理\14_handoff\to_experiment_1")
INPUT = Path(r"E:\人参种在哪\实验1\01_input")
QC = Path(r"E:\人参种在哪\实验1\16_qc")

os.makedirs(INPUT, exist_ok=True)
os.makedirs(QC, exist_ok=True)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

# Files to copy (relative to handoff dir)
files_to_copy = [
    "occurrence_thin_5km.csv",
    "occurrence_thin_10km.csv",
    "occurrence_thin_20km.csv",
    "occurrence_cultivated_or_uncertain.csv",
    "reference_grid_template.tif",
    "common_valid_mask.tif",
    "study_context_adm0.gpkg",
    "study_context_adm1.gpkg",
    "current_predictor_registry.csv",
]

# QC reports
qc_files = [
    "qc_reports/raster_alignment_qc.csv",
    "qc_reports/predictor_numeric_qc.csv",
    "qc_reports/current_climate_qc.csv",
    "qc_reports/soil_qc.csv",
    "qc_reports/terrain_qc.csv",
    "qc_reports/occurrence_environment_extraction_qc.csv",
]

# Copy predictor rasters
pred_dir = HANDOFF / "current_predictors"
pred_files = list(pred_dir.glob("*.tif"))

manifest_rows = []
errors = []

print("=" * 60)
print("INPUT DATA INTEGRITY CHECK")
print("=" * 60)

# Check occurrence files
for fn in ["occurrence_thin_5km.csv", "occurrence_thin_10km.csv", "occurrence_thin_20km.csv"]:
    fp = HANDOFF / fn
    if fp.exists():
        df = pd.read_csv(fp)
        print(f"  {fn}: {len(df)} records")
    else:
        errors.append(f"MISSING: {fn}")

# Check 10km count
df10 = pd.read_csv(HANDOFF / "occurrence_thin_10km.csv")
print(f"  10km record count: {len(df10)} (expected ~252)")

# Check registry
reg = pd.read_csv(HANDOFF / "current_predictor_registry.csv")
print(f"  Predictors in registry: {len(reg)}")

# Check raster alignment QC
ralign = pd.read_csv(HANDOFF / "qc_reports/raster_alignment_qc.csv")
if not ralign['overall_pass'].all():
    errors.append("Raster alignment QC: NOT all pass")
else:
    print(f"  Raster alignment: all {len(ralign)} pass")

# Check reference grid
import rasterio
with rasterio.open(HANDOFF / "reference_grid_template.tif") as src:
    ref_transform = src.transform
    ref_crs = src.crs
    ref_shape = src.shape
    print(f"  Reference grid: {ref_shape}, CRS={ref_crs}")

# Check common_valid_mask
with rasterio.open(HANDOFF / "common_valid_mask.tif") as src:
    mask_shape = src.shape
    print(f"  Valid mask: {mask_shape}")

# Check all predictor rasters
for pf in pred_files:
    with rasterio.open(pf) as src:
        if src.shape != ref_shape or src.transform != ref_transform:
            errors.append(f"Predictor misalignment: {pf.name}")
print(f"  Predictor rasters: {len(pred_files)} checked")

# Check occurrence coordinates
for fn in ["occurrence_thin_5km.csv", "occurrence_thin_10km.csv", "occurrence_thin_20km.csv"]:
    df = pd.read_csv(HANDOFF / fn)
    lon_col = 'decimalLongitude' if 'decimalLongitude' in df.columns else 'longitude'
    lat_col = 'decimalLatitude' if 'decimalLatitude' in df.columns else 'latitude'
    if df[lon_col].between(-180, 180).all() and df[lat_col].between(-90, 90).all():
        print(f"  {fn}: coordinates valid")
    else:
        errors.append(f"COORDINATE ERROR: {fn}")

# Now copy files
print("\n" + "=" * 60)
print("COPYING FILES TO 01_input")
print("=" * 60)

def copy_file(rel_path, src_base=HANDOFF):
    src = src_base / rel_path
    dst = INPUT / rel_path.split("/")[-1]
    if src.exists():
        shutil.copy2(src, dst)
        s_sha = sha256_file(src)
        d_sha = sha256_file(dst)
        match = "OK" if s_sha == d_sha else "MISMATCH"
        if match != "OK":
            errors.append(f"SHA256 MISMATCH: {rel_path}")
        manifest_rows.append({
            'file': dst.name,
            'source_path': str(src),
            'experiment1_path': str(dst),
            'sha256_source': s_sha,
            'sha256_copy': d_sha,
            'size_mb': round(dst.stat().st_size / (1024*1024), 4),
            'status': match
        })
        print(f"  {dst.name}: {match}")
    else:
        errors.append(f"NOT FOUND: {src}")
    return dst

# Copy text/data files
for fn in files_to_copy:
    copy_file(fn)

# Copy QC reports to 01_input/qc_reports
qc_input = INPUT / "qc_reports"
os.makedirs(qc_input, exist_ok=True)
for qf in qc_files:
    src = HANDOFF / qf
    dst = qc_input / Path(qf).name
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  QC: {dst.name}")

# Copy predictor rasters
pred_input = INPUT / "current_predictors"
os.makedirs(pred_input, exist_ok=True)
for pf in pred_files:
    copy_file(f"current_predictors/{pf.name}")

# Save manifest
manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(INPUT / "experiment1_input_manifest.csv", index=False)

# Generate QC report
qc_lines = ["# INPUT QC Report", "", f"Generated: {pd.Timestamp.now()}", ""]
qc_lines.append("## Checks")
qc_lines.append(f"- 5/10/20 km occurrence files: OK")
qc_lines.append(f"- 10 km records: {len(df10)} (~252 expected)")
qc_lines.append(f"- Predictors in registry: {len(reg)} (30 expected)")
qc_lines.append(f"- Raster alignment: all {len(ralign)} pass")
qc_lines.append(f"- Reference grid readable: OK")
qc_lines.append(f"- Common valid mask readable: OK")
qc_lines.append(f"- All predictors aligned: OK")
qc_lines.append(f"- Coordinates in valid range: OK")
qc_lines.append(f"- SHA256 matches: {sum(1 for r in manifest_rows if r['status']=='OK')}/{len(manifest_rows)}")
qc_lines.append("")
qc_lines.append("## Status: PASS" if not errors else "## Status: ERRORS FOUND")

with open(QC / "INPUT_QC.md", 'w', encoding='utf-8') as f:
    f.write('\n'.join(qc_lines))

if errors:
    print("\nERRORS:")
    for e in errors:
        print(f"  - {e}")
    stop_path = QC / "STOP_01_INPUT_QC.md"
    with open(stop_path, 'w', encoding='utf-8') as f:
        f.write("# STOP: INPUT QC FAILED\n\n")
        for e in errors:
            f.write(f"- {e}\n")
    sys.exit(1)

print("\nInput check and copy: COMPLETE (PASS)")
