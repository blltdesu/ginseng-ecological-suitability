"""Step 8: Process SoilGrids data - unit detection, depth processing, and alignment."""
import os
import csv
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from datetime import datetime

RAW_SOIL = r"E:\人参种在哪\数据\土壤数据"
OUT_DIR = r"E:\人参种在哪\00_统一数据预处理\06_soil"
ALIGN_DIR = os.path.join(OUT_DIR, "aligned")
REF_META_PATH = r"E:\人参种在哪\00_统一数据预处理\09_reference_grid\reference_grid_metadata.json"
LOG_PATH = r"E:\人参种在哪\00_统一数据预处理\logs\soil.log"
QC_DIR = r"E:\人参种在哪\00_统一数据预处理\11_qc"

os.makedirs(ALIGN_DIR, exist_ok=True)
os.makedirs(QC_DIR, exist_ok=True)

log_lines = []
log_lines.append(f"[{datetime.now().isoformat()}] SOIL PROCESSING STARTED\n")

# SoilGrids official units and conversion factors for storage integers
# SoilGrids 2.0 stores values as integers with scale factors:
# pH (phh2o): stored as pH*10 → divide by 10
# SOC: stored as dg/kg → divide by 10 for g/kg
# CEC: stored as mmol(c)/kg → divide by 10 for cmol(c)/kg
# Clay: stored as % → divide by 10
# Sand: stored as % → divide by 10
# Nitrogen: stored as cg/kg → divide by 100 for g/kg... actually it's dg/kg → divide by 10
# BDOD: stored as kg/m3 → divide by 100 for kg/dm3

# Official SoilGrids 2.0 scale factors (what to divide by to get target unit):
SOIL_VARIABLES = {
    "phh2o": {"target_unit": "pH", "scale_divisor": 10, "description": "Soil pH (H2O)"},
    "soc": {"target_unit": "g/kg", "scale_divisor": 10, "description": "Soil Organic Carbon"},
    "cec": {"target_unit": "cmol(c)/kg", "scale_divisor": 10, "description": "Cation Exchange Capacity"},
    "clay": {"target_unit": "%", "scale_divisor": 10, "description": "Clay Content"},
    "sand": {"target_unit": "%", "scale_divisor": 10, "description": "Sand Content"},
    "nitrogen": {"target_unit": "g/kg", "scale_divisor": 100, "description": "Soil Nitrogen"},
    "bdod": {"target_unit": "kg/dm3", "scale_divisor": 100, "description": "Bulk Density"},
}


def log(msg):
    print(msg)
    log_lines.append(f"[{datetime.now().isoformat()}] {msg}\n")


def detect_soil_files():
    """Find all SoilGrids TIFF files."""
    files = {}
    for f in os.listdir(RAW_SOIL):
        if not f.endswith(".tif"):
            continue
        # e.g., SoilGrids_phh2o_mean_EastAsia.tif
        for var in SOIL_VARIABLES:
            if var in f.lower():
                files[var] = os.path.join(RAW_SOIL, f)
                break
    return files


def process_soil_variable(var_name, src_path, ref_meta):
    """Process a single soil variable: read depths, compute 0-15cm, align."""
    log(f"\nProcessing {var_name}...")

    with rasterio.open(src_path) as src:
        n_bands = src.count
        band_descs = [src.descriptions[i] for i in range(n_bands)]
        crs = str(src.crs)
        shape = src.shape
        src_transform = src.transform

        log(f"  Bands: {n_bands}, descriptions: {band_descs}")

        # Parse depth bands
        # Expected: "var_0-5cm_mean", "var_5-15cm_mean", etc.
        depth_bands = {}
        for i, desc in enumerate(band_descs):
            if not desc:
                continue
            import re
            m = re.search(r"(\d+)-(\d+)cm", desc)
            if m:
                d0, d1 = int(m.group(1)), int(m.group(2))
                depth_bands[(d0, d1)] = i + 1  # 1-indexed band

        log(f"  Depth bands: {sorted(depth_bands.keys())}")

        # Check data values to detect if still in storage units
        # Read band 1 to check raw values
        data_sample = src.read(1, masked=True)
        raw_min = float(data_sample.min())
        raw_max = float(data_sample.max())
        raw_median = float(np.ma.median(data_sample))

        # Determine conversion
        scale_divisor = SOIL_VARIABLES[var_name]["scale_divisor"]
        target_unit = SOIL_VARIABLES[var_name]["target_unit"]

        # Detect if already converted: check value ranges
        detected_unit = "storage_int"
        conversion_needed = True
        decision_basis = "Default: assume SoilGrids storage integers"

        # Heuristic: if values look like they're already in target range
        if var_name == "phh2o" and raw_max < 15:
            detected_unit = "pH (already converted?)"
            conversion_needed = False
            decision_basis = f"Values in pH range [0-14]: max={raw_max}"
        elif var_name == "clay" and raw_max <= 100:
            detected_unit = "% (already converted?)"
            conversion_needed = False
            decision_basis = f"Values in percentage range [0-100]: max={raw_max}"
        elif var_name == "sand" and raw_max <= 100:
            detected_unit = "% (already converted?)"
            conversion_needed = False
            decision_basis = f"Values in percentage range [0-100]: max={raw_max}"

        factor = scale_divisor if conversion_needed else 1
        log(f"  Raw range: [{raw_min}, {raw_max}], median={raw_median}")
        log(f"  Conversion: {'divide by ' + str(factor) if conversion_needed else 'none needed'}")
        log(f"  Decision basis: {decision_basis}")

        # Compute 0-15 cm depth-weighted mean
        # Need bands: 0-5cm and 5-15cm
        has_0_5 = (0, 5) in depth_bands
        has_5_15 = (5, 15) in depth_bands

        if has_0_5 and has_5_15:
            band_0_5 = src.read(depth_bands[(0, 5)], masked=True).astype(np.float32)
            band_5_15 = src.read(depth_bands[(5, 15)], masked=True).astype(np.float32)
            # Weighted mean: (5 * 0-5 + 10 * 5-15) / 15
            result = (5 * band_0_5 + 10 * band_5_15) / 15.0
            depth_method = "weighted mean: (5*b0_5 + 10*b5_15)/15"
        elif has_0_5:
            result = src.read(depth_bands[(0, 5)], masked=True).astype(np.float32)
            depth_method = "0-5cm only (5-15cm not available)"
        elif has_5_15:
            result = src.read(depth_bands[(5, 15)], masked=True).astype(np.float32)
            depth_method = "5-15cm only (0-5cm not available)"
        else:
            # Use first band
            result = data_sample.astype(np.float32)
            depth_method = f"first band only: {band_descs[0]}"

        log(f"  Depth method: {depth_method}")

        # Apply conversion
        if conversion_needed:
            result = result / factor

        # Write unit decision
        unit_decision = {
            "variable": var_name,
            "source_file": src_path,
            "raw_min": raw_min,
            "raw_median": raw_median,
            "raw_max": raw_max,
            "detected_storage_unit": detected_unit,
            "target_unit": target_unit,
            "conversion_factor": factor,
            "conversion_applied": conversion_needed,
            "decision_basis": decision_basis,
        }

        return result, src_transform, crs, shape, unit_decision, depth_method


def align_to_reference(data, src_transform, src_crs, ref_meta, var_name):
    """Reproject and resample to match reference grid."""
    ref_transform = rasterio.transform.from_origin(
        ref_meta["bounds"]["left"],
        ref_meta["bounds"]["top"],
        ref_meta["resolution"][0],
        abs(ref_meta["resolution"][1]),
    )
    ref_width = ref_meta["width"]
    ref_height = ref_meta["height"]
    ref_crs = ref_meta["crs"]

    # Create output array
    dst_data = np.full((ref_height, ref_width), np.nan, dtype=np.float32)

    # Reproject
    reproject(
        source=data.filled(np.nan),
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
    )

    # Mask nodata
    dst_data = np.ma.masked_invalid(dst_data)

    return dst_data, ref_transform


def main():
    log("=" * 50)
    log("SOIL PROCESSING")

    # Load reference metadata
    import json
    with open(REF_META_PATH) as f:
        ref_meta = json.load(f)
    log(f"Reference grid: {ref_meta['width']}x{ref_meta['height']}, CRS={ref_meta['crs']}")

    # Detect soil files
    soil_files = detect_soil_files()
    log(f"Found {len(soil_files)} soil variables: {list(soil_files.keys())}")

    unit_decisions = []
    qc_records = []
    depth_registry = []

    for var_name, src_path in soil_files.items():
        result, src_transform, src_crs, src_shape, unit_decision, depth_method = \
            process_soil_variable(var_name, src_path, ref_meta)

        unit_decisions.append(unit_decision)

        # Align to reference
        aligned, dst_transform = align_to_reference(
            result, src_transform, src_crs, ref_meta, var_name
        )

        # Write aligned GeoTIFF
        out_path = os.path.join(ALIGN_DIR, f"{var_name}_0_15cm.tif")
        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=ref_meta["height"],
            width=ref_meta["width"],
            count=1,
            dtype="float32",
            crs=ref_meta["crs"],
            transform=dst_transform,
            nodata=-9999,
            compress="lzw",
        ) as dst:
            dst.write(aligned.filled(-9999).astype(np.float32), 1)

        log(f"  Saved: {out_path}")

        # QC stats
        valid = aligned.compressed()
        qc_records.append({
            "variable": var_name,
            "unit": unit_decision["target_unit"],
            "min": round(float(valid.min()), 4),
            "q01": round(float(np.percentile(valid, 1)), 4),
            "median": round(float(np.ma.median(aligned)), 4),
            "mean": round(float(aligned.mean()), 4),
            "q99": round(float(np.percentile(valid, 99)), 4),
            "max": round(float(valid.max()), 4),
            "nodata_pct": round(float((aligned.count() / aligned.size * 100) - 100) * -1, 2),
            "conversion_applied": unit_decision["conversion_applied"],
            "depth_processing": depth_method,
        })

        depth_registry.append({
            "variable": var_name,
            "depth_processing": depth_method,
            "source_bands": "0-5cm + 5-15cm",
            "conversion": f"divide by {unit_decision['conversion_factor']}" if unit_decision["conversion_applied"] else "none",
        })

    # Save unit decisions
    unit_path = os.path.join(OUT_DIR, "soil_unit_decision.csv")
    with open(unit_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=unit_decisions[0].keys())
        writer.writeheader()
        writer.writerows(unit_decisions)
    log(f"Unit decisions saved: {unit_path}")

    # Save depth registry
    depth_path = os.path.join(OUT_DIR, "soil_depth_registry.csv")
    with open(depth_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=depth_registry[0].keys())
        writer.writeheader()
        writer.writerows(depth_registry)

    # Save QC
    qc_path = os.path.join(QC_DIR, "soil_qc.csv")
    with open(qc_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=qc_records[0].keys())
        writer.writeheader()
        writer.writerows(qc_records)
    log(f"QC saved: {qc_path}")

    # Log
    log_lines.append(f"[{datetime.now().isoformat()}] SOIL PROCESSING COMPLETED\n")
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    print(f"\nSoil processing complete. Output: {ALIGN_DIR}")


if __name__ == "__main__":
    main()
