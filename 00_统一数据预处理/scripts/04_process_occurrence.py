"""Step 4: Process GBIF occurrence records for Panax ginseng."""
import os
import csv
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
from collections import Counter

# Paths
RAW_DATA = r"E:\人参种在哪\数据"
EXTRACT_DIR = r"E:\人参种在哪\00_统一数据预处理\01_inventory\extracted"
OUT_DIR = r"E:\人参种在哪\00_统一数据预处理\02_occurrence"
LOG_PATH = r"E:\人参种在哪\00_统一数据预处理\logs\occurrence.log"

# Target species
TARGET_SPECIES = "Panax ginseng"
TARGET_GENUS = "Panax"
TARGET_EPITHET = "ginseng"
EXCLUDE_SPECIES = [
    "Panax quinquefolius", "Panax notoginseng",
    "Panax japonicus", "Panax vietnamensis"
]

RANDOM_SEED = 20260807

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

log_lines = []
log_lines.append(f"[{datetime.now().isoformat()}] OCCURRENCE PROCESSING STARTED\n")
stats = {}  # Track record counts at each stage


def log(msg):
    print(msg)
    log_lines.append(f"[{datetime.now().isoformat()}] {msg}\n")


def load_gbif_data():
    """Find and load the GBIF CSV file."""
    # Find the extracted GBIF CSV
    gbif_dirs = [d for d in os.listdir(EXTRACT_DIR) if d.startswith("0002333-")]
    if not gbif_dirs:
        log("ERROR: GBIF directory not found in extracted!")
        return None

    gbif_dir = os.path.join(EXTRACT_DIR, gbif_dirs[0])
    csv_files = [f for f in os.listdir(gbif_dir) if f.endswith(".csv")]
    if not csv_files:
        log("ERROR: No CSV in GBIF directory!")
        return None

    csv_path = os.path.join(gbif_dir, csv_files[0])
    log(f"Loading GBIF data from: {csv_path}")

    # Read with auto-detection
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8",
                     on_bad_lines="warn")
    stats["raw"] = len(df)
    log(f"Raw records: {len(df)}, columns: {len(df.columns)}")
    return df


def map_fields(df):
    """Standardize field names and create field mapping."""
    # Standard GBIF Darwin Core fields should be present
    # Create mapping of what we have to standard names
    field_mapping = {}
    expected_fields = [
        "gbifID", "scientificName", "acceptedScientificName", "taxonKey",
        "species", "genus", "decimalLongitude", "decimalLatitude",
        "coordinateUncertaintyInMeters", "countryCode", "stateProvince",
        "locality", "year", "eventDate", "basisOfRecord", "occurrenceStatus",
        "establishmentMeans", "occurrenceRemarks", "habitat",
        "datasetKey", "institutionCode", "collectionCode", "issues"
    ]

    for std in expected_fields:
        if std in df.columns:
            field_mapping[std] = std
        else:
            # Try case-insensitive match
            matches = [c for c in df.columns if c.lower() == std.lower()]
            if matches:
                field_mapping[std] = matches[0]
            else:
                field_mapping[std] = None

    # Save field mapping
    mapping_path = os.path.join(OUT_DIR, "occurrence_field_mapping.csv")
    with open(mapping_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["standard_field", "source_field", "present"])
        for std, src in field_mapping.items():
            writer.writerow([std, src, "Yes" if src else "No"])

    present = sum(1 for v in field_mapping.values() if v)
    log(f"Field mapping: {present}/{len(expected_fields)} standard fields matched")
    return field_mapping


def taxonomic_filter(df):
    """Filter for Panax ginseng only."""
    fname = "occurrence_taxonomic_check.csv"

    # Determine the best species column
    species_col = None
    for col in ["species", "scientificName"]:
        if col in df.columns:
            species_col = col
            break

    if species_col is None:
        log("WARNING: No species column found!")
        return df

    df["raw_taxon_name"] = df[species_col].astype(str).str.strip()

    # Normalize: remove author citation, standardize spaces
    df["normalized_taxon_name"] = df["raw_taxon_name"].apply(
        lambda x: "Panax ginseng" if "panax" in x.lower() and "ginseng" in x.lower() else x
    )

    # Check for target species
    is_target = df["raw_taxon_name"].str.lower().str.contains(
        "panax ginseng", na=False
    )
    # Exclude other Panax species
    is_other_panax = df["raw_taxon_name"].str.lower().str.contains(
        "quinquefolius|notoginseng|japonicus|vietnamensis", na=False, regex=True
    )

    df["taxon_status"] = "unknown"
    df.loc[is_target & ~is_other_panax, "taxon_status"] = "target"
    df.loc[is_other_panax, "taxon_status"] = "other_panax_species"
    df.loc[~is_target & ~is_other_panax, "taxon_status"] = "not_panax"

    df["taxon_keep"] = df["taxon_status"] == "target"

    log(f"Taxonomic filter: target={is_target.sum()}, "
        f"other Panax={is_other_panax.sum()}, other={len(df) - is_target.sum() - is_other_panax.sum()}")

    stats["taxonomy_retained"] = is_target.sum()
    return df


def coordinate_filter(df):
    """Filter records with valid coordinates."""
    lat_col = "decimalLatitude"
    lon_col = "decimalLongitude"

    if lat_col not in df.columns or lon_col not in df.columns:
        log("ERROR: Lat/lon columns not found!")
        return df

    # Convert to numeric
    df["decimalLatitude"] = pd.to_numeric(df["decimalLatitude"], errors="coerce")
    df["decimalLongitude"] = pd.to_numeric(df["decimalLongitude"], errors="coerce")

    # Flag issues
    df["coord_valid"] = True
    df.loc[df["decimalLatitude"].isna(), "coord_valid"] = False
    df.loc[df["decimalLongitude"].isna(), "coord_valid"] = False
    df.loc[df["decimalLatitude"].abs() > 90, "coord_valid"] = False
    df.loc[df["decimalLongitude"].abs() > 180, "coord_valid"] = False
    # Zero-zero coordinates
    df.loc[(df["decimalLatitude"] == 0) & (df["decimalLongitude"] == 0), "coord_valid"] = False

    log(f"Valid coordinates: {df['coord_valid'].sum()}/{len(df)}")
    stats["valid_coordinates"] = df["coord_valid"].sum()
    return df


def coordinate_quality(df):
    """Assess coordinate uncertainty."""
    unc_col = "coordinateUncertaintyInMeters"
    if unc_col not in df.columns or df[unc_col].isna().all():
        log("WARNING: coordinateUncertaintyInMeters not available")
        df["coordinate_quality"] = "unknown"
        return df

    df[unc_col] = pd.to_numeric(df[unc_col], errors="coerce")

    def classify_uncertainty(val):
        if pd.isna(val):
            return "unknown"
        if val <= 1000:
            return "high"
        elif val <= 5000:
            return "medium"
        elif val <= 10000:
            return "low"
        else:
            return "very_low"

    df["coordinate_quality"] = df[unc_col].apply(classify_uncertainty)

    qual_counts = Counter(df["coordinate_quality"])
    log(f"Coordinate quality: {dict(qual_counts)}")

    # Flag for main analysis: <=10000m or unknown
    df["coord_quality_pass"] = df["coordinate_quality"].isin(["high", "medium", "low", "unknown"])

    return df


def duplicate_filter(df):
    """Remove duplicate records."""
    # Remove exact coordinate duplicates
    dup_subset = df[df["coord_valid"]]
    dup_mask = dup_subset.duplicated(subset=["decimalLongitude", "decimalLatitude"], keep="first")
    dup_indices = dup_subset[dup_mask].index
    df["coord_duplicate"] = False
    df.loc[dup_indices, "coord_duplicate"] = True

    # Remove duplicate gbifID
    if "gbifID" in df.columns:
        gbif_dup = df["gbifID"].duplicated(keep="first")
        df.loc[gbif_dup, "coord_duplicate"] = True

    log(f"Coordinate duplicates: {df['coord_duplicate'].sum()}")
    return df


def occurrence_status_filter(df):
    """Filter by occurrenceStatus."""
    if "occurrenceStatus" not in df.columns:
        return df

    non_present = df["occurrenceStatus"].str.upper().str.contains("ABSENT", na=False)
    df["status_absent"] = non_present
    log(f"Non-PRESENT records: {non_present.sum()}")
    return df


def cultivation_flag(df):
    """Flag potentially cultivated/garden records."""
    cultivate_keywords = [
        "cultivated", "cultivation", "garden", "botanical garden",
        "farm", "plantation", "nursery", "experimental field",
        "种植", "栽培", "植物园", "药材基地", "试验田", "农场"
    ]

    search_fields = []
    for fld in ["establishmentMeans", "occurrenceRemarks", "habitat", "locality", "institutionCode"]:
        if fld in df.columns:
            search_fields.append(fld)

    if not search_fields:
        log("WARNING: No fields to search for cultivation keywords")
        df["cultivation_flag"] = False
        df["cultivation_reason"] = ""
        return df

    df["cultivation_flag"] = False
    df["cultivation_reason"] = ""

    for kw in cultivate_keywords:
        for fld in search_fields:
            mask = df[fld].astype(str).str.lower().str.contains(kw.lower(), na=False)
            if mask.any():
                df.loc[mask, "cultivation_flag"] = True
                df.loc[mask, "cultivation_reason"] = df.loc[mask, "cultivation_reason"] + f" {fld}:{kw};"

    log(f"Cultivation flagged: {df['cultivation_flag'].sum()}")
    return df


def geographic_anomaly_check(df):
    """Check for geographic anomalies."""
    df["geo_flag"] = ""
    flags = []

    # Check for 0,0 coordinates
    zero_zero = (df["decimalLatitude"] == 0) & (df["decimalLongitude"] == 0)
    if zero_zero.any():
        df.loc[zero_zero, "geo_flag"] += "zero_coord;"
        flags.append(f"zero_coords: {zero_zero.sum()}")

    # Check for coordinate at country centroid (approximate)
    # China: ~35.86, 104.20
    # Skip detailed check for now - will be done after spatial join with boundaries

    # Large number of identical coordinates (>10)
    coord_groups = df[df["coord_valid"]].groupby(["decimalLongitude", "decimalLatitude"]).size()
    high_dup = coord_groups[coord_groups > 10]
    if len(high_dup) > 0:
        flags.append(f"high_dup_locations(>10): {len(high_dup)} locations")

    if flags:
        log(f"Geographic flags: {'; '.join(flags)}")
    else:
        log("Geographic anomaly check: no major issues")

    # Save geographic flags
    flags_path = os.path.join(OUT_DIR, "occurrence_geographic_flags.csv")
    flag_cols = ["gbifID", "decimalLongitude", "decimalLatitude", "geo_flag"]
    flag_cols = [c for c in flag_cols if c in df.columns]
    df[flag_cols].to_csv(flags_path, index=False, encoding="utf-8-sig")

    return df


def spatial_thinning(df, distance_km, ref_grid_shape=(3341, 5010)):
    """Spatial thinning of occurrence points by distance.
    Uses a simple grid-based approach for efficiency.
    """
    # Convert km to degrees (approximate at mid-latitudes)
    # 1 degree ≈ 111 km
    deg_per_km = 1.0 / 111.0
    cell_size_deg = distance_km * deg_per_km

    valid = df[df["is_final_candidate"]].copy()
    if len(valid) == 0:
        return pd.DataFrame()

    # Assign each point to a grid cell
    valid["cell_x"] = (valid["decimalLongitude"] / cell_size_deg).round().astype(int)
    valid["cell_y"] = (valid["decimalLatitude"] / cell_size_deg).round().astype(int)

    # Keep one point per cell (first occurrence)
    thinned = valid.groupby(["cell_x", "cell_y"]).first().reset_index()
    thinned = thinned.drop(columns=["cell_x", "cell_y"], errors="ignore")

    log(f"  Thinning {distance_km}km: {len(valid)} -> {len(thinned)} records")
    return thinned


def main():
    log("="*60)
    log("OCCURRENCE PROCESSING")
    log("="*60)

    # 1. Load data
    df = load_gbif_data()
    if df is None:
        return

    # 2. Map fields
    fm = map_fields(df)

    # 3. Taxonomic filter
    df = taxonomic_filter(df)

    # 4. Coordinate filter
    df = coordinate_filter(df)

    # 5. Coordinate quality
    df = coordinate_quality(df)

    # 6. Duplicate filter
    df = duplicate_filter(df)

    # 7. Occurrence status
    df = occurrence_status_filter(df)

    # 8. Cultivation flag
    df = cultivation_flag(df)

    # 9. Geographic anomaly check
    df = geographic_anomaly_check(df)

    # ---- Build final datasets ----

    # Filter for the main workflow
    df_valid = df[
        df["taxon_keep"] &
        df["coord_valid"] &
        ~df["coord_duplicate"]
    ].copy()

    # Separate non-cultivated candidates and cultivated/uncertain
    noncult_mask = ~df_valid["cultivation_flag"]
    df_noncult = df_valid[noncult_mask].copy()
    df_cult = df_valid[~noncult_mask].copy()

    # Add final candidate flag
    df["is_final_candidate"] = False
    df.loc[df_noncult.index, "is_final_candidate"] = True

    stats["noncultivated_candidates"] = len(df_noncult)
    stats["cultivated_or_uncertain"] = len(df_cult)
    log(f"Non-cultivated candidates: {len(df_noncult)}")
    log(f"Cultivated/uncertain: {len(df_cult)}")

    # ---- Save standardized table ----
    csv_path = os.path.join(OUT_DIR, "occurrence_standardized.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log(f"Saved: {csv_path}")

    # Save noncultivated candidates
    df_noncult.to_csv(
        os.path.join(OUT_DIR, "occurrence_noncultivated_candidate.csv"),
        index=False, encoding="utf-8-sig"
    )

    # Save cultivated/uncertain
    df_cult.to_csv(
        os.path.join(OUT_DIR, "occurrence_cultivated_or_uncertain.csv"),
        index=False, encoding="utf-8-sig"
    )

    # ---- Spatial thinning ----
    thinning_distances = [5, 10, 20]
    for dist_km in thinning_distances:
        thinned = spatial_thinning(df, dist_km)
        out_path = os.path.join(OUT_DIR, f"occurrence_thin_{dist_km}km.csv")
        thinned.to_csv(out_path, index=False, encoding="utf-8-sig")
        stats[f"thin_{dist_km}km"] = len(thinned)

    # ---- Generate statistics for P1 flowchart ----
    flow_data = [
        {"stage": "Raw", "n_records": stats.get("raw", 0), "n_removed": 0},
        {"stage": "Taxonomy retained", "n_records": stats.get("taxonomy_retained", 0),
         "n_removed": stats.get("raw", 0) - stats.get("taxonomy_retained", 0)},
        {"stage": "Valid coordinates", "n_records": stats.get("valid_coordinates", 0),
         "n_removed": stats.get("taxonomy_retained", 0) - stats.get("valid_coordinates", 0)},
        {"stage": "Non-cultivated candidate", "n_records": stats.get("noncultivated_candidates", 0),
         "n_removed": stats.get("valid_coordinates", 0) - stats.get("noncultivated_candidates", 0) - stats.get("cultivated_or_uncertain", 0)},
        {"stage": "Grid deduplicated", "n_records": stats.get("noncultivated_candidates", 0),
         "n_removed": 0},
        {"stage": "5 km thinning", "n_records": stats.get("thin_5km", 0), "n_removed": 0},
        {"stage": "10 km thinning", "n_records": stats.get("thin_10km", 0), "n_removed": 0},
        {"stage": "20 km thinning", "n_records": stats.get("thin_20km", 0), "n_removed": 0},
    ]
    flow_path = os.path.join(OUT_DIR, "occurrence_filtering_flow.csv")
    flow_df = pd.DataFrame(flow_data)
    flow_df.to_csv(flow_path, index=False, encoding="utf-8-sig")

    # Copy to figure data
    fig_data_dir = r"E:\人参种在哪\00_统一数据预处理\13_figure_data"
    os.makedirs(fig_data_dir, exist_ok=True)
    flow_df.to_csv(os.path.join(fig_data_dir, "P1_occurrence_filtering_flow.csv"),
                   index=False, encoding="utf-8-sig")

    # Save map points for P2
    map_cols = ["gbifID", "decimalLongitude", "decimalLatitude", "species",
                "taxon_status", "coordinate_quality", "cultivation_flag", "is_final_candidate"]
    map_cols = [c for c in map_cols if c in df.columns]
    df[map_cols].to_csv(os.path.join(fig_data_dir, "P2_occurrence_map_points.csv"),
                         index=False, encoding="utf-8-sig")

    # Thinning counts for P3
    thin_data = [
        {"distance_km": 5, "n_records": stats.get("thin_5km", 0)},
        {"distance_km": 10, "n_records": stats.get("thin_10km", 0)},
        {"distance_km": 20, "n_records": stats.get("thin_20km", 0)},
    ]
    pd.DataFrame(thin_data).to_csv(
        os.path.join(fig_data_dir, "P3_thinning_counts.csv"),
        index=False, encoding="utf-8-sig"
    )

    # ---- Write log ----
    log_lines.append(f"\n=== FINAL STATISTICS ===\n")
    for k, v in stats.items():
        log_lines.append(f"  {k}: {v}\n")
    log_lines.append(f"[{datetime.now().isoformat()}] OCCURRENCE PROCESSING COMPLETED\n")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    print(f"\n{'='*50}")
    print("OCCURRENCE PROCESSING COMPLETE")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nOutput directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
