"""Step 1: Scan raw data directory and generate file inventory."""
import os
import hashlib
import csv
import zipfile
import tarfile
from datetime import datetime

RAW_DIR = r"E:\人参种在哪\数据"
OUT_DIR = r"E:\人参种在哪\00_统一数据预处理\01_inventory"
LOG_PATH = r"E:\人参种在哪\00_统一数据预处理\logs\inventory.log"

ARCHIVE_EXTS = {".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz"}
RASTER_EXTS = {".tif", ".tiff", ".nc", ".grd", ".bil", ".asc", ".img", ".hdf", ".h5"}
VECTOR_EXTS = {".shp", ".gpkg", ".geojson", ".json", ".kml", ".gml", ".sqlite"}
TABULAR_EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".dbf", ".parquet", ".feather"}
SCRIPT_EXTS = {".py", ".r", ".R", ".ipynb"}

def get_sha256(filepath):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

def check_gdal_readable(filepath):
    """Check if file can be read by GDAL (raster)."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in RASTER_EXTS:
            import rasterio
            with rasterio.open(filepath) as src:
                return True, f"raster: {src.count} band(s), CRS={src.crs}, shape={src.shape}"
    except Exception as e:
        return False, str(e)
    return False, "Not a raster extension"

def check_ogr_readable(filepath):
    """Check if file can be read by OGR (vector)."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in VECTOR_EXTS:
            import geopandas as gpd
            gdf = gpd.read_file(filepath)
            return True, f"vector: {len(gdf)} features, CRS={gdf.crs}"
    except Exception:
        # Try fiona directly for shapefiles
        if ext == ".shp":
            try:
                import fiona
                with fiona.open(filepath) as src:
                    return True, f"vector: {len(src)} features, CRS={src.crs}"
            except Exception as e:
                return False, str(e)
    return False, "Not a vector extension"

def check_archive(filepath):
    """Check if file is an archive and list contents."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".zip":
        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                names = zf.namelist()
                return True, f"zip: {len(names)} entries"
        except Exception as e:
            return True, f"zip (corrupted?): {e}"
    return False, ""

def classify_file(filename, dirpath, gdal_info, ogr_info):
    """Classify a file into a possible category based on keywords."""
    full_path = os.path.join(dirpath, filename).lower()
    fname = filename.lower()
    parent = os.path.basename(dirpath).lower()

    # Build keyword evidence
    scores = {
        "occurrence": 0,
        "boundary": 0,
        "current_climate": 0,
        "future_climate": 0,
        "soil": 0,
        "terrain": 0,
        "landcover": 0,
    }

    # Occurrence keywords
    occ_kw = ["gbif", "occurrence", "panax", "ginseng", "species", "dwca"]
    for kw in occ_kw:
        if kw in fname or kw in parent:
            scores["occurrence"] += 1

    # Current climate keywords
    cc_kw = ["worldclim", "wc2", "bio_", "bioclim", "current", "historical", "bio1", "bio2", "bio3"]
    for kw in cc_kw:
        if kw in fname or kw in parent:
            scores["current_climate"] += 1

    # Future climate keywords
    fc_kw = ["cmip", "ssp126", "ssp245", "ssp370", "ssp585", "2041", "2060", "2061", "2080",
             "access", "miroc", "mpi", "mri", "bcc", "future"]
    for kw in fc_kw:
        if kw in fname or kw in parent:
            scores["future_climate"] += 1

    # Soil keywords
    soil_kw = ["soilgrids", "phh2o", "soc", "cec", "clay", "sand", "nitrogen", "bdod", "cfvo", "soil"]
    for kw in soil_kw:
        if kw in fname or kw in parent:
            scores["soil"] += 1

    # Terrain keywords
    terr_kw = ["srtm", "dem", "elevation", "slope", "aspect", "terrain", "topo"]
    for kw in terr_kw:
        if kw in fname or kw in parent:
            scores["terrain"] += 1

    # Landcover keywords
    lc_kw = ["mcd12", "modis", "landcover", "lc_type", "land_cover"]
    for kw in lc_kw:
        if kw in fname or kw in parent:
            scores["landcover"] += 1

    # Boundary keywords
    bd_kw = ["boundary", "adm0", "adm1", "geoboundaries", "gadm"]
    for kw in bd_kw:
        if kw in fname or kw in parent:
            scores["boundary"] += 1

    # Determine best category
    max_score = max(scores.values())
    if max_score == 0:
        return "unknown", 0.0

    best_cats = [k for k, v in scores.items() if v == max_score]
    confidence = max_score / max(3, 1)  # rough confidence from keyword count
    confidence = min(confidence, 1.0)

    if len(best_cats) == 1:
        return best_cats[0], confidence
    else:
        return f"ambiguous: {'/'.join(best_cats)}", confidence * 0.5


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    log_lines = []
    log_lines.append(f"[{datetime.now().isoformat()}] INVENTORY SCAN STARTED\n")

    inventory = []
    file_id = 0

    for dirpath, dirnames, filenames in os.walk(RAW_DIR):
        for fname in filenames:
            file_id += 1
            abs_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(abs_path, RAW_DIR)
            ext = os.path.splitext(fname)[1].lower()
            size_mb = os.path.getsize(abs_path) / (1024 * 1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(abs_path)).isoformat()

            # Compute hash
            print(f"[{file_id}] Hashing: {rel_path}")
            sha256 = get_sha256(abs_path)
            if sha256.startswith("ERROR"):
                log_lines.append(f"WARNING: Cannot hash {rel_path}: {sha256}\n")

            # Check archive
            is_archive = ext in ARCHIVE_EXTS
            archive_info = ""
            if is_archive:
                is_archive_flag, archive_info = check_archive(abs_path)
                is_archive = is_archive_flag

            # Check GDAL
            gdal_readable, gdal_info = check_gdal_readable(abs_path)

            # Check OGR
            ogr_readable, ogr_info = check_ogr_readable(abs_path)

            # Classify
            category, confidence = classify_file(fname, dirpath, gdal_info, ogr_info)

            # If parent dir name strongly suggests category, use that
            parent_cat_map = {
                "分布记录gbif": "occurrence",
                "行政边界": "boundary",
                "气候数据": "current_climate",
                "未来天气": "future_climate",
                "土壤数据": "soil",
                "地形数据": "terrain",
                "土地覆盖": "landcover",
            }
            parent_dir = os.path.basename(dirpath)
            if parent_dir in parent_cat_map and confidence < 0.9:
                category = parent_cat_map[parent_dir]
                confidence = 0.9

            notes = ""
            if gdal_readable:
                notes += f"GDAL: {gdal_info}; "
            if ogr_readable:
                notes += f"OGR: {ogr_info}; "
            if is_archive:
                notes += f"Archive: {archive_info}; "
            if ext in SCRIPT_EXTS:
                notes += "Script file; "

            row = {
                "file_id": file_id,
                "absolute_path": abs_path,
                "relative_path": rel_path,
                "filename": fname,
                "extension": ext,
                "size_mb": round(size_mb, 4),
                "modified_time": mtime,
                "sha256": sha256,
                "is_archive": is_archive,
                "readable_by_gdal": gdal_readable,
                "readable_by_ogr": ogr_readable,
                "possible_category": category,
                "category_confidence": round(confidence, 2),
                "notes": notes.strip()
            }
            inventory.append(row)

            log_lines.append(f"  {rel_path} -> {category} (conf={confidence:.2f})\n")

    # Write CSV
    csv_path = os.path.join(OUT_DIR, "raw_file_inventory.csv")
    fieldnames = [
        "file_id", "absolute_path", "relative_path", "filename", "extension",
        "size_mb", "modified_time", "sha256", "is_archive",
        "readable_by_gdal", "readable_by_ogr",
        "possible_category", "category_confidence", "notes"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)

    # Write SHA256 file
    sha_path = os.path.join(OUT_DIR, "raw_sha256.csv")
    with open(sha_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "sha256", "filename"])
        for row in inventory:
            writer.writerow([row["relative_path"], row["sha256"], row["filename"]])

    # Write log
    log_lines.append(f"[{datetime.now().isoformat()}] INVENTORY SCAN COMPLETED: {file_id} files\n")
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    # Summary
    from collections import Counter
    cat_counts = Counter(row["possible_category"] for row in inventory)
    print(f"\n{'='*50}")
    print(f"INVENTORY SCAN COMPLETE: {file_id} files found")
    print(f"Category breakdown:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")
    print(f"\nOutput: {csv_path}")
    print(f"SHA256: {sha_path}")
    print(f"Log: {LOG_PATH}")

if __name__ == "__main__":
    main()
