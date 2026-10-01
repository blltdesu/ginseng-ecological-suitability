"""Step 2: Extract archives safely to work directory."""
import os
import csv
import zipfile
import shutil
from datetime import datetime

RAW_DIR = r"E:\人参种在哪\数据"
EXTRACT_DIR = r"E:\人参种在哪\00_统一数据预处理\01_inventory\extracted"
INVENTORY_CSV = r"E:\人参种在哪\00_统一数据预处理\01_inventory\raw_file_inventory.csv"
OUT_CSV = r"E:\人参种在哪\00_统一数据预处理\01_inventory\extracted_file_inventory.csv"
LOG_PATH = r"E:\人参种在哪\00_统一数据预处理\logs\inventory.log"


def extract_zip(zip_path, dest_dir):
    """Extract a zip file to destination directory."""
    results = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                # Skip __MACOSX and hidden files
                if member.startswith("__MACOSX") or os.path.basename(member).startswith("._"):
                    continue
                try:
                    zf.extract(member, dest_dir)
                    results.append(os.path.join(dest_dir, member))
                except Exception as e:
                    print(f"  WARNING: Could not extract {member}: {e}")
        return True, results
    except Exception as e:
        return False, str(e)


def main():
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Read inventory
    archives = []
    with open(INVENTORY_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["is_archive"] == "True":
                archives.append(row)

    print(f"Found {len(archives)} archives to process")

    extracted_files = []
    file_id = 0

    for arch in archives:
        zip_path = arch["absolute_path"]
        sha_prefix = arch["sha256"][:8] if len(arch["sha256"]) >= 8 else arch["filename"]
        # Create unique subdirectory
        arch_name = os.path.splitext(arch["filename"])[0]
        dest_subdir = os.path.join(EXTRACT_DIR, f"{arch_name}_{sha_prefix}")
        os.makedirs(dest_subdir, exist_ok=True)

        print(f"\nExtracting: {arch['relative_path']} -> {dest_subdir}")

        success, result = extract_zip(zip_path, dest_subdir)

        if success:
            for extracted_path in result:
                # Skip directories
                if os.path.isdir(extracted_path):
                    continue
                file_id += 1
                fname = os.path.basename(extracted_path)
                abs_path = extracted_path
                rel_to_extract = os.path.relpath(abs_path, EXTRACT_DIR)
                ext = os.path.splitext(fname)[1].lower()
                size_mb = os.path.getsize(abs_path) / (1024 * 1024)

                row = {
                    "file_id": file_id,
                    "source_archive": arch["filename"],
                    "absolute_path": abs_path,
                    "relative_path_in_extracted": rel_to_extract,
                    "filename": fname,
                    "extension": ext,
                    "size_mb": round(size_mb, 4),
                }
                extracted_files.append(row)
        else:
            print(f"  ERROR: {result}")

    # Write CSV
    fieldnames = [
        "file_id", "source_archive", "absolute_path",
        "relative_path_in_extracted", "filename", "extension", "size_mb"
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_files)

    # Append to log
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] ARCHIVE EXTRACTION COMPLETED: {file_id} files extracted\n")

    # Summary by source archive
    from collections import Counter
    arch_counts = Counter(row["source_archive"] for row in extracted_files)

    print(f"\n{'='*50}")
    print(f"EXTRACTION COMPLETE: {file_id} files extracted")
    print(f"By source archive:")
    for arch_name, count in arch_counts.most_common():
        print(f"  {arch_name}: {count} files")
    print(f"\nOutput: {OUT_CSV}")

if __name__ == "__main__":
    main()
