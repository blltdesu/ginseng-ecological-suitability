"""Step 0: Environment check and diagnostics."""
import sys
import os
import platform
import hashlib
from datetime import datetime

def main():
    report_path = r"E:\人参种在哪\00_统一数据预处理\00_config\environment_report.txt"
    log_path = r"E:\人参种在哪\00_统一数据预处理\logs\preprocessing_master.log"

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("ENVIRONMENT REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 60)

    lines.append(f"\nOperating System: {platform.system()} {platform.release()} {platform.version()}")
    lines.append(f"Python Version: {sys.version}")
    lines.append(f"Python Executable: {sys.executable}")

    packages = [
        "pandas", "numpy", "geopandas", "shapely", "pyproj",
        "rasterio", "rioxarray", "xarray", "netCDF4",
        "scipy", "sklearn", "matplotlib", "openpyxl",
        "pyogrio", "fiona", "tqdm", "rich", "yaml"
    ]

    lines.append("\n--- Core Packages ---")
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            lines.append(f"  {pkg}: {ver}")
        except ImportError:
            lines.append(f"  {pkg}: NOT INSTALLED")

    # GDAL check via rasterio
    try:
        import rasterio
        lines.append(f"\n  GDAL (via rasterio): {rasterio.__gdal_version__}")
    except Exception:
        lines.append("\n  GDAL: UNKNOWN")

    # Path checks
    lines.append("\n--- Path Checks ---")
    paths_to_check = [
        r"E:\人参种在哪",
        r"E:\人参种在哪\数据",
        r"E:\人参种在哪\00_统一数据预处理",
    ]
    for p in paths_to_check:
        exists = os.path.exists(p)
        writable = os.access(p, os.W_OK) if exists else False
        lines.append(f"  {p}: exists={exists}, writable={writable}")

    # Write report
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Master log start
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] PREPROCESSING STARTED\n")
        f.write(f"[{datetime.now().isoformat()}] Environment report: {report_path}\n")

    print(report_text)
    print(f"\nReport saved to: {report_path}")

    # Save pip freeze
    freeze_path = r"E:\人参种在哪\00_统一数据预处理\00_config\python_packages_freeze.txt"
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    with open(freeze_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)
    print(f"Package freeze saved to: {freeze_path}")

if __name__ == "__main__":
    main()
