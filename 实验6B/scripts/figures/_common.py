#!/usr/bin/env python3
"""公共绘图工具: crop窗口加载、地图底图绘制"""
import json
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(r"E:\人参种在哪\实验6B")
FIG_DIR = ROOT / "13_figures"
FIG_DATA = ROOT / "14_figure_data"
BND_ADM0 = Path(r"E:\人参种在哪\00_统一数据预处理\03_boundaries\study_context_adm0.gpkg")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def load_crop():
    with open(FIG_DATA / "crop_window.json") as f:
        return json.load(f)


def load_raster(path, crop=None):
    with rasterio.open(path) as src:
        d = src.read(1).astype(np.float32)
        tr = src.transform
    if crop:
        d = d[crop["row0"]:crop["row1"], crop["col0"]:crop["col1"]]
    return d, tr


def extent(crop):
    return [crop["lon_min"], crop["lon_max"], crop["lat_min"], crop["lat_max"]]


def draw_boundaries(ax, crop):
    try:
        import geopandas as gpd
        g = gpd.read_file(BND_ADM0, bbox=(crop["lon_min"], crop["lat_min"], crop["lon_max"], crop["lat_max"]))
        g.boundary.plot(ax=ax, linewidth=0.7, color="0.35", zorder=5)
    except Exception:
        pass


def setup_map(ax, crop, title):
    ax.set_xlim(crop["lon_min"], crop["lon_max"])
    ax.set_ylim(crop["lat_min"], crop["lat_max"])
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    draw_boundaries(ax, crop)


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png (600dpi) + .pdf")


def pixel_centers(mask, crop):
    """mask为crop后的布尔数组 → 返回像元中心(lons, lats)"""
    with open(FIG_DATA / "crop_window.json") as f:
        c = json.load(f)
    res = (c["lon_max"] - c["lon_min"]) / (c["col1"] - c["col0"])
    rows, cols = np.where(mask)
    lons = c["lon_min"] + (cols + 0.5) * res
    lats = c["lat_max"] - (rows + 0.5) * res
    return lons, lats
