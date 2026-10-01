#!/usr/bin/env python3
"""
Figure E3-5: Spatial group contribution map (3+1 panel).
A. Climate spatial gain
B. Soil spatial gain
C. Terrain spatial gain
D. Dominant group map
"""
import sys, os, logging, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验3"
FIG_DIR = os.path.join(ROOT, "14_figures")
DATA_DIR = os.path.join(ROOT, "15_figure_data")
SPATIAL_DIR = os.path.join(ROOT, "11_spatial_group_contribution")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "fig5.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})
for font in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [font]
        break
    except:
        continue


def load_raster(path):
    """Load raster data and metadata."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        data[data < -1e37] = np.nan
        transform = src.transform
        crs = src.crs
        extent = (
            src.bounds.left / 100000, src.bounds.right / 100000,
            src.bounds.bottom / 100000, src.bounds.top / 100000,
        )
    return data, extent


def add_colorbar_axis(fig, im, ax, label, x, y, w, h):
    """Add a colorbar at a specific position."""
    cax = fig.add_axes([x, y, w, h])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return cb


def main():
    log.info("=" * 60)
    log.info("GENERATING FIGURE E3-5: SPATIAL GROUP CONTRIBUTION")
    log.info("=" * 60)

    # Load gain rasters
    gain_files = {
        'Climate': os.path.join(SPATIAL_DIR, 'climate_spatial_gain.tif'),
        'Soil': os.path.join(SPATIAL_DIR, 'soil_spatial_gain.tif'),
        'Terrain': os.path.join(SPATIAL_DIR, 'terrain_spatial_gain.tif'),
    }
    dominant_file = os.path.join(SPATIAL_DIR, 'dominant_group_ablation.tif')

    gains = {}
    extent = None
    for group, path in gain_files.items():
        if os.path.exists(path):
            data, ext = load_raster(path)
            gains[group] = data
            if extent is None:
                extent = ext
            log.info(f"  Loaded {group} gain: shape={data.shape}, mean={np.nanmean(data):.4f}, range=[{np.nanmin(data):.4f}, {np.nanmax(data):.4f}]")
        else:
            log.warning(f"  Missing: {path}")

    dominant_data = None
    if os.path.exists(dominant_file):
        dominant_data, _ = load_raster(dominant_file)
        log.info(f"  Loaded dominant group map: unique values={np.unique(dominant_data[~np.isnan(dominant_data)])}")

    if not gains:
        log.error("No spatial gain files found. Cannot generate figure.")
        return 1

    # Subsample for faster plotting (every 4th pixel)
    step = 4

    # Create figure: 2x2 panel
    fig = plt.figure(figsize=(16, 12))

    # --- Color scales ---
    # Find symmetric range for gain maps
    all_gains = np.concatenate([g[~np.isnan(g)].ravel()[::10] for g in gains.values()])
    vmax = max(abs(np.nanpercentile(all_gains, 2)), abs(np.nanpercentile(all_gains, 98)))
    vmin = -vmax

    gain_cmap = plt.cm.RdBu_r

    # Dominant group colors: 1=Climate, 2=Soil, 3=Terrain
    dom_colors = ['#E74C3C', '#8E44AD', '#27AE60']
    dom_cmap = ListedColormap(dom_colors)
    dom_bounds = [0.5, 1.5, 2.5, 3.5]
    dom_norm = BoundaryNorm(dom_bounds, len(dom_colors))

    panels = [
        ('Climate', 'A. Climate Spatial Gain\n(CST − ST)'),
        ('Soil', 'B. Soil Spatial Gain\n(CST − CT)'),
        ('Terrain', 'C. Terrain Spatial Gain\n(CST − CS)'),
    ]

    # Plot the three gain maps (top row + bottom-left)
    for idx, (group, title) in enumerate(panels):
        ax = fig.add_subplot(2, 3, idx + 1)
        data = gains[group][::step, ::step]
        # Mask NaN for display
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap=gain_cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', origin='upper', interpolation='bilinear')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add mean annotation
        mean_val = np.nanmean(gains[group])
        ax.text(0.02, 0.98, f'Mean: {mean_val:+.4f}', transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Colorbar for the third panel (or each panel's own)
        if idx == 0:
            cax = fig.add_axes([0.92, 0.55, 0.008, 0.35])
            cb = fig.colorbar(im, cax=cax, orientation='vertical')
            cb.set_label('ΔSuitability', fontsize=8)
            cb.ax.tick_params(labelsize=7)

    # Panel D: Dominant group map (bottom-right, spanning)
    ax_dom = fig.add_subplot(2, 3, 4)
    if dominant_data is not None:
        dom_sub = dominant_data[::step, ::step]
        masked_dom = np.ma.masked_invalid(dom_sub)
        im_dom = ax_dom.imshow(masked_dom, cmap=dom_cmap, norm=dom_norm,
                               aspect='auto', origin='upper', interpolation='nearest')
        ax_dom.set_title('D. Dominant Environmental Group\n(based on |spatial gain|)', fontweight='bold', fontsize=10)
        ax_dom.set_xticks([])
        ax_dom.set_yticks([])

        # Legend for dominant group
        legend_patches = [
            mpatches.Patch(color='#E74C3C', label=f'Climate'),
            mpatches.Patch(color='#8E44AD', label=f'Soil'),
            mpatches.Patch(color='#27AE60', label=f'Terrain'),
        ]
        ax_dom.legend(handles=legend_patches, fontsize=8, loc='lower right')

    # Panel E: Area percentage bar chart (bottom-right second panel)
    ax_pct = fig.add_subplot(2, 3, 5)
    if dominant_data is not None:
        valid = dominant_data[~np.isnan(dominant_data)]
        total_valid = len(valid)
        if total_valid > 0:
            pcts = {}
            for i, group in enumerate(['Climate', 'Soil', 'Terrain']):
                pcts[group] = (valid == (i + 1)).sum() / total_valid * 100
            bars = ax_pct.bar(pcts.keys(), pcts.values(),
                             color=['#E74C3C', '#8E44AD', '#27AE60'],
                             alpha=0.7, edgecolor='black', linewidth=0.5)
            for bar, (g, p) in zip(bars, pcts.items()):
                ax_pct.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{p:.1f}%', ha='center', fontsize=10, fontweight='bold')
            ax_pct.set_title('E. Dominant Group Area Proportion', fontweight='bold', fontsize=10)
            ax_pct.set_ylabel('Area (%)')
            ax_pct.set_ylim(0, max(pcts.values()) * 1.2 + 2)
            ax_pct.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel F: Gain distribution histogram (bottom-right third panel)
    ax_hist = fig.add_subplot(2, 3, 6)
    for group, color in [('Climate', '#E74C3C'), ('Soil', '#8E44AD'), ('Terrain', '#27AE60')]:
        if group in gains:
            vals = gains[group][~np.isnan(gains[group])]
            # Subsample for histogram
            sample = np.random.RandomState(42).choice(vals, size=min(50000, len(vals)), replace=False)
            ax_hist.hist(sample, bins=60, alpha=0.4, color=color, label=group, density=True)
    ax_hist.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_hist.set_xlabel('ΔSuitability')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('F. Spatial Gain Distributions', fontweight='bold', fontsize=10)
    ax_hist.legend(fontsize=8, loc='upper right')
    ax_hist.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle('Figure E3-5: Spatial Environmental Group Contributions', fontweight='bold', y=0.99, fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.91, 0.98])

    for fmt in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'E3_Fig5_spatial_group_contribution.{fmt}')
        fig.savefig(path)
        log.info(f"  Saved: {path}")
    plt.close(fig)

    log.info("E3-Fig5 complete!")
    return 0


if __name__ == "__main__":
    main()
