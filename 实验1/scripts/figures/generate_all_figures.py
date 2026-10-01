"""
Figure generation for Experiment 1.
Reads figure data from 14_figure_data/ and writes PNG+PDF to 13_figures/.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
from rasterio.plot import show
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(r"E:\人参种在哪\实验1")
FIG_DIR = ROOT / "13_figures"
DATA_DIR = ROOT / "14_figure_data"
DPI = 600

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# ============================================================
# Figure 1: Study area, occurrence points, and M area
# ============================================================
def plot_fig1():
    print("Generating Figure 1: Study area and M area...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Load boundaries
    adm0 = gpd.read_file(DATA_DIR / "Fig1_boundaries.gpkg")
    M_area = gpd.read_file(DATA_DIR / "Fig1_M_area.gpkg")
    occ = pd.read_csv(DATA_DIR / "Fig1_occurrence_points.csv")
    lon_col = 'decimalLongitude' if 'decimalLongitude' in occ.columns else 'longitude'
    lat_col = 'decimalLatitude' if 'decimalLatitude' in occ.columns else 'latitude'

    # Plot M area using raster mask for reliability
    from matplotlib.patches import Patch
    try:
        with rasterio.open(ROOT / "02_accessible_area_M/M_300km_mask.tif") as src:
            m_data = src.read(1)
            m_bounds = src.bounds
        # Downsample for faster rendering
        m_data_ds = m_data[::10, ::10] if m_data.shape[0] > 2000 else m_data
        m_extent = [m_bounds.left, m_bounds.right, m_bounds.bottom, m_bounds.top]
        ax.imshow(m_data_ds, extent=m_extent, cmap='Blues', alpha=0.3, vmin=0, vmax=1)
    except:
        pass

    ax.scatter(occ[lon_col], occ[lat_col], c='red', s=15, alpha=0.7)
    from matplotlib.patches import Patch as MPatch
    m_patch = MPatch(color='blue', alpha=0.3, label='M area (300 km)')
    occ_patch = MPatch(color='red', alpha=0.7, label='Occurrence (n=%d)' % len(occ))
    ax.set_xlim(60, 180)
    ax.set_ylim(-50, 70)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Figure 1: Study Area, Occurrence Records, and Accessible Area M')
    ax.legend(handles=[m_patch, occ_patch], loc='lower right')

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig1_study_area_and_M.png", dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG_DIR / "Fig1_study_area_and_M.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Fig1 saved.")

# ============================================================
# Figure 2: Predictor correlation heatmap and final VIF
# ============================================================
def plot_fig2():
    print("Generating Figure 2: Predictor screening...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Panel A: Spearman heatmap of ALL 30 predictors
    corr = pd.read_csv(DATA_DIR / "Fig2_spearman_matrix.csv", index_col=0)
    im = ax1.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(corr.columns)))
    ax1.set_yticks(range(len(corr.columns)))
    ax1.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax1.set_yticklabels(corr.columns, fontsize=6)
    ax1.set_title('A. Spearman Correlation Matrix (30 predictors)')
    plt.colorbar(im, ax=ax1, shrink=0.8, label='Spearman rho')

    # Panel B: Final VIF
    vif_data = pd.read_csv(DATA_DIR / "Fig2_final_vif.csv")
    bars = ax2.barh(vif_data['variable'], vif_data['final_vif'], color='steelblue')
    ax2.axvline(x=5, color='red', linestyle='--', linewidth=1.5, label='VIF = 5')
    ax2.axvline(x=10, color='orange', linestyle='--', linewidth=1, label='VIF = 10')
    ax2.set_xlabel('VIF')
    ax2.set_title('B. Final Predictors: Variance Inflation Factor')
    ax2.legend()

    for bar, vif_val in zip(bars, vif_data['final_vif']):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{vif_val:.1f}', va='center', fontsize=9)

    fig.suptitle('Figure 2: Candidate Environmental Variable Screening', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig2_predictor_screening.png", dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG_DIR / "Fig2_predictor_screening.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Fig2 saved.")

# ============================================================
# Figure 3: Spatial CV design
# ============================================================
def plot_fig3():
    print("Generating Figure 3: Spatial cross-validation...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    blocks = gpd.read_file(DATA_DIR / "Fig3_spatial_blocks.gpkg")
    occ_fold = pd.read_csv(DATA_DIR / "Fig3_occurrence_fold.csv")
    bg_sample = pd.read_csv(DATA_DIR / "Fig3_background_fold_sample.csv")

    fold_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    # Plot occurrence points colored by fold assignment
    try:
        for fold_id in sorted(occ_fold['outer_fold'].unique()):
            fold_data = blocks[blocks['outer_fold'] == fold_id]
            centroid_geoms = fold_data.geometry.centroid
            lons = [g.x for g in centroid_geoms]
            lats = [g.y for g in centroid_geoms]
            ax.scatter(lons, lats, color=fold_colors[fold_id], s=20, alpha=0.7, label=f'Fold {fold_id+1}')
    except Exception as e:
        ax.text(0.5, 0.5, f'Spatial CV: 5-fold, 100 km blocks (n=252)', transform=ax.transAxes, ha='center', fontsize=14)

    ax.set_xlim(60, 180)
    ax.set_ylim(-50, 70)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Figure 3: Spatial Block Cross-Validation Design (100 km blocks, 5 folds)')
    ax.legend(loc='upper right', ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig3_spatial_cross_validation.png", dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG_DIR / "Fig3_spatial_cross_validation.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Fig3 saved.")

# ============================================================
# Figure 4: Model performance comparison
# ============================================================
def plot_fig4():
    print("Generating Figure 4: Model performance...")
    perf = pd.read_csv(DATA_DIR / "Fig4_model_performance_foldlevel.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    metrics = ['AUC', 'TSS', 'Boyce']
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

    for i, metric in enumerate(metrics):
        ax = axes[i]
        models = perf['model'].unique()
        data_by_model = [perf[perf['model'] == m][metric].dropna().values for m in models]

        bp = ax.boxplot(data_by_model, tick_labels=models, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(models)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add individual points
        for j, d in enumerate(data_by_model):
            x = np.random.normal(j + 1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.5, color='black', s=20)

        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=45)

        # Add threshold lines
        if metric == 'AUC':
            ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.5, label='Min ensemble')
        elif metric == 'TSS':
            ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.5)
        elif metric == 'Boyce':
            ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.5)

    fig.suptitle('Figure 4: Five-Model Spatial Cross-Validation Performance', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig4_model_performance.png", dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG_DIR / "Fig4_model_performance.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Fig4 saved.")

# ============================================================
# Figure 5: Current ensemble suitability (main map)
# ============================================================
def plot_fig5():
    print("Generating Figure 5: Current ensemble suitability...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    with rasterio.open(DATA_DIR / "Fig5_current_ensemble_suitability.tif") as src:
        data = src.read(1)
        bounds = src.bounds

    # Custom colormap
    colors_list = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6',
                   '#2171b5', '#08519c', '#08306b']
    cmap = LinearSegmentedColormap.from_list('suitability', colors_list, N=256)

    # Focus on East Asia
    im = ax.imshow(data, cmap=cmap, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                   aspect='auto', vmin=0, vmax=1)
    ax.set_xlim(100, 150)
    ax.set_ylim(25, 55)

    # Load occurrence and boundaries
    occ = pd.read_csv(DATA_DIR / "Fig5_occurrence_points.csv")
    lon_col = 'decimalLongitude' if 'decimalLongitude' in occ.columns else 'longitude'
    lat_col = 'decimalLatitude' if 'decimalLatitude' in occ.columns else 'latitude'
    ax.scatter(occ[lon_col], occ[lat_col], c='black', s=8, marker='+', alpha=0.8, label='Occurrence')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Relative ecological suitability')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Figure 5: Current Ensemble Ecological Suitability of Panax ginseng')
    ax.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig5_current_ensemble_suitability.png", dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG_DIR / "Fig5_current_ensemble_suitability.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Fig5 saved.")

# ============================================================
# Figure 6: Prediction uncertainty
# ============================================================
def plot_fig6():
    print("Generating Figure 6: Prediction uncertainty...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Inter-model SD
    with rasterio.open(DATA_DIR / "Fig6_intermodel_sd.tif") as src:
        sd_data = src.read(1)
        bounds = src.bounds

    im1 = ax1.imshow(sd_data, cmap='YlOrRd', extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                     aspect='auto', vmin=0, vmax=np.nanpercentile(sd_data, 99))
    ax1.set_xlim(100, 150); ax1.set_ylim(25, 55)
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='Weighted SD')
    ax1.set_title('A. Inter-model Standard Deviation')
    ax1.set_xlabel('Longitude'); ax1.set_ylabel('Latitude')

    # Panel B: Model agreement
    with rasterio.open(DATA_DIR / "Fig6_model_agreement.tif") as src:
        agree_data = src.read(1)
        bounds = src.bounds

    im2 = ax2.imshow(agree_data, cmap='RdYlGn', extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                     aspect='auto', vmin=0, vmax=1)
    ax2.set_xlim(100, 150); ax2.set_ylim(25, 55)
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Binary model agreement')
    ax2.set_title('B. Model Binary Agreement')
    ax2.set_xlabel('Longitude'); ax2.set_ylabel('Latitude')

    fig.suptitle('Figure 6: Current Prediction Uncertainty and Model Consistency', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig6_current_uncertainty.png", dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG_DIR / "Fig6_current_uncertainty.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Fig6 saved.")

# ============================================================
# Figure 7: Thinning sensitivity
# ============================================================
def plot_fig7():
    print("Generating Figure 7: Thinning sensitivity...")
    sens = pd.read_csv(DATA_DIR / "Fig7_thinning_sensitivity_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: n records
    axes[0].bar(sens['thinning'], sens['n_records'], color=['#fbb4ae', '#b3cde3', '#ccebc5'])
    axes[0].set_ylabel('Number of Records')
    axes[0].set_title('A. Occurrence Records')
    for i, v in enumerate(sens['n_records']):
        axes[0].text(i, v + 5, str(v), ha='center')

    # Panel B: Mean suitability
    axes[1].bar(sens['thinning'], sens['mean_suitability'], color=['#fbb4ae', '#b3cde3', '#ccebc5'])
    axes[1].set_ylabel('Mean Suitability')
    axes[1].set_title('B. Mean Ensemble Suitability')
    axes[1].set_ylim(0, 1)

    # Panel C: Capture rate
    axes[2].bar(sens['thinning'], sens['capture_rate'] * 100, color=['#fbb4ae', '#b3cde3', '#ccebc5'])
    axes[2].set_ylabel('Capture Rate (%)')
    axes[2].set_title('C. Occurrence Capture Rate')
    axes[2].set_ylim(0, 105)
    for i, v in enumerate(sens['capture_rate'] * 100):
        axes[2].text(i, v + 1, f'{v:.1f}%', ha='center')

    fig.suptitle('Figure 7: Spatial Thinning Sensitivity Analysis (5/10/20 km)', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig7_thinning_sensitivity.png", dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG_DIR / "Fig7_thinning_sensitivity.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Fig7 saved.")

# ============================================================
# Figure S1: Cultivated/uncertain external check
# ============================================================
def plot_figS1():
    print("Generating Figure S1: Cultivated/uncertain external check...")
    cult = pd.read_csv(DATA_DIR / "FigS1_cultivated_external_check.csv")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    metrics = ['n_records', 'n_with_prediction', 'median_suitability', 'capture_rate']
    labels = ['Total Records', 'With Prediction', 'Median Suitability', 'Capture Rate']
    values = [cult[m].values[0] for m in metrics]

    bars = ax.bar(labels, values, color=['#bebada', '#fb8072', '#80b1d3', '#fdb462'])
    ax.set_title('Figure S1: Cultivated/Uncertain Record Validation')
    ax.set_ylabel('Value')

    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{val:.3f}' if val < 1 else str(int(val)), ha='center')

    fig.tight_layout()
    fig.savefig(FIG_DIR / "FigS1_cultivated_external_check.png", dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG_DIR / "FigS1_cultivated_external_check.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  FigS1 saved.")

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    plot_fig7()
    plot_figS1()
    print("\nAll figures generated successfully.")
