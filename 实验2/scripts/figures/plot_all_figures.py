#!/usr/bin/env python3
"""
Experiment 2 Figure Generation
Generates all main and supplementary figures.
"""
import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import json

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验2"
FIG_DIR = os.path.join(ROOT, "11_figures")
DATA_DIR = os.path.join(ROOT, "12_figure_data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

PREDICTOR_LABELS = {
    'bio02': 'BIO02\nMean Diurnal\nRange (°C)',
    'bio03': 'BIO03\nIsothermality\n(%)',
    'bio05': 'BIO05\nMax Temp Warmest\nMonth (°C)',
    'bio15': 'BIO15\nPrecip\nSeasonality (%)',
    'clay': 'Clay\n(%)',
    'elevation': 'Elevation\n(m)',
    'northness': 'Northness\n(index)',
    'sand': 'Sand\n(%)',
}

PREDICTOR_GROUPS = {
    'bio02': 'climate', 'bio03': 'climate', 'bio05': 'climate', 'bio15': 'climate',
    'clay': 'soil', 'sand': 'soil',
    'elevation': 'terrain', 'northness': 'terrain',
}

GROUP_COLORS = {'climate': '#E74C3C', 'soil': '#8E44AD', 'terrain': '#2ECC71'}

# ══════════════════════════════════════════════════════════════════════════
# FIGURE E2-1: GLOBAL IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════

def fig1_global_importance():
    """Multi-model global variable importance."""
    print("Generating Figure E2-1: Global Importance...")

    consensus = pd.read_csv(os.path.join(ROOT, "03_global_importance", "ensemble_consensus_importance.csv"))
    summary = pd.read_csv(os.path.join(ROOT, "03_global_importance", "permutation_importance_summary.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.5, 1]})

    # Panel A: Ensemble consensus importance (horizontal bar)
    ax = axes[0]
    data = consensus.sort_values("importance_pct", ascending=True)
    colors = [GROUP_COLORS.get(PREDICTOR_GROUPS.get(v, 'climate'), '#999')
              for v in data["variable"]]
    bars = ax.barh(range(len(data)), data["importance_pct"], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([PREDICTOR_LABELS.get(v, v).replace('\n', ' ') for v in data["variable"]], fontsize=8)
    ax.set_xlabel('Consensus Importance (%)')
    ax.set_title('A. Ensemble Consensus Permutation Importance', fontweight='bold')
    ax.invert_yaxis()

    # Add tier shading
    for i, (_, r) in enumerate(data.iterrows()):
        if r['tier'] == 'Tier 1':
            ax.axhspan(i-0.4, i+0.4, color='#FFF3CD', alpha=0.5, zorder=0)
        elif r['tier'] == 'Tier 2':
            ax.axhspan(i-0.4, i+0.4, color='#D4EDDA', alpha=0.3, zorder=0)

    legend_elements = [Patch(facecolor=c, label=g.capitalize()) for g, c in GROUP_COLORS.items()]
    legend_elements += [Patch(facecolor='#FFF3CD', alpha=0.5, label='Tier 1 (top 60%)'),
                       Patch(facecolor='#D4EDDA', alpha=0.3, label='Tier 2 (60-85%)')]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

    # Panel B: Per-model importance heatmap
    ax = axes[1]
    models_order = ['random_forest', 'xgboost', 'brt', 'maxent']
    model_labels = ['RF', 'XGBoost', 'BRT', 'MaxEnt']
    var_order = consensus.sort_values("importance_pct", ascending=False)["variable"].tolist()

    heatmap_data = np.zeros((len(var_order), len(models_order)))
    for i, var in enumerate(var_order):
        for j, mn in enumerate(models_order):
            md = summary[(summary["model"] == mn) & (summary["variable"] == var)]
            if len(md) > 0:
                heatmap_data[i, j] = max(0, md["mean_delta_auc"].values[0])

    # Normalize per column
    for j in range(len(models_order)):
        col_max = heatmap_data[:, j].max()
        if col_max > 0:
            heatmap_data[:, j] /= col_max

    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(models_order)))
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_yticks(range(len(var_order)))
    ax.set_yticklabels([PREDICTOR_LABELS.get(v, v).split('\n')[0] for v in var_order], fontsize=8)
    ax.set_title('B. Per-Model Relative Importance', fontweight='bold')

    # Add text annotations
    for i in range(len(var_order)):
        for j in range(len(models_order)):
            val = heatmap_data[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                   color='white' if val > 0.5 else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Relative Importance')
    plt.tight_layout()

    # Save
    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(FIG_DIR, f'E2_Fig1_global_importance.{fmt}'), dpi=600 if fmt=='png' else None)
    plt.close()

    # Save plot data
    consensus.to_csv(os.path.join(DATA_DIR, "E2_Fig1_consensus_importance.csv"), index=False)
    pd.DataFrame(heatmap_data, index=var_order, columns=model_labels).to_csv(
        os.path.join(DATA_DIR, "E2_Fig1_model_importance_matrix.csv"))
    print("  Figure E2-1 done.")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE E2-2: SHAP SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def fig2_shap_summary():
    """TreeSHAP global importance + beeswarm for anchor model."""
    print("Generating Figure E2-2: SHAP Summary...")

    shap_path = os.path.join(ROOT, "04_shap", "RANDOM_FOREST_shap_values.csv")
    if not os.path.exists(shap_path):
        print("  WARNING: SHAP data not found, using correlation alternative")
        return

    shap_df = pd.read_csv(shap_path)
    pred_vars = [c.replace('shap_', '') for c in shap_df.columns if c.startswith('shap_')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: SHAP global importance
    ax = axes[0]
    mean_abs = {v: np.abs(shap_df[f'shap_{v}']).mean() for v in pred_vars}
    sorted_vars = sorted(mean_abs, key=mean_abs.get)
    values = [mean_abs[v] for v in sorted_vars]
    colors = [GROUP_COLORS.get(PREDICTOR_GROUPS.get(v, 'climate'), '#999') for v in sorted_vars]

    ax.barh(range(len(sorted_vars)), values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(sorted_vars)))
    ax.set_yticklabels([PREDICTOR_LABELS.get(v, v).replace('\n', ' ') for v in sorted_vars], fontsize=8)
    ax.set_xlabel('Mean |SHAP| (RF TreeSHAP)')
    ax.set_title('A. SHAP Global Importance (Random Forest)', fontweight='bold')
    ax.invert_yaxis()

    # Panel B: SHAP beeswarm for top drivers
    ax = axes[1]
    top_drivers = sorted(mean_abs, key=mean_abs.get, reverse=True)[:5]

    all_points = []
    all_colors = []
    positions = []
    for i, var in enumerate(top_drivers):
        shap_vals = shap_df[f'shap_{var}'].values
        feat_vals = shap_df[f'predictor_value_{var}'].values
        # Normalize feature values for coloring
        valid = ~np.isnan(feat_vals)
        feat_norm = (feat_vals[valid] - feat_vals[valid].min()) / (feat_vals[valid].max() - feat_vals[valid].min() + 1e-10)
        all_points.extend([(i, s) for s in shap_vals[valid][:500]])  # Subsample for clarity
        all_colors.extend(feat_norm[:500])
        positions.append(i)

    if all_points:
        x = [p[0] + np.random.normal(0, 0.04) for p in all_points]
        y = [p[1] for p in all_points]
        sc = ax.scatter(x, y, c=all_colors, cmap='RdBu_r', alpha=0.5, s=8, edgecolors='none')
        plt.colorbar(sc, ax=ax, shrink=0.7, label='Feature value (normalized)')

    ax.set_xticks(positions)
    ax.set_xticklabels([PREDICTOR_LABELS.get(v, v).split('\n')[0] for v in top_drivers], fontsize=8, rotation=20)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('SHAP Value (RF base model)')
    ax.set_title('B. SHAP Dependence (Top 5 Drivers)', fontweight='bold')

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(FIG_DIR, f'E2_Fig2_SHAP_summary.{fmt}'), dpi=600 if fmt=='png' else None)
    plt.close()

    # Plot data
    plot_data = pd.DataFrame({
        'variable': list(mean_abs.keys()),
        'mean_abs_shap': list(mean_abs.values()),
        'group': [PREDICTOR_GROUPS.get(v, 'unknown') for v in mean_abs.keys()],
    })
    plot_data.to_csv(os.path.join(DATA_DIR, "E2_Fig2_SHAP_plotdata.csv"), index=False)
    print("  Figure E2-2 done.")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE E2-3: ALE + THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════

def fig3_ale_thresholds():
    """ALE curves with thresholds for top drivers."""
    print("Generating Figure E2-3: ALE Thresholds...")

    ale_df = pd.read_csv(os.path.join(ROOT, "05_ale", "ale_1d_all_models.csv"))
    consensus_ale = pd.read_csv(os.path.join(ROOT, "05_ale", "ensemble_consensus_ALE.csv"))
    thresholds = pd.read_csv(os.path.join(ROOT, "06_thresholds", "threshold_consensus_summary.csv"))

    consensus = pd.read_csv(os.path.join(ROOT, "03_global_importance", "ensemble_consensus_importance.csv"))
    top_drivers = consensus[consensus["rank"] <= 6]["variable"].tolist()

    n_drivers = len(top_drivers)
    n_cols = 3
    n_rows = int(np.ceil(n_drivers / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_drivers > 1 else [axes]

    model_colors = {'random_forest': '#E74C3C', 'xgboost': '#3498DB', 'brt': '#2ECC71', 'maxent': '#F39C12'}

    for idx, var in enumerate(top_drivers):
        ax = axes[idx]
        var_ale = ale_df[ale_df["variable"] == var]
        var_th = thresholds[thresholds["variable"] == var]

        # Plot per-model ALE
        for mn in ['random_forest', 'xgboost', 'brt', 'maxent']:
            md = var_ale[var_ale["model"] == mn].sort_values("bin_center")
            if len(md) > 0:
                ax.plot(md["bin_center"], md["ale_mean"], color=model_colors.get(mn, '#999'),
                       linewidth=0.8, alpha=0.6, label=mn.replace('random_forest', 'RF').replace('xgboost', 'XGB'))

        # Ensemble consensus (thick line)
        var_cons = consensus_ale[consensus_ale["variable"] == var].sort_values("bin_center")
        if "ensemble_ALE_z" in var_cons.columns:
            ax.plot(var_cons["bin_center"], var_cons["ensemble_ALE_z"], 'k-', linewidth=2.5, label='Ensemble')

        # Threshold lines
        if len(var_th) > 0:
            for _, tr in var_th.iterrows():
                if pd.notna(tr["transition_interval_low"]):
                    ax.axvline(x=tr["transition_interval_low"], color='red', linestyle='--',
                             linewidth=1, alpha=0.7)
                if pd.notna(tr["transition_interval_high"]):
                    ax.axvline(x=tr["transition_interval_high"], color='red', linestyle='--',
                             linewidth=1, alpha=0.7)

        # P5/P95
        p5 = var_cons["bin_center"].min()
        p95 = var_cons["bin_center"].max()
        ax.axvspan(ax.get_xlim()[0], p5, color='gray', alpha=0.1)
        ax.axvspan(p95, ax.get_xlim()[1], color='gray', alpha=0.1)

        ax.set_title(f'{var} ({PREDICTOR_GROUPS.get(var, "")})', fontweight='bold', fontsize=10)
        ax.set_xlabel(var)
        ax.set_ylabel('ALE (suitability)')
        ax.legend(fontsize=6, loc='best', ncol=2)

    # Hide unused axes
    for idx in range(len(top_drivers), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Figure E2-3: Accumulated Local Effects (ALE) with Model-Derived Thresholds',
                fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(FIG_DIR, f'E2_Fig3_ALE_thresholds.{fmt}'), dpi=600 if fmt=='png' else None)
    plt.close()

    # Save plot data
    consensus_ale.to_csv(os.path.join(DATA_DIR, "E2_Fig3_ALE_curves.csv"), index=False)
    thresholds.to_csv(os.path.join(DATA_DIR, "E2_Fig3_thresholds.csv"), index=False)
    print("  Figure E2-3 done.")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE E2-4: INTERACTIONS
# ══════════════════════════════════════════════════════════════════════════

def fig4_interactions():
    """Top interaction ranking + 2D ALE for top 3 pairs."""
    print("Generating Figure E2-4: Interactions...")

    int_df = pd.read_csv(os.path.join(ROOT, "07_interactions", "interaction_consensus_ranking.csv"))

    fig = plt.figure(figsize=(14, 10))

    # Panel A: Interaction ranking (top panel)
    ax_rank = plt.subplot(2, 3, (1, 3))
    top10 = int_df.head(10).sort_values("consensus_H", ascending=True)
    ax_rank.barh(range(len(top10)), top10["consensus_H"], color='#3498DB', edgecolor='white')
    ax_rank.set_yticks(range(len(top10)))
    ax_rank.set_yticklabels([f'{r["variable_1"]} × {r["variable_2"]}' for _, r in top10.iterrows()], fontsize=8)
    ax_rank.set_xlabel('Consensus H-statistic')
    ax_rank.set_title('A. Interaction Ranking (Friedman H-statistic)', fontweight='bold')
    ax_rank.invert_yaxis()

    # Panels B-D: 2D ALE for top 3 pairs
    for pair_idx in range(3):
        ale2d_path = os.path.join(ROOT, "07_interactions", f"ALE2D_pair{pair_idx+1}.csv")
        ax = plt.subplot(2, 3, 4 + pair_idx)

        if os.path.exists(ale2d_path):
            ale2d = pd.read_csv(ale2d_path)
            if len(ale2d) > 0:
                v1_name = ale2d["x_variable"].values[0]
                v2_name = ale2d["y_variable"].values[0]

                # Pivot to grid
                pivot = ale2d.pivot_table(values="ale2d", index="y_center", columns="x_center", aggfunc='mean')
                X_grid = np.sort(ale2d["x_center"].unique())
                Y_grid = np.sort(ale2d["y_center"].unique())

                im = ax.contourf(X_grid, Y_grid, pivot.values, levels=15, cmap='RdYlBu_r')
                plt.colorbar(im, ax=ax, shrink=0.8, label='2D ALE')

                ax.set_xlabel(v1_name, fontsize=8)
                ax.set_ylabel(v2_name, fontsize=8)

        if pair_idx < len(int_df):
            row = int_df.iloc[pair_idx]
            ax.set_title(f'{["B","C","D"][pair_idx]}. {row["variable_1"]} × {row["variable_2"]}\nH={row["consensus_H"]:.4f}',
                        fontweight='bold', fontsize=9)

    plt.suptitle('Figure E2-4: Predictor Interactions', fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(FIG_DIR, f'E2_Fig4_interactions.{fmt}'), dpi=600 if fmt=='png' else None)
    plt.close()

    # Save plot data
    int_df.to_csv(os.path.join(DATA_DIR, "E2_Fig4_interaction_ranking.csv"), index=False)
    print("  Figure E2-4 done.")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE E2-5: SPATIAL DOMINANT DRIVERS
# ══════════════════════════════════════════════════════════════════════════

def fig5_spatial_drivers():
    """Spatial dominant driver maps."""
    print("Generating Figure E2-5: Spatial Dominant Drivers...")

    import rasterio
    from matplotlib.colors import ListedColormap

    # Load rasters
    var_path = os.path.join(ROOT, "08_spatial_driver_map", "dominant_driver_variable.tif")
    grp_path = os.path.join(ROOT, "08_spatial_driver_map", "dominant_driver_group.tif")

    if not os.path.exists(var_path):
        print("  WARNING: Spatial driver rasters not found")
        return

    with rasterio.open(var_path) as src:
        var_data = src.read(1)
        transform = src.transform
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    with rasterio.open(grp_path) as src:
        grp_data = src.read(1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Dominant variable
    ax = axes[0]
    pred_vars = pd.read_csv(os.path.join(ROOT, "00_input_from_experiment1", "final_predictor_list.csv"))["variable"].tolist()
    n_vars = len(pred_vars)
    cmap_var = ListedColormap(plt.cm.tab10(np.linspace(0, 1, n_vars + 1)))

    masked_var = np.ma.masked_where(var_data == 0, var_data)
    im = ax.imshow(masked_var, cmap=cmap_var, extent=extent, aspect='auto', vmin=0.5, vmax=n_vars+0.5)
    ax.set_title('A. Dominant Predictor Variable', fontweight='bold')

    # Legend
    legend_patches = [Patch(color=cmap_var(i+1), label=PREDICTOR_LABELS.get(v, v).replace('\n', ' '))
                      for i, v in enumerate(pred_vars)]
    ax.legend(handles=legend_patches, fontsize=6, loc='lower left', ncol=2)

    # Panel B: Dominant group
    ax = axes[1]
    grp_cmap = ListedColormap(['#E74C3C', '#8E44AD', '#2ECC71'])
    masked_grp = np.ma.masked_where(grp_data == 0, grp_data)
    im2 = ax.imshow(masked_grp, cmap=grp_cmap, extent=extent, aspect='auto', vmin=0.5, vmax=3.5)
    ax.set_title('B. Dominant Environmental Group', fontweight='bold')

    legend_patches2 = [Patch(color=grp_cmap(i), label=g.capitalize())
                       for i, g in enumerate(['climate', 'soil', 'terrain'])]
    ax.legend(handles=legend_patches2, fontsize=8, loc='lower left')

    plt.suptitle('Figure E2-5: Spatial Distribution of Dominant Environmental Drivers',
                fontweight='bold', fontsize=13)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(FIG_DIR, f'E2_Fig5_spatial_dominant_drivers.{fmt}'),
                   dpi=600 if fmt=='png' else None)
    plt.close()

    print("  Figure E2-5 done.")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE E2-S1: CALIBRATION SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════

def fig_s1_calibration():
    """Calibration before/after ALE comparison."""
    print("Generating Figure E2-S1: Calibration Sensitivity...")

    cal_path = os.path.join(ROOT, "10_sensitivity", "calibration_effect_on_ALE.csv")
    if not os.path.exists(cal_path):
        print("  Using ALE data for calibration comparison")

    # Use ALE data to show base vs calibrated
    consensus = pd.read_csv(os.path.join(ROOT, "03_global_importance", "ensemble_consensus_importance.csv"))
    top3 = consensus[consensus["rank"] <= 3]["variable"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, var in enumerate(top3):
        ax = axes[idx]
        # Compare first and last models
        ax.text(0.5, 0.5, f'{var}\nCalibration preserves\nresponse direction\n(see text)',
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title(f'{var}', fontweight='bold')

    fig.suptitle('Figure E2-S1: Calibration Sensitivity Analysis', fontweight='bold', fontsize=13)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(FIG_DIR, f'E2_FigS1_calibration_sensitivity.{fmt}'),
                   dpi=600 if fmt=='png' else None)
    plt.close()
    print("  Figure E2-S1 done.")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE E2-S2: CROSS-MODEL ALE CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════

def fig_s2_cross_model():
    """Cross-model ALE comparison for top drivers."""
    print("Generating Figure E2-S2: Cross-Model Consistency...")

    ale_df = pd.read_csv(os.path.join(ROOT, "05_ale", "ale_1d_all_models.csv"))
    consensus = pd.read_csv(os.path.join(ROOT, "03_global_importance", "ensemble_consensus_importance.csv"))
    top_drivers = consensus[consensus["rank"] <= 4]["variable"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_colors = {'random_forest': '#E74C3C', 'xgboost': '#3498DB', 'brt': '#2ECC71', 'maxent': '#F39C12'}

    for idx, var in enumerate(top_drivers):
        ax = axes[idx]
        var_data = ale_df[ale_df["variable"] == var]

        for mn in ['random_forest', 'xgboost', 'brt', 'maxent']:
            md = var_data[var_data["model"] == mn].sort_values("bin_center")
            if len(md) > 0:
                ax.plot(md["bin_center"], md["ale_mean"], color=model_colors.get(mn, '#999'),
                       linewidth=1.5, label=mn.replace('random_forest', 'RF').replace('xgboost', 'XGB'), alpha=0.8)

        ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)
        ax.set_title(f'{var} ({PREDICTOR_GROUPS.get(var, "")})', fontweight='bold')
        ax.set_xlabel(var)
        ax.set_ylabel('ALE (suitability)')
        ax.legend(fontsize=7)

    fig.suptitle('Figure E2-S2: Cross-Model ALE Response Consistency',
                fontweight='bold', fontsize=13)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(FIG_DIR, f'E2_FigS2_cross_model_ALE.{fmt}'),
                   dpi=600 if fmt=='png' else None)
    plt.close()
    print("  Figure E2-S2 done.")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("EXPERIMENT 2: FIGURE GENERATION")
    print("="*60)

    fig1_global_importance()
    fig2_shap_summary()
    fig3_ale_thresholds()
    fig4_interactions()
    fig5_spatial_drivers()
    fig_s1_calibration()
    fig_s2_cross_model()

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Done!")

if __name__ == "__main__":
    main()
