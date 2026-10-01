#!/usr/bin/env python3
"""
Supplementary Figure E3-S2: Per-algorithm group-level Shapley contributions.
Shows drop-one loss and Shapley % for each of the 4 model types.
"""
import sys, os, logging, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

ROOT = r"E:\人参种在哪\实验3"
FIG_DIR = os.path.join(ROOT, "14_figures")
DATA_DIR = os.path.join(ROOT, "15_figure_data")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "figS2.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
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

COLORS = {'Climate': '#E74C3C', 'Soil': '#8E44AD', 'Terrain': '#27AE60'}


def main():
    log.info("=" * 60)
    log.info("GENERATING FIGURE E3-S2: ALGORITHM GROUP CONTRIBUTIONS")
    log.info("=" * 60)

    algo_file = os.path.join(ROOT, "13_sensitivity", "group_contribution_by_algorithm.csv")
    if not os.path.exists(algo_file):
        log.error(f"Missing data file: {algo_file}")
        return 1

    df = pd.read_csv(algo_file)
    log.info(f"Loaded: {len(df)} rows, columns={list(df.columns)}")

    # Save figure data
    df.to_csv(os.path.join(DATA_DIR, 'E3_FigS2_algorithm_group_contributions.csv'), index=False)

    models = ['maxent', 'random_forest', 'xgboost', 'brt']
    model_labels = ['MaxEnt', 'Random Forest', 'XGBoost', 'BRT']
    groups = ['Climate', 'Soil', 'Terrain']

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for col, (model, mlab) in enumerate(zip(models, model_labels)):
        # Row 1: Shapley %
        ax = axes[0, col]
        shap_df = df[(df['Model'] == model) & (df['Type'] == 'Shapley %')]
        values = []
        for g in groups:
            row = shap_df[shap_df['Group'] == g]
            values.append(row['Value'].values[0] if len(row) > 0 else 0)
        colors = [COLORS[g] for g in groups]
        bars = ax.bar(groups, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, values):
            y_pos = bar.get_height() + (2 if val >= 0 else -8)
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold',
                   color='green' if val > 50 else 'black' if val > 0 else 'red')
        ax.set_title(f'{mlab}', fontweight='bold', fontsize=11)
        if col == 0:
            ax.set_ylabel('Shapley Contribution (%)')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ylim_max = max(abs(max(values, key=abs)), 20) * 1.3
        ax.set_ylim(min(-ylim_max * 0.3, min(values) * 1.5), ylim_max)

        # Row 2: Drop-one loss
        ax2 = axes[1, col]
        drop_df = df[(df['Model'] == model) & (df['Type'] == 'Drop-one Loss')]
        values2 = []
        for g in groups:
            row = drop_df[drop_df['Group'] == g]
            values2.append(row['Value'].values[0] if len(row) > 0 else 0)
        colors2 = [COLORS[g] for g in groups]
        bars2 = ax2.bar(groups, values2, color=colors2, alpha=0.7, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars2, values2):
            y_pos = bar.get_height() + (0.05 if val >= 0 else -0.12)
            ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{val:.3f}', ha='center', fontsize=8)
        if col == 0:
            ax2.set_ylabel('Drop-one Loss (ΔBoyce)')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ylim_max2 = max(abs(max(values2, key=abs)), 0.2) * 1.3
        ax2.set_ylim(min(-ylim_max2 * 0.3, min(values2) * 1.5), ylim_max2)

    axes[0, 0].text(-0.15, 0.5, 'Shapley %', transform=axes[0, 0].transAxes,
                    fontsize=11, fontweight='bold', va='center', rotation=90)
    axes[1, 0].text(-0.15, 0.5, 'Drop-one Loss', transform=axes[1, 0].transAxes,
                    fontsize=11, fontweight='bold', va='center', rotation=90)

    fig.suptitle('Figure E3-S2: Algorithm Sensitivity — Per-Model Group Contributions',
                 fontweight='bold', y=1.01, fontsize=13)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'E3_FigS2_algorithm_group_contributions.{fmt}')
        fig.savefig(path)
        log.info(f"  Saved: {path}")
    plt.close(fig)

    log.info("E3-S2 complete!")
    return 0


if __name__ == "__main__":
    main()
