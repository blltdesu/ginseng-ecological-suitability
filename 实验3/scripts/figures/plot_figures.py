#!/usr/bin/env python3
"""
Generate all figures for Experiment 3.
E3_Fig1: Design & group structure
E3_Fig2: Subset performance comparison
E3_Fig3: Group contribution decomposition (Standalone + Drop-one + Shapley)
E3_Fig4: Complementarity / synergy
E3_Fig5: Spatial group contribution map
"""
import sys, os, logging, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
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
        logging.FileHandler(os.path.join(LOG_DIR, "figures.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Matplotlib style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})
# Try to use a CJK-capable font
for font in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [font]
        break
    except:
        continue

COLORS = {
    'Climate': '#E74C3C',
    'Soil': '#8E44AD',
    'Terrain': '#27AE60',
    'Full': '#2C3E50',
}


def fig1_design_structure():
    """Figure E3-1: Experimental design and group structure."""
    log.info("Generating Figure E3-1: Design structure...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Variable grouping
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A. Environment Group Partition', fontweight='bold', loc='left')

    groups = {
        'Climate': ['bio02', 'bio03', 'bio05', 'bio15'],
        'Soil': ['clay', 'sand'],
        'Terrain': ['elevation', 'northness'],
    }
    y_positions = {'Climate': 7.5, 'Soil': 4.5, 'Terrain': 1.5}
    colors = {'Climate': '#E74C3C', 'Soil': '#8E44AD', 'Terrain': '#27AE60'}

    for group, vars_list in groups.items():
        y = y_positions[group]
        ax.text(1, y + 0.5, group, fontweight='bold', fontsize=12, color=colors[group], va='bottom')
        rect = FancyBboxPatch((0.5, y - 0.3), 9, len(vars_list) * 0.8 + 0.8,
                              boxstyle="round,pad=0.1", facecolor=colors[group], alpha=0.15,
                              edgecolor=colors[group], linewidth=1.5)
        ax.add_patch(rect)
        for j, var in enumerate(vars_list):
            ax.text(1.5, y + 0.1 + j * 0.8, var, fontsize=9, va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=colors[group], alpha=0.8))

    # Panel B: Subset model structure
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('B. Subset Model Design (7 non-null + NULL)', fontweight='bold', loc='left')

    subsets = [
        ('C', 8.5, '#E74C3C'), ('S', 7.0, '#8E44AD'), ('T', 5.5, '#27AE60'),
        ('CS', 4.0, '#7B241C'), ('CT', 2.8, '#1E8449'), ('ST', 1.6, '#6C3483'),
        ('CST', 0.4, '#2C3E50'),
    ]
    for label, y, color in subsets:
        rect = FancyBboxPatch((0.5, y - 0.25), 9, 0.6,
                              boxstyle="round,pad=0.05", facecolor=color, alpha=0.2,
                              edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(5, y + 0.05, f'Subset: {label}', ha='center', va='center',
               fontweight='bold', fontsize=10, color=color)

    ax.text(5, 9.3, 'Single groups', ha='center', fontsize=9, fontstyle='italic', color='gray')
    ax.text(5, 3.8, 'Two-group combinations', ha='center', fontsize=9, fontstyle='italic', color='gray')
    ax.text(5, 0.6, 'Full model', ha='center', fontsize=9, fontstyle='italic', color='gray')

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'E3_Fig1_group_partition_design.{fmt}')
        fig.savefig(path)
        log.info(f"  Saved: {path}")
    plt.close(fig)

    # Save figure data
    group_data = []
    for g, vs in groups.items():
        for v in vs:
            group_data.append({'Group': g, 'Variable': v})
    pd.DataFrame(group_data).to_csv(os.path.join(DATA_DIR, 'E3_Fig1_group_definition.csv'), index=False)

    registry = pd.read_csv(os.path.join(ROOT, "03_subset_datasets", "subset_registry.csv"))
    registry.to_csv(os.path.join(DATA_DIR, 'E3_Fig1_subset_registry.csv'), index=False)


def fig2_subset_performance():
    """Figure E3-2: Subset model performance comparison (Boyce + TSS + AUC)."""
    log.info("Generating Figure E3-2: Subset performance...")

    ens_fold = pd.read_csv(os.path.join(ROOT, "07_performance_comparison", "ensemble_subset_fold_metrics.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    subset_order = ['NULL', 'C', 'S', 'T', 'CS', 'CT', 'ST', 'CST']
    colors_list = ['#95A5A6', '#E74C3C', '#8E44AD', '#27AE60', '#C0392B', '#1E8449', '#6C3483', '#2C3E50']

    for idx, metric in enumerate(['boyce', 'tss', 'auc']):
        ax = axes[idx]
        data_by_subset = []
        positions = []
        labels = []
        for i, ss in enumerate(subset_order):
            vals = ens_fold[ens_fold['subset'] == ss][metric].dropna()
            if len(vals) > 0:
                data_by_subset.append(vals.values)
                positions.append(i)
                labels.append(ss)

        bp = ax.boxplot(data_by_subset, positions=positions, widths=0.5,
                        patch_artist=True, showfliers=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black'))

        for patch, color in zip(bp['boxes'], colors_list[:len(data_by_subset)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        metric_label = {'boyce': "Boyce's Index", 'tss': 'TSS', 'auc': 'AUC'}[metric]
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle('Figure E3-2: Ensemble Subset Model Performance (Spatial CV)', fontweight='bold', y=1.01)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'E3_Fig2_subset_performance.{fmt}')
        fig.savefig(path)
        log.info(f"  Saved: {path}")
    plt.close(fig)

    ens_fold.to_csv(os.path.join(DATA_DIR, 'E3_Fig2_subset_fold_metrics.csv'), index=False)


def fig3_group_contribution():
    """Figure E3-3: Group contribution decomposition."""
    log.info("Generating Figure E3-3: Group contribution...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Standalone predictive power
    ax = axes[0]
    standalone_file = os.path.join(ROOT, "08_ablation", "standalone_predictive_power.csv")
    if os.path.exists(standalone_file):
        sp = pd.read_csv(standalone_file)
        groups = sp['Group'].tolist()
        values = sp['Standalone_Power'].tolist()
        colors_a = [COLORS[g] for g in groups]
        bars = ax.bar(groups, values, color=colors_a, alpha=0.7, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', fontsize=9)
        ax.set_ylabel('Standalone Power (normalized)')
        ax.set_title('A. Standalone Predictive Power', fontweight='bold', loc='left')
        ax.set_ylim(0, max(values) * 1.2 + 0.05)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel B: Drop-one-group loss
    ax = axes[1]
    abl_file = os.path.join(ROOT, "08_ablation", "drop_one_group_loss.csv")
    if os.path.exists(abl_file):
        abl = pd.read_csv(abl_file)
        groups = abl['Group'].tolist()
        losses = abl['Drop_Loss'].tolist()
        ci_low = abl['CI_low'].tolist()
        ci_high = abl['CI_high'].tolist()
        colors_b = [COLORS[g] for g in groups]
        yerr = [[abs(l - cl) for l, cl in zip(losses, ci_low)],
                [abs(ch - l) for l, ch in zip(losses, ci_high)]]
        bars = ax.bar(groups, losses, color=colors_b, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.errorbar(range(len(groups)), losses, yerr=yerr, fmt='none',
                   ecolor='black', capsize=5, linewidth=1)
        for bar, loss in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{loss:.4f}', ha='center', fontsize=9)
        ax.set_ylabel('Drop-one Loss (ΔBoyce)')
        ax.set_title('B. Drop-One-Group Ablation', fontweight='bold', loc='left')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel C: Shapley contribution
    ax = axes[2]
    shap_file = os.path.join(ROOT, "09_group_shapley", "group_shapley_contributions.csv")
    if os.path.exists(shap_file):
        shap = pd.read_csv(shap_file)
        groups = shap['Group'].tolist()
        percents = shap['Shapley_percent'].tolist()
        ci_low_s = shap['CI_low'].tolist()
        ci_high_s = shap['CI_high'].tolist()
        colors_c = [COLORS[g] for g in groups]
        yerr_s = [[abs(p - cl) for p, cl in zip(percents, ci_low_s)],
                  [abs(ch - p) for p, ch in zip(percents, ci_high_s)]]
        bars = ax.bar(groups, percents, color=colors_c, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.errorbar(range(len(groups)), percents, yerr=yerr_s, fmt='none',
                   ecolor='black', capsize=5, linewidth=1)
        for bar, pct in zip(bars, percents):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold')
        ax.set_ylabel('Shapley Contribution (%)')
        ax.set_title('C. Group-Level Shapley Decomposition', fontweight='bold', loc='left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle('Figure E3-3: Environmental Group Contribution Decomposition', fontweight='bold', y=1.01)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'E3_Fig3_group_contribution.{fmt}')
        fig.savefig(path)
        log.info(f"  Saved: {path}")
    plt.close(fig)

    # Save figure data
    if os.path.exists(standalone_file):
        pd.read_csv(standalone_file).to_csv(os.path.join(DATA_DIR, 'E3_Fig3_standalone.csv'), index=False)
    if os.path.exists(abl_file):
        pd.read_csv(abl_file).to_csv(os.path.join(DATA_DIR, 'E3_Fig3_drop_one.csv'), index=False)
    if os.path.exists(shap_file):
        pd.read_csv(shap_file).to_csv(os.path.join(DATA_DIR, 'E3_Fig3_shapley.csv'), index=False)


def fig4_complementarity():
    """Figure E3-4: Pairwise complementarity / redundancy."""
    log.info("Generating Figure E3-4: Complementarity...")

    synergy_file = os.path.join(ROOT, "10_complementarity", "pairwise_synergy.csv")
    inc_file = os.path.join(ROOT, "10_complementarity", "incremental_gain_paths.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Pairwise synergy
    ax = axes[0]
    if os.path.exists(synergy_file):
        syn = pd.read_csv(synergy_file)
        pairs = syn['Pair'].tolist()
        values = syn['Synergy_Boyce'].tolist()
        ci_low = syn['CI_low'].tolist()
        ci_high = syn['CI_high'].tolist()

        colors_s = ['#E74C3C' if 'Climate' in p else '#8E44AD' if 'Soil' in p else '#27AE60' for p in pairs]
        pair_colors = []
        for p in pairs:
            if 'Climate' in p and 'Soil' in p:
                pair_colors.append('#C0392B')
            elif 'Climate' in p and 'Terrain' in p:
                pair_colors.append('#1E8449')
            else:
                pair_colors.append('#6C3483')

        x = range(len(pairs))
        bars = ax.bar(x, values, color=pair_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        yerr = [[abs(v - cl) for v, cl in zip(values, ci_low)],
                [abs(ch - v) for v, ch in zip(values, ci_high)]]
        ax.errorbar(x, values, yerr=yerr, fmt='none', ecolor='black', capsize=5, linewidth=1)

        for i, (bar, val, interp) in enumerate(zip(bars, values, syn['Interpretation'])):
            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.01,
                   f'{val:.4f}\n({interp[:12]})', ha='center', fontsize=8, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=30, ha='right')
        ax.set_ylabel('Pairwise Synergy (Boyce)')
        ax.set_title('A. Pairwise Complementarity', fontweight='bold', loc='left')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel B: Incremental gain paths
    ax = axes[1]
    if os.path.exists(inc_file):
        inc = pd.read_csv(inc_file)
        x_pos = []
        labels = []
        gains = []
        path_colors = []
        for i, row in inc.iterrows():
            x_pos.extend([i * 2 + 0.5, i * 2 + 1.5])
            labels.extend([f'Step1', f'Step2'])
            gains.extend([row['Step1_Gain'], row['Step2_Gain']])
            if 'C →' in row['Path']:
                path_colors.extend(['#E74C3C', '#C0392B'])
            elif 'S →' in row['Path']:
                path_colors.extend(['#8E44AD', '#6C3483'])
            else:
                path_colors.extend(['#27AE60', '#1E8449'])

        ax.bar(x_pos, gains, color=path_colors, alpha=0.7, edgecolor='black', linewidth=0.5, width=0.8)
        ax.set_xticks([1, 3, 5, 7])
        ax.set_xticklabels(inc['Path'].tolist(), rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Incremental Boyce Gain')
        ax.set_title('B. Incremental Gain Paths', fontweight='bold', loc='left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Legend for Step1/Step2
        legend_patches = [
            mpatches.Patch(color='#E74C3C', alpha=0.7, label='Step 1 (add one group)'),
            mpatches.Patch(color='#C0392B', alpha=0.7, label='Step 2 (add second group)'),
        ]
        ax.legend(handles=legend_patches, fontsize=8, loc='upper right')

    fig.suptitle('Figure E3-4: Environmental Group Complementarity and Redundancy', fontweight='bold', y=1.01)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'E3_Fig4_complementarity.{fmt}')
        fig.savefig(path)
        log.info(f"  Saved: {path}")
    plt.close(fig)

    # Save figure data
    if os.path.exists(synergy_file):
        pd.read_csv(synergy_file).to_csv(os.path.join(DATA_DIR, 'E3_Fig4_pairwise_synergy.csv'), index=False)
    if os.path.exists(inc_file):
        pd.read_csv(inc_file).to_csv(os.path.join(DATA_DIR, 'E3_Fig4_incremental_gain.csv'), index=False)


def main():
    log.info("=" * 60)
    log.info("GENERATING EXPERIMENT 3 FIGURES")
    log.info("=" * 60)

    fig1_design_structure()
    fig2_subset_performance()
    fig3_group_contribution()
    fig4_complementarity()

    log.info("\nAll figures generated!")
    log.info("Note: Figure E3-5 (spatial maps) requires raster data from spatial_analysis.py")
    log.info("Note: Supplementary figures E3-S1, E3-S2 require additional data files")
    return 0


if __name__ == "__main__":
    main()
