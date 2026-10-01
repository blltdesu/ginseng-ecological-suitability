"""Figure 1B: Occurrence record cleaning + final predictor VIF.
Left panel: horizontal funnel of occurrence filtering stages.
Right panel: VIF horizontal bar chart grouped by Climate/Soil/Terrain.
Soil depth: 0–15 cm (thickness-weighted mean, SoilGrids 2.0).
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

FIG_DIR = r"E:\人参种在哪\00_统一数据预处理\12_figures"
FIG_DATA_DIR = r"E:\人参种在哪\00_统一数据预处理\13_figure_data"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(FIG_DATA_DIR, exist_ok=True)

# Matplotlib settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False

# ── colour palette ──────────────────────────────────────────────
STAGE_COLORS = {
    "Raw GBIF":                  "#4472C4",
    "Valid coordinates":         "#5B9BD5",
    "Non-cultivated candidates": "#70AD47",
    "5 km thinned":              "#FFC000",
    "10 km thinned (primary)":   "#ED7D31",
    "20 km thinned":             "#A5A5A5",
}

GROUP_COLORS = {
    "Climate": "#4472C4",
    "Soil":    "#70AD47",
    "Terrain": "#ED7D31",
}

# ── data ─────────────────────────────────────────────────────────
def load_occurrence_data():
    """Load occurrence counts from figure-data CSV; fall back to hard-coded."""
    csv_path = os.path.join(FIG_DATA_DIR, "Fig1B_occurrence_counts.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return list(zip(df["stage"], df["count"]))
    # fallback
    return [
        ("Raw GBIF",                  922),
        ("Valid coordinates",         564),
        ("Non-cultivated candidates", 413),
        ("5 km thinned",              290),
        ("10 km thinned (primary)",   252),
        ("20 km thinned",             212),
    ]


def load_vif_data():
    """Load VIF data from figure-data CSV; fall back to hard-coded."""
    csv_path = os.path.join(FIG_DATA_DIR, "Fig1B_predictor_vif.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return list(zip(df["predictor"], df["group"], df["VIF"]))
    return [
        ("bio02",     "Climate", 2.13),
        ("bio03",     "Climate", 1.87),
        ("bio05",     "Climate", 1.25),
        ("bio15",     "Climate", 1.81),
        ("clay",      "Soil",    1.96),
        ("sand",      "Soil",    2.11),
        ("elevation", "Terrain", 1.68),
        ("northness", "Terrain", 1.00),
    ]


# ── main figure ──────────────────────────────────────────────────
def plot_Fig1B():
    print("Plotting Fig1B: Occurrence QC + Predictor VIF …")

    occ_data = load_occurrence_data()
    vif_data = load_vif_data()

    # ── layout: two panels side by side ──────────────────────────
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(16, 7),
        gridspec_kw={"width_ratios": [1, 1.15]},
    )

    # =============================================================
    # LEFT PANEL — occurrence cleaning funnel (horizontal bars)
    # =============================================================
    stages     = [d[0] for d in occ_data]
    counts     = [d[1] for d in occ_data]
    n_stages   = len(stages)

    # Reverse so the widest bar is at the top (funnel look)
    y_pos      = list(range(n_stages))
    bar_height = 0.55

    for i, (stage, count) in enumerate(zip(stages, counts)):
        color = STAGE_COLORS.get(stage, "#CCCCCC")
        is_primary = "primary" in stage.lower() or "10 km" in stage

        axL.barh(
            y_pos[i], count, height=bar_height,
            color=color,
            edgecolor="black",
            linewidth=2.5 if is_primary else 0.8,
            zorder=5,
        )
        # count label inside / at end of bar
        label_x = count + 18
        axL.text(
            label_x, y_pos[i], str(count),
            va="center", ha="left",
            fontweight="bold" if is_primary else "normal",
            fontsize=13 if is_primary else 11,
            color="black",
        )

    # Y-axis stage labels
    axL.set_yticks(y_pos)
    axL.set_yticklabels(stages, fontsize=10)
    axL.invert_yaxis()                     # top = raw

    # X-axis — omit axis line; keep only enough to anchor labels
    axL.set_xlim(0, max(counts) * 1.32)
    axL.xaxis.set_visible(False)
    for spine in ("top", "right", "bottom"):
        axL.spines[spine].set_visible(False)
    axL.spines["left"].set_visible(False)
    axL.tick_params(left=False)

    # Arrow connectors between bars
    for i in range(n_stages - 1):
        mid_x = max(counts[i], counts[i + 1]) + 55
        mid_y = (y_pos[i] + y_pos[i + 1]) / 2
        axL.annotate(
            "", xy=(mid_x, y_pos[i + 1] + bar_height / 2),
            xytext=(mid_x, y_pos[i] - bar_height / 2),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        )

    axL.set_title("Occurrence record cleaning", fontsize=13,
                  fontweight="bold", loc="left", pad=12)

    # Legend for 10 km primary line
    primary_patch = mpatches.Patch(
        edgecolor="black", linewidth=2.5, facecolor=STAGE_COLORS["10 km thinned (primary)"],
        label="10 km = primary analysis",
    )
    axL.legend(handles=[primary_patch], loc="lower right", fontsize=8.5,
               framealpha=0.9)

    # =============================================================
    # RIGHT PANEL — VIF horizontal bar chart grouped
    # =============================================================
    # Group order: Climate → Soil → Terrain
    group_order = ["Climate", "Soil", "Terrain"]
    # Within each group, sort by VIF descending
    grouped = {}
    for pred, grp, vif in vif_data:
        grouped.setdefault(grp, []).append((pred, vif))
    for grp in grouped:
        grouped[grp].sort(key=lambda x: x[1], reverse=True)

    y = 0
    y_ticks     = []
    y_labels    = []
    group_spans = []   # (y_center, label)

    y_group_start = {}
    for grp in group_order:
        if grp not in grouped:
            continue
        items = grouped[grp]
        g_start = y
        for pred, vif in items:
            y_ticks.append(y)
            y_labels.append(pred)
            color = GROUP_COLORS.get(grp, "#999999")
            axR.barh(y, vif, height=0.55, color=color,
                     edgecolor="white", linewidth=0.8, zorder=5)
            # VIF value label
            axR.text(vif + 0.06, y, f"{vif:.2f}",
                     va="center", ha="left", fontsize=10,
                     fontweight="bold")
            y += 1
        g_end = y - 1
        group_spans.append(((g_start + g_end) / 2, grp))
        y_group_start[grp] = g_start
        y += 0.6  # gap between groups

    # Group brackets & labels
    for y_center, grp in group_spans:
        g_start = y_group_start[grp]
        g_end   = g_start + len(grouped[grp]) - 1
        axR.axhline(y=g_start - 0.45, xmin=0, xmax=1, color="grey",
                    linewidth=0.6, alpha=0.5)
        axR.text(
            -0.02, y_center, grp,
            transform=axR.get_yaxis_transform(),
            ha="right", va="center", fontsize=10,
            fontweight="bold", color="dimgrey",
        )

    # VIF = 5 reference line
    axR.axvline(x=5, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    axR.text(5.08, y - 0.3, "VIF = 5", fontsize=9, color="red",
             alpha=0.8, fontstyle="italic")

    axR.set_yticks(y_ticks)
    axR.set_yticklabels(y_labels, fontsize=10)
    axR.set_xlabel("Variance Inflation Factor (VIF)", fontsize=12)
    axR.set_xlim(0, max(v for _, _, v in vif_data) * 1.28)
    axR.invert_yaxis()

    # Light grid
    axR.xaxis.grid(True, alpha=0.25, linestyle="--")
    axR.set_axisbelow(True)
    for spine in ("top", "right"):
        axR.spines[spine].set_visible(False)

    axR.set_title("Final predictor VIF", fontsize=13,
                  fontweight="bold", loc="left", pad=12)

    # Group legend (colour swatches)
    legend_handles = [
        mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in group_order
    ]
    axR.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
               framealpha=0.9, title="Predictor group")

    # ── overall title & footnote ─────────────────────────────────
    fig.suptitle(
        "Fig. 1B  |  Occurrence data quality control & final predictor set (VIF)",
        fontsize=15, fontweight="bold", y=1.02,
    )

    # Footnote
    fig.text(
        0.5, -0.04,
        "Soil variables: 0–15 cm thickness-weighted mean (SoilGrids 2.0).  "
        "Occurrence records: GBIF download ID 0002333-260803105813674, "
        "Panax ginseng C.A.Mey. only.  "
        "VIF computed from 10 km thinned occurrence + 30 current predictors; "
        "8 retained after collinearity screening.",
        ha="center", fontsize=8, fontstyle="italic", color="dimgrey",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # ── save ─────────────────────────────────────────────────────
    png_path = os.path.join(FIG_DIR, "Fig1B_data_QC_and_VIF.png")
    pdf_path = os.path.join(FIG_DIR, "Fig1B_data_QC_and_VIF.pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"  Fig1B saved:  {png_path}")
    print(f"  Fig1B saved:  {pdf_path}")


if __name__ == "__main__":
    plot_Fig1B()
