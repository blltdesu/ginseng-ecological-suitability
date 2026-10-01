# -*- coding: utf-8 -*-
"""
Prepare Fig3C material package (relative suitable-area loss).

Builds two tidy CSVs from the Experiment 4 area statistics:
  - Fig3C_area_loss_summary.csv : one row per SSP x Period (8 rows) with
    mean / SD / min / max of the relative loss percentage across GCMs.
  - Fig3C_area_loss_by_gcm.csv  : one row per GCM x SSP x Period (36 rows)
    for the jittered scatter / spread overlay.

Only relative percentages are exported; absolute km2 values are kept out
of the plotting materials because they depend on the global prediction
domain (spatial-scope audit for Figure 3).

Output folder: E:\\人参种在哪\\绘图\\Fig3C_绘图素材包
"""
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(r"E:\人参种在哪\实验4\10_area_statistics\scenario_area_statistics.csv")
OUT = Path(r"E:\人参种在哪\绘图\Fig3C_绘图素材包")

SSP_LABEL = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5",
             "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with open(SRC, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # GCM-level tidy table (relative values only)
    gcm_rows = []
    for r in rows:
        gcm_rows.append({
            "gcm": r["gcm"],
            "ssp": r["ssp"],
            "ssp_label": SSP_LABEL[r["ssp"]],
            "period": r["period"],
            "loss_pct": round(float(r["loss_pct"]), 3),
            "persistence_pct": round(float(r["persistence_pct"]), 3),
        })
    with open(OUT / "Fig3C_area_loss_by_gcm.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gcm_rows[0].keys()))
        w.writeheader()
        w.writerows(gcm_rows)

    # per SSP x Period summary
    groups = {}
    for r in gcm_rows:
        groups.setdefault((r["ssp"], r["period"]), []).append(r["loss_pct"])

    summary = []
    for (ssp, period), vals in sorted(groups.items()):
        n = len(vals)
        mean = sum(vals) / n
        sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
        summary.append({
            "ssp": ssp,
            "ssp_label": SSP_LABEL[ssp],
            "period": period,
            "n_gcms": n,
            "loss_pct_mean": round(mean, 2),
            "loss_pct_sd": round(sd, 3),
            "loss_pct_min": round(min(vals), 2),
            "loss_pct_max": round(max(vals), 2),
        })
    with open(OUT / "Fig3C_area_loss_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    for s in summary:
        print(s)
    print(f"total GCM-level rows: {len(gcm_rows)}")


if __name__ == "__main__":
    main()
