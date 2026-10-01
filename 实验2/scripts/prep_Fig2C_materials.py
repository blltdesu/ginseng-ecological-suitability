#!/usr/bin/env python3
"""
Prepare Fig2C plotting material package.
Outputs tidy data files for the interaction proxy (H*) figure.
"""
import os
import numpy as np
import pandas as pd

SRC = r"E:\人参种在哪\实验2\07_interactions"
OUT = r"E:\人参种在哪\绘图\Fig2C_绘图素材包"
DATA_DIR = os.path.join(OUT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

ALL_VARS = ["bio02", "bio03", "bio05", "bio15", "clay", "elevation", "northness", "sand"]
VAR_GROUP = {
    "bio02": "climate", "bio03": "climate", "bio05": "climate", "bio15": "climate",
    "clay": "soil", "sand": "soil",
    "elevation": "terrain", "northness": "terrain",
}
VAR_LABEL = {
    "bio02": "BIO02 Mean diurnal range",
    "bio03": "BIO03 Isothermality",
    "bio05": "BIO05 Max temp. of warmest month",
    "bio15": "BIO15 Precipitation seasonality",
    "clay": "Clay content",
    "sand": "Sand content",
    "elevation": "Elevation",
    "northness": "Northness",
}

def main():
    rank = pd.read_csv(os.path.join(SRC, "interaction_consensus_ranking.csv"))
    rank = rank.sort_values("interaction_rank").reset_index(drop=True)

    # ---- File 1: tidy Top-10 interaction table (lollipop-ready) ----
    tidy = pd.DataFrame({
        "rank": rank["interaction_rank"].astype(int),
        "var1": rank["variable_1"],
        "var2": rank["variable_2"],
        "pair_label": rank["variable_1"] + " x " + rank["variable_2"],
        "H_star": rank["consensus_H"].round(6),
        "support_models": rank["n_models_positive_H"].astype(int),
        "group_pair": rank["variable_1"].map(VAR_GROUP) + "-" + rank["variable_2"].map(VAR_GROUP),
        "involves_bio15": rank["variable_1"].eq("bio15") | rank["variable_2"].eq("bio15"),
    })
    tidy.to_csv(os.path.join(DATA_DIR, "Fig2C_interaction_top10_tidy.csv"), index=False)

    # ---- File 2: 8x8 symmetric heatmap matrix (H*), NaN = not evaluated ----
    mat = pd.DataFrame(np.nan, index=ALL_VARS, columns=ALL_VARS, dtype=float)
    for _, r in rank.iterrows():
        v1, v2 = r["variable_1"], r["variable_2"]
        mat.loc[v1, v2] = r["consensus_H"]
        mat.loc[v2, v1] = r["consensus_H"]
    mat.to_csv(os.path.join(DATA_DIR, "Fig2C_heatmap_matrix_8x8.csv"))

    # Support matrix (n models supporting), for annotation overlay
    sup = pd.DataFrame(np.nan, index=ALL_VARS, columns=ALL_VARS, dtype=float)
    for _, r in rank.iterrows():
        v1, v2 = r["variable_1"], r["variable_2"]
        sup.loc[v1, v2] = r["n_models_positive_H"]
        sup.loc[v2, v1] = r["n_models_positive_H"]
    sup.to_csv(os.path.join(DATA_DIR, "Fig2C_heatmap_support_matrix_8x8.csv"))

    # ---- File 3: network edges ----
    edges = tidy[["var1", "var2", "H_star", "support_models", "rank", "group_pair"]].copy()
    edges.to_csv(os.path.join(DATA_DIR, "Fig2C_network_edges.csv"), index=False)

    # ---- File 4: network nodes ----
    node_rows = []
    for v in ALL_VARS:
        sub = rank[(rank["variable_1"] == v) | (rank["variable_2"] == v)]
        node_rows.append({
            "variable": v,
            "label": VAR_LABEL[v],
            "group": VAR_GROUP[v],
            "n_interactions_evaluated": len(sub),
            "n_top3_interactions": int((sub["interaction_rank"] <= 3).sum()),
            "max_H_star": round(sub["consensus_H"].max(), 6) if len(sub) else np.nan,
            "mean_H_star": round(sub["consensus_H"].mean(), 6) if len(sub) else np.nan,
            "mean_support_models": round(sub["n_models_positive_H"].mean(), 3) if len(sub) else np.nan,
        })
    nodes = pd.DataFrame(node_rows)
    nodes.to_csv(os.path.join(DATA_DIR, "Fig2C_network_nodes.csv"), index=False)

    # ---- File 5: per-model H values (for optional support annotation) ----
    per_model = pd.read_csv(os.path.join(SRC, "H_statistic_by_model.csv"))
    pm = per_model.pivot_table(index=["variable_1", "variable_2"], columns="model",
                               values="H_statistic").reset_index()
    pm.columns.name = None
    pm = pm.rename(columns={"variable_1": "var1", "variable_2": "var2"})
    pm = pm.merge(tidy[["var1", "var2", "rank", "H_star", "support_models"]],
                  on=["var1", "var2"], how="right")
    pm = pm.sort_values("rank")
    pm.to_csv(os.path.join(DATA_DIR, "Fig2C_per_model_H_values.csv"), index=False)

    # ---- File 6: variable metadata ----
    meta = pd.DataFrame([{"variable": v, "label": VAR_LABEL[v], "group": VAR_GROUP[v]}
                         for v in ALL_VARS])
    meta.to_csv(os.path.join(DATA_DIR, "Fig2C_variable_metadata.csv"), index=False)

    print("Files written to", DATA_DIR)
    for f in sorted(os.listdir(DATA_DIR)):
        print("  -", f)
    print("\nSanity check (Top 3):")
    print(tidy.head(3).to_string(index=False))
    print("\nbio15 involvement:", int(tidy["involves_bio15"].sum()), "of", len(tidy), "pairs")
    print("Nodes:", nodes[["variable", "n_top3_interactions", "max_H_star"]].to_string(index=False))

if __name__ == "__main__":
    main()
