"""
umap_3d_cluster_visual.py
==========================
Simple 3D UMAP visualization colored by cluster-level model accuracy.
No analysis, just the visual.

Colors each HDBSCAN cluster by its average model accuracy.
Green = model does well in this sentiment regime.
Red = model does poorly.

Outputs → data/results_umap/umap_3d_cluster_accuracy.html
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import os

TRAIN_EMBEDDINGS = "data/train_embeddings.npy"
TRAIN_TIMESTAMPS = "data/train_timestamps.npy"
TEST_EMBEDDINGS  = "data/test_embeddings.npy"
TEST_TIMESTAMPS  = "data/test_timestamps.npy"
PRICE_CSV        = "data/btc_data_hourly.csv"
UMAP_CSV         = "data/results_umap/umap_data.csv"
OUTPUT_DIR       = "data/results_umap"

N_COMPONENTS  = 5
LAG           = 1
SMOOTH_WINDOW = 4


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        import plotly.graph_objects as go
        import hdbscan
        import umap as umap_lib
    except ImportError:
        os.system("pip install hdbscan plotly umap-learn -q")
        import plotly.graph_objects as go
        import hdbscan
        import umap as umap_lib

    # ── load everything ────────────────────────────────────────────────────────
    print("Loading data...")
    X_train  = np.load(TRAIN_EMBEDDINGS)
    X_test   = np.load(TEST_EMBEDDINGS)
    ts_train = np.load(TRAIN_TIMESTAMPS, allow_pickle=True)
    ts_test  = np.load(TEST_TIMESTAMPS,  allow_pickle=True)
    X_all    = np.vstack([X_train, X_test])
    ts_all   = list(ts_train) + list(ts_test)

    df = pd.DataFrame({"timestamp": pd.to_datetime(ts_all).floor("h")})
    sort_idx = df["timestamp"].argsort().values
    df    = df.iloc[sort_idx].reset_index(drop=True)
    X_all = X_all[sort_idx]

    price_df = pd.read_csv(PRICE_CSV)
    price_df["timestamp"] = pd.to_datetime(price_df["Timestamp"]).dt.floor("h")
    price_df = price_df.sort_values("timestamp").reset_index(drop=True)
    price_df["return"]     = price_df["Close"].astype(float).pct_change()
    price_df["return_fwd"] = price_df["return"].shift(-1).rolling(
        SMOOTH_WINDOW, min_periods=SMOOTH_WINDOW).mean()
    price_df = price_df[["timestamp", "return_fwd"]].dropna()

    # ── fit PLS and predict ────────────────────────────────────────────────────
    print("Fitting PLS and predicting...")
    train_df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts_train).floor("h")})
    train_df["idx"] = range(len(X_train))
    train_df["ts_target"] = train_df["timestamp"] + pd.Timedelta(hours=LAG)
    merged_tr = train_df.merge(
        price_df, left_on="ts_target", right_on="timestamp", how="inner"
    ).dropna(subset=["return_fwd"])

    scaler = StandardScaler()
    scaler.fit(X_train)
    pls = PLSRegression(n_components=N_COMPONENTS)
    pls.fit(scaler.transform(X_train[merged_tr["idx"].values]),
            merged_tr["return_fwd"].values)

    df["idx_all"] = range(len(df))
    df["ts_target"] = df["timestamp"] + pd.Timedelta(hours=LAG)
    merged_all = df.merge(
        price_df, left_on="ts_target", right_on="timestamp", how="inner"
    ).dropna(subset=["return_fwd"])

    y_pred = pls.predict(scaler.transform(X_all)).flatten()
    merged_all["pred"]    = y_pred[merged_all["idx_all"].values]
    merged_all["correct"] = (np.sign(merged_all["pred"]) ==
                              np.sign(merged_all["return_fwd"])).astype(int)

    # ── load 2D UMAP for clustering, compute 3D ───────────────────────────────
    print("Loading 2D UMAP coords for clustering...")
    umap_df = pd.read_csv(UMAP_CSV)
    umap_df["timestamp"] = pd.to_datetime(umap_df["timestamp"])

    merged = umap_df.merge(
        merged_all[["ts_target", "pred", "correct", "return_fwd"]],
        left_on="timestamp", right_on="ts_target", how="inner"
    )
    print(f"  {len(merged):,} hours")

    # cluster on 2D coords
    print("Clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=10)
    merged["cluster"] = clusterer.fit_predict(
        merged[["umap_x", "umap_y"]].values)

    n_clusters = len(set(merged["cluster"])) - (1 if -1 in merged["cluster"].values else 0)
    print(f"  {n_clusters} clusters found")

    # per-cluster accuracy
    cluster_acc = merged[merged["cluster"] >= 0].groupby("cluster")["correct"].mean()
    merged["cluster_acc"] = merged["cluster"].map(cluster_acc)

    # ── compute 3D UMAP ───────────────────────────────────────────────────────
    print("Computing 3D UMAP...")
    reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1,
                             n_components=3, random_state=42, verbose=False)
    coords3 = reducer.fit_transform(
        StandardScaler().fit_transform(X_all))

    # align 3D coords with merged df
    coord_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "x3": coords3[:, 0],
        "y3": coords3[:, 1],
        "z3": coords3[:, 2]
    })
    merged = merged.merge(coord_df, on="timestamp", how="left")

    # ── build 3D plot ─────────────────────────────────────────────────────────
    print("Building 3D plot...")
    fig = go.Figure()

    # noise points
    noise = merged[merged["cluster"] == -1]
    if len(noise) > 0:
        fig.add_trace(go.Scatter3d(
            x=noise["x3"], y=noise["y3"], z=noise["z3"],
            mode="markers",
            marker=dict(color="rgba(120,120,120,0.15)", size=2),
            name="Unclassified", hoverinfo="skip"
        ))

    # one trace per cluster, colored by accuracy
    for cid in sorted(cluster_acc.index):
        subset = merged[merged["cluster"] == cid]
        acc    = cluster_acc[cid]
        n      = len(subset)

        # green for high accuracy, red for low, centered at 50%
        if acc >= 0.5:
            intensity = min(1.0, (acc - 0.5) * 4)
            r = int(30  * (1 - intensity))
            g = int(200 * intensity + 80 * (1 - intensity))
            b = int(60  * (1 - intensity))
        else:
            intensity = min(1.0, (0.5 - acc) * 4)
            r = int(220 * intensity + 80 * (1 - intensity))
            g = int(60  * (1 - intensity))
            b = int(60  * (1 - intensity))
        color = f"rgb({r},{g},{b})"

        fig.add_trace(go.Scatter3d(
            x=subset["x3"], y=subset["y3"], z=subset["z3"],
            mode="markers",
            marker=dict(color=color, size=3, opacity=0.75,
                        line=dict(width=0)),
            text=[f"Cluster {cid}<br>Accuracy: {acc:.1%}<br>N={n}"
                  for _ in range(len(subset))],
            hovertemplate="%{text}<extra></extra>",
            name=f"C{cid}  {acc:.0%}  n={n}"
        ))

    fig.update_layout(
        title=dict(
            text="Crypto Twitter Sentiment Clusters — 3D UMAP<br>"
                 "<sub>Each cluster colored by model accuracy | "
                 "Green = model predicts well here | Red = model struggles</sub>",
            x=0.5, font=dict(size=14)),
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            bgcolor="rgb(8, 10, 18)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            zaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        ),
        width=1100, height=750,
        template="plotly_dark",
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            font=dict(size=10),
            title=dict(text="Cluster / Accuracy / N")
        )
    )

    out = f"{OUTPUT_DIR}/umap_3d_cluster_accuracy.html"
    fig.write_html(out)
    print(f"\nSaved → {out}")
    print("Open in browser to explore interactively.")

    # quick summary
    print(f"\nCluster accuracy range: "
          f"{cluster_acc.min():.1%} — {cluster_acc.max():.1%}")
    print(f"Std dev: {cluster_acc.std():.4f}")


if __name__ == "__main__":
    run()
