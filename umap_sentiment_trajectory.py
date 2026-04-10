"""
umap_sentiment_trajectory.py
=============================
Visualizes the hourly sentiment vector trajectory through embedding space
using UMAP dimensionality reduction.

Each hour is a point in 2D/3D UMAP space. Points are connected by lines
showing how collective crypto Twitter sentiment moved over time.
Colored by price return to reveal whether sentiment trajectory patterns
precede or follow price movements.

Also computes velocity (hour-over-hour change) and visualizes step size
vs actual price return — the "slope of sentiment" analysis.

Outputs → data/results_umap/
  - umap_2d_trajectory.html  (interactive Plotly, colored by return)
  - umap_3d_trajectory.html  (interactive 3D version)
  - velocity_vs_return.png   (scatter of step size vs price return)
  - umap_data.csv            (UMAP coords + metadata for further analysis)
"""

import numpy as np
import pandas as pd
import os

# ── paths ──────────────────────────────────────────────────────────────────────
TRAIN_EMBEDDINGS = "data/train_embeddings.npy"
TRAIN_TIMESTAMPS = "data/train_timestamps.npy"
TEST_EMBEDDINGS  = "data/test_embeddings.npy"
TEST_TIMESTAMPS  = "data/test_timestamps.npy"
PRICE_CSV        = "data/btc_data_hourly.csv"
OUTPUT_DIR       = "data/results_umap"

# ── config ─────────────────────────────────────────────────────────────────────
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.1
UMAP_N_COMPONENTS = 2      # change to 3 for 3D
SAMPLE_EVERY      = 1      # use every hour (set higher to subsample if slow)


def load_all():
    print("Loading embeddings...")
    X_train  = np.load(TRAIN_EMBEDDINGS)
    X_test   = np.load(TEST_EMBEDDINGS)
    ts_train = np.load(TRAIN_TIMESTAMPS, allow_pickle=True)
    ts_test  = np.load(TEST_TIMESTAMPS,  allow_pickle=True)

    X_all  = np.vstack([X_train, X_test])
    ts_all = list(ts_train) + list(ts_test)

    df = pd.DataFrame({"timestamp": pd.to_datetime(ts_all).floor("h")})
    df["split"] = ["train"] * len(X_train) + ["test"] * len(X_test)

    price_df = pd.read_csv(PRICE_CSV)
    price_df["timestamp"] = pd.to_datetime(price_df["Timestamp"]).dt.floor("h")
    price_df = price_df.sort_values("timestamp").reset_index(drop=True)
    price_df["return"]     = price_df["Close"].astype(float).pct_change()
    price_df["return_fwd"] = price_df["return"].shift(-1).rolling(4, min_periods=4).mean()
    price_df["close"]      = price_df["Close"].astype(float)

    df = df.merge(
        price_df[["timestamp", "return", "return_fwd", "close"]],
        on="timestamp", how="left"
    )

    print(f"  Total hours: {len(X_all):,}")
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_all, df


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── install umap if needed ─────────────────────────────────────────────────
    try:
        import umap
    except ImportError:
        print("Installing umap-learn...")
        os.system("pip install umap-learn --break-system-packages -q")
        import umap

    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("Installing plotly...")
        os.system("pip install plotly --break-system-packages -q")
        import plotly.graph_objects as go
        import plotly.express as px

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from sklearn.preprocessing import StandardScaler

    X_all, df = load_all()

    # subsample if needed
    if SAMPLE_EVERY > 1:
        idx = np.arange(0, len(X_all), SAMPLE_EVERY)
        X_all = X_all[idx]
        df    = df.iloc[idx].reset_index(drop=True)

    # sort chronologically
    sort_idx = df["timestamp"].argsort().values
    df    = df.iloc[sort_idx].reset_index(drop=True)
    X_all = X_all[sort_idx]

    # normalize
    print("\nNormalizing embeddings...")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_all)

    # ── UMAP 2D ────────────────────────────────────────────────────────────────
    print(f"Running UMAP (n_neighbors={UMAP_N_NEIGHBORS}, "
          f"min_dist={UMAP_MIN_DIST}, 2D)...")
    reducer_2d = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        random_state=42,
        verbose=True
    )
    embedding_2d = reducer_2d.fit_transform(X_sc)
    print("  UMAP 2D done.")

    df["umap_x"] = embedding_2d[:, 0]
    df["umap_y"] = embedding_2d[:, 1]

    # ── velocity (step size between consecutive hours) ─────────────────────────
    print("\nComputing sentiment velocity in UMAP space...")
    umap_coords = embedding_2d
    velocity    = np.sqrt(np.sum(np.diff(umap_coords, axis=0)**2, axis=1))
    df["velocity"] = np.concatenate([[np.nan], velocity])

    # also compute velocity in original high-dimensional space
    raw_velocity = np.sqrt(np.sum(np.diff(X_all, axis=0)**2, axis=1))
    df["raw_velocity"] = np.concatenate([[np.nan], raw_velocity])

    # ── save CSV ───────────────────────────────────────────────────────────────
    df.to_csv(f"{OUTPUT_DIR}/umap_data.csv", index=False)
    print(f"  Data saved → {OUTPUT_DIR}/umap_data.csv")

    # ── 2D interactive Plotly ─────────────────────────────────────────────────
    print("\nBuilding 2D interactive plot...")
    df_plot = df.dropna(subset=["return_fwd"]).copy()

    # clip returns for color scale
    ret_clip = df_plot["return_fwd"].clip(
        df_plot["return_fwd"].quantile(0.02),
        df_plot["return_fwd"].quantile(0.98)
    )

    fig2d = go.Figure()

    # trajectory line (thin, chronological)
    fig2d.add_trace(go.Scatter(
        x=df_plot["umap_x"],
        y=df_plot["umap_y"],
        mode="lines",
        line=dict(color="rgba(150,150,150,0.15)", width=0.5),
        name="Trajectory",
        hoverinfo="skip",
        showlegend=True
    ))

    # points colored by forward return
    fig2d.add_trace(go.Scatter(
        x=df_plot["umap_x"],
        y=df_plot["umap_y"],
        mode="markers",
        marker=dict(
            color=ret_clip,
            colorscale="RdYlGn",
            size=4,
            opacity=0.7,
            colorbar=dict(title="4h fwd return"),
            line=dict(width=0, color="rgba(0,0,0,0)")
        ),
        text=[
            f"Time: {str(t)[:16]}<br>"
            f"Return: {r*100:.3f}%<br>"
            f"BTC: ${c:,.0f}<br>"
            f"Split: {s}"
            for t, r, c, s in zip(
                df_plot["timestamp"],
                df_plot["return_fwd"].fillna(0),
                df_plot["close"].fillna(0),
                df_plot["split"]
            )
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Hour",
        showlegend=False
    ))

    # mark train/test split
    split_ts = df_plot[df_plot["split"] == "test"]["timestamp"].min()
    split_row = df_plot[df_plot["timestamp"] == split_ts]
    if len(split_row) > 0:
        fig2d.add_trace(go.Scatter(
            x=[split_row["umap_x"].values[0]],
            y=[split_row["umap_y"].values[0]],
            mode="markers",
            marker=dict(size=12, color="black", symbol="star"),
            name="Train/Test split (Feb 2022)",
        ))

    fig2d.update_layout(
        title=dict(
            text="Crypto Twitter Collective Sentiment — UMAP Trajectory<br>"
                 "<sub>Each point = 1 hour of averaged tweet embeddings | "
                 "Color = 4h forward BTC return | Connected chronologically</sub>",
            x=0.5
        ),
        xaxis_title="UMAP dimension 1",
        yaxis_title="UMAP dimension 2",
        width=1000, height=700,
        template="plotly_white",
        legend=dict(x=0.01, y=0.99)
    )

    fig2d.write_html(f"{OUTPUT_DIR}/umap_2d_trajectory.html")
    print(f"  2D plot saved → {OUTPUT_DIR}/umap_2d_trajectory.html")

    # ── 3D interactive Plotly ─────────────────────────────────────────────────
    print("\nRunning UMAP 3D...")
    reducer_3d = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=3,
        random_state=42,
        verbose=False
    )
    embedding_3d = reducer_3d.fit_transform(X_sc)

    df["umap_x3"] = embedding_3d[:, 0]
    df["umap_y3"] = embedding_3d[:, 1]
    df["umap_z3"] = embedding_3d[:, 2]
    df_plot3 = df.dropna(subset=["return_fwd"]).copy()

    ret_clip3 = df_plot3["return_fwd"].clip(
        df_plot3["return_fwd"].quantile(0.02),
        df_plot3["return_fwd"].quantile(0.98)
    )

    fig3d = go.Figure()

    fig3d.add_trace(go.Scatter3d(
        x=df_plot3["umap_x3"],
        y=df_plot3["umap_y3"],
        z=df_plot3["umap_z3"],
        mode="lines",
        line=dict(color="rgba(150,150,150,0.1)", width=1),
        name="Trajectory",
        hoverinfo="skip"
    ))

    fig3d.add_trace(go.Scatter3d(
        x=df_plot3["umap_x3"],
        y=df_plot3["umap_y3"],
        z=df_plot3["umap_z3"],
        mode="markers",
        marker=dict(
            color=ret_clip3,
            colorscale="RdYlGn",
            size=3,
            opacity=0.6,
            colorbar=dict(title="4h fwd return")
        ),
        text=[
            f"Time: {str(t)[:16]}<br>Return: {r*100:.3f}%<br>BTC: ${c:,.0f}"
            for t, r, c in zip(
                df_plot3["timestamp"],
                df_plot3["return_fwd"].fillna(0),
                df_plot3["close"].fillna(0)
            )
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Hour"
    ))

    fig3d.update_layout(
        title=dict(
            text="Crypto Twitter Sentiment — UMAP 3D Trajectory<br>"
                 "<sub>Color = 4h forward BTC return (green=up, red=down)</sub>",
            x=0.5
        ),
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3"
        ),
        width=1000, height=700,
        template="plotly_dark"
    )

    fig3d.write_html(f"{OUTPUT_DIR}/umap_3d_trajectory.html")
    print(f"  3D plot saved → {OUTPUT_DIR}/umap_3d_trajectory.html")

    # ── velocity vs return scatter ─────────────────────────────────────────────
    print("\nBuilding velocity vs return analysis...")
    df_vel = df.dropna(subset=["velocity", "return_fwd"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # UMAP velocity vs return
    sc = axes[0].scatter(
        df_vel["velocity"],
        df_vel["return_fwd"] * 100,
        c=df_vel["return_fwd"],
        cmap="RdYlGn",
        alpha=0.3, s=4,
        vmin=df_vel["return_fwd"].quantile(0.02),
        vmax=df_vel["return_fwd"].quantile(0.98)
    )
    plt.colorbar(sc, ax=axes[0], label="4h fwd return")

    # trend line
    z = np.polyfit(df_vel["velocity"].dropna(),
                   (df_vel["return_fwd"] * 100).dropna(), 1)
    xline = np.linspace(df_vel["velocity"].min(), df_vel["velocity"].max(), 100)
    axes[0].plot(xline, np.poly1d(z)(xline), "k--", linewidth=1.5,
                 label=f"r={np.corrcoef(df_vel['velocity'].dropna(), df_vel['return_fwd'].dropna())[0,1]:.3f}")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].set_xlabel("Sentiment velocity (UMAP step size)")
    axes[0].set_ylabel("4h forward return %")
    axes[0].set_title("Sentiment Velocity vs Forward Return\n(UMAP space)")
    axes[0].legend()

    # raw high-dim velocity vs return magnitude
    df_vel2 = df.dropna(subset=["raw_velocity", "return_fwd"]).copy()
    axes[1].scatter(
        df_vel2["raw_velocity"],
        np.abs(df_vel2["return_fwd"]) * 100,
        alpha=0.2, s=4, color="steelblue"
    )
    corr2 = np.corrcoef(df_vel2["raw_velocity"], np.abs(df_vel2["return_fwd"]))[0, 1]
    z2 = np.polyfit(df_vel2["raw_velocity"],
                    np.abs(df_vel2["return_fwd"]) * 100, 1)
    xline2 = np.linspace(df_vel2["raw_velocity"].min(),
                          df_vel2["raw_velocity"].max(), 100)
    axes[1].plot(xline2, np.poly1d(z2)(xline2), "r--",
                 linewidth=1.5, label=f"r={corr2:.3f}")
    axes[1].set_xlabel("Sentiment velocity (1536-dim step size)")
    axes[1].set_ylabel("|4h forward return| %")
    axes[1].set_title("Sentiment Velocity vs |Return Magnitude|\n"
                       "(original 1536-dim space)")
    axes[1].legend()

    plt.suptitle("Does the speed of sentiment change predict price movement?",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/velocity_vs_return.png", dpi=150)
    plt.close()
    print(f"  Velocity plot saved → {OUTPUT_DIR}/velocity_vs_return.png")

    # ── print velocity correlations ────────────────────────────────────────────
    corr_umap = np.corrcoef(
        df_vel["velocity"], df_vel["return_fwd"])[0, 1]
    corr_raw  = np.corrcoef(
        df_vel2["raw_velocity"], np.abs(df_vel2["return_fwd"]))[0, 1]
    corr_raw_dir = np.corrcoef(
        df_vel2["raw_velocity"], df_vel2["return_fwd"])[0, 1]

    print(f"\n=== Velocity Correlations ===")
    print(f"  UMAP velocity vs return (direction):    {corr_umap:.4f}")
    print(f"  Raw velocity vs |return| (magnitude):   {corr_raw:.4f}")
    print(f"  Raw velocity vs return (direction):     {corr_raw_dir:.4f}")
    print(f"\n  Interpretation:")
    if abs(corr_raw) > 0.05:
        print(f"  → Sentiment velocity IS correlated with move magnitude")
        print(f"    Hours when crypto Twitter shifts topic/tone rapidly")
        print(f"    tend to precede larger price moves")
    else:
        print(f"  → Sentiment velocity not strongly correlated with magnitude")

    print(f"\nAll outputs → {OUTPUT_DIR}/")
    print(f"\nOpen the HTML files in a browser for interactive exploration:")
    print(f"  {OUTPUT_DIR}/umap_2d_trajectory.html")
    print(f"  {OUTPUT_DIR}/umap_3d_trajectory.html")


if __name__ == "__main__":
    run()
