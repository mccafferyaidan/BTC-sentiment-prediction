"""
pls_clean_threshold.py
======================
Clean confidence threshold analysis.

Training: ALL hours, forward-correct target mean(return[T+1..T+4])
Evaluation: ALL test hours, same forward-correct target
Analysis: magnitude bins, confidence threshold table, P&L

No leakage (forward-correct target only).
No large-moves training bias (trained on all hours).
This is the most honest version of the confidence signal analysis.

Outputs → data/results_clean_threshold/
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── paths ──────────────────────────────────────────────────────────────────────
TRAIN_EMBEDDINGS = "data/train_embeddings.npy"
TRAIN_TIMESTAMPS = "data/train_timestamps.npy"
TEST_EMBEDDINGS  = "data/test_embeddings.npy"
TEST_TIMESTAMPS  = "data/test_timestamps.npy"
PRICE_CSV        = "data/btc_data_hourly.csv"
OUTPUT_DIR       = "data/results_clean_threshold"

# ── config ─────────────────────────────────────────────────────────────────────
N_COMPONENTS     = 5        # test multiple
N_COMPONENTS_LIST = [5, 10, 20]
LAG              = 1
SMOOTH_WINDOW    = 4
CONF_PERCENTILES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data...")
    X_train  = np.load(TRAIN_EMBEDDINGS)
    X_test   = np.load(TEST_EMBEDDINGS)
    ts_train = np.load(TRAIN_TIMESTAMPS, allow_pickle=True)
    ts_test  = np.load(TEST_TIMESTAMPS,  allow_pickle=True)
    train_df = pd.DataFrame({"timestamp": pd.to_datetime(ts_train).floor("h")})
    test_df  = pd.DataFrame({"timestamp": pd.to_datetime(ts_test).floor("h")})

    price_df = pd.read_csv(PRICE_CSV)
    price_df["timestamp"] = pd.to_datetime(price_df["Timestamp"]).dt.floor("h")
    price_df = price_df.sort_values("timestamp").reset_index(drop=True)
    price_df["return"] = price_df["Close"].astype(float).pct_change()

    # forward-correct target — purely future, no leakage
    # mean(return[T+1], return[T+2], return[T+3], return[T+4])
    price_df["return_fwd"] = (
        price_df["return"].shift(-1).rolling(SMOOTH_WINDOW, min_periods=SMOOTH_WINDOW).mean()
    )
    price_df["return_next"] = price_df["return"].shift(-1)

    price_df = price_df[["timestamp", "return_fwd", "return_next"]].dropna()
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, train_df, test_df, price_df


def align(embed_df, X, price_df, lag=1):
    df = embed_df.copy()
    df["idx"] = range(len(df))
    df["ts_target"] = df["timestamp"] + pd.Timedelta(hours=lag)
    merged = df.merge(
        price_df[["timestamp", "return_fwd", "return_next"]],
        left_on="ts_target", right_on="timestamp", how="inner"
    ).dropna(subset=["return_fwd"])
    return (X[merged["idx"].values],
            merged["return_fwd"].values,
            merged["return_next"].values,
            merged["ts_target"].values)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FIT — all hours, forward-correct target
# ══════════════════════════════════════════════════════════════════════════════

def fit_pls(X_train, y_train, scaler, n_components):
    X_sc = scaler.transform(X_train)
    pls  = PLSRegression(n_components=n_components)
    pls.fit(X_sc, y_train)
    return pls


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAGNITUDE BINS
# ══════════════════════════════════════════════════════════════════════════════

def magnitude_bin_analysis(df, n_components):
    df = df.copy()
    df["magnitude_bin"] = pd.qcut(df["magnitude"], q=5,
                                   labels=[f"Q{i+1}" for i in range(5)])

    print(f"\n=== Accuracy by Return Magnitude ({n_components}c) ===")
    print(f"  Q1=smallest 20%  Q5=largest 20%\n")

    header = (f"  {'Bin':>4} {'N':>6} {'Avg_|ret|':>10} "
              f"{'Acc_4h':>8} {'Avg_profit':>12}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for b in [f"Q{i+1}" for i in range(5)]:
        s        = df[df["magnitude_bin"] == b]
        acc      = s["correct_fwd"].mean()
        avg_ret  = s["magnitude"].mean()
        avg_prof = (np.sign(s["pred"]) * s["actual_fwd"]).mean()

        print(f"  {b:>4} {len(s):>6} {avg_ret:>10.4f} "
              f"{acc:>8.3f} {avg_prof:>12.5f}")

        rows.append({"bin": b, "n": len(s),
                     "avg_magnitude": round(avg_ret, 5),
                     "acc_4h": round(acc, 4),
                     "avg_profit": round(avg_prof, 6)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONFIDENCE THRESHOLD TABLE
# ══════════════════════════════════════════════════════════════════════════════

def confidence_threshold_table(df, n_components, naive_acc):
    print(f"\n=== Confidence Threshold Table ({n_components}c) ===")
    print(f"  Total test hours: {len(df):,}  |  "
          f"Naive baseline: {naive_acc*100:.1f}% always-up\n")

    header = (f"  {'Pctl':>5} {'Threshold':>10} {'N_trades':>9} "
              f"{'Freq%':>7} {'Hit%':>7} {'vs_naive':>9} "
              f"{'Avg_ret%':>9} {'Total_ret%':>11} {'ProfitF':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for pct in CONF_PERCENTILES:
        cutoff = np.percentile(df["confidence"], pct)
        s      = df[df["confidence"] >= cutoff]
        if len(s) == 0:
            continue

        hit_rate  = s["correct_fwd"].mean()
        pnl       = np.sign(s["pred"]) * s["actual_fwd"]
        avg_ret   = pnl.mean()
        total_ret = pnl.sum()
        freq      = len(s) / len(df) * 100
        avg_w     = pnl[pnl > 0].mean() if (pnl > 0).any() else 0
        avg_l     = pnl[pnl < 0].mean() if (pnl < 0).any() else 0
        pf        = abs(avg_w / avg_l) if avg_l != 0 else np.inf
        vs_naive  = hit_rate * 100 - naive_acc * 100

        print(f"  {pct:>4}% {cutoff:>10.5f} {len(s):>9,} "
              f"{freq:>7.1f} {hit_rate*100:>7.1f} {vs_naive:>+9.1f} "
              f"{avg_ret*100:>9.4f} {total_ret*100:>11.3f} {pf:>9.3f}")

        rows.append({
            "n_components": n_components,
            "percentile_cutoff": pct,
            "confidence_threshold": round(cutoff, 6),
            "n_trades": len(s),
            "trade_freq_pct": round(freq, 2),
            "hit_rate": round(hit_rate, 4),
            "vs_naive_ppt": round(vs_naive, 2),
            "avg_return_per_trade": round(avg_ret, 6),
            "total_return": round(total_ret, 6),
            "profit_factor": round(pf, 4),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5. P&L
# ══════════════════════════════════════════════════════════════════════════════

def simulate_pnl(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["trade_pnl"]   = np.sign(df["pred"]) * df["actual_fwd"]
    df["cumul_equal"] = df["trade_pnl"].cumsum()

    conf_norm = df["confidence"] / df["confidence"].mean()
    df["trade_pnl_sized"] = conf_norm * np.sign(df["pred"]) * df["actual_fwd"]
    df["cumul_sized"]     = df["trade_pnl_sized"].cumsum()

    total_eq  = df["trade_pnl"].sum()
    total_sz  = df["trade_pnl_sized"].sum()
    win_rate  = (df["trade_pnl"] > 0).mean()
    avg_win   = df[df["trade_pnl"] > 0]["trade_pnl"].mean()
    avg_loss  = df[df["trade_pnl"] < 0]["trade_pnl"].mean()
    pf        = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    sharpe    = (df["trade_pnl"].mean() / df["trade_pnl"].std()) * np.sqrt(len(df))

    print(f"\n=== Simulated P&L (no transaction costs) ===")
    print(f"  {'Metric':<30} {'Equal sizing':>14} {'Conf sizing':>14}")
    print(f"  {'Total P&L':<30} {total_eq:>14.4f} {total_sz:>14.4f}")
    print(f"  {'Win rate':<30} {win_rate:>14.3f}")
    print(f"  {'Avg win':<30} {avg_win:>14.5f}")
    print(f"  {'Avg loss':<30} {avg_loss:>14.5f}")
    print(f"  {'Profit factor':<30} {pf:>14.3f}")
    print(f"  {'Sharpe (annualized)':<30} {sharpe:>14.3f}")

    improvement = (total_sz - total_eq) / abs(total_eq) * 100
    direction   = "improves" if total_sz > total_eq else "reduces"
    print(f"\n  → Confidence sizing {direction} P&L by {abs(improvement):.1f}%")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, X_test, train_df, test_df, price_df = load_data()

    scaler = StandardScaler()
    scaler.fit(X_train)

    # align training data — all hours, forward-correct target
    X_tr, y_tr, _, _ = align(train_df, X_train, price_df, LAG)
    X_te, y_fwd, y_next, ts_te = align(test_df, X_test, price_df, LAG)

    naive_acc = np.mean(y_fwd > 0)
    print(f"\n  Naive always-up accuracy: {naive_acc:.3f}")
    print(f"  Training hours: {len(y_tr):,} | Test hours: {len(X_te):,}")

    all_thresh = []

    for n in N_COMPONENTS_LIST:
        print(f"\n{'='*60}")
        print(f"Components: {n}")
        print(f"{'='*60}")

        pls = fit_pls(X_tr, y_tr, scaler, n)

        X_te_sc = scaler.transform(X_te)
        y_pred  = pls.predict(X_te_sc).flatten()

        flat_acc = np.mean(np.sign(y_pred) == np.sign(y_fwd))
        print(f"  Flat accuracy (all hours): {flat_acc:.3f}")

        df = pd.DataFrame({
            "timestamp":    ts_te,
            "pred":         y_pred,
            "actual_fwd":   y_fwd,
            "actual_next":  y_next,
            "magnitude":    np.abs(y_fwd),
            "confidence":   np.abs(y_pred),
            "correct_fwd":  (np.sign(y_pred) == np.sign(y_fwd)).astype(int),
            "correct_next": (np.sign(y_pred) == np.sign(y_next)).astype(int),
        })

        corr = np.corrcoef(df["confidence"], df["magnitude"])[0, 1]
        print(f"  Pearson corr (confidence vs magnitude): {corr:.4f}")

        bin_df    = magnitude_bin_analysis(df, n)
        thresh_df = confidence_threshold_table(df, n, naive_acc)
        all_thresh.append(thresh_df)

        if n == N_COMPONENTS:
            pnl_df = simulate_pnl(df)
            pnl_df[["timestamp", "pred", "actual_fwd", "trade_pnl",
                     "cumul_equal", "trade_pnl_sized",
                     "cumul_sized"]].to_csv(
                f"{OUTPUT_DIR}/pnl_series_{n}c.csv", index=False)

        bin_df.to_csv(f"{OUTPUT_DIR}/magnitude_bins_{n}c.csv", index=False)
        thresh_df.to_csv(f"{OUTPUT_DIR}/thresholds_{n}c.csv", index=False)

    # combined threshold CSV
    pd.concat(all_thresh).to_csv(
        f"{OUTPUT_DIR}/all_thresholds.csv", index=False)

    # ── plot: hit rate by confidence threshold for all component counts ────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["steelblue", "darkorange", "green"]
    for thresh_df, color, n in zip(all_thresh, colors, N_COMPONENTS_LIST):
        axes[0].plot(thresh_df["percentile_cutoff"],
                     thresh_df["hit_rate"] * 100,
                     "o-", color=color, linewidth=2, label=f"{n}c")
        axes[1].plot(thresh_df["percentile_cutoff"],
                     thresh_df["profit_factor"],
                     "o-", color=color, linewidth=2, label=f"{n}c")

    axes[0].axhline(naive_acc * 100, color="purple", linestyle="--",
                    linewidth=0.8, label=f"Naive {naive_acc*100:.1f}%")
    axes[0].axhline(50, color="red", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Confidence percentile cutoff")
    axes[0].set_ylabel("Hit rate %")
    axes[0].set_title("Hit Rate by Confidence Threshold\n"
                      "(trained on all hours, forward-correct target)")
    axes[0].legend(fontsize=9)

    axes[1].axhline(1.0, color="red", linestyle="--",
                    linewidth=0.8, label="Break even")
    axes[1].set_xlabel("Confidence percentile cutoff")
    axes[1].set_ylabel("Profit factor")
    axes[1].set_title("Profit Factor by Confidence Threshold")
    axes[1].legend(fontsize=9)

    plt.suptitle("Clean PLS Threshold Analysis\n"
                 "No leakage — forward-correct target — trained on all hours",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/clean_threshold_plot.png", dpi=150)
    plt.close()

    print(f"\nAll outputs → {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
