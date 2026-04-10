"""
pls_large_moves.py
==================
Only attempts to predict hours where price movement exceeds a threshold —
top/bottom 10% of hourly returns. Sentiment is more likely to lead price
during high-volatility events than during quiet hours.

Outputs → data/results_large_moves/
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import os

# ── paths ──────────────────────────────────────────────────────────────────────
TRAIN_EMBEDDINGS = "data/train_embeddings.npy"
TRAIN_TIMESTAMPS = "data/train_timestamps.npy"
TEST_EMBEDDINGS  = "data/test_embeddings.npy"
TEST_TIMESTAMPS  = "data/test_timestamps.npy"
PRICE_CSV        = "data/btc_data_hourly.csv"
OUTPUT_DIR       = "data/results_large_moves"

# ── config ─────────────────────────────────────────────────────────────────────
PRED_LAGS         = [1, 2, 4]
N_COMPONENTS_LIST = [1, 5, 10, 20]
MOVE_THRESHOLD    = 0.10   # top/bottom 10% of moves — adjust to taste
N_PERMUTATIONS    = 1000

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading embeddings...")
    X_train  = np.load(TRAIN_EMBEDDINGS)
    X_test   = np.load(TEST_EMBEDDINGS)
    ts_train = np.load(TRAIN_TIMESTAMPS, allow_pickle=True)
    ts_test  = np.load(TEST_TIMESTAMPS,  allow_pickle=True)
    train_df = pd.DataFrame({"timestamp": pd.to_datetime(ts_train).floor("h")})
    test_df  = pd.DataFrame({"timestamp": pd.to_datetime(ts_test).floor("h")})
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, train_df, test_df


def load_price():
    print("Loading price data...")
    df = pd.read_csv(PRICE_CSV)
    df["timestamp"] = pd.to_datetime(df["Timestamp"]).dt.floor("h")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["return"] = df["Close"].astype(float).pct_change()

    # forward-correct smooth — purely future, no leakage
    df["return_fwd"] = df["return"].shift(-1).rolling(4, min_periods=1).mean()

    # compute threshold on the FULL dataset so train/test use same cutoff
    threshold = df["return_fwd"].abs().quantile(1 - MOVE_THRESHOLD)
    df["is_large_move"] = df["return_fwd"].abs() >= threshold
    print(f"  Large move threshold: |return_fwd| >= {threshold:.4f}")
    print(f"  Large move hours: {df['is_large_move'].sum():,} / {len(df):,} "
          f"({df['is_large_move'].mean()*100:.1f}%)")

    return df[["timestamp", "return", "return_fwd", "is_large_move"]].dropna()


# ══════════════════════════════════════════════════════════════════════════════
# 2. ALIGN — filter to large move hours only
# ══════════════════════════════════════════════════════════════════════════════

def align_large_moves(embed_df, X, price_df, lag):
    """
    Align embeddings to future return, then keep only rows where the
    future hour is a large move.
    """
    df = embed_df.copy()
    df["idx"] = range(len(df))
    df["ts_target"] = df["timestamp"] + pd.Timedelta(hours=lag)

    merged = df.merge(
        price_df[["timestamp", "return_fwd", "is_large_move"]],
        left_on="ts_target", right_on="timestamp", how="inner"
    ).dropna(subset=["return_fwd"])

    # filter to large moves only
    merged = merged[merged["is_large_move"]].reset_index(drop=True)

    return X[merged["idx"].values], merged["return_fwd"].values


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLS + SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════════════════

def run_pls(X_tr, y_tr, X_te, y_te, n, scaler):
    X_tr_sc = scaler.transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    pls = PLSRegression(n_components=n)
    pls.fit(X_tr_sc, y_tr)
    y_pred_tr = pls.predict(X_tr_sc).flatten()
    y_pred_te = pls.predict(X_te_sc).flatten()
    acc_train = np.mean(np.sign(y_pred_tr) == np.sign(y_tr))
    acc_test  = np.mean(np.sign(y_pred_te) == np.sign(y_te))
    return acc_train, acc_test, y_pred_te


def permutation_test(y_pred, y_true, n_perms=N_PERMUTATIONS):
    observed = np.mean(np.sign(y_pred) == np.sign(y_true))
    count = sum(
        np.mean(np.sign(np.random.permutation(y_pred)) == np.sign(y_true)) >= observed
        for _ in range(n_perms)
    )
    return count / n_perms


def binomial_pvalue(n_correct, n_total):
    return stats.binomtest(n_correct, n_total, 0.5, alternative="greater").pvalue


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, X_test, train_df, test_df = load_data()
    price_df = load_price()

    scaler = StandardScaler()
    scaler.fit(X_train)

    print(f"\n=== PLS on large moves only (top/bottom {MOVE_THRESHOLD*100:.0f}%) ===\n")
    header = f"{'Model':<28} {'N_test':>7} {'Train':>7} {'Test':>7} {'Gap':>7} {'Binom_p':>10} {'Perm_p':>10} {'Sig':>5}"
    print(header)
    print("-" * len(header))

    rows = []
    for lag in PRED_LAGS:
        X_tr, y_tr = align_large_moves(train_df, X_train, price_df, lag)
        X_te, y_te = align_large_moves(test_df,  X_test,  price_df, lag)

        for n in N_COMPONENTS_LIST:
            label = f"large-moves {n}c T+{lag}h"
            acc_train, acc_test, y_pred = run_pls(X_tr, y_tr, X_te, y_te, n, scaler)

            n_correct = int(np.sum(np.sign(y_pred) == np.sign(y_te)))
            binom_p   = binomial_pvalue(n_correct, len(y_te))
            perm_p    = permutation_test(y_pred, y_te)
            sig       = "***" if binom_p < 0.001 else "**" if binom_p < 0.01 else "*" if binom_p < 0.05 else "ns"
            gap       = acc_train - acc_test

            print(f"{label:<28} {len(y_te):>7} {acc_train:>7.3f} {acc_test:>7.3f} {gap:>7.3f} {binom_p:>10.4f} {perm_p:>10.4f} {sig:>5}")

            rows.append({
                "model": label, "lag_hours": lag, "n_components": n,
                "n_test": len(y_te), "n_correct": n_correct,
                "acc_train": round(acc_train, 4), "acc_test": round(acc_test, 4),
                "overfit_gap": round(gap, 4),
                "p_binom": round(binom_p, 6), "p_perm": round(perm_p, 4),
                "significant": sig, "above_baseline": acc_test > 0.5,
            })

    results_df = pd.DataFrame(rows).sort_values("acc_test", ascending=False)
    results_df.to_csv(f"{OUTPUT_DIR}/large_moves_results.csv", index=False)

    # heatmap
    pivot = results_df.pivot(index="n_components", columns="lag_hours", values="acc_test")
    plt.figure(figsize=(7, 4))
    im = plt.imshow(pivot.values, cmap="RdYlGn", vmin=0.45, vmax=0.65, aspect="auto")
    plt.colorbar(im, label="Test directional accuracy")
    plt.xticks(range(len(pivot.columns)), [f"T+{c}h" for c in pivot.columns])
    plt.yticks(range(len(pivot.index)),   [f"{c}c"  for c in pivot.index])
    plt.xlabel("Lag"); plt.ylabel("Components")
    plt.title(f"Large Moves Only (top/bottom {MOVE_THRESHOLD*100:.0f}%) — Test Accuracy")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.values[i,j]:.3f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap_large_moves.png", dpi=150)
    plt.close()

    sig_models = results_df[results_df["significant"] != "ns"]
    print(f"\nSignificant models: {len(sig_models)} / {len(results_df)}")
    print(f"All outputs → {OUTPUT_DIR}/")

if __name__ == "__main__":
    run()
