"""
robust_significance_clean.py
=============================
Three-test significance battery for the clean FWD-PLS model.

Identical structure to robust_significance.py but uses:
  - All training hours (no large-moves filter)
  - Forward-correct target mean(return[T+1..T+4])
  - No leakage, no training distribution bias

This tests whether the clean unconditional model has genuine
statistical significance independent of the large-moves training.

Outputs -> data/results_significance_clean/
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings("ignore")

# -- paths ---------------------------------------------------------------------
TRAIN_EMBEDDINGS = "data/train_embeddings.npy"
TRAIN_TIMESTAMPS = "data/train_timestamps.npy"
TEST_EMBEDDINGS  = "data/test_embeddings.npy"
TEST_TIMESTAMPS  = "data/test_timestamps.npy"
PRICE_CSV        = "data/btc_data_hourly.csv"
OUTPUT_DIR       = "data/results_significance_clean"

# -- config --------------------------------------------------------------------
N_COMPONENTS   = 5
LAG            = 1
SMOOTH_WINDOW  = 4
N_BLOCK_PERMS  = 1000
BLOCK_SIZES    = [6, 12, 24]
WALK_FOLDS     = 5
MIN_TRAIN_HOURS = 1500   # all-hours so we have more to work with


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def load_all():
    print("Loading embeddings and price data...")
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

    # forward-correct target -- no leakage
    price_df["return_fwd"] = (
        price_df["return"].shift(-1)
        .rolling(SMOOTH_WINDOW, min_periods=SMOOTH_WINDOW).mean()
    )
    price_df = price_df[["timestamp", "return_fwd"]].dropna()

    print(f"  Train: {len(X_train):,} hours | Test: {len(X_test):,} hours")
    return X_train, X_test, train_df, test_df, price_df


def align(embed_df, X, price_df, lag=1):
    """Align embeddings to forward return -- ALL hours, no filter."""
    df = embed_df.copy()
    df["idx"] = range(len(df))
    df["ts_target"] = df["timestamp"] + pd.Timedelta(hours=lag)
    merged = df.merge(
        price_df[["timestamp", "return_fwd"]],
        left_on="ts_target", right_on="timestamp", how="inner"
    ).dropna(subset=["return_fwd"])
    return (
        X[merged["idx"].values],
        merged["return_fwd"].values,
        merged["ts_target"].values
    )


# ==============================================================================
# 2. FIT CLEAN MODEL
# ==============================================================================

def fit_clean(X_train, train_df, price_df, scaler, n_components):
    """Fit PLS on ALL training hours with forward-correct target."""
    X_tr, y_tr, _ = align(train_df, X_train, price_df, LAG)
    X_tr_sc = scaler.transform(X_tr)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_tr_sc, y_tr)
    print(f"  Trained on {len(y_tr):,} hours (all hours, forward-correct target)")
    return pls


# ==============================================================================
# 3. BLOCK BOOTSTRAP
# ==============================================================================

def block_bootstrap_test(y_pred, y_true, block_size, n_perms=N_BLOCK_PERMS):
    observed_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
    n = len(y_pred)
    n_blocks = int(np.ceil(n / block_size))
    blocks = [y_pred[i*block_size : min((i+1)*block_size, n)]
              for i in range(n_blocks)]
    count = 0
    for _ in range(n_perms):
        shuffled = [blocks[i] for i in np.random.permutation(n_blocks)]
        y_perm = np.concatenate(shuffled)[:n]
        if np.mean(np.sign(y_perm) == np.sign(y_true)) >= observed_acc:
            count += 1
    return observed_acc, count / n_perms


# ==============================================================================
# 4. DIEBOLD-MARIANO
# ==============================================================================

def diebold_mariano_test(y_pred, y_true, lag=1):
    n = len(y_true)
    majority_sign = np.sign(np.sum(y_true))
    y_naive = np.full(n, majority_sign)

    loss_model = (np.sign(y_pred) != np.sign(y_true)).astype(float)
    loss_naive = (y_naive       != np.sign(y_true)).astype(float)
    d = loss_naive - loss_model
    d_bar = np.mean(d)

    T = len(d)
    h = lag
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = gamma_0
    for k in range(1, h + 1):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2 * (1 - k / (h + 1)) * gamma_k

    var_d_bar = gamma_sum / T
    if var_d_bar <= 0:
        return d_bar, 0.0, np.nan, np.nan, 0.0, 0.0

    dm_stat = d_bar / np.sqrt(var_d_bar)
    hlp_stat = dm_stat * np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    p_value = stats.t.sf(hlp_stat, df=T-1)

    model_acc = 1 - np.mean(loss_model)
    naive_acc = 1 - np.mean(loss_naive)
    return d_bar, p_value, dm_stat, hlp_stat, model_acc, naive_acc


# ==============================================================================
# 5. WALK-FORWARD VALIDATION
# ==============================================================================

def walk_forward_validation(X_all, all_df, price_df,
                             n_folds=WALK_FOLDS,
                             min_train=MIN_TRAIN_HOURS,
                             n_components=N_COMPONENTS):
    print("\n" + "="*60)
    print("TEST 3: Walk-Forward Validation (all hours)")
    print("="*60)

    all_df = all_df.copy()
    all_df["idx"] = range(len(all_df))
    all_df["ts_target"] = all_df["timestamp"] + pd.Timedelta(hours=LAG)
    merged = all_df.merge(
        price_df[["timestamp", "return_fwd"]],
        left_on="ts_target", right_on="timestamp", how="inner"
    ).dropna(subset=["return_fwd"])
    merged = merged.sort_values("ts_target").reset_index(drop=True)

    X_aligned = X_all[merged["idx"].values]
    y_aligned  = merged["return_fwd"].values
    ts_aligned = merged["ts_target"].values

    total     = len(merged)
    fold_size = (total - min_train) // n_folds

    if fold_size < 30:
        print(f"  WARNING: fold_size={fold_size} too small. Reduce n_folds or min_train.")

    print(f"  Total hours: {total:,}")
    print(f"  Min training: {min_train:,}")
    print(f"  Fold size: ~{fold_size}\n")

    header = f"{'Fold':>5} {'Train_N':>9} {'Test_N':>7} {'Acc':>7} {'Binom_p':>10} {'Sig':>5} {'Test_Period'}"
    print(header)
    print("-" * len(header))

    results = []
    for fold in range(n_folds):
        train_end  = min_train + fold * fold_size
        test_start = train_end
        test_end   = min(test_start + fold_size, total)
        if test_end <= test_start:
            break

        X_tr = X_aligned[:train_end]
        y_tr = y_aligned[:train_end]
        X_te = X_aligned[test_start:test_end]
        y_te = y_aligned[test_start:test_end]
        ts_te = ts_aligned[test_start:test_end]

        scaler = StandardScaler()
        scaler.fit(X_tr)
        pls = PLSRegression(n_components=n_components)
        pls.fit(scaler.transform(X_tr), y_tr)
        y_pred = pls.predict(scaler.transform(X_te)).flatten()

        acc = np.mean(np.sign(y_pred) == np.sign(y_te))
        n_correct = int(np.sum(np.sign(y_pred) == np.sign(y_te)))
        binom_p = stats.binomtest(n_correct, len(y_te), 0.5,
                                  alternative="greater").pvalue
        sig = "***" if binom_p < 0.001 else "**" if binom_p < 0.01 \
              else "*" if binom_p < 0.05 else "ns"

        period = f"{str(ts_te[0])[:10]} to {str(ts_te[-1])[:10]}"
        print(f"{fold+1:>5} {train_end:>9,} {len(y_te):>7} {acc:>7.3f} "
              f"{binom_p:>10.4f} {sig:>5}  {period}")

        results.append({
            "fold": fold+1, "train_n": train_end, "test_n": len(y_te),
            "acc": round(acc, 4), "p_binom": round(binom_p, 6),
            "significant": sig, "period": period
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  ERROR: No folds generated.")
        return df

    n_sig    = (df["significant"] != "ns").sum()
    mean_acc = df["acc"].mean()
    print(f"\n  Significant folds: {n_sig} / {len(df)}")
    print(f"  Mean accuracy: {mean_acc:.3f}")
    print(f"  Range: {df['acc'].min():.3f} -- {df['acc'].max():.3f}")

    if mean_acc > 0.5 and n_sig >= len(df) // 2:
        print(f"\n  Signal is CONSISTENT across time periods")
    elif mean_acc > 0.5:
        print(f"\n  Signal present but INCONSISTENT across folds")
    else:
        print(f"\n  Signal does NOT hold across time periods")
    return df
# ==============================================================================
# 7. TEMPORAL STABILITY TEST (HALVES + QUARTERS)
# ==============================================================================

def temporal_stability_test(y_pred, y_true):

    print("\n" + "="*60)
    print("TEST 4: Temporal Stability (Halves and Quarters)")
    print("="*60)

    n = len(y_true)

    splits = {
        "Full Sample": (0, n),
        "First Half": (0, n//2),
        "Second Half": (n//2, n),
        "Q1": (0, n//4),
        "Q2": (n//4, n//2),
        "Q3": (n//2, 3*n//4),
        "Q4": (3*n//4, n)
    }

    header = f"{'Segment':>12} {'N':>7} {'Acc':>7} {'Binom_p':>10} {'Sig':>5}"
    print(header)
    print("-"*len(header))

    results = []

    for name, (start, end) in splits.items():

        y_p = y_pred[start:end]
        y_t = y_true[start:end]

        if len(y_t) == 0:
            continue

        correct = np.sum(np.sign(y_p) == np.sign(y_t))
        acc = correct / len(y_t)

        p_val = stats.binomtest(correct, len(y_t), 0.5,
                                alternative="greater").pvalue

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 \
              else "*" if p_val < 0.05 else "ns"

        print(f"{name:>12} {len(y_t):>7} {acc:>7.3f} {p_val:>10.4f} {sig:>5}")

        results.append({
            "segment": name,
            "n": len(y_t),
            "accuracy": acc,
            "p_value": p_val,
            "significant": sig
        })

    return pd.DataFrame(results)

# ==============================================================================
# 6. MAIN
# ==============================================================================

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, X_test, train_df, test_df, price_df = load_all()

    scaler = StandardScaler()
    scaler.fit(X_train)

    print(f"\nFitting clean FWD-PLS ({N_COMPONENTS}c, all hours, forward-correct target)...")
    pls = fit_clean(X_train, train_df, price_df, scaler, N_COMPONENTS)

    print("\nApplying to all test hours...")
    X_te, y_te, ts_te = align(test_df, X_test, price_df, LAG)
    y_pred = pls.predict(scaler.transform(X_te)).flatten()

    base_acc  = np.mean(np.sign(y_pred) == np.sign(y_te))
    naive_acc = np.mean(y_te > 0)
    print(f"  Test accuracy (all hours): {base_acc:.3f}")
    print(f"  Naive always-up accuracy:  {naive_acc:.3f}")

    # -- TEST 4: Temporal Stability ----------------------------------------------
    stab_df = temporal_stability_test(y_pred, y_te)
    stab_df.to_csv(f"{OUTPUT_DIR}/temporal_stability_results.csv", index=False)
    block_results = []

    # -- TEST 1: Block Bootstrap -----------------------------------------------
    print("\n" + "="*60)
    print("TEST 1: Block Bootstrap Permutation Test")
    print("="*60)
    print(f"  {N_BLOCK_PERMS} permutations per block size\n")

    header = f"{'Block_Size':>12} {'Obs_Acc':>9} {'p_value':>10} {'Sig':>5}"
    print(header)
    print("-" * len(header))



    for bs in BLOCK_SIZES:
        obs_acc, p_val = block_bootstrap_test(y_pred, y_te, block_size=bs)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 \
              else "*" if p_val < 0.05 else "ns"
        print(f"{bs:>10}h   {obs_acc:>9.3f} {p_val:>10.4f} {sig:>5}")
        block_results.append({
            "block_size_hours": bs, "observed_acc": round(obs_acc, 4),
            "p_value": round(p_val, 4), "significant": sig
        })

    pd.DataFrame(block_results).to_csv(
        f"{OUTPUT_DIR}/block_bootstrap_results.csv", index=False)

    # -- TEST 2: Diebold-Mariano -----------------------------------------------
    print("\n" + "="*60)
    print("TEST 2: Diebold-Mariano Test (HAC variance)")
    print("="*60)
    print(f"  Model vs naive baseline (always predict majority class)\n")

    d_bar, p_dm, dm_stat, hlp_stat, model_acc, naive_acc_dm = \
        diebold_mariano_test(y_pred, y_te, lag=1)

    sig_dm = "***" if p_dm < 0.001 else "**" if p_dm < 0.01 \
             else "*" if p_dm < 0.05 else "ns"

    print(f"  Model accuracy:      {model_acc:.3f}")
    print(f"  Naive accuracy:      {naive_acc_dm:.3f}")
    print(f"  Mean loss diff:      {d_bar:.4f}  (positive = model better)")
    print(f"  DM statistic:        {dm_stat:.4f}")
    print(f"  HLN statistic:       {float(hlp_stat):.4f}")
    print(f"  p-value (one-sided): {p_dm:.6f}  {sig_dm}")

    pd.DataFrame([{
        "model_acc": round(model_acc, 4),
        "naive_acc": round(naive_acc_dm, 4),
        "mean_loss_diff": round(d_bar, 6),
        "dm_statistic": round(dm_stat, 4),
        "hln_statistic": round(float(hlp_stat), 4),
        "p_value_onesided": round(p_dm, 6),
        "significant": sig_dm
    }]).to_csv(f"{OUTPUT_DIR}/diebold_mariano_results.csv", index=False)

    # -- TEST 3: Walk-Forward --------------------------------------------------
    X_all  = np.vstack([X_train, X_test])
    all_ts = list(train_df["timestamp"]) + list(test_df["timestamp"])
    all_df = pd.DataFrame({"timestamp": pd.to_datetime(all_ts).floor("h")})

    wf_df = walk_forward_validation(X_all, all_df, price_df)
    wf_df.to_csv(f"{OUTPUT_DIR}/walk_forward_results.csv", index=False)

    # -- PLOT ------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    bdf = pd.DataFrame(block_results)
    colors = ["green" if s != "ns" else "gray" for s in bdf["significant"]]
    ax1.bar([f"{b}h blocks" for b in bdf["block_size_hours"]],
            -np.log10(bdf["p_value"].clip(1e-4)), color=colors, alpha=0.8)
    ax1.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=0.8, label="p=0.05")
    ax1.axhline(-np.log10(0.001), color="purple", linestyle="--", linewidth=0.8, label="p=0.001")
    ax1.set_ylabel("-log10(p-value)")
    ax1.set_title("Block Bootstrap\n(clean FWD-PLS, all hours)")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(["Model", "Naive"], [model_acc, naive_acc_dm],
            color=["steelblue", "salmon"], alpha=0.8, width=0.4)
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    ax2.set_ylim(0.4, 0.7)
    ax2.set_ylabel("Directional Accuracy")
    ax2.set_title(f"Diebold-Mariano\np={p_dm:.4f} {sig_dm}")
    for i, v in enumerate([model_acc, naive_acc_dm]):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)

    if len(wf_df) > 0:
        ax3 = fig.add_subplot(gs[1, :])
        fold_colors = ["green" if s != "ns" else "steelblue"
                       for s in wf_df["significant"]]
        ax3.bar(range(len(wf_df)), wf_df["acc"], color=fold_colors, alpha=0.8)
        ax3.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="50% baseline")
        ax3.axhline(base_acc, color="black", linestyle="--", linewidth=0.8,
                    label=f"Full test acc ({base_acc:.3f})")
        ax3.set_xticks(range(len(wf_df)))
        ax3.set_xticklabels([f"Fold {r['fold']}\n{r['period']}"
                             for _, r in wf_df.iterrows()], fontsize=7)
        ax3.set_ylabel("Accuracy")
        ax3.set_ylim(0.3, 0.75)
        ax3.set_title("Walk-Forward Validation (clean FWD-PLS)")
        ax3.legend(fontsize=8)
        for i, (acc, sig) in enumerate(zip(wf_df["acc"], wf_df["significant"])):
            ax3.text(i, acc + 0.01, f"{acc:.3f}\n{sig}", ha="center", fontsize=8)

    plt.suptitle(f"Robust Significance — Clean FWD-PLS {N_COMPONENTS}c\n"
                 f"(all hours, forward-correct target, no leakage)", fontsize=11)
    plt.savefig(f"{OUTPUT_DIR}/robust_significance_clean.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # -- SUMMARY ---------------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL SUMMARY — Clean FWD-PLS")
    print("="*60)
    print(f"\nModel: FWD-PLS {N_COMPONENTS}c T+1h (all hours, forward-correct target)")
    print(f"Test accuracy: {base_acc:.3f} on {len(y_te):,} test hours\n")

    print("Test 1 -- Block Bootstrap:")
    for r in block_results:
        print(f"  {r['block_size_hours']}h blocks: p={r['p_value']:.4f} {r['significant']}")

    print(f"\nTest 2 -- Diebold-Mariano:")
    print(f"  p={p_dm:.6f} {sig_dm} | HLN={float(hlp_stat):.4f}")

    if len(wf_df) > 0:
        print(f"\nTest 3 -- Walk-Forward:")
        print(f"  Mean accuracy: {wf_df['acc'].mean():.3f}")
        print(f"  Significant folds: {(wf_df['significant'] != 'ns').sum()} / {len(wf_df)}")

    print(f"\nAll outputs -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
