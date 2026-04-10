"""
pls_momentum_test.py
====================
Tests whether the PLS sentiment model is capturing genuine sentiment
signal or merely learning market momentum.

Four tests:

1. Pure Momentum Baseline
   Predict next 4h return direction using only lagged price returns.
   If this matches PLS accuracy, sentiment adds nothing.

2. PLS vs Momentum Comparison
   Direct accuracy comparison: momentum only vs sentiment only vs combined.

3. Granger Causality
   Tests in both directions:
     - Does sentiment Granger-cause price? (our claim)
     - Does price Granger-cause sentiment? (the concern)
   Uses the first PLS component score as the sentiment proxy.

4. Regime Analysis
   Splits test period into trending vs choppy regimes.
   Momentum capture would show high accuracy in trends, poor in chop.
   Genuine sentiment signal should be more consistent.

Outputs -> data/results_momentum/
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import os
warnings.filterwarnings("ignore")

TRAIN_EMBEDDINGS = "data/train_embeddings.npy"
TRAIN_TIMESTAMPS = "data/train_timestamps.npy"
TEST_EMBEDDINGS  = "data/test_embeddings.npy"
TEST_TIMESTAMPS  = "data/test_timestamps.npy"
PRICE_CSV        = "data/btc_data_hourly.csv"
OUTPUT_DIR       = "data/results_momentum"

N_COMPONENTS   = 5
LAG            = 1
SMOOTH_WINDOW  = 4
N_LAGS_MOMENTUM = 4      # how many lagged returns to use in momentum model
N_LAGS_GRANGER  = 8      # max lag for Granger test
TREND_WINDOW    = 12     # hours to compute trend strength


# ==============================================================================
# LOAD
# ==============================================================================

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

    # forward-correct target
    price_df["return_fwd"] = (
        price_df["return"].shift(-1)
        .rolling(SMOOTH_WINDOW, min_periods=SMOOTH_WINDOW).mean()
    )

    # lagged returns for momentum model
    for i in range(1, N_LAGS_MOMENTUM + 1):
        price_df[f"return_lag{i}"] = price_df["return"].shift(i)

    # trend strength: absolute value of rolling mean return (high = trending)
    price_df["trend_strength"] = (
        price_df["return"].rolling(TREND_WINDOW, min_periods=TREND_WINDOW).mean().abs()
    )

    price_df = price_df.dropna()
    return X_train, X_test, train_df, test_df, price_df


def align(embed_df, X, price_df):
    df = embed_df.copy()
    df["idx"] = range(len(df))
    df["ts_target"] = df["timestamp"] + pd.Timedelta(hours=LAG)
    lag_cols = ["return_fwd"] + [f"return_lag{i}" for i in range(1, N_LAGS_MOMENTUM+1)] + ["trend_strength"]
    merged = df.merge(
        price_df[["timestamp"] + lag_cols],
        left_on="ts_target", right_on="timestamp", how="inner"
    ).dropna()
    return X[merged["idx"].values], merged, merged["ts_target"].values


# ==============================================================================
# FIT MODELS
# ==============================================================================

def fit_pls(X_tr, y_tr, scaler, n_components):
    pls = PLSRegression(n_components=n_components)
    pls.fit(scaler.transform(X_tr), y_tr)
    return pls


def fit_momentum(X_momentum_tr, y_tr):
    """Logistic regression on lagged returns only — no embeddings."""
    scaler_m = StandardScaler()
    X_sc = scaler_m.fit_transform(X_momentum_tr)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_sc, np.sign(y_tr))
    return lr, scaler_m


# ==============================================================================
# TEST 1 & 2: MOMENTUM BASELINE + COMPARISON
# ==============================================================================

def momentum_comparison(X_train, train_merged, X_test, test_merged,
                        scaler, n_components):
    print(f"\n{'='*65}")
    print("TEST 1 & 2: Momentum Baseline vs Sentiment vs Combined")
    print(f"{'='*65}")

    y_tr = train_merged["return_fwd"].values
    y_te = test_merged["return_fwd"].values
    lag_cols = [f"return_lag{i}" for i in range(1, N_LAGS_MOMENTUM+1)]

    # ── pure momentum model ───────────────────────────────────────────────────
    X_mom_tr = train_merged[lag_cols].values
    X_mom_te = test_merged[lag_cols].values
    lr_mom, scaler_mom = fit_momentum(X_mom_tr, y_tr)
    y_pred_mom = lr_mom.predict(scaler_mom.transform(X_mom_te))
    acc_mom = np.mean(y_pred_mom == np.sign(y_te))

    # ── pure sentiment model (PLS) ────────────────────────────────────────────
    pls = fit_pls(X_train, y_tr, scaler, n_components)
    y_pred_pls_raw = pls.predict(scaler.transform(X_test)).flatten()
    acc_pls = np.mean(np.sign(y_pred_pls_raw) == np.sign(y_te))

    # ── combined model: PLS score + lagged returns ───────────────────────────
    pls_scores_tr = pls.transform(scaler.transform(X_train))
    pls_scores_te = pls.transform(scaler.transform(X_test))

    X_comb_tr = np.hstack([pls_scores_tr, X_mom_tr])
    X_comb_te = np.hstack([pls_scores_te, X_mom_te])
    lr_comb, scaler_comb = fit_momentum(X_comb_tr, y_tr)
    y_pred_comb = lr_comb.predict(scaler_comb.transform(X_comb_te))
    acc_comb = np.mean(y_pred_comb == np.sign(y_te))

    # naive
    naive_acc = np.mean(y_te > 0)

    print(f"\n  Test hours: {len(y_te):,}")
    print(f"  Naive (always-up) accuracy: {naive_acc:.3f}\n")
    print(f"  {'Model':<40} {'Accuracy':>10} {'vs Naive':>10} {'Binom_p':>10}")
    print(f"  " + "-" * 74)

    results = {}
    for name, acc in [
        ("Pure Momentum (lagged returns only)", acc_mom),
        ("Pure Sentiment (PLS embeddings only)", acc_pls),
        ("Combined (PLS + momentum)", acc_comb),
    ]:
        n_correct = int(acc * len(y_te))
        p = stats.binomtest(n_correct, len(y_te), 0.5, alternative="greater").pvalue
        vs = (acc - naive_acc) * 100
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {name:<40} {acc:>10.3f} {vs:>+10.2f}pp {p:>8.4f} {sig}")
        results[name] = {"accuracy": acc, "vs_naive": vs, "p": p}

    print(f"\n  Key question: does PLS outperform pure momentum?")
    gap = acc_pls - acc_mom
    if gap > 0.01:
        print(f"  -> YES: PLS beats momentum by {gap*100:.1f}pp")
        print(f"     Sentiment adds information beyond price history")
    elif gap > -0.01:
        print(f"  -> SIMILAR: PLS and momentum perform comparably ({gap*100:.1f}pp gap)")
        print(f"     Sentiment may be partially capturing momentum")
    else:
        print(f"  -> NO: Momentum outperforms PLS by {-gap*100:.1f}pp")
        print(f"     WARNING: PLS may be capturing momentum signal")

    print(f"\n  Does combining PLS + momentum beat either alone?")
    if acc_comb > max(acc_pls, acc_mom) + 0.005:
        print(f"  -> YES: Combined ({acc_comb:.3f}) > best single model")
        print(f"     Sentiment and momentum contain independent information")
    else:
        print(f"  -> MARGINAL: Combined ({acc_comb:.3f}) not clearly better")

    return results, y_pred_pls_raw, y_te, test_merged


# ==============================================================================
# TEST 3: GRANGER CAUSALITY
# ==============================================================================

def granger_causality_test(price_df, train_df, X_train, scaler, n_components):
    print(f"\n{'='*65}")
    print("TEST 3: Granger Causality")
    print(f"{'='*65}")
    print(f"  Does sentiment Granger-cause price? (our claim)")
    print(f"  Does price Granger-cause sentiment? (the concern)\n")

    # get PLS component 1 score for all training hours as sentiment proxy
    train_df2 = train_df.copy()
    train_df2["idx"] = range(len(train_df2))
    train_df2["ts_target"] = train_df2["timestamp"] + pd.Timedelta(hours=LAG)

    price_merge = pd.read_csv(PRICE_CSV)
    price_merge["timestamp"] = pd.to_datetime(price_merge["Timestamp"]).dt.floor("h")
    price_merge = price_merge.sort_values("timestamp").reset_index(drop=True)
    price_merge["return"] = price_merge["Close"].astype(float).pct_change()
    price_merge["return_fwd"] = (
        price_merge["return"].shift(-1)
        .rolling(SMOOTH_WINDOW, min_periods=SMOOTH_WINDOW).mean()
    )
    price_merge = price_merge[["timestamp", "return", "return_fwd"]].dropna()

    merged = train_df2.merge(
        price_merge, left_on="ts_target", right_on="timestamp", how="inner"
    ).dropna()

    # fit PLS and extract component 1 scores
    y_tr = merged["return_fwd"].values
    X_tr = scaler.transform(X_train[merged["idx"].values])
    pls  = PLSRegression(n_components=n_components)
    pls.fit(X_tr, y_tr)
    sentiment_scores = pls.transform(X_tr)[:, 0]  # component 1

    returns = merged["return"].values

    # build bivariate dataframe
    n = min(len(sentiment_scores), len(returns))
    granger_df = pd.DataFrame({
        "sentiment": sentiment_scores[:n],
        "price_return": returns[:n]
    }).dropna()

    print(f"  Series length: {len(granger_df):,} hours")
    print(f"  Max lag tested: {N_LAGS_GRANGER} hours\n")

    # test 1: does price return Granger-cause sentiment?
    print(f"  H0: Price does NOT Granger-cause sentiment")
    print(f"  (if rejected, sentiment is reactive to price)\n")
    try:
        gc1 = grangercausalitytests(
            granger_df[["sentiment", "price_return"]],
            maxlag=N_LAGS_GRANGER, verbose=False
        )
        print(f"  {'Lag':>5} {'F-stat':>10} {'p-value':>10} {'Verdict':>12}")
        print(f"  " + "-" * 42)
        reactive_lags = []
        for lag, res in gc1.items():
            f_stat = res[0]["ssr_ftest"][0]
            p_val  = res[0]["ssr_ftest"][1]
            verdict = "REACTIVE" if p_val < 0.05 else "ok"
            if p_val < 0.05:
                reactive_lags.append(lag)
            print(f"  {lag:>5} {f_stat:>10.3f} {p_val:>10.4f} {verdict:>12}")
        if reactive_lags:
            print(f"\n  -> Price Granger-causes sentiment at lags: {reactive_lags}")
            print(f"     Sentiment is partially REACTIVE to price at these lags")
        else:
            print(f"\n  -> Price does NOT Granger-cause sentiment at any lag")
            print(f"     No evidence sentiment is merely tracking price")
    except Exception as e:
        print(f"  ERROR: {e}")

    # test 2: does sentiment Granger-cause price?
    print(f"\n  H0: Sentiment does NOT Granger-cause price return")
    print(f"  (if rejected, sentiment leads price — our claim)\n")
    try:
        gc2 = grangercausalitytests(
            granger_df[["price_return", "sentiment"]],
            maxlag=N_LAGS_GRANGER, verbose=False
        )
        print(f"  {'Lag':>5} {'F-stat':>10} {'p-value':>10} {'Verdict':>12}")
        print(f"  " + "-" * 42)
        predictive_lags = []
        for lag, res in gc2.items():
            f_stat = res[0]["ssr_ftest"][0]
            p_val  = res[0]["ssr_ftest"][1]
            verdict = "PREDICTIVE" if p_val < 0.05 else "ns"
            if p_val < 0.05:
                predictive_lags.append(lag)
            print(f"  {lag:>5} {f_stat:>10.3f} {p_val:>10.4f} {verdict:>12}")
        if predictive_lags:
            print(f"\n  -> Sentiment Granger-causes price at lags: {predictive_lags}")
            print(f"     Sentiment has genuine PREDICTIVE content for price")
        else:
            print(f"\n  -> Sentiment does NOT Granger-cause price at any lag")
    except Exception as e:
        print(f"  ERROR: {e}")


# ==============================================================================
# TEST 4: REGIME ANALYSIS
# ==============================================================================

def regime_analysis(y_pred_pls, y_te, test_merged):
    print(f"\n{'='*65}")
    print("TEST 4: Trending vs Choppy Regime Analysis")
    print(f"{'='*65}")
    print(f"  Momentum capture -> high accuracy in trends, poor in chop")
    print(f"  Genuine sentiment -> more consistent across regimes\n")

    trend_strength = test_merged["trend_strength"].values
    median_trend   = np.median(trend_strength)

    trending = trend_strength >= median_trend
    choppy   = ~trending

    results = []
    for name, mask in [("Trending (strong momentum)", trending),
                        ("Choppy (weak momentum)", choppy)]:
        if mask.sum() < 10:
            continue
        y_p = y_pred_pls[mask]
        y_t = y_te[mask]
        acc = np.mean(np.sign(y_p) == np.sign(y_t))
        n   = mask.sum()
        p   = stats.binomtest(int(acc*n), n, 0.5, alternative="greater").pvalue
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {name:<35} N={n:>5}  Acc={acc:.3f}  p={p:.4f} {sig}")
        results.append({"regime": name, "n": n, "accuracy": round(acc, 4),
                         "p": round(p, 4), "sig": sig})

    df = pd.DataFrame(results)
    if len(df) == 2:
        trend_acc = df[df["regime"].str.contains("Trending")]["accuracy"].values[0]
        chop_acc  = df[df["regime"].str.contains("Choppy")]["accuracy"].values[0]
        gap = trend_acc - chop_acc
        print(f"\n  Accuracy gap (trending - choppy): {gap*100:+.1f}pp")
        if abs(gap) < 0.02:
            print(f"  -> CONSISTENT: model performs similarly in both regimes")
            print(f"     Strong evidence against pure momentum capture")
        elif gap > 0.02:
            print(f"  -> REGIME DEPENDENT: model does better in trending markets")
            print(f"     Some momentum capture likely — note this as a limitation")
        else:
            print(f"  -> COUNTER-MOMENTUM: model does better in choppy markets")
            print(f"     Sentiment signal is contrarian to momentum")
    return df


# ==============================================================================
# MAIN
# ==============================================================================

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        os.system("pip install statsmodels -q")

    X_train, X_test, train_df, test_df, price_df = load_data()

    scaler = StandardScaler()
    scaler.fit(X_train)

    # align both sets
    X_tr, train_merged, _ = align(train_df, X_train, price_df)
    X_te, test_merged,  _ = align(test_df,  X_test,  price_df)

    # run all tests
    model_results, y_pred_pls, y_te, test_merged = momentum_comparison(
        X_tr, train_merged, X_te, test_merged, scaler, N_COMPONENTS)

    granger_causality_test(price_df, train_df, X_train, scaler, N_COMPONENTS)

    regime_df = regime_analysis(y_pred_pls, y_te, test_merged)

    # save results
    pd.DataFrame([
        {"test": k, **v} for k, v in model_results.items()
    ]).to_csv(f"{OUTPUT_DIR}/momentum_comparison.csv", index=False)

    regime_df.to_csv(f"{OUTPUT_DIR}/regime_analysis.csv", index=False)

    print(f"\n{'='*65}")
    print(f"All outputs -> {OUTPUT_DIR}/")
    print(f"\nSummary of momentum capture concern:")
    print(f"  If PLS > momentum AND sentiment Granger-causes price")
    print(f"  AND accuracy is consistent across regimes:")
    print(f"  -> Sentiment signal is genuine, not momentum capture")


if __name__ == "__main__":
    run()
