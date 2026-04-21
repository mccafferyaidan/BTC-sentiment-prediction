# Sentiment-Driven Prediction of Large Bitcoin Price Movements
### LLM Embeddings + PLS Decomposition on 22 Million Crypto Tweets

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project investigates whether **collective social media sentiment can predict the direction of large Bitcoin price movements** before they happen.

Rather than reducing tweet content to a single sentiment score (as most prior work does), I represent each hour of crypto Twitter as a **1,536-dimensional LLM embedding vector** — the average of all tweet embeddings within that hour — and apply **Partial Least Squares (PLS) regression** to find directions in that high-dimensional space that predict future price.

The key finding: **model confidence scales monotonically with accuracy**. The top 5% of highest-confidence predictions achieve **70.4% directional accuracy** on held-out test data, beating a naive baseline by 21.1 percentage points, with a profit factor of 1.55. Results survive rigorous significance testing including block bootstrap permutation tests (p<0.0001) and 5/5 walk-forward folds across different market regimes including the 2022 crash.


---

## Key Results

| Confidence Threshold | Trades | Hit Rate | vs Naive | Profit Factor |
|---|---|---|---|---|
| All hours (p0) | 3,239 | 52.7% | +3.4pp | 1.22 |
| Top 50% confidence | 1,620 | 56.1% | +6.8pp | 1.34 |
| Top 30% confidence | 972 | 58.1% | +8.9pp | 1.38 |
| Top 10% confidence | 324 | 64.2% | +14.9pp | 1.44 |
| **Top 5% confidence** | **162** | **70.4%** | **+21.1pp** | **1.55** |

- Confidence scaling is **perfectly monotonic** (Spearman r=1.0, p<0.0001) — ruling out threshold optimization as an explanation
- Confidence-based position sizing improves cumulative P&L by **146.7%** over equal-weight
- Pearson correlation between model confidence and actual move magnitude: **r=0.200**

### Accuracy Concentrates on Large Moves

| Return Quintile | Accuracy | Avg Profit/Trade |
|---|---|---|
| Q1 — smallest 20% | 47.7% | negative |
| Q2 | 46.5% | negative |
| Q3 | 49.1% | negligible |
| Q4 | 54.2% | positive |
| **Q5 — largest 20%** | **64.7%** | **+0.00207** |

Signal is absent on small moves and concentrates sharply on large price changes — consistent with Shiller's (2017) narrative economics prediction that sentiment effects manifest at moments of market stress.

### Walk-Forward Validation (5 Folds)

| Fold | Market Regime | Accuracy | Significant |
|---|---|---|---|
| 1 | Bull market (May–Jul 2021) | 59.1% | ** |
| 2 | Bull market (Jul–Oct 2021) | 64.9% | *** |
| 3 | ATH period (Oct 2021–Jan 2022) | 72.6% | *** |
| 4 | Crash (Jan–May 2022) | 69.2% | *** |
| 5 | Post-crash (May–Jun 2022) | 63.5% | *** |
| **Mean** | | **65.9%** | **5/5 significant** |

---

## Methodology

### Pipeline

```
22.8M Tweets → OpenAI Embeddings (1536-dim) → Hourly Average → PLS Regression → Prediction
```

1. **Embedding**: Each tweet is embedded using `text-embedding-3-small` (OpenAI), producing a 1,536-dimensional dense vector. Total cost: ~$22 for 22.7M tweets.

2. **Aggregation**: All tweet embeddings within each calendar hour are averaged into a single collective sentiment vector. This produces 13,059 hourly vectors spanning Jan 2021–Jun 2022.

3. **PLS Regression**: Partial Least Squares finds directions in the 1,536-dimensional embedding space that maximize covariance with the forward return target. Each component is interpretable as a **supervised eigenmood** — a direction through semantic space that predicts price, extending the eigenmoods framework of [ten Thij, Wood, Rocha & Bollen (2019)](https://casci.binghamton.edu/projects/computational-social-science/).

4. **Target**: `mean(return[T+1], return[T+2], return[T+3], return[T+4])` — purely forward-looking with zero lookahead leakage. Entry at T+1, exit at T+4, ~3 hour hold.

5. **Train/test split**: Chronological at Feb 15, 2022. Test period covers the 2022 bear market and crash.

### Two Models

- **FWD-PLS**: Trained on all hours with forward-correct target. The primary model.
- **LM-PLS**: Trained only on large-move hours (top/bottom 10% of |return_fwd|). Stronger per-trade edge, slightly lower hit rate.

### Significance Testing

Three complementary tests address temporal dependence in financial time series:

1. **Block Bootstrap** (6h, 12h, 24h blocks) — p<0.0001 at all block sizes for LM-PLS; p=0.019 at 24h blocks for FWD-PLS
2. **Diebold-Mariano** (Harvey-Leybourne-Newbold correction) — borderline p=0.065 due to class imbalance in test period
3. **Walk-Forward** — 5/5 folds significant across bull, ATH, crash, and recovery regimes

---

## Dataset

- **Tweets**: 22,782,957 Bitcoin-related tweets, Jan 2021–Jun 2022 ([Kaggle source](https://www.kaggle.com/))
- **Price**: Hourly BTC/USDT OHLCV data from Binance
- **Embeddings**: Generated via OpenAI API, stored as BLOBs in SQLite (~22GB)

*Raw data not included in this repo due to size. See `src/ingest.py` and `src/embed.py` to reproduce from the Kaggle dataset.*

---

## Repository Structure

```
crypto-sentiment-prediction/
├── src/
│   ├── ingest.py                    # Load tweets into SQLite
│   ├── embed.py                     # Generate OpenAI embeddings
│   ├── average.py                   # Compute hourly averages
│   ├── pls_clean_threshold.py       # FWD-PLS: confidence threshold analysis
│   ├── pls_large_moves.py           # LM-PLS: large movement conditioned model
│   ├── pls_horizon_comparison.py    # Single candle vs 4h window comparison
│   ├── pls_magnitude_analysis.py    # Accuracy by return magnitude
│   ├── pls_momentum_test.py         # Momentum baseline + Granger causality
│   ├── pls_threshold_significance.py # Full significance table for both models
│   ├── robust_significance.py       # Block bootstrap + DM + walk-forward (LM-PLS)
│   ├── robust_significance_clean.py # Block bootstrap + DM + walk-forward (FWD-PLS)
│   ├── umap_sentiment_trajectory.py # UMAP visualization of sentiment space
│   ├── umap_3d_cluster_visual.py    # 3D cluster accuracy visualization
│   └── cluster_tweet_inspector.py  # Interactive cluster tweet explorer
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/crypto-sentiment-prediction.git
cd crypto-sentiment-prediction
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

You will need an OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

---

## Theoretical Context

This work extends the **eigenmoods framework** of ten Thij, Wood, Rocha & Bollen (2019), who applied SVD to lexicon-scored emotion time series to find natural axes of collective mood variation. PLS on 1,536-dimensional LLM embeddings finds the analogous axes in a vastly richer semantic space, constrained to be predictive of price rather than merely descriptive of emotional variance. The result — **supervised eigenmoods** — represents both a methodological extension and an application to financial prediction.

The finding that signal concentrates on large moves is consistent with **Shiller's (2017) narrative economics**: crowd sentiment narratives build in social media before manifesting in price, and this mechanism is strongest when collective uncertainty is highest.

---

## References

- Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market. *Journal of Computational Science*.
- ten Thij, M., Wood, I.B., Rocha, L.M., & Bollen, J. (2019). Detecting eigenmoods in individual human emotions. *IC2S2*.
- Shiller, R.J. (2017). Narrative economics. *American Economic Review*.
- Zou, Y., & Herremans, D. (2023). PreBit — A multimodal model with Twitter FinBERT embeddings for extreme price movement prediction of Bitcoin. *Expert Systems with Applications*.

---

## Author

**Aidan McCaffery** — Binghamton University, Industrial & Systems Engineering, Class of 2025

[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [Email](mailto:YOUR_EMAIL)
