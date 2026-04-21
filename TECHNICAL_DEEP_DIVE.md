# Technical Deep Dive — causal-stock-predictor

This document explains every component of the system in detail: what it does, how it works, why it was built this way, and what each result means. It is intended as a companion to the research paper.

---

## Table of Contents

1. [The Core Problem](#1-the-core-problem)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Data Collection and Feature Engineering](#3-data-collection-and-feature-engineering)
4. [Target Variable Design](#4-target-variable-design)
5. [Causal Discovery — Granger Causality](#5-causal-discovery--granger-causality)
6. [Causal Discovery — PCMCI](#6-causal-discovery--pcmci)
7. [Feature Selection Logic](#7-feature-selection-logic)
8. [PCMCI Stability Analysis](#8-pcmci-stability-analysis)
9. [The Ensemble Model](#9-the-ensemble-model)
10. [The Stacking Meta-Learner](#10-the-stacking-meta-learner)
11. [Data Splits and Leakage Prevention](#11-data-splits-and-leakage-prevention)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Regime-Based Evaluation](#13-regime-based-evaluation)
14. [Statistical Significance Tests](#14-statistical-significance-tests)
15. [Confidence Calibration](#15-confidence-calibration)
16. [Paper Validation Checks](#16-paper-validation-checks)
17. [Design Decisions and Tradeoffs](#17-design-decisions-and-tradeoffs)
18. [Known Limitations](#18-known-limitations)

---

## 1. The Core Problem

Standard machine learning models for stock prediction face two deep problems:

**Problem 1 — Spurious correlations.** Most features are correlated with stock returns during the training period simply because they share a common trend, or because a confounding variable drives both. Correlation is not causation. A model built on all correlated features learns relationships that may not hold out-of-sample, particularly during market regime shifts (crashes, rate hike cycles, etc.).

**Problem 2 — Regime instability.** Even if a feature is genuinely predictive in one market regime (e.g., a bull market), it may stop working in another regime (e.g., a rate hike cycle). Models that overfit to regime-specific correlations collapse when the regime changes.

**The hypothesis this project tests**: If we first filter features to only those that are *causally* linked to returns (using PCMCI conditional independence testing), the resulting model will:
1. Use fewer, more robust features.
2. Degrade less across regime shifts compared to a model trained on all correlated features.

This is the central empirical claim of the research paper.

---

## 2. Pipeline Overview

The pipeline runs in six sequential steps:

```
Step 1: Load historical data
  └── yFinance for US stocks, uploaded CSV for India
  └── Saved to ml/data/raw/

Step 2: Feature engineering
  └── ~30 candidate features across 6 categories
  └── Saved to ml/data/processed/features/{TICKER}_features.csv

Step 3: Causal discovery (on training split only)
  ├── Granger causality — baseline
  └── PCMCI / ParCorr — primary method
  └── Selector combines both → causal_features_{TICKER}.json

Step 4: Train ensemble (on training split)
  ├── LightGBM
  ├── XGBoost
  ├── ARIMA
  └── Ridge meta-learner (on validation set predictions)

Step 5: Live prediction + regime evaluation
  ├── Live inference (US stocks only)
  └── Option B regime performance table

Step 6: Full paper regime backtest (opt-in, --paper-eval)
  └── Option B: train on pre-regime data, test on each regime
```

Everything except Step 6 runs automatically when you call `python run_pipeline.py --ticker AAPL`.

---

## 3. Data Collection and Feature Engineering

### Data sources

**US stocks**: yFinance for OHLCV (Open/High/Low/Close/Volume) from 2010 to present. Finnhub for news sentiment.

**India (NIFTY)**: Uploaded CSV files with pre-built features. Real-time P/E, P/B, and news headlines are not available for the NIFTY index, so live prediction is skipped for Indian market tickers.

### Feature categories

The system builds approximately 30 candidate features across six categories. These are the inputs to causal discovery — not all of them will survive.

**Technical features** (`ml/src/features/technical.py`):
- RSI (relative strength index) — momentum indicator
- MACD (moving average convergence/divergence) — trend strength
- Bollinger Band position — measures where price sits relative to recent range
- Short-term momentum (5-day, 20-day price change)
- Historical volatility (rolling standard deviation of daily returns)
- Volume change

**Macro features** (`ml/src/features/macro.py`):
- VIX change — the market's "fear gauge"; spikes signal regime stress
- Yield spread (10Y minus 2Y Treasury) — inverted yield curves historically precede recessions
- DXY (US Dollar Index) — strong dollar often hurts multinationals and commodities
- Oil price change — cost input for many sectors
- Gold return — risk-off / safe haven signal
- S&P 500 excess return vs stock — measures relative performance

**Sentiment features** (`ml/src/features/sentiment.py`):
- Finnhub news sentiment score (rolling mean and std)
- Sentiment momentum (change in sentiment over 5-day window)
- Optional FinBERT scoring for headlines (`--finbert` flag); see below

**Earnings features** (`ml/src/features/earnings.py`):
- EPS surprise (actual minus estimate)
- Forward P/E ratio
- P/B ratio

**Options features** (`ml/src/features/options.py`):
- Put/call ratio — measures market hedging activity
- Implied volatility — market-priced uncertainty

**Sector features** (`ml/src/features/sector.py`):
- Sector relative performance vs market
- Sector momentum

### FinBERT sentiment

FinBERT is a BERT-based model fine-tuned on financial text. When `--finbert` is passed, headline text is scored using FinBERT rather than the simpler Finnhub sentiment score. This is slower (GPU recommended) but produces more semantically nuanced sentiment features. Enable with:

```bash
python run_pipeline.py --ticker AAPL --finbert
```

---

## 4. Target Variable Design

The model predicts a **5-day forward log return**. There are two variants:

### `excess_return_5d` (default for US individual stocks)

```
excess_return_5d[t] = log(Price[t+5] / Price[t]) - log(Benchmark[t+5] / Benchmark[t])
```

Where `Benchmark` is the S&P 500 (SPY). This measures *alpha* — the stock's return above and beyond what the overall market did. It removes market-wide movements, so if the whole market went up 2%, only the stock-specific component remains.

**Why this is better for individual stock prediction**: Most of the variance in a single stock's 5-day return comes from market-wide moves. A model that just "predicted the market went up" would look good without learning anything stock-specific. Excess return forces the model to learn what drives *this* stock beyond market beta.

### `log_return_5d` (for indexes like NIFTY)

```
log_return_5d[t] = log(Price[t+5] / Price[t])
```

For market indexes, subtracting the index from itself yields zero. So NIFTY uses raw log return. The pipeline automatically detects this case and applies the correct target.

### Why log returns (not price or percentage returns)?

Log returns have desirable statistical properties: they are approximately normally distributed, they are additive over time, and they do not diverge like price levels. This makes them well-suited for the regression and statistical testing framework used here.

### The forward-looking problem

Both target columns contain **future price information** — they are computed using `shift(-5)` internally. This means these columns must be **excluded from the causal discovery step** and treated with care during training. If either target column is included as a feature in PCMCI or Granger, the algorithm will trivially find it as causal (it literally is the answer). This is handled explicitly in Step 3.

---

## 5. Causal Discovery — Granger Causality

**File**: `ml/src/causal/granger.py`

### What Granger causality tests

For each candidate feature X, Granger causality tests the hypothesis:

> H₀: Past values of X do NOT help predict future values of Y, beyond Y's own past values.

If we can reject H₀ (p-value below the threshold), we call X "Granger-causal" for Y. Under the hood, this fits two VAR (Vector Autoregression) models:
- **Restricted**: Y predicted from its own lags only
- **Unrestricted**: Y predicted from its own lags + lags of X

Then an F-test (or chi-squared test) compares the two. A significantly better fit when X is included means X has predictive content for Y.

### Implementation details

The system tests each feature independently against the target, using lags 1 through `max_lag` (from config). The minimum p-value across all tested lags is taken as the feature's Granger score.

**Multiple testing correction**: With ~30 features being tested, there is a high risk of false positives (by chance, some features will appear significant). The code applies Benjamini-Hochberg FDR (False Discovery Rate) correction at the configured significance level. This reduces the expected proportion of false discoveries among the features marked causal.

**Contiguous block handling**: If a feature has gaps (NaN values), the test runs on the longest contiguous block of non-missing data, rather than dropping all rows globally (which would distort the time structure).

### Why Granger is the baseline, not the primary method

Granger causality has a critical weakness: it does not control for confounders. If both X and Y are driven by a common cause Z, Granger will incorrectly flag X as causing Y even though the link is entirely through Z. For example, VIX and many individual stocks' volatility are both driven by market-wide stress. Granger might flag VIX as causal for a stock's return just because both react to the same underlying events.

This is precisely why PCMCI is used as the primary method.

---

## 6. Causal Discovery — PCMCI

**File**: `ml/src/causal/pcmci.py`

### What PCMCI tests

PCMCI (Peter-Clark Momentary Conditional Independence) is an algorithm from the `tigramite` library. It tests **conditional independence** in time series: X causes Y at lag τ only if the link holds after conditioning on *all other variables* in the system.

Formally, it tests:

> X(t-τ) ⊥⊥ Y(t) | **Z**

Where **Z** is the set of all other candidate variables at relevant lags. This controls for confounders — if X and Y both react to Z, conditioning on Z removes that shared component, and the X→Y link disappears.

### How PCMCI works (two phases)

**Phase 1 — PC (Peter-Clark) skeleton**: For each variable pair, PCMCI builds a causal skeleton by iteratively conditioning on increasingly large sets of variables and removing links that become independent. This prunes the search space.

**Phase 2 — MCI (Momentary Conditional Independence)**: For each remaining link X(t-τ) → Y(t), PCMCI tests conditional independence while controlling for:
- All parents of X at all tested lags
- All parents of Y at all tested lags

This two-phase approach is computationally feasible while still providing rigorous confounder control.

### Conditional independence test: ParCorr

The default test is **ParCorr** (Partial Correlation). This is a linear test that measures whether X and Y are correlated after removing the linear effects of the conditioning set. For the 5-day return prediction task (which is approximately linear in the features), ParCorr is appropriate and fast.

Alternative tests configured in `config.yaml`:
- `GPDC` — Gaussian Process Distance Correlation: non-linear, but much slower
- `CMIknn` — k-nearest-neighbour CMI: fully non-parametric, slowest

For publication, ParCorr is the standard choice in financial time series applications due to its well-understood analytical significance thresholds.

### The `exclude_target` flag

In `run_pipeline.py`, PCMCI is called with `exclude_target=True`. This removes the target variable from the set of variables PCMCI tests over. Why?

The target (`excess_return_5d`) is a forward-looking variable — it contains information about the *future*. If it is included in the PCMCI variable set, PCMCI's conditional independence tests for other features will condition on a variable that contains future information. This can create spurious links or suppress real ones through "collider bias" — a well-known causal structure problem.

When `exclude_target=True`, PCMCI runs on the feature set only, producing an outgoing-link summary for each feature (strongest link to any other variable). The Granger test (which explicitly tests feature→target) is then used as the direct feature-to-target signal. The selector combines both.

### Output

For each feature, PCMCI produces:
- `causal`: bool — is the link significant?
- `pval`: minimum p-value across all tested lags
- `val`: the test statistic value
- `best_lag`: the lag (in days) at which the causal link is strongest

These are saved in `ml/saved_models/causal_features_{TICKER}.json` with full metadata.

---

## 7. Feature Selection Logic

**File**: `ml/src/causal/selector.py`

The selector combines Granger and PCMCI results into the final feature list. This is what actually gets saved and used for model training.

### Default strategy: Intersection

A feature must appear causal in **both** Granger **and** PCMCI to be selected. This is the strictest filter:

```
selected = {f : granger_causal(f) AND pcmci_causal(f)}
```

Intersection makes the strongest paper claim: the selected features pass two independent statistical tests with different assumptions (predictive causality AND conditional independence).

### Adaptive fallback chain

When intersection yields too few features (below `min_causal_features` in config), the selector does not fail — it relaxes criteria in order:

1. **Adaptive intersection**: Iteratively relax Granger and PCMCI p-value thresholds (e.g., p < 0.05 → p < 0.10) until enough features pass.
2. **Union fallback**: Include features passing **either** Granger or PCMCI at any threshold.
3. **Top-N by PCMCI p-value**: Take the N most significant PCMCI features regardless of the Granger result.

Every fallback is recorded in the output JSON under `selection_info`, so it is transparent which path was taken. This matters for the paper: if union was used, that should be disclosed.

For NIFTY (which has less data than 15 years of US market history), the selector may fall back to union. This is expected and documented.

### Output JSON structure

```json
{
  "ticker": "AAPL",
  "strategy": "intersection",
  "n_features": 8,
  "selection_info": {
    "method": "intersection_bool",
    "n": 8
  },
  "features": [
    {
      "name": "vix_change_1d",
      "granger_causal": true,
      "granger_pval": 0.003,
      "pcmci_causal": true,
      "pcmci_pval": 0.008,
      "pcmci_lag": 1
    },
    ...
  ]
}
```

The `pcmci_lag` field indicates how many trading days ago this feature has its strongest causal link to returns. A `lag=2` for VIX means that 2 days ago's VIX change is the most predictive signal, not yesterday's.

---

## 8. PCMCI Stability Analysis

**File**: `ml/src/causal/stability.py`

### The problem it solves

Running PCMCI once on the full training set and claiming causal discovery is statistically weak. If the same features are not consistently discovered across different time periods, then the "causal" claim is really just "the algorithm happened to find these features on this particular data window." The paper needs to demonstrate that the selected features are *stable*.

### How it works

The training data is split into `n_windows` (default: 3) non-overlapping sub-windows. PCMCI runs independently on each window. The output is the set of causal features for each window.

**Jaccard similarity** between two feature sets A and B is:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Jaccard = 1.0 means perfect agreement. Jaccard = 0.0 means no overlap. All pairwise Jaccard scores are computed and averaged.

### Stability thresholds

| Mean Jaccard | Verdict | Recommended action |
|---|---|---|
| ≥ 0.60 | STABLE | Use intersection strategy; strong causal claim |
| 0.35 – 0.60 | MODERATE | Use majority-core strategy; acknowledge partial instability |
| < 0.35 | UNSTABLE | Use union strategy; disclose instability in paper |

### Core feature sets

- **Stable core**: Features appearing in ALL windows — the most robust claim
- **Majority core**: Features appearing in more than 50% of windows — a softer claim
- **Any window (union)**: All features discovered in at least one window

For the paper, the stable core is the ideal set to report in Table 1. If the stable core is empty or very small, the paper should use the majority core and report the stability statistics honestly.

---

## 9. The Ensemble Model

**File**: `ml/src/ensemble.py`

Three base models are combined because no single model dominates across all market conditions.

### LightGBM (`ml/src/models/lgbm_model.py`)

LightGBM is a gradient boosting framework that builds an ensemble of decision trees. It is the primary model (highest weight) because:

- It handles non-linear interactions between features
- It is robust to outliers (common in financial returns)
- It naturally provides feature importance scores
- It integrates with SHAP for explainability

**SHAP values** (SHapley Additive exPlanations) are computed for each prediction. These decompose the prediction into per-feature contributions, providing the "causal drivers" shown in the live output. A positive SHAP value for VIX means "this stock's VIX reading pushed the predicted return upward for this particular prediction."

### XGBoost (`ml/src/models/xgb_model.py`)

XGBoost is an alternative gradient boosting framework. It uses a slightly different tree-building algorithm (level-wise vs. leaf-wise in LightGBM). Including both:

- Adds diversity to the ensemble — they make different errors on different inputs
- Provides a robustness check: if LightGBM and XGBoost agree on a prediction, it's more reliable

### ARIMA (`ml/src/models/arima_model.py`)

ARIMA (AutoRegressive Integrated Moving Average) captures **linear autocorrelation** in the return series itself. While stock returns are largely uncorrelated over short horizons (consistent with weak-form market efficiency), there are sometimes small but exploitable autocorrelations, particularly at the 5-day horizon.

**Critical implementation detail**: ARIMA is fitted on the training set only. During validation and testing, ARIMA uses **rolling one-step-ahead forecasting** (`predict_val_set`): it predicts the next value, then incorporates the actual observation to update its state before predicting the next one. This is proper online time series forecasting — it is not data leakage. The predict-then-update structure means the ARIMA never sees future data.

A constant ARIMA prediction (all zeros) is a known failure mode when the model cannot find any autocorrelation. The paper validation check (Check 4) explicitly tests for this.

### Optional models

**LSTM** (`ml/src/models/lstm_model.py`): If PyTorch is installed, a Long Short-Term Memory network can be added as a fourth base learner. LSTMs model non-linear temporal dependencies over longer sequences. The ensemble degrades gracefully to the three-model version if LSTM is not available.

**TFT** (`ml/src/models/tft_model.py`): Temporal Fusion Transformer — a more powerful temporal model. Also optional.

---

## 10. The Stacking Meta-Learner

**File**: `ml/src/models/stacking_meta_learner.py`

### Why stacking instead of fixed weights?

The original design used fixed weights (LightGBM 50%, XGBoost 35%, ARIMA 15%). These were hand-tuned and do not adapt to the ticker or the amount of data available. A learned meta-learner can discover that for a particular stock, ARIMA actually contributes very little (common when returns are truly unpredictable at the linear level), or that XGBoost is more reliable than LightGBM for a specific feature set.

### How the meta-learner is trained

1. Base models (LightGBM, XGBoost, ARIMA) are trained on the **training split**
2. Each base model generates predictions on the **validation split** (data it never trained on)
3. A **Ridge Regression** meta-learner is trained to map `[lgbm_pred, xgb_pred, arima_pred]` → `actual_return` using the validation set predictions and actuals

Ridge regression is chosen because:
- It learns weights that can go negative (if a model is anti-predictive, weight it down)
- The L2 regularization prevents any single base model from dominating
- It is fast and interpretable — the learned coefficients are directly the model weights

### Fallback to fixed weights

If no meta-learner file exists on disk (e.g., when loading an older model), the ensemble falls back to the config-specified fixed weights (0.50/0.35/0.15). This ensures backward compatibility.

### Learned weights interpretation

The meta-learner's learned coefficients can be inspected after training:

```python
weights = ensemble.meta_learner.learned_weights()
# e.g., {'lgbm': 0.62, 'xgb': 0.31, 'arima': 0.07}
```

If the ARIMA weight is near zero, it means ARIMA's rolling forecasts were not adding value for this ticker beyond what LightGBM + XGBoost already captured. This is informative and should be reported in the paper's ablation.

---

## 11. Data Splits and Leakage Prevention

### Three-way split

The data is split into three non-overlapping chronological segments:

| Split | Proportion | Purpose |
|---|---|---|
| Training | 70% | Base model training (LightGBM, XGBoost, ARIMA) |
| Validation | 15% | Meta-learner training; hyperparameter selection |
| Test | 15% | Final evaluation; never touched during training |

The test set is the last 15% of the time series — the most recent data. This is the correct chronological order for financial data (you can only train on the past and test on the future).

### Causal discovery runs on training split only

PCMCI and Granger run only on the training split (`train_ratio` from config, typically 85% of data). The validation and test splits are never seen during feature selection. This prevents the feature selector from accidentally selecting features that happen to correlate with future test data.

### Target column exclusion in causal discovery

Both `excess_return_5d` and `log_return_5d` are **dropped** before the data is passed to PCMCI or Granger:

```python
cols_to_drop = [c for c in [TARGET_COL, AUXILIARY_TARGET, "excess_return_5d"]
                if c in df_train.columns]
df_causal = df_train.drop(columns=cols_to_drop)
```

This is critical. These columns contain `shift(-5)` values — i.e., future prices. Including them in the causal discovery step would cause the algorithm to find spurious "causal" links from features that correlate with the future target through mathematical construction rather than genuine causal relationships.

The target is re-added only for the Granger step (which specifically tests feature→target relationships), and even then through a separate data copy.

### ARIMA one-step-ahead forecasting

ARIMA's `predict_val_set(y_val)` implements rolling online forecasting: predict t, observe y[t], update model, predict t+1. The observation of y[t] happens **after** the prediction for t is made. This is not leakage — it is standard online time series evaluation.

### Calibration fitted on validation, scored on test

Confidence calibration (Check 5) fits the Isotonic calibrator on the validation split predictions and evaluates it on the test split. Using validation data for calibration and test data for evaluation ensures the reported calibration statistics are genuinely out-of-sample.

---

## 12. Evaluation Metrics

**File**: `ml/src/evaluation/metrics.py`

### Directional accuracy (DA)

```
DA = (number of correct UP/DOWN calls) / (total predictions)
```

Random baseline = 0.50. A DA of 0.52 means the model is right 52% of the time on direction. For financial prediction, small improvements above 0.50 are practically significant because a strategy can be run many times.

### Sharpe ratio (annualized, turnover-corrected)

The Sharpe ratio measures return per unit of risk. For this system:

```
Sharpe = (mean(excess_return) / std(excess_return)) * sqrt(trading_days_per_year)
```

Where `excess_return` is the strategy's return minus the risk-free rate.

**Key correction**: Transaction costs (10bps per trade) are only charged when the model **changes direction** — not on every period. The original implementation charged 10bps every period regardless, which is equivalent to 600bps/year even for a model that holds the same position continuously. That severely overstated costs. The corrected version only charges when `sign(pred[t]) ≠ sign(pred[t-1])`.

**HAC correction for overlapping returns**: 5-day returns overlap — `return[t]` and `return[t+1]` share 4 of the same 5 days. This induces autocorrelation that inflates the standard Sharpe ratio estimate by 2–3x. The code uses Newey-West HAC (heteroskedasticity and autocorrelation consistent) variance estimation with 4 lags (horizon - 1) to correct for this.

### Three Sharpe variants

| Method | Description | Use |
|---|---|---|
| `sharpe_ratio_original` | Flat 10bps every period | Paper comparison: shows original vs corrected |
| `sharpe_ratio` | Turnover-corrected + HAC | Correct version for paper Table 2 |
| `sharpe_ratio_scaled` | Confidence-weighted positions | Shows real-world practitioner Sharpe |

### Max drawdown

```
Max drawdown = min over time of (equity[t] - peak_equity[t]) / peak_equity[t]
```

This measures the worst peak-to-trough decline of the strategy's equity curve. It is typically negative (a loss). A max drawdown of -0.15 means the strategy lost at most 15% from its peak at some point.

### Calmar ratio

```
Calmar ratio = annualized_return / |max_drawdown|
```

Return per unit of drawdown risk. Higher is better. Useful for comparing strategies with similar returns but different risk profiles.

### RMSE, MAE, R²

Standard regression metrics on log returns. Important note: **R² of 0.01–0.10 is good for financial return prediction**. Do not compare this to price-level R² (~0.99). Predicting log returns is genuinely hard — a small R² means the model explains some variance beyond the near-zero mean.

---

## 13. Regime-Based Evaluation

**File**: `ml/src/evaluation/regime_splitter.py`, `ml/src/evaluation/backtester.py`

### Why regime evaluation matters

A model can appear to work well when evaluated on the full test set while actually only working in one specific regime. For example, a model might achieve DA = 0.58 overall but DA = 0.47 during the COVID crash and DA = 0.63 during the bull market. These averages would tell very different stories about robustness.

The paper's central claim is that PCMCI-selected features degrade **less** across regime shifts. This can only be demonstrated with regime-specific evaluation.

### Defined regimes

| Regime | Period | Characteristic |
|---|---|---|
| Bull market | 2010 – 2019 | Low volatility, steady uptrend, low rates |
| COVID crash | Jan 2020 – Jun 2020 | Extreme volatility, structural break |
| Recovery | Jul 2020 – Dec 2021 | High growth, stimulus-driven |
| Rate hike cycle | 2022 | Rising rates, growth stocks punished |
| AI bull run | 2023 – 2025 | Large-cap tech dominated, rate plateau |

### Option B evaluation protocol

The paper uses **Option B**: train the model on all data *before* a regime starts, then test it on the regime period. This is the most realistic evaluation — the model is trained as if we are at the start of each regime, with no knowledge of what is about to happen.

For example:
- COVID crash regime: train on 2010 – Dec 2019, test on Jan 2020 – Jun 2020
- Rate hike regime: train on 2010 – Dec 2021, test on 2022

This is computationally expensive because it requires retraining the model for each regime, but it is the correct evaluation framework for demonstrating regime robustness.

### HMM regime detection

An alternative to manual regime labelling is Hidden Markov Model (HMM) based regime detection (`ml/src/evaluation/hmm_regime_detector.py`). With `n_states=2`, the HMM learns to identify "low volatility" and "high volatility" regimes from the return series, without using the manually defined date boundaries.

The paper validation check (Check 7) compares HMM-detected regimes against the manual labels. High agreement between the two (measured via a cross-tabulation) validates that the manually defined regimes correspond to real market state changes rather than being arbitrary date cuts.

---

## 14. Statistical Significance Tests

**File**: `ml/src/evaluation/significance.py`

These tests are what separate a publishable result from an observation. Every directional accuracy number in the paper needs a p-value.

### Test 1: Binomial test (DA vs random)

**Question**: Is the model's directional accuracy significantly above the 50% random baseline?

**How it works**: Under the null hypothesis that the model is pure noise, each prediction is correct with probability 0.50. With N predictions and K correct calls, the probability of observing K or more correct calls by chance is computed using the binomial distribution.

```
H₀: DA ≤ 0.50 (no better than random)
Hₐ: DA > 0.50 (better than random)
```

A p-value < 0.05 means we can reject the hypothesis that the model is random at the 5% significance level.

**Why this matters**: Without this test, a DA of 0.53 with N=100 predictions might just be noise. With N=252 (one trading year), it starts becoming significant. The test accounts for sample size.

### Test 2: McNemar's test (model A vs model B)

**Question**: Is the PCMCI-selected model significantly better than the all-features model?

**How it works**: For each time step t, we know:
- Whether Model A (PCMCI) got it right or wrong
- Whether Model B (all features) got it right or wrong

This creates a 2×2 contingency table:

```
          B correct   B wrong
A correct    n11         n10
A wrong      n01         n00
```

McNemar's test focuses on the discordant pairs (n10 and n01). If A is genuinely better, we'd expect n10 > n01 (A correct when B is wrong, more often than the reverse).

**Why NOT a two-sample proportion test**: Both models make predictions on the same dates, in the same market conditions. Their errors are correlated. McNemar's test correctly handles this pairing — it is designed for paired binary comparisons.

### Test 3: Block bootstrap confidence interval for DA

**Question**: What is the 95% confidence interval for the model's true directional accuracy?

**How it works**: Financial returns are autocorrelated and fat-tailed. The standard normal approximation for a proportion (DA ± 1.96 × sqrt(DA(1-DA)/n)) assumes independent draws — which is violated. Instead:

1. Divide the test period into blocks of consecutive days (block size = sqrt(n))
2. Resample whole blocks (not individual days) to preserve temporal autocorrelation
3. Compute DA on each resampled dataset — 2000 times
4. Report the 2.5th and 97.5th percentile of the resulting distribution

If the lower end of the 95% CI is above 0.50, the model is reliably better than random even accounting for uncertainty.

### Test 4: Diebold-Mariano test (regression comparison)

**Question**: Does Model A have significantly lower prediction error (RMSE) than Model B?

**How it works**: Let `dₜ = (y_true[t] - y_pred_A[t])² - (y_true[t] - y_pred_B[t])²` be the loss differential at time t. If A is better, d̄ should be negative (A's squared errors are smaller).

The DM statistic is `d̄ / sqrt(Var(d)/n)`, tested against a t-distribution. The variance is estimated using Newey-West HAC with h-1 lags to account for autocorrelation in 5-day-ahead forecast errors.

This is the standard test for comparing time series forecasting models on regression accuracy, as opposed to directional accuracy.

---

## 15. Confidence Calibration

**File**: `ml/src/models/calibration.py`

### What calibration means

A well-calibrated model's stated confidence should match empirical accuracy. If the model says "I'm 70% confident" on 100 predictions, roughly 70 of them should be correct. If only 55 are correct, the model is overconfident. If 85 are correct, it is underconfident.

### ECE (Expected Calibration Error)

ECE measures the average gap between predicted confidence and empirical accuracy across confidence bins:

```
ECE = Σ_b (|b| / n) × |accuracy(b) - confidence(b)|
```

Where the sum is over confidence bins (e.g., [0.5, 0.6), [0.6, 0.7), ...).

A lower ECE is better. ECE = 0 means perfect calibration.

### Isotonic calibration

After the ensemble produces confidence scores, Isotonic Regression is fitted as a post-hoc calibrator. It maps raw confidence scores to calibrated probabilities using a non-parametric monotone function. Unlike Platt scaling (logistic regression), Isotonic regression makes no assumption about the shape of the calibration curve.

**Important**: The calibrator is fitted on the **validation split** and evaluated on the **test split**. This is explicit OOS calibration — fitting and evaluating on the same data would be circular.

### What counts as valid confidence for the paper

The paper validation checks explicitly warn when confidence scores are not valid. Only confidence scores that come from the ensemble's actual internal agreement (or a proper probabilistic model) are valid for paper evidence. Using `|predicted_return| * 10` as a proxy for confidence is flagged as `INVALID_synthetic_proxy` — it's not a calibrated probability and should not appear in a paper's calibration table.

---

## 16. Paper Validation Checks

**File**: `run_paper_validation.py`

This single script runs all critical checks and saves results to `ml/logs/paper_validation_{TICKER}_{timestamp}.json`. It is designed so that every paper claim has a corresponding logged verification.

### Check 1: PCMCI Stability

Runs `PCMCIStabilityAnalyzer` with 3 windows on the training split. Reports mean Jaccard similarity and the stability verdict. If verdict is UNSTABLE, the paper must disclose this.

### Check 2: ATR Diagnostic (Nifty only)

The NIFTY feature matrix computes ATR (Average True Range) using High and Low prices. But in some NIFTY data sources, High = Low = Close (approximate tick data). In this case, ATR is just a scaled version of Close, making it mathematically correlated with the target by construction — not causally informative. The check flags if ATR/Close correlation exceeds 0.8, meaning ATR should be excluded from causal discovery.

### Check 3: Statistical Significance

Runs the binomial test and block bootstrap CI on the test set (final 15%). Reports whether DA is significantly above 0.50. If the binomial test is not significant, the paper's directional accuracy claim is the critical weakness.

### Check 4: ARIMA Variance

Verifies that ARIMA's `predict_raw()` output has non-zero standard deviation. A constant ARIMA output (all values identical) means the ARIMA component contributes nothing to the ensemble. The meta-learner will effectively assign it zero weight, reducing the ensemble to LightGBM + XGBoost only.

### Check 5: Confidence Calibration ECE

Fits Isotonic calibration on validation predictions and scores it on the test split. Reports the reliability table and ECE. Also checks whether the confidence source is valid (actual ensemble confidence vs proxy).

### Check 6: Ablation Table (--full flag only)

Runs the full feature ablation: trains and evaluates three model variants:
1. **All features** — baseline with no feature selection
2. **Granger-only selected features** — correlation-based causal selection
3. **PCMCI-selected features** — the proposed method

The ablation demonstrates whether PCMCI selection actually provides value over simpler alternatives.

### Check 7: HMM Regime Robustness

Fits a 2-state HMM on the return series and compares the data-driven regime labels against the manually defined regimes. High alignment validates that the manual regimes are not arbitrary. Also evaluates model performance within each HMM-detected regime.

### Check 8: Sharpe Comparison

Computes all three Sharpe variants (original flat-cost, turnover-corrected, confidence-weighted) on the test set. Reports the gap between original and corrected — a large gap means the original evaluation significantly overstated strategy performance due to unrealistic transaction cost accounting.

---

## 17. Design Decisions and Tradeoffs

### Why 5-day returns, not 1-day or 20-day?

- **1-day returns**: Too noisy. PCMCI requires sufficient data for conditional independence tests, and 1-day return prediction has very low signal-to-noise ratio.
- **5-day returns**: Aligns with a typical institutional trade horizon. Enough signal for Granger/PCMCI to work. Still short enough that regime changes happen across the test period.
- **20-day returns**: Fewer independent observations per year (12 vs 52 for weekly). Less statistical power for significance tests.

### Why Ridge regression for the meta-learner?

Ridge allows negative weights (if a model hurts, assign it negative weight) without the instability of unconstrained OLS. Lasso would zero out some base models, which is a viable alternative but removes the ensemble diversity benefit. Ridge preserves all models with partial weights.

### Why Granger as baseline rather than just PCMCI?

1. Granger is a well-established, understood test — reviewers will know what it means
2. Granger's FDR-corrected p-values provide a second, independent statistical filter
3. The intersection of Granger + PCMCI is a stronger claim than PCMCI alone: two tests with different assumptions both agree

### Why not include LSTM by default?

LSTM requires PyTorch, adds training time, and the benefit on the ~30-feature input is not consistently positive (gradient boosting methods already handle temporal patterns through lag features). LSTM is available as an option when you believe there are long-range temporal dependencies the lag-based features don't capture.

### Why block bootstrap rather than standard bootstrap for DA CI?

Standard bootstrap assumes independent and identically distributed (IID) samples. Financial returns are neither: they exhibit volatility clustering (autocorrelation in squared returns) and trending. The block bootstrap preserves the temporal autocorrelation structure by resampling contiguous blocks of size `sqrt(n)`.

---

## 18. Known Limitations

### Survivorship bias

The tickers tested (AAPL, large-cap US stocks, NIFTY) are all entities that have survived to the present. Stocks that were delisted, went bankrupt, or were acquired are not included. This means the feature distributions in the training data may not represent the full population of stocks a practitioner would encounter.

**For the paper**: Include at least one robustness check with de-listed or poor-performing tickers, or clearly document this limitation in the threat-to-validity section.

### "Conditional independence" ≠ "interventional causality"

PCMCI establishes that features have conditional independence links to returns. This is a form of statistical causality — it means the link cannot be explained away by any other variable in the system. It does NOT mean that intervening to change the VIX would change a stock's return in a predictable way. The paper should use careful language: "PCMCI-selected features (conditional independence)" not "causally linked features."

### NIFTY live prediction not available

Real-time P/E, P/B, and news sentiment are not available for the NIFTY index. The pipeline skips live prediction for NIFTY and reports only historical model performance. Any production use case for the Indian market would need a real-time data feed.

### Feature set size sensitivity

The adaptive selection strategy means different runs (with different random seeds or PCMCI parameter settings) may produce different feature sets, especially for tickers with limited data. The PCMCI Stability Analysis (Check 1) directly measures this and should be run before making any paper claims about specific selected features.

### Overlapping 5-day returns create false precision

Because `return[t]` and `return[t+1]` overlap by 4 days, there are effectively only `N/5` independent observations in a dataset of N rows. This inflates all test statistics by a factor of `sqrt(5) ≈ 2.2`. The HAC corrections in the Sharpe ratio and DM test address this for those metrics, but the sample size reported in the binomial test still uses the raw count N — which should be divided by 5 to get the effective sample size when interpreting the result.
