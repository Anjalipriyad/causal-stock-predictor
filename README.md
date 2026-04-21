# causal-stock-predictor

> **PCMCI-based Feature Selection for Robust Equity Return Prediction Under Market Regime Shifts**

A full-stack stock prediction system built around a reproducible PCMCI-based pipeline — using causal-discovery tooling to identify conditionally independent predictors rather than relying on correlation alone. Designed for both a research paper and a production web app.

---

## What It Does

Given a stock ticker (e.g. `AAPL` or `NIFTY`), the system:

1. Identifies which macro + sentiment features are causally linked to returns via Granger + PCMCI
2. Trains a LightGBM + XGBoost + ARIMA ensemble **only on PCMCI-selected features**
3. Predicts **5-day forward log return** with a confidence band
4. Explains **why** it made the prediction (causal drivers via SHAP)
5. Shows robustness across market regimes (bull, crash, recovery, rate hike, AI bull)
6. Validates paper claims with a single reproducible script

---

## Project Structure

```
causal-stock-predictor/
├── ml/
│   ├── src/
│   │   ├── causal/          ← Granger, PCMCI, selector, stability
│   │   ├── data/            ← DataLoader, NiftyLoader, validator
│   │   ├── evaluation/      ← backtester, metrics, regime_splitter,
│   │   │                       significance, ablation, hmm_regime_detector
│   │   ├── features/        ← technical, macro, sentiment, finbert,
│   │   │                       earnings, options, sector, pipeline
│   │   ├── improvements/    ← HPO, meta-tuning, feature augmentation,
│   │   │                       diagnostics, target transform
│   │   ├── models/          ← lgbm, xgb, arima, lstm, tft, calibration,
│   │   │                       stacking, regime_model
│   │   └── ensemble.py      ← weighted ensemble + live/historical predict
│   ├── tests/               ← pytest test suite
│   └── scripts/             ← ablation, improvement scripts
├── run_pipeline.py          ← end-to-end pipeline runner
├── run_paper_validation.py  ← paper-critical validation checks
└── README.md
```

---

## Quickstart

### 1. Clone + setup environment

```bash
git clone https://github.com/your-username/causal-stock-predictor.git
cd causal-stock-predictor

cp .env.example .env
# add your FINNHUB_API_KEY to .env
```

### 2. Install ML dependencies

```bash
cd ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
# US stock (downloads data automatically)
python run_pipeline.py --ticker AAPL

# Indian market (Nifty 50 index)
python run_pipeline.py --ticker NIFTY --market india

# Individual NSE stock
python run_pipeline.py --ticker RELIANCE.NS --market india
```

The pipeline auto-detects what already exists on disk and skips completed steps. Use `--full-retrain` to force a complete rebuild.

### 4. Pipeline flags

| Flag | Description |
|---|---|
| `--predict-only` | Run live prediction only (models must exist) |
| `--refresh-data` | Re-download data even if already cached |
| `--full-retrain` | Force rebuild features + causal discovery + retrain |
| `--paper-eval` | Run full Option B regime backtest (~1hr) |
| `--with-retrain` | Walk-forward retraining (~1hr) |
| `--with-regime` | Regime-aware model training (~40min) |
| `--finbert` | Use FinBERT for sentiment scoring (GPU recommended) |

### 5. Run paper validation

```bash
# Quick mode (skips slow PCMCI stability check)
python run_paper_validation.py --ticker AAPL --quick

# Full validation (all checks)
python run_paper_validation.py --ticker AAPL

# Indian market
python run_paper_validation.py --ticker NIFTY --market india --quick
```

---

## Pipeline Steps

The pipeline runs these steps in order, skipping any that are already complete:

```
Step 1  Load historical data         (yFinance / uploaded CSV for India)
Step 2  Feature engineering          (~30 candidate features)
Step 3  Causal discovery             (Granger + PCMCI on training split only)
Step 4  Train models                 (LightGBM, XGBoost, ARIMA ensemble)
Step 5  Live prediction + reporting  (Option B regime performance table)
Step 6  Full regime backtest         (opt-in via --paper-eval)
```

---

## ML Architecture

```
Raw Data (yFinance + Finnhub)
        ↓
Feature Engineering (~30 candidate features)
  ├── Technical  — RSI, MACD, Bollinger, momentum, volatility
  ├── Macro      — VIX, yield spread, DXY, oil, gold, S&P 500
  ├── Sentiment  — Finnhub news sentiment, rolling averages
  ├── Earnings   — EPS surprise, P/E, P/B
  ├── Options    — put/call ratio, implied volatility
  └── Sector     — sector relative performance
        ↓
Causal Discovery  (training split only — no test leakage)
  ├── Granger Causality    (baseline, full training set)
  └── PCMCI / ParCorr      (last 50% of training set, via tigramite)
        ↓
Adaptive Feature Selection
  ├── Default: intersection of Granger + PCMCI findings
  ├── Fallback: iteratively relax p-value thresholds → union → top-N by PCMCI p-value
  └── Saved to  ml/saved_models/causal_features_{TICKER}.json
        ↓
Ensemble Model
  ├── LightGBM  (weight: 0.50)
  ├── XGBoost   (weight: 0.35)
  └── ARIMA     (weight: 0.15)
        ↓
PredictionResult
  ├── predicted_return   (5-day log / excess return)
  ├── predicted_price
  ├── direction          UP | DOWN
  ├── confidence         0–1
  ├── upper_band / lower_band
  └── causal_drivers     [ {feature, impact, shap} ]
```

---

## Paper Validation Checks

`run_paper_validation.py` runs all paper-critical checks and saves results to `ml/logs/`:

| Check | Description |
|---|---|
| **1 — PCMCI Stability** | Sliding-window Jaccard similarity; flags unstable feature sets |
| **2 — ATR Diagnostic** | Nifty H/L contamination check (skipped for non-Nifty tickers) |
| **3 — Statistical Significance** | Binomial test + bootstrap CI for directional accuracy vs 50% baseline |
| **4 — ARIMA Variance** | Verifies ARIMA predictions are non-constant |
| **5 — Confidence Calibration** | Isotonic calibration ECE fitted on val, scored on OOS test |
| **6 — Ablation Table** | Full-feature vs PCMCI-only vs Granger-only comparison (--full) |
| **7 — HMM Regime Robustness** | HMM-detected regimes vs manual regime labels (skipped in --quick) |
| **8 — Sharpe Comparison** | Original flat-cost vs turnover-corrected vs confidence-weighted Sharpe |

---

## Research Approach

The core novelty is a **two-stage pipeline**:

**Stage 1 — PCMCI Discovery**: PCMCI identifies lagged conditional-independence links between macro/sentiment features and stock returns. Unlike simple pairwise correlation, PCMCI controls for confounders and provides a stricter statistical filter. This is described in the paper as "PCMCI-selected features (conditional independence)" — not interventional causality.

**Stage 2 — Regime Robustness**: Models trained on PCMCI-selected features degrade less under regime shifts than models trained on all correlated features. Bootstrap confidence intervals test whether observed differences are statistically significant.

### Target Variable

| Target | Use case |
|---|---|
| `excess_return_5d` | Individual US equities — stock return minus benchmark (SPY). **Recommended default.** |
| `log_return_5d` | Market indexes (e.g. NIFTY) — raw 5-day log return; subtracting the index from itself is meaningless. |

NIFTY automatically uses `log_return_5d` via a local config override (never mutating global config).

### Evaluation Regimes

| Regime | Period |
|---|---|
| Bull market | 2010 – 2019 |
| COVID crash | Jan 2020 – Jun 2020 |
| Recovery | Jul 2020 – Dec 2021 |
| Rate hike cycle | 2022 |
| AI bull run | 2023 – 2025 |

### Key Metrics

- Directional accuracy (% correct up/down calls)
- Sharpe ratio (annualised, turnover-corrected)
- RMSE + MAPE on log returns
- Max drawdown / Calmar ratio

---

## Data Leakage Prevention

Several explicit safeguards prevent leakage:

- Causal discovery runs on the **training split only** (controlled by `train_ratio` in `config.yaml`)
- Both `excess_return_5d` and `log_return_5d` (forward-looking columns) are **dropped** before feeding data into PCMCI/Granger
- PCMCI runs with `exclude_target=True` to avoid target-variable contamination inside the conditional independence graph
- Calibration is fitted on the val split and scored on the held-out test split

---

## Tech Stack

### ML

| Library | Purpose |
|---|---|
| `yfinance` | Historical OHLCV data |
| `requests` | Finnhub API calls |
| `pandas` / `numpy` | Feature engineering |
| `tigramite` | PCMCI causal discovery |
| `statsmodels` | Granger causality + ARIMA |
| `lightgbm` | Primary forecasting model |
| `xgboost` | Ensemble model |
| `scikit-learn` | Preprocessing + metrics |
| `shap` | Model interpretability |
| `pmdarima` | Auto ARIMA |
| `hmmlearn` | HMM regime detection (optional) |

### Backend (Phase 2)

| Library | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `pydantic` | Request/response schemas |

### Frontend (Phase 3)

| Library | Purpose |
|---|---|
| `react` + `vite` | UI framework |
| `recharts` | Price + prediction charts |
| `zustand` | Global state |
| `axios` | API calls |

---

## Build Phases

| Phase | Status | Description |
|---|---|---|
| **Phase 1 — ML** | In progress | Causal discovery + model training |
| **Phase 2 — Backend** | Not started | FastAPI wrapper around ML layer |
| **Phase 3 — Frontend** | Not started | React dashboard |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `FINNHUB_API_KEY` | Phase 1 | Free key at finnhub.io |
| `ML_SAVED_MODELS_DIR` | Phase 2 | Path to saved model weights |
| `ML_CONFIG_PATH` | Phase 2 | Path to config.yaml |
| `BACKEND_HOST` | Phase 2 | FastAPI host |
| `BACKEND_PORT` | Phase 2 | FastAPI port |
| `VITE_API_BASE_URL` | Phase 3 | Backend URL for frontend |

---

## Known Limitations

- **Survivorship bias**: ticker selection uses large-cap surviving names. For paper submission include a robustness check with de-listed or underperforming tickers, or document this limitation explicitly.
- **NIFTY live prediction**: skipped — real-time P/E, P/B, and news headlines are not available for the index.
- **PCMCI instability**: if the Jaccard similarity across windows is low (verdict = UNSTABLE), the paper should disclose this and use the `union` selection strategy.

---

## License

MIT
