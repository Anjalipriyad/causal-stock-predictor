# causal-stock-predictor

> **Causal Feature Selection for Robust Equity Return Prediction Under Market Regime Shifts**

A full-stack stock prediction system built around a novel causal ML approach — using PCMCI causal discovery to identify *what actually causes* price movement, not just what correlates with it. Designed for both a research paper and a production web app.

---

## What It Does

Given a stock ticker (e.g. `AAPL`), the system:
1. Identifies which macro + sentiment features **causally drive** its returns using PCMCI
2. Trains a LightGBM + XGBoost + ARIMA ensemble **only on causal features**
3. Predicts **5-day forward log return** with a confidence band
4. Explains **why** it made the prediction (causal drivers)
5. Shows robustness across market regimes (bull, crash, recovery, rate hike, AI bull)

---

## Project Structure

```
causal-stock-predictor/
├── ml/               ← Phase 1 — causal ML layer (build + lock first)
├── backend/          ← Phase 2 — FastAPI REST API
├── frontend/         ← Phase 3 — React dashboard
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Quickstart

### 1. Clone + setup environment
```bash
git clone https://github.com/your-username/causal-stock-predictor.git
cd causal-stock-predictor

cp .env.example .env
# open .env and add your FINNHUB_API_KEY
```

### 2. Install ML dependencies
```bash
cd ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pull historical data
```bash
python -m src.data.loader --ticker AAPL
```
This pulls OHLCV (2010–2025) from yFinance + sentiment from Finnhub.
Saved to `ml/data/raw/`. Run once — never again unless you add a new ticker.

### 4. Run feature engineering
```bash
python -m src.features.pipeline --ticker AAPL
```
Saved to `ml/data/processed/features/AAPL_features.csv`

### 5. Run causal discovery
```bash
python -m src.causal.selector --ticker AAPL
```
Runs Granger + PCMCI. Saves causal feature list to:
`ml/saved_models/causal_features_AAPL.json`

### 6. Train models
```bash
python -m src.models.lgbm_model --ticker AAPL
python -m src.models.xgb_model  --ticker AAPL
python -m src.models.arima_model --ticker AAPL
```
Saves trained weights to `ml/saved_models/`

### 7. Run backtesting
```bash
python -m src.evaluation.backtester --ticker AAPL
```
Prints regime-split evaluation table.

---

## ML Architecture

```
Raw Data (yFinance + Finnhub)
        ↓
Feature Engineering (~30 candidate features)
  ├── Technical  — RSI, MACD, Bollinger, momentum, volatility
  ├── Macro      — VIX, yield spread, DXY, oil, gold, S&P 500
  └── Sentiment  — Finnhub news sentiment, rolling averages
        ↓
Causal Discovery
  ├── Granger Causality    (baseline)
  └── PCMCI / ParCorr      (primary — via tigramite)
        ↓
Causal Feature Set (6–12 features)
  └── saved to causal_features_{ticker}.json
        ↓
Ensemble Model
  ├── LightGBM  (weight: 0.50)
  ├── XGBoost   (weight: 0.35)
  └── ARIMA     (weight: 0.15)
        ↓
PredictionResult
  ├── predicted_return   (5-day log return)
  ├── predicted_price
  ├── direction          UP | DOWN
  ├── confidence         0–1
  ├── upper_band
  ├── lower_band
  └── causal_drivers     [ {feature, impact} ]
```

---

## Research Approach

The core novelty is the **two-stage pipeline**:

1. **Causal Discovery** — PCMCI identifies lagged causal links between macro/sentiment features and stock returns. Unlike Granger causality, PCMCI controls for confounders and finds conditional independence, producing a stricter causal graph.
   
  Adaptive selection: the code now implements an adaptive feature-selection fallback. By default the pipeline uses a strict `intersection` of Granger + PCMCI findings; if too few features are selected the selector will iteratively relax Granger/PCMCI p-value thresholds, then fall back to `union`, and finally pick top features by PCMCI p-value if needed. Selection metadata is recorded in `saved_models/causal_features_{TICKER}.json` for transparency.

2. **Regime Robustness** — Models trained on causal features degrade *less* under regime shifts than models trained on all correlated features. This is the central claim of the paper.

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
- Sharpe ratio (annualised)
- RMSE + MAPE on log returns
- Max drawdown
- Calmar ratio

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
| `xgboost` | Comparison model |
| `scikit-learn` | Preprocessing + metrics |
| `shap` | Model interpretability |
| `pmdarima` | Auto ARIMA |

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
| **Phase 1 — ML** | 🔄 In progress | Causal discovery + model training |
| **Phase 2 — Backend** | ⬜ Not started | FastAPI wrapper around ML layer |
| **Phase 3 — Frontend** | ⬜ Not started | React dashboard |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `FINNHUB_API_KEY` | ✅ Phase 1 | Get free key at finnhub.io |
| `ML_SAVED_MODELS_DIR` | Phase 2 | Path to saved model weights |
| `ML_CONFIG_PATH` | Phase 2 | Path to config.yaml |
| `BACKEND_HOST` | Phase 2 | FastAPI host |
| `BACKEND_PORT` | Phase 2 | FastAPI port |
| `VITE_API_BASE_URL` | Phase 3 | Backend URL for frontend |

---

## License

MIT

---

## Target variable choice

For the paper and experiments we predict a fixed forward-return target (the
`model.target` value in `config.yaml`). The code supports two commonly used
targets:

- `excess_return_5d`: market-adjusted 5-day log return (stock return minus
  benchmark return) — appropriate when the goal is to predict alpha (stock
  performance relative to the market). This is the recommended default for
  individual US equities where a reliable market benchmark (e.g. SPY) exists.
- `log_return_5d`: raw 5-day log return — appropriate for market indexes
  (e.g. NIFTY) where subtracting the index from itself is meaningless.

The repository no longer performs silent in-place mutations of the loaded
configuration. If a per-market override is required (for example, predicting
index returns for NIFTY), a local config override is used at runtime and is
logged explicitly. For paper submission you should *choose one* target and
report the choice and its justification in the Methods section (recommended:
use `excess_return_5d` for single-stock alpha prediction; use
`log_return_5d` only when modelling index-level returns).

If you'd like, I can (1) enforce a single target across all experiments by
updating `config.yaml` and the paper text, or (2) add a clear CLI/config
switch for per-market targets and surface that choice in exported experiment
metadata. Which would you prefer?
