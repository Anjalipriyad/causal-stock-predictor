causal-stock-predictor
Stock return prediction using causal feature selection (PCMCI + Granger) instead of correlation — producing models that are more robust across market regime shifts.
What It Does
Given a ticker, predicts whether the stock will outperform or underperform the market over the next 5 trading days, with a confidence score and explanation of what drove the prediction.
==================================================
  LIVE PREDICTION — AAPL   (2026-03-15)
==================================================
  Current price:    $250.12
  Predicted price:  $252.34
  Direction:        UP
  Expected return:  +0.89%
  Confidence:       61%
  Range (90% CI):   $243.10 — $259.80
  Horizon:          5 trading days
  Model:            ensemble(lgbm+xgb+arima)

  Causal Drivers:
    ▲ sector_rel_momentum_20d      positive  (+0.023)
    ▼ vix_change_1d                negative  (-0.018)
    ▲ rsi_14                       positive  (+0.014)
==================================================
Core Idea
Most ML stock models learn correlations that break when market regimes shift. We use PCMCI causal discovery to find features that genuinely cause returns to move — not just correlate with them. Causal mechanisms are more stable across regime transitions than statistical correlations.
Results
AAPLNVDAGOOGLDirectional accuracy54.0%54.9%54.3%Sharpe ratio0.761.331.15Best regimeRecoveryCOVID crash (64%)COVID crash (57%)Worst regime—AI bull (40%)Rate hike (43%)
Random baseline = 50%. Evaluated out-of-sample across 5 market regimes.
Stack

Causal discovery — tigramite (PCMCI), statsmodels (Granger)
Models — LightGBM, XGBoost, ARIMA (pmdarima)
Data — yFinance (prices, macro), Finnhub (sentiment)
Explainability — SHAP

Quickstart
bashgit clone https://github.com/anjalipriyad/causal-stock-predictor.git
cd causal-stock-predictor

cp .env.example .env
# add your FINNHUB_API_KEY to .env

pip install -r ml/requirements.txt

# Run full pipeline (first time ~1hr due to PCMCI)
python run_pipeline.py --ticker AAPL

# Daily prediction after first run (~10 seconds)
python run_pipeline.py --ticker AAPL --predict-only
Pipeline
Step 1  Load historical data (yFinance + Finnhub)
Step 2  Feature engineering (~50 candidate signals)
Step 3  Causal discovery (Granger + PCMCI) → saves causal_features_AAPL.json
Step 4  Train ensemble (LightGBM + XGBoost + ARIMA)
Step 5  Live prediction
The pipeline is smart — it auto-detects what already exists and skips completed steps. Only downloads data and runs PCMCI once.
Commands
bashpython run_pipeline.py --ticker AAPL                # smart run
python run_pipeline.py --ticker AAPL --predict-only # just prediction
python run_pipeline.py --ticker AAPL --full-retrain # retrain everything
python run_pipeline.py --ticker AAPL --with-retrain # + walk-forward retraining
python run_pipeline.py --ticker AAPL --with-regime  # + regime-aware models
Project Structure
causal-stock-predictor/
├── ml/
│   ├── src/
│   │   ├── data/          # loader, validator
│   │   ├── features/      # technical, macro, sentiment, sector, pipeline
│   │   ├── causal/        # granger, pcmci, selector
│   │   ├── models/        # lgbm, xgb, arima, tft, regime_model, base_model
│   │   └── evaluation/    # metrics, backtester, regime_splitter, retrain_schedule
│   ├── configs/config.yaml
│   └── saved_models/      # trained weights + causal_features_{ticker}.json
├── run_pipeline.py
├── .env.example
└── README.md
Tests
bashcd causal-stock-predictor
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest ml/tests/ -v
# 127/127 passing
