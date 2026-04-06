"""
diagnostics.py
----------------
Baseline diagnostics and per-regime evaluation helpers.

Small utilities to compute baseline metrics, per-regime tables and
print concise diagnostic output for a ticker. Meant to be non-invasive
— uses existing `Ensemble`, `Metrics`, and `RegimeSplitter` utilities.

Usage:
    from ml.src.improvements.diagnostics import run_baseline
    run_baseline('NIFTY', market='india', force_retrain=False)
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.src.data.loader import _load_config
from ml.src.features.pipeline import FeaturePipeline
from ml.src.data.nifty_loader import NiftyLoader
from ml.src.causal.selector import CausalSelector
from ml.src.ensemble import Ensemble
from ml.src.evaluation.metrics import Metrics
from ml.src.evaluation.regime_splitter import RegimeSplitter

logger = logging.getLogger(__name__)


def _load_feature_matrix(ticker: str, market: str = "us") -> pd.DataFrame:
    pipeline = FeaturePipeline()
    if market == "india" or ticker.upper() in ("NIFTY", "^NSEI", "NIFTY50"):
        feat_path = NiftyLoader().out_dir / "NIFTY_features.csv"
    else:
        feat_path = pipeline.features_dir / f"{ticker}_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"No feature matrix at {feat_path}")
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    return df


def run_baseline(ticker: str, market: str = "us", force_retrain: bool = False) -> dict:
    """
    Run baseline diagnostics for `ticker`.

    Steps:
      - load feature matrix
      - load causal features (fallback to top numeric features if missing)
      - load or train ensemble (train if missing or force_retrain=True)
      - compute overall test set metrics (last `test_ratio` fraction)
      - compute per-regime metrics using RegimeSplitter

    Returns a dict with 'overall' and per-regime metrics.
    """
    ticker = ticker.upper()
    cfg = _load_config()

    df = _load_feature_matrix(ticker, market)

    selector = CausalSelector()
    try:
        causal_features = selector.load(ticker)
    except Exception:
        logger.warning("No causal features saved — using top numeric features as fallback.")
        # Fallback: pick top numeric columns excluding target
        target = cfg["model"]["target"]
        causal_features = [c for c in df.select_dtypes(include=["number"]).columns if c != target]
        causal_features = causal_features[: cfg["causal"]["selector"]["max_causal_features"]]

    ensemble = Ensemble(cfg=cfg)
    models_dir = Path(ensemble.models_dir)

    # Try load saved models; if not available, train (unless caller asked not to)
    try:
        ensemble.load(ticker)
        logger.info(f"[diagnostics] Loaded saved models for {ticker}.")
    except Exception as e:
        logger.warning(f"[diagnostics] Could not load saved models: {e}")
        if force_retrain:
            logger.info(f"[diagnostics] Training models for {ticker} (force_retrain=True)...")
            ensemble.train_all(df, ticker, causal_features)
        else:
            logger.info(f"[diagnostics] Training models for {ticker} to produce baseline...")
            ensemble.train_all(df, ticker, causal_features)

    # Build test slice (last `test_ratio` fraction)
    n = len(df)
    test_ratio = cfg["model"].get("test_ratio", 0.15)
    test_start = int(n * (1 - test_ratio))
    test_df = df.iloc[test_start:]

    preds = ensemble.predict_historical(test_df, causal_features)
    m = Metrics()
    results = {"overall": m.compute_all(preds["predicted_return"], preds.get("actual_return", pd.Series([])), label="overall")}

    # Per-regime diagnostics
    splitter = RegimeSplitter()
    regimes = splitter.split_all(df)
    for regime_name, regime_df in regimes.items():
        if len(regime_df) < 30:
            continue
        try:
            preds_r = ensemble.predict_historical(regime_df, causal_features)
            if "actual_return" in preds_r.columns:
                results[regime_name] = m.compute_all(preds_r["predicted_return"], preds_r["actual_return"], label=regime_name)
        except Exception as e:
            logger.warning(f"[diagnostics] Regime {regime_name} evaluation failed: {e}")

    # Print concise table
    print(f"\nBaseline diagnostics for {ticker} — overall + per-regime metrics:\n")
    for k, v in results.items():
        da = v.get("directional_accuracy", float("nan"))
        rmse = v.get("rmse", float("nan"))
        print(f"  {k:20s}  DA={da:.4f}  RMSE={rmse:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline diagnostics for a ticker")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--market", type=str, default="us", choices=["us", "india"]) 
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    run_baseline(args.ticker, market=args.market, force_retrain=args.force_retrain)
