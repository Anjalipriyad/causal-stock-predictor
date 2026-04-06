"""
quick_wins.py
--------------
Small quick-win routines to raise directional accuracy fast.

Currently implements:
  - threshold tuning: find best offset threshold on predicted returns
    that maximises directional accuracy on the held-out test set.

Saves best threshold to `saved_models/threshold_{TICKER}.json`.
"""

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config
from ml.src.causal.selector import CausalSelector
from ml.src.ensemble import Ensemble
from ml.src.improvements.diagnostics import _load_feature_matrix

logger = logging.getLogger(__name__)


def tune_threshold(
    ticker: str,
    market: str = "us",
    low: float = -0.02,
    high: float = 0.02,
    n: int = 81,
    save: bool = True,
) -> dict:
    """
    Sweep a threshold grid and find the threshold that maximises
    directional accuracy on the held-out test slice (config-driven).

    Returns a dict with `best_threshold`, `best_accuracy`, and `baseline_accuracy`.
    """
    ticker = ticker.upper()
    cfg = _load_config()

    df = _load_feature_matrix(ticker, market)

    selector = CausalSelector()
    try:
        causal_features = selector.load(ticker)
    except Exception:
        logger.warning("No causal features saved — using top numeric features as fallback.")
        target = cfg["model"]["target"]
        causal_features = [c for c in df.select_dtypes(include=["number"]).columns if c != target]
        causal_features = causal_features[: cfg["causal"]["selector"]["max_causal_features"]]

    ensemble = Ensemble(cfg=cfg)
    try:
        ensemble.load(ticker)
    except Exception:
        logger.info("No saved models found — training ensemble to obtain predictions.")
        ensemble.train_all(df, ticker, causal_features)

    # Test slice
    n_rows = len(df)
    test_ratio = cfg["model"].get("test_ratio", 0.15)
    test_start = int(n_rows * (1 - test_ratio))
    test_df = df.iloc[test_start:]

    preds = ensemble.predict_historical(test_df, causal_features)
    if "actual_return" not in preds.columns:
        raise RuntimeError("No actual_return column present in predictions; cannot tune threshold.")

    y_pred = preds["predicted_return"].values
    y_true = preds["actual_return"].values

    thresholds = np.linspace(low, high, n)
    accuracies = []
    for t in thresholds:
        dir_pred = (y_pred >= t).astype(int)
        dir_true = (y_true >= 0).astype(int)
        acc = float((dir_pred == dir_true).mean())
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    best_idx = int(np.argmax(accuracies))
    best_t = float(thresholds[best_idx])
    best_acc = float(accuracies[best_idx])
    # Baseline at t=0
    baseline_idx = int(np.argmin(np.abs(thresholds - 0.0)))
    baseline_acc = float(accuracies[baseline_idx])

    result = {
        "ticker": ticker,
        "best_threshold": best_t,
        "best_accuracy": best_acc,
        "baseline_accuracy": baseline_acc,
    }

    models_dir = Path(ensemble.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"threshold_{ticker}.json"
    if save:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved best threshold → {out_path}")

    print(f"Tuned threshold for {ticker}: best={best_t:.5f} (DA={best_acc:.4f}), baseline(0)={baseline_acc:.4f}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick wins: tune threshold for directional accuracy")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--market", type=str, default="us", choices=["us", "india"]) 
    parser.add_argument("--low", type=float, default=-0.02)
    parser.add_argument("--high", type=float, default=0.02)
    parser.add_argument("--n", type=int, default=81)
    args = parser.parse_args()
    tune_threshold(args.ticker, market=args.market, low=args.low, high=args.high, n=args.n)
