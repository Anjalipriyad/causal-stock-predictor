"""
monitor.py
----------
Lightweight monitoring and fallback mechanism.

Checks recent model performance (directional accuracy) and creates a
simple fallback rule if performance drops below a configured threshold.
"""

import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from ml.src.data.loader import _load_config
from ml.src.evaluation.metrics import Metrics
from ml.src.improvements.diagnostics import _load_feature_matrix
from ml.src.ensemble import Ensemble

logger = logging.getLogger(__name__)


def monitor_model(ticker: str, market: str = "us", threshold: float = 0.60, test_period_days: int = 252) -> Dict:
    """
    Evaluate the saved ensemble on the most recent `test_period_days` and
    emit a simple fallback if directional accuracy < threshold.

    Fallback policy (simple): use 5-day momentum sign as prediction.
    """
    ticker = ticker.upper()
    cfg = _load_config()
    df = _load_feature_matrix(ticker, market)

    # Recent window
    recent_df = df.tail(test_period_days)
    ensemble = Ensemble(cfg=cfg)
    try:
        ensemble.load(ticker)
    except Exception as e:
        logger.warning(f"monitor: could not load ensemble for {ticker}: {e}")
        return {"status": "no_model"}

    # Predictions on recent data
    try:
        preds = ensemble.predict_historical(recent_df)
    except Exception as e:
        logger.warning(f"monitor: prediction failed: {e}")
        return {"status": "predict_failed"}

    m = Metrics()
    if "actual_return" not in preds.columns:
        return {"status": "no_actuals"}

    scores = m.compute_all(preds["predicted_return"], preds["actual_return"], label="monitor_recent")
    da = scores.get("directional_accuracy", 0.0)

    out = {"ticker": ticker, "directional_accuracy": da, "threshold": threshold}

    if da < threshold:
        # Create simple fallback: 5-day momentum rule
        fallback = {"type": "momentum", "window": 5}
        models_dir = Path(ensemble.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"fallback_{ticker}.json"
        with open(out_path, "w") as f:
            json.dump(fallback, f, indent=2)
        logger.warning(f"monitor: DA below threshold ({da:.3f} < {threshold}) — saved fallback → {out_path}")
        out["fallback_saved"] = str(out_path)
    else:
        logger.info(f"monitor: DA OK ({da:.3f} >= {threshold})")

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor model and save fallback if needed")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--market", choices=["us", "india"], default="us")
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()
    print(monitor_model(args.ticker, market=args.market, threshold=args.threshold))
