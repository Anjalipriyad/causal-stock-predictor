"""
target_transform.py
--------------------
Utilities to create/adjust target columns used by models.

Provides safe, reversible transforms such as:
  - smoothing the target via rolling mean
  - creating a binary `direction` label (UP/DOWN)
  - discretizing returns into quantiles

Saves transformed matrix as `{ticker}_features_target.csv` by default.
"""

import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

from ml.src.improvements.diagnostics import _load_feature_matrix
from ml.src.features.pipeline import FeaturePipeline
from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


def transform_target(
    ticker: str,
    market: str = "us",
    smoothing_window: Optional[int] = None,
    threshold: float = 0.0,
    discretize_quantiles: Optional[int] = None,
    out_suffix: str = "_target",
) -> Dict:
    """
    Apply simple target transforms and save a copy of the matrix.

    - `smoothing_window`: if provided, replace target with rolling mean
    - `threshold`: used for binary direction label (>= threshold => UP)
    - `discretize_quantiles`: if provided, add `target_q` column with quantile bins
    """
    ticker = ticker.upper()
    cfg = _load_config()
    df = _load_feature_matrix(ticker, market)

    target_col = cfg["model"]["target"]
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in feature matrix.")

    df_out = df.copy()

    if smoothing_window is not None and smoothing_window > 1:
        df_out[f"{target_col}_sm_{smoothing_window}"] = df_out[target_col].rolling(smoothing_window).mean()
        # Optionally use the smoothed target as the active target
        df_out[ target_col ] = df_out[f"{target_col}_sm_{smoothing_window}"]

    # Binary direction label
    df_out["direction"] = (df_out[target_col] >= threshold).astype(int)

    # Quantile discretization (optional)
    if discretize_quantiles is not None and discretize_quantiles > 1:
        df_out["target_q"] = pd.qcut(df_out[target_col].rank(method="first"), discretize_quantiles, labels=False)

    # Drop NA introduced by smoothing
    df_out = df_out.dropna()

    pipeline = FeaturePipeline()
    out_dir = pipeline.features_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}_features{out_suffix}.csv"
    df_out.to_csv(out_path)

    report = {
        "ticker": ticker,
        "saved_path": str(out_path),
        "rows": len(df_out),
        "columns": len(df_out.columns),
    }

    logger.info(f"Saved target-transformed feature matrix → {out_path}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transform target column (smoothing, direction label)")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--market", choices=["us", "india"], default="us")
    parser.add_argument("--smoothing-window", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--discretize-quantiles", type=int, default=None)
    args = parser.parse_args()
    r = transform_target(
        args.ticker,
        market=args.market,
        smoothing_window=args.smoothing_window,
        threshold=args.threshold,
        discretize_quantiles=args.discretize_quantiles,
    )
    print(r)
