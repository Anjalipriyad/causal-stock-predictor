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

    # Store raw target BEFORE any smoothing for the direction label
    raw_target = df_out[target_col].copy()

    if smoothing_window is not None and smoothing_window > 1:
        # ┌────────────────────────────────────────────────────────────────┐
        # │ WARNING — EXPERIMENTAL ONLY, DO NOT USE FOR PAPER RESULTS     │
        # │                                                               │
        # │ The target (e.g. excess_return_5d) already encodes future     │
        # │ prices via shift(-horizon). Rolling mean with window=W on     │
        # │ this target blends returns from t+5, t+6, ..., t+4+W into    │
        # │ the label at time t. A model trained on this smoothed target  │
        # │ and evaluated against actual_return will show overstated      │
        # │ accuracy because the training objective leaks future data     │
        # │ beyond the stated horizon.                                    │
        # │                                                               │
        # │ This function is kept for ablation experiments only.          │
        # └────────────────────────────────────────────────────────────────┘
        import warnings
        warnings.warn(
            "target_transform: smoothing a forward-looking target introduces "
            "additional lookahead beyond the stated horizon. Results using "
            "smoothed targets are EXPERIMENTAL ONLY and must NOT be used as "
            "paper evidence.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(
            "[target_transform] EXPERIMENTAL: smoothing target with window=%d. "
            "This introduces lookahead beyond the stated horizon.",
            smoothing_window,
        )
        df_out[f"{target_col}_sm_{smoothing_window}"] = df_out[target_col].rolling(smoothing_window).mean()
        # Optionally use the smoothed target as the active target
        df_out[ target_col ] = df_out[f"{target_col}_sm_{smoothing_window}"]

    # Binary direction label — ALWAYS uses raw (unsmoothed) target to prevent
    # the direction label from encoding smoothed future returns (Issue #12)
    df_out["direction"] = (raw_target >= threshold).astype(int)

    # Quantile discretization (optional)
    if discretize_quantiles is not None and discretize_quantiles > 1:
        # Prevent look-ahead bias using rolling percentile ranking instead of global pd.qcut
        roll_pct = df_out[target_col].rolling(window=252, min_periods=20).rank(pct=True)
        # Map percentile [0,1] to discretize bins [0, discretize_quantiles - 1]
        df_out["target_q"] = (roll_pct * discretize_quantiles).apply(np.floor).clip(0, discretize_quantiles - 1).fillna(0).astype(int)

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
