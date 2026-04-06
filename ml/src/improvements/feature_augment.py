"""
feature_augment.py
-------------------
Conservative feature augmentation utilities used in a short "feature
engineering sprint". Adds lagged features, rolling volatility and simple
momentum features to an existing feature matrix and saves an augmented
copy `{ticker}_features_aug.csv`.

This is intentionally non-destructive: the original feature matrix is
left untouched and the augmented matrix is saved separately.
"""

import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from ml.src.improvements.diagnostics import _load_feature_matrix
from ml.src.causal.selector import CausalSelector
from ml.src.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


def augment_features(
    ticker: str,
    market: str = "us",
    lag_days: List[int] = [1, 5],
    vol_windows: List[int] = [5, 10],
    momentum_windows: List[int] = [5, 10],
    out_suffix: str = "_aug",
) -> Dict:
    """
    Add lag, volatility, and momentum features and save augmented matrix.

    Returns a small report dict describing saved file and number of new cols.
    """
    ticker = ticker.upper()
    df = _load_feature_matrix(ticker, market)

    # Decide which numeric columns to augment: causal features if present
    try:
        causal = CausalSelector().load(ticker)
        cols = [c for c in causal if c in df.columns]
        if not cols:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
    except Exception:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    cols = [c for c in cols if c in df.columns]
    new_cols = []

    # Lag features
    for lag in lag_days:
        for c in cols:
            name = f"{c}_lag_{lag}"
            df[name] = df[c].shift(lag)
            new_cols.append(name)

    # Rolling volatility on the target (if present) and on top numeric cols
    target = None
    try:
        from ml.src.data.loader import _load_config
        cfg = _load_config()
        target = cfg["model"]["target"] if cfg else None
    except Exception:
        target = None

    vol_base = target if target in df.columns else (cols[0] if cols else None)
    for w in vol_windows:
        if vol_base is None:
            break
        name = f"{vol_base}_vol_{w}"
        df[name] = df[vol_base].rolling(w).std()
        new_cols.append(name)

    # Momentum (rolling mean of returns)
    for w in momentum_windows:
        for c in cols:
            name = f"{c}_mom_{w}"
            df[name] = df[c].rolling(window=w).mean()
            new_cols.append(name)

    # Drop rows with NaNs introduced by shifts/rolling at the top
    df = df.dropna()

    # Save augmented matrix
    pipeline = FeaturePipeline()
    out_dir = pipeline.features_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}_features{out_suffix}.csv"
    df.to_csv(out_path)

    report = {
        "ticker": ticker,
        "saved_path": str(out_path),
        "new_columns": len(new_cols),
        "final_columns": len(df.columns),
        "dropped_rows": None,
    }

    logger.info(f"Saved augmented feature matrix → {out_path}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment feature matrix")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--market", choices=["us", "india"], default="us")
    args = parser.parse_args()
    r = augment_features(args.ticker, market=args.market)
    print(r)
