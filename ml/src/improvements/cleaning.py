"""
cleaning.py
-----------
Feature-matrix cleaning utilities.

Provides a safe, conservative cleaning step that:
  - drops columns with > `drop_nan_ratio` NaNs
  - drops constant columns
  - fills remaining NaNs (ffill, bfill, median)
  - replaces inf with NaN

Saves a cleaned feature matrix as `{ticker}_features_cleaned.csv` in the
features directory so the original matrix is preserved.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ml.src.improvements.diagnostics import _load_feature_matrix
from ml.src.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


def clean_feature_matrix(
    ticker: str,
    market: str = "us",
    drop_nan_ratio: float = 0.01,
    out_suffix: str = "_cleaned",
) -> Dict:
    """
    Clean the feature matrix conservatively and save a cleaned copy.

    Returns a small report dict with summary of actions taken.
    """
    ticker = ticker.upper()
    df = _load_feature_matrix(ticker, market)

    orig_cols = list(df.columns)
    report = {"orig_columns": len(orig_cols)}

    # Replace infinite values
    numeric = df.select_dtypes(include=[np.number])
    inf_mask = np.isinf(numeric.values)
    if inf_mask.any():
        logger.warning(f"Found {inf_mask.sum()} infinite values — replacing with NaN")
        df = df.replace([np.inf, -np.inf], np.nan)

    # Drop columns with too many NaNs
    nan_ratio = df.isnull().mean()
    drop_cols = nan_ratio[nan_ratio > drop_nan_ratio].index.tolist()
    df = df.drop(columns=drop_cols)
    report["dropped_for_nan"] = drop_cols

    # Drop constant columns
    numeric = df.select_dtypes(include=[np.number])
    constant_cols = [c for c in numeric.columns if numeric[c].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
    report["dropped_constant"] = constant_cols

    # Fill remaining NaNs conservatively: ffill -> bfill -> median
    df = df.fillna(method="ffill").fillna(method="bfill")
    for c in df.columns:
        if df[c].isnull().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                med = df[c].median()
                df[c] = df[c].fillna(med)
            else:
                df[c] = df[c].fillna("")

    # Save cleaned matrix
    pipeline = FeaturePipeline()
    out_dir = pipeline.features_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}_features{out_suffix}.csv"
    df.to_csv(out_path)
    report["cleaned_path"] = str(out_path)
    report["final_columns"] = len(df.columns)

    logger.info(f"Saved cleaned feature matrix → {out_path}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean feature matrix for a ticker")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--market", choices=["us", "india"], default="us")
    parser.add_argument("--drop-nan-ratio", type=float, default=0.01)
    args = parser.parse_args()
    r = clean_feature_matrix(args.ticker, market=args.market, drop_nan_ratio=args.drop_nan_ratio)
    print(r)
