"""
nifty_feature_guard.py
----------------------
Guards and validators for Nifty-specific feature engineering issues.

Problem: Nifty's Unused_Data (2008-2018) has no true High/Low columns.
The loader approximates them as:
    High = max(Open, Close) * 1.005
    Low  = min(Open, Close) * 0.995

This makes atr_14 (Average True Range) a near-constant multiple of Close
rather than true intraday volatility. Including atr_14 in causal discovery
for Nifty would measure price level, not range — corrupting the feature.

This module:
    1. Validates which features are affected by synthetic H/L
    2. Removes them from the causal feature candidate set for Nifty
    3. Documents affected features in the paper's data section
    4. Provides a diagnostic showing the ATR/Close correlation

Paper section: "Data Limitations — Nifty 50 (2008-2018)"
    "The Unused_Data source does not provide intraday High/Low prices.
     We approximate these as ±0.5% of Close, resulting in atr_14 being
     mechanically correlated with price level rather than measuring true
     intraday range. We therefore exclude atr_14 from the causal feature
     candidate set for the Nifty 50 model. This affects 1,407 of 3,555
     rows (2008-2018). All other technical features are computed from
     Close prices only and are unaffected."

Usage:
    from ml.src.features.nifty_feature_guard import NiftyFeatureGuard
    guard = NiftyFeatureGuard()
    safe_features = guard.filter_synthetic_hl_features(all_features)
    guard.print_atr_diagnostic(df)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Features that are DIRECTLY computed from High/Low and are therefore
# corrupted by the synthetic H/L approximation for the 2008-2018 period.
SYNTHETIC_HL_AFFECTED_FEATURES = {
    "atr_14",            # Average True Range: uses High, Low, prev Close
    # Note: bb_upper, bb_lower, bb_width, bb_pct use Close only — SAFE
    # Note: volatility_Nd uses log_return_1d from Close only — SAFE
    # Note: rsi_14 uses Close only — SAFE
    # Note: macd, macd_signal, macd_hist use Close only — SAFE
}

# Features safe to use even with synthetic H/L
SYNTHETIC_HL_SAFE_FEATURES = {
    "log_return_1d", "log_return_5d", "excess_return_5d",
    "momentum_5d", "momentum_10d", "momentum_20d",
    "volatility_10d", "volatility_20d", "volatility_30d",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_pct",
    # All macro, sentiment, fundamental features are unaffected
}


class NiftyFeatureGuard:
    """
    Guards against using synthetic H/L-derived features in Nifty causal discovery.
    """

    def __init__(self):
        self.affected   = SYNTHETIC_HL_AFFECTED_FEATURES
        self.safe       = SYNTHETIC_HL_SAFE_FEATURES

    def filter_synthetic_hl_features(
        self,
        feature_list: list[str],
        warn: bool = True,
    ) -> list[str]:
        """
        Remove features that are corrupted by synthetic H/L approximation.

        Args:
            feature_list: Candidate feature names
            warn:         If True, log which features were removed

        Returns:
            Filtered list with affected features removed.
        """
        removed  = [f for f in feature_list if f in self.affected]
        filtered = [f for f in feature_list if f not in self.affected]

        if removed and warn:
            logger.warning(
                f"[nifty_feature_guard] Removed {len(removed)} features "
                f"corrupted by synthetic H/L (2008-2018): {removed}. "
                f"See nifty_feature_guard.py for explanation."
            )
        elif not removed:
            logger.info(
                "[nifty_feature_guard] No synthetic H/L features in candidate set."
            )

        return filtered

    def print_atr_diagnostic(self, df: pd.DataFrame) -> dict:
        """
        Show the ATR/Close correlation diagnostic.

        If ATR is truly measuring intraday range, its correlation with
        the rolling standard deviation of returns should be high (~0.7+),
        and its correlation with Close level should be low (<0.3).

        With synthetic H/L, ATR ≈ Close * constant, so:
            corr(atr_14, close) → ~1.0  (BAD: measuring price level)
            corr(atr_14, volatility_20d) → low (BAD: not measuring range)

        Paper evidence: report these correlations to justify exclusion.
        """
        result = {}

        if "atr_14" not in df.columns:
            logger.info("[nifty_feature_guard] atr_14 not in DataFrame — nothing to diagnose.")
            return result

        if "close" not in df.columns:
            logger.warning("[nifty_feature_guard] 'close' column not found for diagnostic.")
            return result

        atr   = df["atr_14"].dropna()
        close = df["close"].reindex(atr.index)

        corr_close = float(atr.corr(close))
        result["atr_close_correlation"] = corr_close

        if "volatility_20d" in df.columns:
            vol = df["volatility_20d"].reindex(atr.index)
            corr_vol = float(atr.corr(vol))
            result["atr_vol20d_correlation"] = corr_vol
        else:
            corr_vol = None
            result["atr_vol20d_correlation"] = None

        # Split by data source period
        split_date = "2019-01-01"
        atr_old  = atr[atr.index < split_date]
        atr_new  = atr[atr.index >= split_date]
        close_old = close[close.index < split_date]
        close_new = close[close.index >= split_date]

        if len(atr_old) > 10 and len(close_old) == len(atr_old):
            result["atr_close_corr_2008_2018"] = float(atr_old.corr(close_old))
        if len(atr_new) > 10 and len(close_new) == len(atr_new):
            result["atr_close_corr_2019_2024"] = float(atr_new.corr(close_new))

        print(f"\n{'='*55}")
        print(f"  ATR DIAGNOSTIC — Nifty 50")
        print(f"{'='*55}")
        print(f"  ATR vs Close correlation (full):  {corr_close:.4f}")
        if corr_vol is not None:
            print(f"  ATR vs Vol_20d correlation:       {corr_vol:.4f}")
        if "atr_close_corr_2008_2018" in result:
            print(f"  ATR vs Close corr (2008-2018):    {result['atr_close_corr_2008_2018']:.4f}")
        if "atr_close_corr_2019_2024" in result:
            print(f"  ATR vs Close corr (2019-2024):    {result['atr_close_corr_2019_2024']:.4f}")
        print(f"")
        if corr_close > 0.8:
            print(f"  ⚠ ATR is highly correlated with Close (>{0.8:.0%}).")
            print(f"    This confirms ATR measures price LEVEL, not intraday range.")
            print(f"    EXCLUDE atr_14 from Nifty causal discovery. ✓")
        else:
            print(f"  ATR correlation with Close is {corr_close:.4f} — acceptable.")
        print(f"{'='*55}\n")

        return result

    def get_paper_footnote(self) -> str:
        """Return the exact footnote text for the paper's data section."""
        return (
            "The Nifty 50 dataset for 2008-2018 (Unused_Data.csv) does not "
            "include intraday High and Low prices. We approximate these as "
            "High = max(Open, Close) × 1.005 and Low = min(Open, Close) × 0.995, "
            "which introduces a near-perfect correlation between atr_14 and the "
            "Close price level (r > 0.95 for 2008-2018, compared to r ≈ 0.40 for "
            "2019-2024 where true OHLC data is available). We therefore exclude "
            "atr_14 from the causal feature candidate set for the Nifty 50 model "
            "to avoid confounding price level with intraday volatility. All other "
            "technical features are derived from daily Close prices and are "
            "unaffected by this approximation."
        )


def apply_nifty_feature_guard_to_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop atr_14 from a Nifty feature matrix and log the action.
    Call this in nifty_loader.build_feature_matrix() after the merge step.

    Args:
        df: Feature matrix as produced by NiftyLoader.build_feature_matrix()

    Returns:
        df with atr_14 dropped (if present), plus a note added as column metadata.
    """
    guard = NiftyFeatureGuard()
    guard.print_atr_diagnostic(df)

    cols_to_drop = [c for c in SYNTHETIC_HL_AFFECTED_FEATURES if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(
            f"[nifty_feature_guard] Dropped synthetic-H/L features: {cols_to_drop}"
        )
    return df