"""
sentiment.py
------------
Computes sentiment features from raw Finnhub sentiment data.

Features produced:
    Raw
        sentiment_score             — daily avg_sentiment from loader
        article_count               — number of articles that day

    Rolling averages of sentiment score
        sentiment_ma_3d             — 3-day rolling mean
        sentiment_ma_5d             — 5-day rolling mean
        sentiment_ma_10d            — 10-day rolling mean

    Rolling averages of article count (buzz)
        buzz_ma_3d                  — 3-day rolling mean article count
        buzz_ma_5d                  — 5-day rolling mean article count

    Derived
        sentiment_momentum          — sentiment_ma_3d - sentiment_ma_10d
                                      (short vs long-term sentiment shift)
        sentiment_std_5d            — 5-day rolling std (sentiment uncertainty)

Usage:
    from ml.src.features.sentiment import SentimentFeatures
    sent = SentimentFeatures()
    df_sent = sent.compute(sentiment_df)
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class SentimentFeatures:
    """
    Computes sentiment features from raw sentiment DataFrame.

    Input:  DataFrame with DatetimeIndex and columns:
                avg_sentiment  (float, -1 to +1)
                article_count  (int)
    Output: DataFrame with all sentiment feature columns.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg  = _load_config(config_path)
        sent = cfg["features"]["sentiment"]

        self.rolling_windows = sent["rolling_windows"]    # [3, 5, 10]
        self.buzz_windows    = sent["buzz_windows"]       # [3, 5]

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all sentiment features from raw sentiment DataFrame.
        Returns a new DataFrame — original is not modified.

        If df is empty (e.g. no Finnhub key), returns empty DataFrame.
        The feature pipeline handles missing sentiment gracefully.
        """
        if df is None or df.empty:
            logger.warning(
                "[sentiment] Empty sentiment data — sentiment features will be zeroed."
            )
            return pd.DataFrame()

        df = df.copy().sort_index()

        logger.info("[sentiment] Computing sentiment features ...")

        df = self._rename_raw(df)
        df = self._rolling_sentiment(df)
        df = self._rolling_buzz(df)
        df = self._derived(df)

        # Drop raw columns — only keep engineered features
        df = df.drop(columns=["avg_sentiment", "article_count"], errors="ignore")

        logger.info(f"[sentiment] Done. {len(df)} rows, {len(df.columns)} features.")
        return df

    def feature_names(self) -> list[str]:
        """Return the list of feature column names this module produces."""
        names = ["sentiment_score", "article_count"]
        names += [f"sentiment_ma_{w}d" for w in self.rolling_windows]
        names += [f"buzz_ma_{w}d"      for w in self.buzz_windows]
        names += ["sentiment_momentum", "sentiment_std_5d"]
        return names

    def fill_missing(
        self, df: pd.DataFrame, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        When sentiment data is unavailable, return a zero-filled DataFrame
        with the correct index. Called by pipeline.py when sentiment is missing.
        """
        names = self.feature_names()
        result = pd.DataFrame(0.0, index=index, columns=names)
        logger.warning(
            "[sentiment] No sentiment data — all sentiment features set to 0."
        )
        return result

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _rename_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep raw columns with cleaner names for intermediate use."""
        rename_map = {}
        if "avg_sentiment" in df.columns:
            rename_map["avg_sentiment"] = "sentiment_score"
        if "article_count" in df.columns:
            rename_map["article_count"] = "article_count"
        return df.rename(columns=rename_map)

    def _rolling_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling means of daily sentiment score."""
        if "sentiment_score" not in df.columns:
            return df
        for w in self.rolling_windows:
            df[f"sentiment_ma_{w}d"] = (
                df["sentiment_score"]
                .rolling(window=w, min_periods=1)
                .mean()
            )
        return df

    def _rolling_buzz(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling means of article count (news buzz)."""
        if "article_count" not in df.columns:
            return df
        for w in self.buzz_windows:
            df[f"buzz_ma_{w}d"] = (
                df["article_count"]
                .rolling(window=w, min_periods=1)
                .mean()
            )
        return df

    def _derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        sentiment_momentum: short-term vs long-term sentiment shift.
            Positive = sentiment improving recently vs trend.
            Negative = sentiment deteriorating.

        sentiment_std_5d: rolling 5-day std of sentiment score.
            High value = high uncertainty / mixed signals in news.
        """
        short_col = f"sentiment_ma_{min(self.rolling_windows)}d"
        long_col  = f"sentiment_ma_{max(self.rolling_windows)}d"

        if short_col in df.columns and long_col in df.columns:
            df["sentiment_momentum"] = df[short_col] - df[long_col]
        else:
            df["sentiment_momentum"] = 0.0

        if "sentiment_score" in df.columns:
            df["sentiment_std_5d"] = (
                df["sentiment_score"]
                .rolling(window=5, min_periods=1)
                .std()
                .fillna(0)
            )
        else:
            df["sentiment_std_5d"] = 0.0

        return df