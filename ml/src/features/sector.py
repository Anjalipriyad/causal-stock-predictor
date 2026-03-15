"""
sector.py
---------
Additional high-signal features not in technical.py or macro.py.

Features produced:
    Sector relative momentum
        sector_rel_momentum_5d   — stock return vs sector ETF (5-day)
        sector_rel_momentum_20d  — stock return vs sector ETF (20-day)

    Volume anomaly
        volume_ratio_20d         — today's volume / 20-day avg volume
        volume_trend_5d          — 5-day volume trend (slope)

    Volatility regime
        vol_regime               — current vol vs 60-day avg (high=1, low=0)
        vol_of_vol               — rolling std of volatility (uncertainty signal)

    Price momentum vs market
        beta_adjusted_momentum   — momentum adjusted for market beta
        idiosyncratic_momentum   — momentum unexplained by market

Usage:
    from ml.src.features.sector import SectorFeatures
    sector = SectorFeatures()
    df_extra = sector.compute(price_df, spy_df, sector_etf_df)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class SectorFeatures:
    """
    Computes sector-relative and volume-based features.

    Input:
        price_df:      OHLCV DataFrame for the target stock
        spy_df:        S&P 500 close prices (^GSPC)
        sector_etf_df: Sector ETF close prices (e.g. XLK for tech)

    Output: DataFrame with all sector/volume features
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg = _load_config(config_path)
        self.momentum_windows = cfg["features"]["technical"]["momentum_windows"]

    def compute(
        self,
        price_df: pd.DataFrame,
        spy_df: Optional[pd.DataFrame] = None,
        sector_etf_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute all sector + volume features.
        All inputs must have DatetimeIndex.
        Missing inputs are handled gracefully — features just won't be computed.
        """
        logger.info("[sector] Computing sector/volume features ...")
        frames = []

        frames.append(self._volume_features(price_df))

        if spy_df is not None and not spy_df.empty:
            frames.append(self._market_relative(price_df, spy_df))

        if sector_etf_df is not None and not sector_etf_df.empty:
            frames.append(self._sector_relative(price_df, sector_etf_df))

        frames.append(self._volatility_regime(price_df))

        result = pd.concat(
            [f for f in frames if f is not None and not f.empty],
            axis=1
        ).sort_index()

        logger.info(f"[sector] Done. {len(result.columns)} features computed.")
        return result

    # -----------------------------------------------------------------------
    # Volume features
    # -----------------------------------------------------------------------

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume ratio and trend.
        High volume on up days = bullish confirmation.
        High volume on down days = bearish confirmation.
        """
        if "volume" not in df.columns:
            return pd.DataFrame()

        vol = df["volume"].astype(float)
        result = pd.DataFrame(index=df.index)

        # Volume ratio: today vs 20-day average
        result["volume_ratio_20d"] = vol / vol.rolling(20).mean()

        # Volume trend: slope of last 5 days (positive = increasing interest)
        result["volume_trend_5d"] = (
            vol.rolling(5).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean()
                if x.mean() > 0 else 0,
                raw=True
            )
        )

        # On-balance volume signal (normalised)
        direction = np.sign(df["close"].diff())
        obv       = (vol * direction).cumsum()
        result["obv_momentum_10d"] = obv.diff(10) / vol.rolling(10).mean().replace(0, np.nan)

        return result

    # -----------------------------------------------------------------------
    # Market-relative features
    # -----------------------------------------------------------------------

    def _market_relative(
        self, price_df: pd.DataFrame, spy_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Beta-adjusted momentum and idiosyncratic return.
        Separates stock-specific signal from market noise.
        """
        result = pd.DataFrame(index=price_df.index)

        stock_ret = np.log(price_df["close"] / price_df["close"].shift(1))
        spy_close = spy_df["close"] if "close" in spy_df.columns else spy_df.iloc[:, 0]
        spy_ret   = np.log(spy_close / spy_close.shift(1))
        spy_ret   = spy_ret.reindex(price_df.index).ffill()

        # Rolling beta (60-day)
        cov    = stock_ret.rolling(60).cov(spy_ret)
        var    = spy_ret.rolling(60).var()
        beta   = cov / var.replace(0, np.nan)
        result["rolling_beta"] = beta.clip(-3, 3)

        # Idiosyncratic momentum: stock momentum - beta * market momentum
        for w in [5, 20]:
            stock_mom  = stock_ret.rolling(w).sum()
            market_mom = spy_ret.rolling(w).sum()
            result[f"idiosyncratic_momentum_{w}d"] = stock_mom - beta * market_mom

        return result

    # -----------------------------------------------------------------------
    # Sector-relative features
    # -----------------------------------------------------------------------

    def _sector_relative(
        self, price_df: pd.DataFrame, sector_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Return relative to sector ETF.
        Captures sector rotation and stock-specific outperformance.
        """
        result = pd.DataFrame(index=price_df.index)

        stock_ret  = np.log(price_df["close"] / price_df["close"].shift(1))
        sect_close = sector_df["close"] if "close" in sector_df.columns else sector_df.iloc[:, 0]
        sect_ret   = np.log(sect_close / sect_close.shift(1))
        sect_ret   = sect_ret.reindex(price_df.index).ffill()

        for w in [5, 20]:
            result[f"sector_rel_momentum_{w}d"] = (
                stock_ret.rolling(w).sum() - sect_ret.rolling(w).sum()
            )

        return result

    # -----------------------------------------------------------------------
    # Volatility regime
    # -----------------------------------------------------------------------

    def _volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility regime and vol-of-vol.
        High vol regime = model should be less confident.
        """
        result = pd.DataFrame(index=df.index)

        log_ret = np.log(df["close"] / df["close"].shift(1))
        vol_20  = log_ret.rolling(20).std() * np.sqrt(252)
        vol_60  = log_ret.rolling(60).std() * np.sqrt(252)

        # Vol regime: 1 = high vol (current > 60-day avg), 0 = low vol
        result["vol_regime"] = (vol_20 > vol_60).astype(float)

        # Vol of vol: rolling std of 20-day vol (measures vol uncertainty)
        result["vol_of_vol"] = vol_20.rolling(20).std()

        # Vol percentile rank (0-1): where is today's vol in 252-day history
        result["vol_percentile"] = vol_20.rolling(252).rank(pct=True)

        return result