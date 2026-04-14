"""
technical.py
------------
Computes technical indicator features from OHLCV price data.
All parameters are read from config.yaml — nothing hardcoded.

Features produced:
    Momentum
        momentum_5d, momentum_10d, momentum_20d     — price / price_N_days_ago - 1
    Volatility
        volatility_10d, volatility_20d, volatility_30d  — rolling std of log returns
    RSI
        rsi_14                                      — Relative Strength Index
    MACD
        macd, macd_signal, macd_hist                — MACD line, signal, histogram
    Bollinger Bands
        bb_upper, bb_lower, bb_mid, bb_width        — bands + bandwidth
        bb_pct                                      — %B: where price sits in band
    ATR
        atr_14                                      — Average True Range (normalised)
    Log return
        log_return_1d                               — daily log return (used as input feature)
        log_return_5d                               — TARGET variable (forward 5-day return)

Usage:
    from ml.src.features.technical import TechnicalFeatures
    tech = TechnicalFeatures()
    df_features = tech.compute(price_df)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """
    Computes all technical features from a raw OHLCV DataFrame.
    Input:  DataFrame with columns [open, high, low, close, volume]
            and a DatetimeIndex.
    Output: DataFrame with original OHLCV + all technical features.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg  = _load_config(config_path)
        tech = cfg["features"]["technical"]

        self.rsi_period         = tech["rsi_period"]
        self.macd_fast          = tech["macd_fast"]
        self.macd_slow          = tech["macd_slow"]
        self.macd_signal        = tech["macd_signal"]
        self.bb_window          = tech["bollinger_window"]
        self.bb_std             = tech["bollinger_std"]
        self.momentum_windows   = tech["momentum_windows"]
        self.volatility_windows = tech["volatility_windows"]
        self.atr_period         = tech["atr_period"]
        self.horizon            = cfg["model"]["horizon_days"]

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all technical feature computations on an OHLCV DataFrame.
        Returns a new DataFrame — original is not modified.
        """
        df = df.copy()
        df = df.sort_index()

        logger.info("[technical] Computing technical features ...")

        df = self._log_returns(df)
        df = self._momentum(df)
        df = self._volatility(df)
        df = self._rsi(df)
        df = self._macd(df)
        df = self._bollinger(df)
        df = self._atr(df)
        df = self._target(df)

        n_before = len(df)
        df = df.dropna()
        n_after  = len(df)
        logger.info(
            f"[technical] Done. {n_after} rows after dropna "
            f"({n_before - n_after} dropped for warmup)."
        )
        return df

    def feature_names(self) -> list[str]:
        """Return the list of feature column names this module produces."""
        names = ["log_return_1d"]
        names += [f"momentum_{w}d"   for w in self.momentum_windows]
        names += [f"volatility_{w}d" for w in self.volatility_windows]
        names += ["rsi_14"]
        names += ["macd", "macd_signal", "macd_hist"]
        names += ["bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_pct"]
        names += ["atr_14"]
        return names

    # -----------------------------------------------------------------------
    # Private computations
    # -----------------------------------------------------------------------

    def _log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Daily log return: ln(close_t / close_t-1)"""
        df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))
        return df

    def _momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Price momentum over N days: close_t / close_{t-N} - 1
        Captures short, medium, and longer-term trend direction.
        """
        for w in self.momentum_windows:
            df[f"momentum_{w}d"] = df["close"].pct_change(periods=w)
        return df

    def _volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling standard deviation of log returns.
        Represents realised volatility over N-day window.
        Annualised by multiplying by sqrt(252).
        """
        for w in self.volatility_windows:
            df[f"volatility_{w}d"] = (
                df["log_return_1d"]
                .rolling(window=w)
                .std()
                * np.sqrt(252)
            )
        return df

    def _rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Relative Strength Index (Wilder smoothing).
        RSI = 100 - 100 / (1 + avg_gain / avg_loss)
        Values: 0-100. Overbought > 70, oversold < 30.
        """
        period = self.rsi_period
        delta  = df["close"].diff()
        gain   = delta.clip(lower=0)
        loss   = -delta.clip(upper=0)

        # Wilder's smoothing (equivalent to EMA with alpha=1/period)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs           = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        return df

    def _macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal_period)
        Histogram = MACD - Signal
        """
        ema_fast = df["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow, adjust=False).mean()

        df["macd"]        = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # Normalise MACD by price level so it's comparable across tickers
        df["macd"]        = df["macd"]        / df["close"]
        df["macd_signal"] = df["macd_signal"] / df["close"]
        df["macd_hist"]   = df["macd_hist"]   / df["close"]
        return df

    def _bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bollinger Bands:
            mid   = SMA(close, window)
            upper = mid + std * multiplier
            lower = mid - std * multiplier
            width = (upper - lower) / mid     — normalised bandwidth
            pct   = (close - lower) / (upper - lower)  — %B position
        """
        window = self.bb_window
        mult   = self.bb_std

        mid   = df["close"].rolling(window=window).mean()
        std   = df["close"].rolling(window=window).std()
        upper = mid + mult * std
        lower = mid - mult * std

        df["bb_mid"]   = mid
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["bb_width"] = (upper - lower) / mid.replace(0, np.nan)
        df["bb_pct"]   = (df["close"] - lower) / (upper - lower).replace(0, np.nan)

        # Normalise band levels by price
        df["bb_upper"] = df["bb_upper"] / df["close"]
        df["bb_lower"] = df["bb_lower"] / df["close"]
        df["bb_mid"]   = df["bb_mid"]   / df["close"]
        return df

    def _atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Average True Range — measures market volatility.
        TR = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = EMA(TR, period)
        Normalised by close price so it's comparable across tickers.
        """
        period     = self.atr_period
        prev_close = df["close"].shift(1)

        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)

        df["atr_14"] = (
            tr.ewm(span=period, adjust=False).mean()
            / df["close"].replace(0, np.nan)
        )
        return df

    def _target(self, df: pd.DataFrame, spy_close: pd.Series = None) -> pd.DataFrame:
        """
        Forward 5-day excess return — the prediction target.
        excess_return_5d(t) = log(stock_t+5/stock_t) - log(SPY_t+5/SPY_t)

        Removing market-wide moves isolates stock-specific signal.
        Falls back to raw log return if SPY data not available.

        WARNING — BOTH excess_return_5d AND log_return_5d contain
        future prices via shift(-horizon). They are target columns
        and must NEVER be included as features during causal discovery
        or model training. Use drop_leaky_columns() to strip them.
        """
        stock_ret = np.log(df["close"].shift(-self.horizon) / df["close"])

        if spy_close is not None and not spy_close.empty:
            # Align SPY to same index
            spy_aligned = spy_close.reindex(df.index).ffill()
            spy_ret = np.log(spy_aligned.shift(-self.horizon) / spy_aligned)
            df["excess_return_5d"] = stock_ret - spy_ret
        else:
            # Fallback to raw return if SPY not available
            df["excess_return_5d"] = stock_ret

        # Keep raw return as auxiliary target column (NOT a feature)
        df["log_return_5d"] = stock_ret
        return df

    @staticmethod
    def drop_leaky_columns(df: pd.DataFrame, keep_target: str = None) -> pd.DataFrame:
        """
        Remove forward-looking target columns from a DataFrame.

        Both excess_return_5d and log_return_5d contain future prices
        (shift(-horizon)) and must be stripped before causal discovery
        or any analysis that should not see the future.

        Args:
            df:          DataFrame to clean
            keep_target: If set, keep this one column (e.g. for Granger
                         which needs the target column in the DataFrame)

        Returns:
            DataFrame with leaky columns removed
        """
        leaky = ["excess_return_5d", "log_return_5d"]
        to_drop = [c for c in leaky if c in df.columns and c != keep_target]
        return df.drop(columns=to_drop)

    def set_spy_close(self, spy_close: pd.Series) -> None:
        """Set SPY close prices for excess return calculation."""
        self._spy_close = spy_close

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Override to pass spy_close to _target
        df = df.copy()
        df = df.sort_index()
        df = self._log_returns(df)
        df = self._momentum(df)
        df = self._volatility(df)
        df = self._rsi(df)
        df = self._macd(df)
        df = self._bollinger(df)
        df = self._atr(df)
        spy_close = getattr(self, "_spy_close", None)
        df = self._target(df, spy_close=spy_close)
        n_before = len(df)
        df = df.dropna(subset=[c for c in df.columns if c != "excess_return_5d"])
        df = df.dropna()
        n_after = len(df)
        import logging
        logging.getLogger(__name__).info(
            f"[technical] Done. {n_after} rows after dropna "
            f"({n_before - n_after} dropped for warmup)."
        )
        return df