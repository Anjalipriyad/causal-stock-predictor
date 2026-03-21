"""
earnings.py
-----------
Earnings surprise and post-earnings drift features.

Post-earnings drift (PEAD) is one of the strongest documented
anomalies in finance — stocks drift in the direction of the
earnings surprise for days after the announcement.

Features produced:
    earnings_surprise         — (actual - estimate) / |estimate|
    earnings_surprise_abs     — absolute magnitude of surprise
    days_since_earnings       — days since last earnings report
    earnings_drift_signal     — surprise × time_decay
                                positive = positive surprise fading in
                                negative = negative surprise fading in
    beat_miss                 — 1 if beat, -1 if miss, 0 if no data

Data source: Finnhub /stock/earnings endpoint (free tier)

Usage:
    from ml.src.features.earnings import EarningsFeatures
    earnings = EarningsFeatures()
    df_earnings = earnings.compute(ticker, price_df)
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class EarningsFeatures:
    """
    Computes earnings surprise and post-earnings drift features.
    Fetches earnings data from Finnhub and aligns to daily price index.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg = _load_config(config_path)

        self.finnhub_key   = cfg["finnhub"]["api_key"]
        self.finnhub_base  = cfg["finnhub"]["base_url"]
        self.timeout       = cfg["finnhub"]["timeout_seconds"]
        self.request_delay = cfg["finnhub"]["request_delay_seconds"]

        earnings_cfg       = cfg["features"].get("earnings", {})
        self.decay_days    = earnings_cfg.get("decay_days", 30)
        self.threshold     = earnings_cfg.get("surprise_threshold", 0.05)
        self.enabled       = earnings_cfg.get("enabled", True)

        self.root      = Path(__file__).resolve().parents[3]
        self.cache_dir = self.root / cfg["data"]["raw_dir"] / "earnings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def compute(
        self, ticker: str, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute earnings features aligned to price_df index.

        Args:
            ticker:   Stock ticker e.g. "AAPL"
            price_df: OHLCV DataFrame with DatetimeIndex

        Returns:
            DataFrame with earnings features, same index as price_df
        """
        if not self.enabled:
            return self._empty(price_df.index)

        if not self.finnhub_key:
            logger.warning("[earnings] No FINNHUB_API_KEY — skipping earnings features.")
            return self._empty(price_df.index)

        logger.info(f"[earnings] Computing earnings features for {ticker}...")

        # Load or fetch earnings data
        earnings_df = self._load_or_fetch(ticker)

        if earnings_df is None or earnings_df.empty:
            logger.warning(f"[earnings] No earnings data for {ticker}.")
            return self._empty(price_df.index)

        # Align to price index
        result = self._align_to_index(earnings_df, price_df.index)
        logger.info(f"[earnings] Done. {result.notna().any(axis=1).sum()} days with earnings signal.")
        return result

    # -----------------------------------------------------------------------
    # Private — data fetching
    # -----------------------------------------------------------------------

    def _load_or_fetch(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load from cache or fetch from Finnhub."""
        cache_path = self.cache_dir / f"{ticker}_earnings.csv"

        # Use cache if less than 7 days old
        if cache_path.exists():
            age_days = (datetime.now() - datetime.fromtimestamp(
                cache_path.stat().st_mtime)).days
            if age_days < 7:
                logger.info(f"[earnings] Loading {ticker} earnings from cache.")
                return pd.read_csv(cache_path, parse_dates=["date"])

        return self._fetch_from_finnhub(ticker, cache_path)

    def _fetch_from_finnhub(
        self, ticker: str, cache_path: Path
    ) -> Optional[pd.DataFrame]:
        """Fetch earnings history from Finnhub."""
        try:
            resp = requests.get(
                f"{self.finnhub_base}/stock/earnings",
                params={"symbol": ticker, "limit": 40, "token": self.finnhub_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            time.sleep(self.request_delay)

            if not data:
                return None

            records = []
            for item in data:
                try:
                    actual   = item.get("actual")
                    estimate = item.get("estimate")
                    period   = item.get("period", "")
                    surprise = item.get("surprise")
                    surprise_pct = item.get("surprisePercent")

                    if actual is None or estimate is None:
                        continue

                    # Compute surprise if not provided
                    if surprise_pct is None and estimate != 0:
                        surprise_pct = ((actual - estimate) / abs(estimate)) * 100

                    records.append({
                        "date":         pd.to_datetime(period),
                        "actual_eps":   float(actual),
                        "estimate_eps": float(estimate),
                        "surprise_pct": float(surprise_pct) if surprise_pct else 0.0,
                    })
                except Exception:
                    continue

            if not records:
                return None

            df = pd.DataFrame(records).dropna(subset=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df.to_csv(cache_path, index=False)
            logger.info(f"[earnings] Fetched {len(df)} earnings reports for {ticker}.")
            return df

        except Exception as e:
            logger.warning(f"[earnings] Fetch failed for {ticker}: {e}")
            return None

    # -----------------------------------------------------------------------
    # Private — feature computation
    # -----------------------------------------------------------------------

    def _align_to_index(
        self, earnings_df: pd.DataFrame, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        For each trading day, compute earnings features based on
        the most recent earnings report before that day.
        """
        result = pd.DataFrame(index=index)
        result["earnings_surprise"]     = 0.0
        result["earnings_surprise_abs"] = 0.0
        result["days_since_earnings"]   = np.nan
        result["earnings_drift_signal"] = 0.0
        result["beat_miss"]             = 0.0

        earnings_dates = pd.to_datetime(earnings_df["date"].values)

        for i, trade_date in enumerate(index):
            # Find most recent earnings before this date
            past_earnings = earnings_df[
                pd.to_datetime(earnings_df["date"]) <= trade_date
            ]

            if past_earnings.empty:
                continue

            latest = past_earnings.iloc[-1]
            earnings_date = pd.to_datetime(latest["date"])
            days_since    = (trade_date - earnings_date).days

            if days_since > self.decay_days * 2:
                continue

            surprise_pct = float(latest["surprise_pct"]) / 100

            # Time decay: signal fades linearly over decay_days
            decay = max(0, 1 - days_since / self.decay_days)

            result.at[trade_date, "earnings_surprise"]     = surprise_pct
            result.at[trade_date, "earnings_surprise_abs"] = abs(surprise_pct)
            result.at[trade_date, "days_since_earnings"]   = days_since
            result.at[trade_date, "earnings_drift_signal"] = surprise_pct * decay
            result.at[trade_date, "beat_miss"]             = (
                1.0 if surprise_pct > self.threshold
                else -1.0 if surprise_pct < -self.threshold
                else 0.0
            )

        return result

    def _empty(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return zero-filled DataFrame when earnings data unavailable."""
        return pd.DataFrame({
            "earnings_surprise":     0.0,
            "earnings_surprise_abs": 0.0,
            "days_since_earnings":   np.nan,
            "earnings_drift_signal": 0.0,
            "beat_miss":             0.0,
        }, index=index)