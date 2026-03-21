"""
options.py
----------
Implied volatility features from options market data.

The IV-RV spread (implied minus realised volatility) is one of the
strongest predictors in finance — when options traders expect bigger
moves than recent history, they're usually right.

Features produced:
    iv_30d                — 30-day implied volatility (from ATM options)
    iv_60d                — 60-day implied volatility
    iv_rv_spread_20       — IV(30d) minus realised vol(20d)
    iv_rv_spread_30       — IV(30d) minus realised vol(30d)
    iv_percentile_252     — where IV sits in last 252 days (0-1)
    iv_change_5d          — 5-day change in IV
    iv_term_structure     — IV(60d) minus IV(30d) — slope of vol curve

Data source: yFinance options chain (free)

Usage:
    from ml.src.features.options import OptionsFeatures
    options = OptionsFeatures()
    df_options = options.compute(ticker, price_df)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class OptionsFeatures:
    """
    Computes implied volatility features from yFinance options chain.

    Note: yFinance only provides current options chain, not historical IV.
    For historical IV we use a proxy: reconstruct from put-call prices
    at each date using Black-Scholes approximation.
    For live inference: use current options chain directly.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg = _load_config(config_path)

        options_cfg         = cfg["features"].get("options", {})
        self.enabled        = options_cfg.get("enabled", True)
        self.expiry_windows = options_cfg.get("expiry_windows", [30, 60])
        self.iv_rv_windows  = options_cfg.get("iv_rv_windows", [20, 30])

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def compute_live(self, ticker: str, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute IV features using current options chain.
        Used for live inference only — not historical training.

        Returns single-row DataFrame with current IV features.
        """
        if not self.enabled:
            return self._empty_live()

        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            return self._extract_iv_features(stock, price_df)
        except Exception as e:
            logger.warning(f"[options] Live IV fetch failed for {ticker}: {e}")
            return self._empty_live()

    def compute_historical(
        self, ticker: str, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute historical IV proxy from realised volatility and VIX.

        Since we can't get historical options prices for free, we use
        a VIX-beta proxy — the stock's historical beta to VIX gives
        an estimate of how the stock's IV moved over time.

        This is an approximation. A paid data source (CBOE, OptionMetrics)
        would give exact historical IV.
        """
        if not self.enabled:
            return self._empty_historical(price_df.index)

        try:
            return self._compute_iv_proxy(ticker, price_df)
        except Exception as e:
            logger.warning(f"[options] Historical IV computation failed: {e}")
            return self._empty_historical(price_df.index)

    # -----------------------------------------------------------------------
    # Private — live IV extraction
    # -----------------------------------------------------------------------

    def _extract_iv_features(
        self, stock, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract IV from current options chain."""
        import yfinance as yf

        result = {}

        try:
            expirations = stock.options
            if not expirations:
                return self._empty_live()

            current_price = float(price_df["close"].iloc[-1])
            ivs_by_expiry = {}

            for exp_date in expirations[:4]:   # check first 4 expiries
                try:
                    chain    = stock.option_chain(exp_date)
                    exp_dt   = datetime.strptime(exp_date, "%Y-%m-%d")
                    dte      = (exp_dt - datetime.now()).days

                    if dte < 7:
                        continue

                    # Get ATM options (closest to current price)
                    calls = chain.calls
                    puts  = chain.puts

                    if calls.empty or puts.empty:
                        continue

                    # Find ATM strike
                    calls["dist"] = abs(calls["strike"] - current_price)
                    puts["dist"]  = abs(puts["strike"]  - current_price)

                    atm_call = calls.loc[calls["dist"].idxmin()]
                    atm_put  = puts.loc[puts["dist"].idxmin()]

                    # Average call and put IV
                    call_iv = atm_call.get("impliedVolatility", np.nan)
                    put_iv  = atm_put.get("impliedVolatility",  np.nan)

                    if not np.isnan(call_iv) and not np.isnan(put_iv):
                        ivs_by_expiry[dte] = (call_iv + put_iv) / 2

                except Exception:
                    continue

            if not ivs_by_expiry:
                return self._empty_live()

            # Interpolate to standard 30d and 60d
            dtes = sorted(ivs_by_expiry.keys())
            ivs  = [ivs_by_expiry[d] for d in dtes]

            iv_30d = float(np.interp(30, dtes, ivs)) if min(dtes) <= 30 <= max(dtes) else ivs[0]
            iv_60d = float(np.interp(60, dtes, ivs)) if min(dtes) <= 60 <= max(dtes) else ivs[-1]

            # Realised volatility
            log_ret = np.log(price_df["close"] / price_df["close"].shift(1))
            rv_20   = float(log_ret.rolling(20).std().iloc[-1] * np.sqrt(252))
            rv_30   = float(log_ret.rolling(30).std().iloc[-1] * np.sqrt(252))

            # IV change
            iv_5d_ago = iv_30d   # placeholder — no historical chain

            result = {
                "iv_30d":            iv_30d,
                "iv_60d":            iv_60d,
                "iv_rv_spread_20":   iv_30d - rv_20,
                "iv_rv_spread_30":   iv_30d - rv_30,
                "iv_percentile_252": np.nan,   # requires historical IV
                "iv_change_5d":      0.0,
                "iv_term_structure": iv_60d - iv_30d,
            }

            return pd.DataFrame([result])

        except Exception as e:
            logger.warning(f"[options] IV extraction failed: {e}")
            return self._empty_live()

    # -----------------------------------------------------------------------
    # Private — historical IV proxy
    # -----------------------------------------------------------------------

    def _compute_iv_proxy(
        self, ticker: str, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Historical IV proxy using VIX-beta approach.

        IV_stock ≈ beta_to_vix × VIX + idiosyncratic_vol_premium

        This is approximate but captures the main regime variation in IV.
        """
        import yfinance as yf

        result = pd.DataFrame(index=price_df.index)

        # Fetch VIX
        try:
            vix = yf.download(
                "^VIX",
                start=str(price_df.index.min().date()),
                end=str(price_df.index.max().date()),
                progress=False,
                auto_adjust=True,
                multi_level_index=False,
            )["close"].reindex(price_df.index).ffill()
        except Exception:
            return self._empty_historical(price_df.index)

        # Realised volatility
        log_ret = np.log(price_df["close"] / price_df["close"].shift(1))
        rv_20   = log_ret.rolling(20).std() * np.sqrt(252)
        rv_30   = log_ret.rolling(30).std() * np.sqrt(252)

        # VIX-based IV proxy (VIX is in percentage points, convert to decimal)
        vix_decimal = vix / 100

        # Rolling beta of stock vol to VIX
        vix_returns   = vix_decimal.pct_change()
        stock_vol_chg = rv_20.pct_change()
        beta_to_vix   = stock_vol_chg.rolling(60).cov(vix_returns) / \
                        vix_returns.rolling(60).var()
        beta_to_vix   = beta_to_vix.clip(0.5, 3.0)   # reasonable bounds

        # IV proxy
        iv_30d_proxy = vix_decimal * beta_to_vix
        iv_30d_proxy = iv_30d_proxy.fillna(rv_20)

        # Features
        result["iv_30d"]            = iv_30d_proxy
        result["iv_60d"]            = iv_30d_proxy.rolling(5).mean()  # smoothed
        result["iv_rv_spread_20"]   = iv_30d_proxy - rv_20
        result["iv_rv_spread_30"]   = iv_30d_proxy - rv_30
        result["iv_percentile_252"] = iv_30d_proxy.rolling(252).rank(pct=True)
        result["iv_change_5d"]      = iv_30d_proxy.diff(5)
        result["iv_term_structure"] = result["iv_60d"] - result["iv_30d"]

        return result.fillna(0)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _empty_live(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "iv_30d":            0.0,
            "iv_60d":            0.0,
            "iv_rv_spread_20":   0.0,
            "iv_rv_spread_30":   0.0,
            "iv_percentile_252": 0.0,
            "iv_change_5d":      0.0,
            "iv_term_structure": 0.0,
        }])

    def _empty_historical(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        cols = ["iv_30d","iv_60d","iv_rv_spread_20","iv_rv_spread_30",
                "iv_percentile_252","iv_change_5d","iv_term_structure"]
        return pd.DataFrame(0.0, index=index, columns=cols)