"""
options.py  (CORRECTED)
-----------------------
Replaces the original options.py.

Key fix: adds a mode guard that prevents compute_live() output from
ever being used in a multi-row historical context.

Original bug in pipeline.py:
    options_live = self.options.compute_live(ticker, price_df)
    options_df = pd.DataFrame(
        [options_live.iloc[0].to_dict()] * len(tech_df),  # ← LOOKAHEAD BUG
        index=tech_df.index
    )

This broadcasts today's implied volatility to ALL historical rows,
meaning every row in a backtesting context would have future IV.
The fix: raise an error if compute_live() output is ever replicated
across multiple rows, and enforce strict mode separation.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)

# ── Lookahead guard sentinel ───────────────────────────────────────────────
_LIVE_IV_SENTINEL_COL = "_live_iv_do_not_broadcast"


class OptionsFeatures:
    """
    Implied volatility features with strict live/historical mode separation.

    CRITICAL DESIGN RULE:
        compute_live()       → returns a SINGLE ROW with today's IV
                               MUST NOT be replicated to multiple rows
        compute_historical() → returns a FULL SERIES of historical IV proxy
                               Safe to use in any backtesting context

    The _check_no_live_broadcast() method raises if code tries to use
    live IV in a multi-row historical context.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg = _load_config(config_path)

        options_cfg         = cfg["features"].get("options", {})
        self.enabled        = options_cfg.get("enabled", True)
        self.expiry_windows = options_cfg.get("expiry_windows", [30, 60])
        self.iv_rv_windows  = options_cfg.get("iv_rv_windows", [20, 30])

    # -----------------------------------------------------------------------
    # Live inference — returns SINGLE ROW only
    # -----------------------------------------------------------------------

    def compute_live(self, ticker: str, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute IV features using current options chain.
        Returns a SINGLE ROW DataFrame.

        IMPORTANT: The returned DataFrame contains a sentinel column
        _LIVE_IV_SENTINEL_COL that will trigger an assertion if this
        single-row result is broadcast to multiple rows (which would
        create a lookahead bias in backtesting).

        To use in pipeline.build_live(), call:
            options_row = options.compute_live(ticker, price_df)
            # Correct: join to a single-row inference DataFrame only
            # WRONG: replicate to len(tech_df) rows
        """
        if not self.enabled:
            result = self._empty_live()
        else:
            try:
                import yfinance as yf
                stock  = yf.Ticker(ticker)
                result = self._extract_iv_features(stock, price_df)
            except Exception as e:
                logger.warning(f"[options] Live IV fetch failed for {ticker}: {e}")
                result = self._empty_live()

        # Add sentinel — broadcasts of this row will be caught
        result[_LIVE_IV_SENTINEL_COL] = 1
        return result

    # -----------------------------------------------------------------------
    # Historical — returns FULL SERIES, safe for backtesting
    # -----------------------------------------------------------------------

    def compute_historical(
        self, ticker: str, price_df: pd.DataFrame, vix_series: pd.Series = None
    ) -> pd.DataFrame:
        """
        Compute historical IV proxy from realised volatility and VIX.
        Returns full-length DataFrame — safe for all backtesting uses.

        CRITICAL FIX: vix_series must be passed from the data loader to
        avoid look-ahead bias. Never download data inside feature compute.
        """
        if not self.enabled:
            return self._empty_historical(price_df.index)
        try:
            return self._compute_iv_proxy(ticker, price_df, vix_series)
        except Exception as e:
            logger.warning(f"[options] Historical IV computation failed: {e}")
            return self._empty_historical(price_df.index)

    # -----------------------------------------------------------------------
    # Guard: catch lookahead bias at the source
    # -----------------------------------------------------------------------

    @staticmethod
    def check_no_live_broadcast(df: pd.DataFrame, context: str = "") -> None:
        """
        Assert that a live IV row has NOT been replicated across multiple rows.

        Call this in pipeline.py before joining options_df to the main matrix.
        If triggered, it means compute_live() output is being used in a
        multi-row context, which would give future IV to historical rows.

        Args:
            df:      The DataFrame to check
            context: Description of where the check is called (for error message)

        Raises:
            ValueError if live IV sentinel is found in a multi-row DataFrame.
        """
        if _LIVE_IV_SENTINEL_COL not in df.columns:
            return   # no live IV present — safe

        if len(df) > 1:
            raise ValueError(
                f"[options] LOOKAHEAD BIAS DETECTED {context}: "
                f"Live IV (compute_live output, containing '{_LIVE_IV_SENTINEL_COL}') "
                f"has been replicated to {len(df)} rows. "
                f"This assigns future implied volatility to historical rows. "
                f"Use compute_historical() for multi-row DataFrames."
            )
        # Single row is fine — strip the sentinel before returning
        df.drop(columns=[_LIVE_IV_SENTINEL_COL], inplace=True, errors="ignore")

    # -----------------------------------------------------------------------
    # Private — live IV extraction (same as original)
    # -----------------------------------------------------------------------

    def _extract_iv_features(self, stock, price_df: pd.DataFrame) -> pd.DataFrame:
        """Extract IV from current options chain."""
        try:
            expirations   = stock.options
            if not expirations:
                return self._empty_live()

            current_price = float(price_df["close"].iloc[-1])
            ivs_by_expiry = {}

            for exp_date in expirations[:4]:
                try:
                    chain  = stock.option_chain(exp_date)
                    exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                    dte    = (exp_dt - datetime.now()).days
                    if dte < 7:
                        continue

                    calls = chain.calls
                    puts  = chain.puts
                    if calls.empty or puts.empty:
                        continue

                    calls = calls.copy()
                    puts  = puts.copy()
                    calls["dist"] = abs(calls["strike"] - current_price)
                    puts["dist"]  = abs(puts["strike"]  - current_price)

                    atm_call = calls.loc[calls["dist"].idxmin()]
                    atm_put  = puts.loc[puts["dist"].idxmin()]

                    call_iv = atm_call.get("impliedVolatility", np.nan)
                    put_iv  = atm_put.get("impliedVolatility",  np.nan)

                    if not np.isnan(call_iv) and not np.isnan(put_iv):
                        ivs_by_expiry[dte] = (call_iv + put_iv) / 2
                except Exception:
                    continue

            if not ivs_by_expiry:
                return self._empty_live()

            dtes  = sorted(ivs_by_expiry.keys())
            ivs   = [ivs_by_expiry[d] for d in dtes]

            iv_30d = float(np.interp(30, dtes, ivs)) if min(dtes) <= 30 <= max(dtes) else ivs[0]
            iv_60d = float(np.interp(60, dtes, ivs)) if min(dtes) <= 60 <= max(dtes) else ivs[-1]

            log_ret = np.log(price_df["close"] / price_df["close"].shift(1))
            rv_20   = float(log_ret.rolling(20).std().iloc[-1] * np.sqrt(252))
            rv_30   = float(log_ret.rolling(30).std().iloc[-1] * np.sqrt(252))

            return pd.DataFrame([{
                "iv_30d":            iv_30d,
                "iv_60d":            iv_60d,
                "iv_rv_spread_20":   iv_30d - rv_20,
                "iv_rv_spread_30":   iv_30d - rv_30,
                "iv_percentile_252": np.nan,
                "iv_change_5d":      0.0,
                "iv_term_structure": iv_60d - iv_30d,
            }])

        except Exception as e:
            logger.warning(f"[options] IV extraction failed: {e}")
            return self._empty_live()

    # -----------------------------------------------------------------------
    # Private — historical IV proxy (same as original)
    # -----------------------------------------------------------------------

    def _compute_iv_proxy(self, ticker: str, price_df: pd.DataFrame, vix_series: pd.Series = None) -> pd.DataFrame:
        """Historical IV proxy using VIX-beta approach."""
        result = pd.DataFrame(index=price_df.index)

        if vix_series is not None:
            vix = vix_series.reindex(price_df.index).ffill()
        else:
            # Fallback: if VIX not provided, warn and return zeros for proxy-linked cols
            logger.warning(f"[options] No VIX series provided for {ticker} IV proxy")
            return self._empty_historical(price_df.index)

        log_ret = np.log(price_df["close"] / price_df["close"].shift(1))
        rv_20   = log_ret.rolling(20).std() * np.sqrt(252)
        rv_30   = log_ret.rolling(30).std() * np.sqrt(252)

        vix_decimal   = vix / 100
        vix_returns   = vix_decimal.pct_change()
        stock_vol_chg = rv_20.pct_change()
        beta_to_vix   = (
            stock_vol_chg.rolling(60).cov(vix_returns) /
            vix_returns.rolling(60).var()
        ).clip(0.5, 3.0)

        iv_30d_proxy = vix_decimal * beta_to_vix
        iv_30d_proxy = iv_30d_proxy.fillna(rv_20)

        result["iv_30d"]            = iv_30d_proxy
        result["iv_60d"]            = iv_30d_proxy.rolling(5).mean()
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
            "iv_30d": 0.0, "iv_60d": 0.0,
            "iv_rv_spread_20": 0.0, "iv_rv_spread_30": 0.0,
            "iv_percentile_252": 0.0, "iv_change_5d": 0.0,
            "iv_term_structure": 0.0,
        }])

    def _empty_historical(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        cols = ["iv_30d","iv_60d","iv_rv_spread_20","iv_rv_spread_30",
                "iv_percentile_252","iv_change_5d","iv_term_structure"]
        return pd.DataFrame(0.0, index=index, columns=cols)