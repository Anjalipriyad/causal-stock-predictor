"""
macro.py
--------
Computes macro and cross-asset features from raw macro DataFrames.
All macro tickers are loaded individually by the DataLoader and merged here.

Features produced:
    VIX
        vix_level                   — raw VIX close
        vix_change_1d               — 1-day change in VIX
        vix_change_5d               — 5-day change in VIX
        vix_ma_5                    — 5-day rolling mean of VIX
        vix_ma_10                   — 10-day rolling mean of VIX

    Yield curve
        yield_10y                   — 10Y treasury yield
        yield_3m                    — 3M treasury yield
        yield_spread                — 10Y - 3M (inversion = recession signal)
        yield_spread_change_1d      — daily change in spread

    Dollar index
        dxy_return_1d               — 1-day log return of DXY

    Commodities
        gold_return_1d              — 1-day log return of Gold
        oil_return_1d               — 1-day log return of Oil

    S&P 500
        sp500_return_1d             — 1-day log return of S&P 500
        sp500_ma_5                  — 5-day rolling mean return
        sp500_ma_10                 — 10-day rolling mean return

    Sector ETFs (for each of XLK, XLF, XLE, XLV, XLI)
        {etf}_return_1d             — 1-day log return
        {etf}_ma_5                  — 5-day rolling mean return

Usage:
    from ml.src.features.macro import MacroFeatures
    macro = MacroFeatures()
    df_macro = macro.compute(macro_dict)
    # macro_dict = {"^VIX": df_vix, "^TNX": df_tnx, ...}
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class MacroFeatures:
    """
    Computes macro features from a dict of raw macro DataFrames.

    Input:  dict mapping symbol → DataFrame (each with DatetimeIndex + 'close' column)
    Output: single merged DataFrame of all macro features aligned to trading days
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg   = _load_config(config_path)
        macro = cfg["features"]["macro"]

        self.vix_change_windows  = macro["vix_change_windows"]
        self.yield_spread        = macro["yield_spread"]
        self.cross_asset_returns = macro["cross_asset_returns"]
        self.rolling_windows     = macro["macro_rolling_windows"]
        self.sector_etfs         = cfg["data"]["tickers"]["sector_etfs"]

        # Symbol map — safe filenames used as keys by DataLoader
        self.VIX    = "^VIX"
        self.TNX    = "^TNX"
        self.IRX    = "^IRX"
        self.DXY    = "DX-Y.NYB"
        self.GOLD   = "GC=F"
        self.OIL    = "CL=F"
        self.SP500  = "^GSPC"

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def compute(self, macro_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Build the full macro feature DataFrame from raw macro data.

        Args:
            macro_dict: {symbol: DataFrame} as returned by DataLoader.read_macro()

        Returns:
            DataFrame with DatetimeIndex and all macro feature columns.
        """
        logger.info("[macro] Computing macro features ...")
        frames = []

        frames.append(self._vix_features(macro_dict))
        frames.append(self._yield_features(macro_dict))
        frames.append(self._cross_asset_features(macro_dict))
        frames.append(self._sector_features(macro_dict))

        # Merge all on date index
        result = pd.concat(
            [f for f in frames if f is not None and not f.empty],
            axis=1,
        )
        result = result.sort_index()

        n_before = len(result)
        result   = result.dropna(how="all")
        logger.info(
            f"[macro] Done. {len(result)} rows "
            f"({n_before - len(result)} all-NaN rows dropped)."
        )
        return result

    def feature_names(self) -> list[str]:
        """Return the list of macro feature column names produced."""
        names = [
            "vix_level", "vix_change_1d", "vix_change_5d",
            "vix_ma_5", "vix_ma_10",
            "yield_10y", "yield_3m", "yield_spread", "yield_spread_change_1d",
            "dxy_return_1d",
            "gold_return_1d",
            "oil_return_1d",
            "sp500_return_1d", "sp500_ma_5", "sp500_ma_10",
        ]
        for etf in self.sector_etfs:
            names += [f"{etf.lower()}_return_1d", f"{etf.lower()}_ma_5"]
        return names

    # -----------------------------------------------------------------------
    # Private — feature groups
    # -----------------------------------------------------------------------

    def _vix_features(
        self, macro_dict: dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """VIX level, changes, and rolling means."""
        df = self._get_close(macro_dict, self.VIX, "vix_level")
        if df is None:
            logger.warning("[macro] VIX data not found — skipping VIX features.")
            return None

        for w in self.vix_change_windows:
            df[f"vix_change_{w}d"] = df["vix_level"].diff(periods=w)

        for w in self.rolling_windows:
            df[f"vix_ma_{w}"] = df["vix_level"].rolling(window=w).mean()

        return df

    def _yield_features(
        self, macro_dict: dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """10Y yield, 3M yield, spread, and spread change."""
        tnx = self._get_close(macro_dict, self.TNX, "yield_10y")
        irx = self._get_close(macro_dict, self.IRX, "yield_3m")

        frames = [f for f in [tnx, irx] if f is not None]
        if not frames:
            logger.warning("[macro] No yield data found — skipping yield features.")
            return None

        result = pd.concat(frames, axis=1)

        if self.yield_spread and "yield_10y" in result and "yield_3m" in result:
            result["yield_spread"] = result["yield_10y"] - result["yield_3m"]
            result["yield_spread_change_1d"] = result["yield_spread"].diff(1)

        return result

    def _cross_asset_features(
        self, macro_dict: dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Log returns for DXY, Gold, Oil, S&P 500."""
        if not self.cross_asset_returns:
            return None

        assets = {
            "dxy_return_1d":   (self.DXY,   None),
            "gold_return_1d":  (self.GOLD,  None),
            "oil_return_1d":   (self.OIL,   None),
            "sp500_return_1d": (self.SP500, self.rolling_windows),
        }

        frames = []
        for feat_name, (symbol, roll_windows) in assets.items():
            raw = self._get_close(macro_dict, symbol)
            if raw is None:
                logger.warning(f"[macro] {symbol} not found — skipping {feat_name}.")
                continue

            col_name = list(raw.columns)[0]
            ret = np.log(raw[col_name] / raw[col_name].shift(1)).rename(feat_name)
            f   = ret.to_frame()

            if roll_windows:
                prefix = feat_name.replace("_return_1d", "")
                for w in roll_windows:
                    f[f"{prefix}_ma_{w}"] = ret.rolling(window=w).mean()

            frames.append(f)

        if not frames:
            return None
        return pd.concat(frames, axis=1)

    def _sector_features(
        self, macro_dict: dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """1-day log return + 5-day rolling mean for each sector ETF."""
        frames = []
        for etf in self.sector_etfs:
            raw = self._get_close(macro_dict, etf)
            if raw is None:
                logger.warning(f"[macro] {etf} not found — skipping sector feature.")
                continue

            col_name  = list(raw.columns)[0]
            etf_lower = etf.lower()
            ret       = np.log(raw[col_name] / raw[col_name].shift(1))

            f = pd.DataFrame({
                f"{etf_lower}_return_1d": ret,
                f"{etf_lower}_ma_5":      ret.rolling(window=5).mean(),
            })
            frames.append(f)

        if not frames:
            return None
        return pd.concat(frames, axis=1)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_close(
        self,
        macro_dict: dict[str, pd.DataFrame],
        symbol: str,
        rename: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Extract the 'close' column for a symbol from macro_dict.
        Returns a single-column DataFrame, optionally renamed.
        """
        df = macro_dict.get(symbol)
        if df is None or df.empty:
            return None
        if "close" not in df.columns:
            return None

        series = df["close"].copy()
        series.name = rename or symbol
        return series.to_frame()